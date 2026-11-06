"""
Edited protocols — an answer written again after the user edited its protocol (the 🔒 switches of the answer page).

The protocol of an answer is its search (concepts, PubMed and ClinicalTrials.gov queries) and the protocol of each
MIMIC-IV study (study design, exploratory SQL, analysis code). An edit changes one of them::

    {"part": "search", "concepts": [...] or None, "question_type": str or None, "queries": {...} or None}
    {"part": "rwd", "study": <study key>, "kind": "design", "spec": {...}}
    {"part": "rwd", "study": <study key>, "kind": "sql", "sql": "SELECT ..."}
    {"part": "rwd", "study": <study key>, "kind": "code", "code": "..."}

Search edits: ``concepts`` are the planner's concept groups (role, label, kind, synonyms); the PubMed and
ClinicalTrials.gov queries are built from them as the planner builds them, unless ``queries`` gives them as written
by hand ({"pubmed_query", "relaxed": [...], "facets": {name: query}, "trial_terms": {...}}).

``iter_rerun`` runs only what the edit changes — the sources whose queries or concepts changed, or the edited study —
keeps the evidence of the other sources from the answer that was edited, and writes the answer again from all of it
(the events of ``pipeline_tools.iter_answer``). The new answer records its protocol in ``MESResult.protocol``: the
version, the changes (before / after, a diff of SQL or code), what ran again and what was kept, the checks the edited
part went through, the versions it descends from, and how the answer changed against the edited one. Edited answers
are not cached.
"""

import copy
import difflib
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime

from mes import config
from mes.tools import rwd_tools, trials_tools
from mes.tools.planner_tools import PLAN_SCHEMA, QUESTION_TYPES, ROLES, build_pubmed_queries, build_trial_terms
from mes.tools.ranking_tools import encode
from mes.tools.result_schema import QueryPlan

CONCEPT_KINDS = tuple(PLAN_SCHEMA["properties"]["concepts"]["items"]["properties"]["kind"]["enum"])
TRIAL_FIELDS = ("condition", "intervention", "outcome", "term")
SOURCE_NAMES = {"literature": "PubMed", "pdfs": "PDFs", "trials": "ClinicalTrials.gov", "kg": "iBKH",
                "rwd": rwd_tools.DATABASE}
NOUNS = {"literature": ("article", "articles"), "pdfs": ("PDF passage", "PDF passages"),
         "trials": ("trial record", "trial records"), "kg": ("relation", "relations"), "rwd": ("study", "studies")}
LABELS = {"search": "Search protocol", "design": "Study design", "sql": "SQL query", "code": "Analysis code"}
MAX_CONCEPTS = 12
MAX_QUERY = 4000
MAX_DIFF = 20000
CHECKS = {
    "design": "The edited design was validated against the MIMIC-IV dictionary and compiled to SQL by the study "
              "templates, which ran it with their diagnostics.",
    "sql": "The edited query passed MES's checks (one read-only SELECT over the documented tables, aggregated "
           "output, counts below 10 suppressed). It was not compared with an independently written query.",
    "code": "The edited code ran in a fresh sandboxed session under the analyst's code policy; its result passed the "
            "result checks and the code check looked up its codes and names again. It was not reviewed "
            "independently after the edit.",
    "queries": "The queries written by hand ran as written; the concepts still guide the ranking, the knowledge "
               "graph and the answer.",
    "unreproduced": "The analysis this code was edited from gave other numbers in a fresh session (it depends on "
                    "state or randomness): estimates can differ from v{v} without being edited.",
}


# ============================================================================ search protocol
def concepts_of(plan):
    """The concepts of a plan (a result's plan dict) as the planner returns them: role, label, kind, synonyms."""
    return [{"role": e.get("role") if e.get("role") in ROLES else "population", "label": e.get("name", ""),
             "kind": e.get("type") if e.get("type") in CONCEPT_KINDS else "other",
             "synonyms": [str(s) for s in e.get("synonyms") or []]} for e in plan.get("entities") or []]


def _norm(text, limit=MAX_QUERY + 1):
    return " ".join(str(text or "").split())[:limit]


def _clean_concepts(concepts):
    """(concepts, errors) of concept rows from the editor: empty rows dropped, synonyms split at ";"."""
    out, errors = [], []
    for n, c in enumerate(concepts or [], start=1):
        if not isinstance(c, dict):
            continue
        synonyms = c.get("synonyms") or []
        if isinstance(synonyms, str):
            synonyms = re.split(r"[;\n]", synonyms)
        synonyms = list(dict.fromkeys(s for s in (_norm(x, 200) for x in synonyms) if s))[:10]
        label = _norm(c.get("label"), 200)
        if not label:
            if synonyms:
                errors.append(f"Concept {n} has synonyms but no name")
            continue
        if c.get("role") not in ROLES:
            errors.append(f"Concept “{label}”: choose its role ({', '.join(ROLES)})")
            continue
        out.append({"role": c["role"], "label": label, "kind": c.get("kind") if c.get("kind") in CONCEPT_KINDS
                    else "other", "synonyms": [s for s in synonyms if s.lower() != label.lower()]})
    if len(out) > MAX_CONCEPTS:
        errors.append(f"At most {MAX_CONCEPTS} concepts ({len(out)} given)")
    return out, errors


def query_problem(query, name):
    """Why a PubMed / ClinicalTrials.gov query cannot be run as written ("" when it can)."""
    if len(query) > MAX_QUERY:
        return f"{name} is too long ({len(query):,} characters, at most {MAX_QUERY:,})"
    depth = 0
    for ch in re.sub(r'"[^"]*"', "", query):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return f"{name}: a closing parenthesis has no opening one"
    if depth:
        return f"{name}: an opening parenthesis is not closed"
    if query.count('"') % 2:
        return f"{name}: a quote is not closed"
    return ""


def _trial_terms(terms):
    """ClinicalTrials.gov terms as they are searched: known fields, spaces collapsed, empty ones left out."""
    return {k: _norm(terms[k]) for k in TRIAL_FIELDS if _norm(terms.get(k))}


def _rebase(query, old, new):
    """A facet query (main concepts AND focus terms) with the main concepts replaced."""
    if old and query.startswith(old):
        rest = query[len(old):]
        if not new:
            return rest[5:] if rest.startswith(" AND ") else rest.strip()
        return new + rest
    if not old and new:
        return f"{new} AND {query}" if query else new
    return query


def _main_query(concepts):
    return build_pubmed_queries([c for c in concepts if c["role"] in ("population", "intervention")])[0]


def _concept_text(c):
    return f"{c['role']}: {c['label']}" + (f" ({'; '.join(c['synonyms'])})" if c["synonyms"] else "")


def plan_changes(base, new):
    """[{what, before, after}] between two QueryPlans."""
    changes = []

    def add(what, before, after):
        if before != after:
            changes.append({"what": what, "before": before, "after": after})
    add("Question type", base.question_type, new.question_type)
    old = {c["label"].lower(): c for c in _clean_concepts(concepts_of(asdict(base)))[0]}
    now = {c["label"].lower(): c for c in _clean_concepts(concepts_of(asdict(new)))[0]}
    for key, c in now.items():
        if key not in old:
            add("Concept added", "", _concept_text(c))
        elif c != old[key]:
            add("Concept changed", _concept_text(old[key]), _concept_text(c))
    for key, c in old.items():
        if key not in now:
            add("Concept removed", _concept_text(c), "")
    add("PubMed query", base.pubmed_query, new.pubmed_query)
    add("Relaxed PubMed queries", "\n".join(base.pubmed_tiers[1:]), "\n".join(new.pubmed_tiers[1:]))
    for f_old, f_new in zip(base.facets, new.facets):
        add(f"Facet query: {f_new.get('name', '')}", f_old.get("pubmed_query", ""), f_new.get("pubmed_query", ""))
    for key in [k for k in TRIAL_FIELDS if k in base.trial_terms or k in new.trial_terms]:
        add(f"ClinicalTrials.gov {key}", base.trial_terms.get(key, ""), new.trial_terms.get(key, ""))
    return changes


def search_edit_plan(plan, edit, sources, question=""):
    """
    (QueryPlan, changes, sources to search again, errors) of ``plan`` (a result's plan dict) with the search ``edit``.
    PubMed runs again when its queries or the concepts change (they also choose the head-to-head and most-cited
    queries and the ranking terms), ClinicalTrials.gov when its terms change, iBKH when the concepts change.
    ``question``: the answer's question, for a stored plan without its own (an old or foreign record).
    """
    fields = {k: v for k, v in (plan or {}).items() if k in QueryPlan.__dataclass_fields__}
    fields["question"] = fields.get("question") or question or ""
    base = QueryPlan(**fields)
    new = copy.deepcopy(base)
    errors = []
    old_concepts = concepts_of(plan or {})
    concepts, concepts_edited = old_concepts, False
    if edit.get("concepts") is not None:
        cleaned, errors = _clean_concepts(edit["concepts"])
        if cleaned != _clean_concepts(old_concepts)[0]:       # concepts given back unchanged leave the plan as it is
            concepts, concepts_edited = cleaned, True
            new.entities = [{"name": c["label"], "type": c["kind"], "role": c["role"], "synonyms": c["synonyms"]}
                            for c in concepts]
            new.pico = {role: "; ".join(c["label"] for c in concepts if c["role"] == role) for role in ROLES}
            new.fallback = False
    if edit.get("question_type"):
        if edit["question_type"] in QUESTION_TYPES:
            new.question_type = edit["question_type"]
        else:
            errors.append(f"Unknown question type “{_norm(edit['question_type'], 40)}”")
    queries = edit.get("queries")
    if isinstance(queries, dict):                  # written by hand; what is given back unchanged stays as it was
        main = _norm(queries.get("pubmed_query"))
        relaxed = [q for q in (_norm(x) for x in queries.get("relaxed") or []) if q]
        new.pubmed_query = base.pubmed_query if main == _norm(base.pubmed_query) else main
        tiers = list(dict.fromkeys(q for q in [main] + relaxed if q))
        if tiers != [_norm(t) for t in base.pubmed_tiers]:
            new.pubmed_tiers, new.pubmed_query_broad = tiers, (tiers[-1] if tiers else "")
        facet_queries = queries.get("facets") or {}
        new.facets = [{**f, "pubmed_query": q if q != _norm(f.get("pubmed_query")) else f.get("pubmed_query", "")}
                      for f, q in ((f, _norm(facet_queries.get(f.get("name"), f.get("pubmed_query", ""))))
                                   for f in base.facets)]
        terms = _trial_terms(queries.get("trial_terms") or {})
        if terms != _trial_terms(base.trial_terms):
            new.trial_terms = terms
        named = [("The PubMed query", main)] + [(f"Relaxed query {n}", q) for n, q in enumerate(relaxed, start=1)] \
            + [(f"The facet query “{f.get('name')}”", f["pubmed_query"]) for f in new.facets] \
            + [(f"The ClinicalTrials.gov {k}", v) for k, v in new.trial_terms.items()]
        errors += [p for p in (query_problem(q, name) for name, q in named) if p]
    elif concepts_edited:                          # built from the concepts, as the planner builds them
        new.pubmed_query, new.pubmed_query_broad, new.pubmed_tiers = build_pubmed_queries(concepts)
        new.trial_terms = build_trial_terms(concepts)
        old_main, new_main = _main_query(old_concepts), _main_query(concepts)
        new.facets = [{**f, "pubmed_query": _rebase(f.get("pubmed_query", ""), old_main, new_main)}
                      for f in base.facets]
    # an edit may not leave a searched source without its query (one the plan had empty already is not the edit's)
    if "literature" in sources and not new.pubmed_query and new.pubmed_query != base.pubmed_query:
        errors.append("PubMed needs a query: add a population or intervention concept, or write the query yourself")
    if "trials" in sources and not _trial_terms(new.trial_terms) and _trial_terms(base.trial_terms):
        errors.append("ClinicalTrials.gov needs a condition, intervention or term")
    changes = plan_changes(base, new)
    concepts_changed = base.entities != new.entities
    pubmed_changed = (base.pubmed_query, base.pubmed_tiers, [f.get("pubmed_query") for f in base.facets]) != \
        (new.pubmed_query, new.pubmed_tiers, [f.get("pubmed_query") for f in new.facets])
    rerun = [s for s, changed in (("literature", pubmed_changed or concepts_changed),
                                  ("trials", _trial_terms(base.trial_terms) != _trial_terms(new.trial_terms)),
                                  ("kg", concepts_changed))
             if changed and s in sources]
    return new, changes, rerun, errors


# ============================================================================ study protocols
DESIGN_FIELDS = (("analysis", "Analysis"), ("base", "Population"), ("include", "Inclusion"), ("exclude", "Exclusion"),
                 ("exposure", "Exposure / treatment"), ("comparator", "Comparator"), ("outcome", "Outcome"),
                 ("grace_h", "Grace period (h)"), ("covariates", "Covariates"), ("stratify", "Break down by"),
                 ("variable", "Variable"), ("sql_question", "Question of the exploratory query"),
                 ("profile", "Patient"), ("k", "Similar patients"), ("new_user", "New users only"))


def _plain_criteria(value):
    """Criteria without their display labels (labels are filled in by the validation)."""
    if isinstance(value, list):
        return [_plain_criteria(v) for v in value]
    if isinstance(value, dict) and "concept" in value:
        return {k: v for k, v in value.items() if k != "label"}
    return value


def _labelled(c):
    """A criterion with its concept's name (a design being edited has none yet: the validation fills them in)."""
    from mes.tools.rwd.spec import Criterion
    c = Criterion.from_dict(c)
    if not c.label:
        try:
            from mes.tools.rwd.concepts import get_index
            c.label = get_index().label(c.concept)
        except Exception:                    # no store here: the concept id stands
            pass
    return c


def _design_text(key, value):
    from mes.tools.rwd.spec import criterion_text
    if value in (None, "", []):
        return "—"
    if key in ("include", "exclude"):
        return "; ".join(criterion_text(_labelled(c)) for c in value)
    if key in ("exposure", "comparator"):
        return criterion_text(_labelled(value))
    from mes.tools.rwd import library as L
    names = {"base": lambda v: L.BASES.get(v, {}).get("label", v), "outcome": lambda v: L.OUTCOMES.get(v, (v,))[0],
             "variable": lambda v: L.VARIABLES.get(v, (v,))[0]}
    if key in ("covariates", "stratify"):
        return ", ".join(L.VARIABLES.get(v, (v,))[0] for v in value)
    if key == "profile":
        from mes.tools.rwd.report import profile_text
        return profile_text({"profile": value}) or "—"
    if key == "new_user":
        return "yes" if value else "no"
    if key in names:
        return names[key](value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def design_changes(before, after):
    """[{what, before, after}] between two study designs (spec dicts)."""
    from mes.tools.rwd.spec import StudySpec
    a, b = StudySpec.from_dict(before).to_dict(), StudySpec.from_dict(after).to_dict()
    return [{"what": label, "before": _design_text(key, a.get(key)), "after": _design_text(key, b.get(key))}
            for key, label in DESIGN_FIELDS if _plain_criteria(a.get(key)) != _plain_criteria(b.get(key))]


def _diff(before, after, label):
    lines = difflib.unified_diff(str(before or "").splitlines(), str(after or "").splitlines(), f"{label} before",
                                 f"{label} after", lineterm="", n=2)
    return "\n".join(lines)[:MAX_DIFF]


def _find_study(base, key):
    return next((s for s in base.rwd_studies or [] if s.get("key") == key), None)


# ============================================================================ an edit
def prepare(base, edit, sources=None):
    """
    What an ``edit`` of the answer ``base`` changes: {plan, changes, rerun, diff, checks, label, study, summary}.
    Raises ValueError when it cannot be run (nothing changed, invalid concepts or queries, a study that is not in
    the answer or cannot be edited that way).
    """
    sources = [s for s in (sources or base.sources or []) if s in SOURCE_NAMES]
    plan_dict = base.plan or {}
    part = (edit or {}).get("part")
    out = {"diff": "", "checks": [], "study": None}
    if part == "search":
        plan, changes, rerun, errors = search_edit_plan(plan_dict, edit, sources, base.question)
        from mes.tools.pipeline_tools import LITERATURE_PARTS
        if "pubmed" not in (base.literature or LITERATURE_PARTS):      # an answer from the PDFs alone stays so
            rerun = [s for s in rerun if s != "literature"]
        if errors:
            raise ValueError("; ".join(errors))
        if not changes:
            raise ValueError("Nothing was changed in the search protocol")
        out.update(plan=plan, changes=changes, rerun=rerun, label=LABELS["search"],
                   checks=[CHECKS["queries"]] if isinstance(edit.get("queries"), dict) else [])
    elif part == "rwd":
        kind = edit.get("kind")
        study = _find_study(base, edit.get("study"))
        if study is None:
            raise ValueError("The study to edit is not part of this answer")
        problem = rwd_tools.editable(study, kind)
        if problem:
            raise ValueError(problem[0].upper() + problem[1:])
        if "rwd" not in sources:
            raise ValueError("Real-world data is not among the sources of this answer")
        if kind == "design":
            changes = design_changes(study["spec"], edit.get("spec") or {})
            if not changes:
                raise ValueError("Nothing was changed in the study design")
        elif kind == "sql":
            before, after = str(study.get("sql") or "").strip(), str(edit.get("sql") or "").strip().rstrip(";").strip()
            if " ".join(before.rstrip(";").split()) == " ".join(after.split()):
                raise ValueError("Nothing was changed in the SQL query")
            changes = [{"what": "SQL query", "before": before, "after": after}]
            out["diff"] = _diff(before, after, "query")
        else:
            before, after = str(study.get("code") or ""), str(edit.get("code") or "")
            if before.strip() == after.strip():
                raise ValueError("Nothing was changed in the analysis code")
            out["diff"] = _diff(before, after, "analysis.py")
            added = sum(1 for line in out["diff"].splitlines() if line.startswith("+") and not line.startswith("+++"))
            removed = sum(1 for line in out["diff"].splitlines() if line.startswith("-") and not line.startswith("---"))
            changes = [{"what": "Analysis code", "before": f"{len(before.splitlines())} lines",
                        "after": f"{len(after.splitlines())} lines (+{added} −{removed})"}]
        checks = [CHECKS[kind]]
        if kind == "code" and study.get("reproduction") and not study["reproduction"].get("reproduced"):
            checks.append(CHECKS["unreproduced"].format(v=int((base.protocol or {}).get("version") or 1)))
        # a stored plan without its question (an old or foreign record) takes the answer's question
        plan_fields = {"question": base.question, **{k: v for k, v in plan_dict.items() if k in QueryPlan.__dataclass_fields__}}
        out.update(plan=QueryPlan(**plan_fields), changes=changes, rerun=["rwd"], label=LABELS[kind], study=study,
                   checks=checks)
    else:
        raise ValueError(f"Unknown protocol part {part!r}")
    whats = list(dict.fromkeys(c["what"].split(":")[0] for c in out["changes"]))
    out["summary"] = ", ".join(whats[:4]) + (f" and {len(whats) - 4} more" if len(whats) > 4 else "")
    return out


def check_edit(base, edit):
    """(errors, preview) of an edit before it runs: preview {changes, rerun, kept} (what would run again)."""
    try:
        p = prepare(base, edit)
    except ValueError as e:
        return [str(e)], None
    return [], {"changes": p["changes"], "rerun": p["rerun"], "kept": _kept(base, p["rerun"])}


def _present(base):
    """The sources of ``base`` that gave evidence or ran (PDFs only when they were searched)."""
    names = [s for s in ("literature", "pdfs", "trials", "kg", "rwd") if s in (base.sources or [])
             or (s == "pdfs" and "pdfs" in (base.source_status or {}))]
    return names


def _kept(base, rerun):
    return [s for s in _present(base) if s not in rerun]


def names(sources):
    """Readable names of sources: "PubMed, iBKH"."""
    return ", ".join(SOURCE_NAMES.get(s, s) for s in sources) or "—"


def _evidence_groups(base):
    """Copies of the evidence of ``base`` by source (PubMed articles and PDFs apart)."""
    groups = {"literature": [], "pdfs": [], "trials": [], "kg": [], "rwd": []}
    for item in base.evidence or []:
        item = copy.deepcopy(item)
        if item.source_type == "literature":
            groups["literature" if str(item.origin).startswith("PubMed") else "pdfs"].append(item)
        elif item.source_type == "trial":
            groups["trials"].append(item)
        elif item.source_type in groups:
            groups[item.source_type].append(item)
    return groups


# ============================================================================ comparison
def _identities(result, source_type):
    out = {}
    for e in result.evidence or []:
        if e.source_type == source_type:
            ids = e.identifiers or {}
            out[ids.get("pmid") or ids.get("nct") or ids.get("file") or e.title] = e.title
    return out


def _estimates(study):
    """{label: value text} of a study's estimates (sensitivity analyses marked)."""
    from mes.tools.rwd.report import estimate_text
    out = {}
    for e in ((study or {}).get("result") or {}).get("estimates") or []:
        label = str(e.get("label", "")) + (" (sensitivity)" if e.get("sensitivity") else "")
        out[label] = estimate_text(e)[len(str(e.get("label", ""))) + 2:]
    return out


def _rows(study):
    return ((study or {}).get("result") or {}).get("tables", {}).get("result")


def compare(base, result, before_study=None, after_study=None):
    """How ``result`` differs from the answer it was edited from: sources added / removed, cited sources, and the
    estimates of the edited study that changed (by label; one only before or after counts as changed) — or, for an
    exploratory query (a result table, no estimates), its rows before and after."""
    out = {"sources": {}, "cited": {"before": len(base.citation_order or []), "after": len(result.citation_order or [])},
           "estimates": [], "tables": []}
    for t in ("literature", "trial", "kg"):
        a, b = _identities(base, t), _identities(result, t)
        added, removed = [b[k] for k in b if k not in a], [a[k] for k in a if k not in b]
        if added or removed:
            out["sources"][t] = {"before": len(a), "after": len(b), "n_added": len(added), "n_removed": len(removed),
                                 "added": added[:10], "removed": removed[:10]}
    if before_study is not None and after_study is not None:
        title = ((after_study.get("result") or {}).get("title") or (before_study.get("result") or {}).get("title")
                 or f"{rwd_tools.database_of(after_study)} study")
        a, b = _estimates(before_study), _estimates(after_study)
        if not a and not b and (_rows(before_study) is not None or _rows(after_study) is not None):
            before, after = _rows(before_study) or [], _rows(after_study) or []
            out["tables"].append({"study": title, "before_rows": len(before), "after_rows": len(after),
                                  "changed": before != after})
            return out
        rows = [{"label": k, "before": a.get(k, ""), "after": v} for k, v in b.items()] + \
            [{"label": k, "before": v, "after": ""} for k, v in a.items() if k not in b]
        changed = [r for r in rows if r["before"] != r["after"]]
        out["estimates"].append({"study": title, "changed": changed[:16], "n_changed": len(changed),
                                 "n_same": len(rows) - len(changed)})
    return out


def record(base, result, edit, prepared, version, new_study=None):
    """``MESResult.protocol`` of ``result``, the answer to ``base`` with ``edit`` (``prepared``: ``prepare``)."""
    from_version = int((base.protocol or {}).get("version") or 1)
    target, label, summary = prepared["study"] or {}, prepared["label"], prepared["summary"]
    title = (target.get("result") or {}).get("title") or (target.get("spec") or {}).get("title") or ""
    return {"version": version, "from_version": from_version, "part": edit.get("part"), "label": label,
            "study": target.get("key"), "study_ref": target.get("evidence_id"), "study_title": title,
            "new_study": (new_study or {}).get("key"),
            "changes": prepared["changes"], "summary": summary, "diff": prepared["diff"], "rerun": prepared["rerun"],
            "kept": _kept(base, prepared["rerun"]), "checks": prepared["checks"],
            "edited_at": datetime.now().isoformat(timespec="seconds"),
            "trail": list((base.protocol or {}).get("trail") or []) + [{"version": version, "from_version": from_version,
                                                                       "label": label, "summary": summary}],
            "comparison": compare(base, result, prepared["study"], new_study)}


# ============================================================================ run
def iter_rerun(question, llm, base, edit, version=None, mode=None, history=None, verify=None, tested=None,
               cancel=None):
    """
    Answer ``question`` again with ``edit`` applied to the protocol of ``base`` (the MESResult it edits); yields the
    events of ``pipeline_tools.iter_answer`` (kept sources get a stage event with status "kept"). ``version``: the
    number of the new protocol (default: the edited one's + 1); ``tested``: the record of the same SQL / code edit
    run by the editor's test (``rwd_tools.rerun_study``). An edited part that fails ends the run with an error
    event: the answer is not written without it.
    """
    from mes.tools import pipeline_tools as P
    mode_cfg = config.get_mode(mode or base.mode)
    verify = mode_cfg.verify_claims if verify is None else bool(verify)
    sources = [s for s in P.AVAILABLE_SOURCES if s in (base.sources or [])]
    history = history or []
    t_start = time.time()
    timings = {}
    from_version = int((base.protocol or {}).get("version") or 1)
    version = int(version or from_version + 1)
    calls_before = llm.n_calls
    part = (edit or {}).get("part")

    yield {"type": "stage", "stage": "plan", "status": "running", "message": "Applying the edited protocol"}
    try:
        p = prepare(base, edit, sources)
    except ValueError as e:
        yield {"type": "stage", "stage": "plan", "status": "error", "message": str(e)[:300]}
        yield {"type": "error", "message": f"The edited protocol cannot be run: {e}"}
        return
    plan, rerun = p["plan"], p["rerun"]
    kept = _kept(base, rerun)
    yield {"type": "stage", "stage": "plan", "status": "edited" if part == "search" else "kept", "elapsed": 0.0,
           "message": (f"Search protocol edited ({p['summary']})" if part == "search" else
                       f"{p['label']} edited; the search is kept from v{from_version}")}

    groups = _evidence_groups(base)
    for name in kept:
        n = len(groups[name]) if name != "rwd" else len(base.rwd_studies or [])
        yield {"type": "stage", "stage": name, "status": "kept",
               "message": f"{n} {NOUNS[name][n != 1]} kept from v{from_version}"}
    query = plan.query
    source_status = {k: {**copy.deepcopy(v), "kept": True} for k, v in (base.source_status or {}).items()
                     if k in kept}
    rwd_studies = copy.deepcopy(base.rwd_studies or [])
    target = p["study"]
    results, new_study = {}, None
    messages = queue.Queue()

    def reporter(stage):
        return lambda msg: messages.put((stage, msg))

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        jobs = {}
        qv = encode([query], is_query=True)[0] if set(rerun) & {"literature", "trials", "kg"} else None
        if "literature" in rerun:
            jobs["literature"] = pool.submit(P._literature, plan, mode_cfg, query, qv, reporter("literature"),
                                             "fulltext" in (base.literature or P.LITERATURE_PARTS))
        if "trials" in rerun:
            jobs["trials"] = pool.submit(trials_tools.search_trials, plan, mode_cfg, query, qv, reporter("trials"))
        if "kg" in rerun:
            jobs["kg"] = pool.submit(P._kg, plan, mode_cfg, query, qv, reporter("kg"))
        if "rwd" in rerun:
            jobs["rwd"] = pool.submit(rwd_tools.rerun_study, target, edit, plan, mode_cfg, llm, reporter("rwd"),
                                      tested)
        started = {name: time.time() for name in jobs}
        for name in jobs:
            yield {"type": "stage", "stage": name, "status": "running", "message": P.SOURCE_LABELS.get(name, name)}
        pending = set(jobs)
        while pending or not messages.empty():
            try:
                stage, msg = messages.get(timeout=0.1)
                yield {"type": "log", "stage": stage, "message": msg}
            except queue.Empty:
                pass
            for name in [n for n in pending if jobs[n].done()]:
                pending.discard(name)
                elapsed = round(time.time() - started[name], 2)
                timings[name] = elapsed
                try:
                    out = jobs[name].result()
                except Exception as e:
                    yield {"type": "stage", "stage": name, "status": "error", "message": str(e)[:300],
                           "elapsed": elapsed}
                    yield {"type": "error", "message": f"The edited protocol could not be run: "
                                                       f"{SOURCE_NAMES.get(name, name)} failed ({str(e)[:300]})"}
                    return
                if name == "rwd":
                    items, new_study = out
                    others = [it for it in groups["rwd"] if it.meta.get("study_key") != target.get("key")]
                    results["rwd"] = others + items
                    rwd_studies = [new_study if s.get("key") == target.get("key") else s for s in rwd_studies]
                    source_status["rwd"] = {"ok": True, "n_items": len(results["rwd"]), "elapsed": elapsed}
                else:
                    items, stats = out
                    results[name] = items
                    source_status[name] = {"ok": True, "n_items": len(items), "elapsed": elapsed,
                                           **{k: v for k, v in stats.items() if not isinstance(v, (dict, list))}}
                    if name == "literature":
                        source_status[name]["queries"] = stats.get("queries", {})
                    if name == "kg":
                        source_status[name]["links"] = stats.get("links", {})
                        source_status[name]["semantic_nodes"] = stats.get("semantic_nodes", [])
                yield {"type": "stage", "stage": name, "status": "done", "message": f"{len(items)} items",
                       "elapsed": elapsed}
            if cancel is not None and cancel.is_set():
                return

        literature = results["literature"] if "literature" in rerun else groups["literature"]
        if "literature" in rerun and mode_cfg.completeness_round:
            yield {"type": "stage", "stage": "coverage", "status": "running", "message": "Checking facet coverage"}
            t = time.time()
            extra = []
            try:
                extra = P._snow_coverage(plan, literature, mode_cfg, query, lambda msg: messages.put(("coverage", msg)))
            except Exception as e:
                P.logger.warning("Coverage round failed: %s", e)
            while not messages.empty():
                stage, msg = messages.get()
                yield {"type": "log", "stage": stage, "message": msg}
            literature = literature + extra
            timings["coverage"] = round(time.time() - t, 2)
            yield {"type": "stage", "stage": "coverage", "status": "done", "elapsed": timings["coverage"],
                   "message": f"{len(extra)} articles added for uncovered facets"}
        literature = literature + groups["pdfs"]
        for name in ("trials", "kg", "rwd"):
            results.setdefault(name, groups[name])
        result = yield from P._write(question, plan, query, llm, mode_cfg, sources, literature, results, source_status,
                                     rwd_studies, history, verify, timings, t_start, calls_before, 0)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if result is None:
        return
    result.literature = list(base.literature or [])
    result.protocol = record(base, result, edit, p, version, new_study)
    yield {"type": "stage", "stage": "done", "status": "done",
           "message": f"Protocol v{version} done in {result.timings['total']:.0f}s with {result.n_llm_calls} LLM calls"}
    yield {"type": "result", "result": result, "final": True}
