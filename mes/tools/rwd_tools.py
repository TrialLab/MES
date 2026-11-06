"""
RWD-Analyst source — real-world evidence from MIMIC-IV (see tools/rwd/ and scripts/build_mimic_db.py).

Contract shared by every MES evidence source::

    search(plan, mode, query, query_vector, report) -> (items: list[EvidenceItem], stats: dict)

``search_rwd`` designs a study for the question (tools/rwd/designer.py: one LLM call for the intent, everything
else deterministic), compiles it to SQL, runs the analysis and returns evidence items with prefix ``R``. The full
study record (spec, SQL, attrition, estimates, diagnostics, tables) is in ``stats["studies"]``; the pipeline copies
it to ``MESResult.rwd_studies`` for the UI. ``spec_override`` re-runs an edited design without the LLM;
``rerun_study`` runs one study again with the user's edit of its design, its exploratory SQL or its analysis code
(an edited protocol, tools/protocol_tools.py).

Privacy: only the schema, dictionary entries and aggregate results reach the LLM; patient-level rows stay local.
"""

import hashlib
import json
import logging
import time

import numpy as np

from mes.tools.result_schema import EvidenceItem, Passage
from mes.tools.rwd import DATABASE, DATASET, database_of  # noqa: F401  (names of the connected database)
from mes.tools.rwd.db import get_store

logger = logging.getLogger(__name__)


def _plain(obj):
    """JSON-safe copy (numpy numbers → Python numbers, NaN → None) so cached results keep their types."""
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    text = json.dumps(obj, default=default).replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null")
    return json.loads(text)


def available():
    return get_store().available


def unavailable_reason():
    """Why the real-world data source is off, naming the folder MES looked in (``MES_MIMIC_DIR``)."""
    from mes import config
    root = get_store().root
    try:
        shown = root.resolve().relative_to(config.ROOT.resolve()).as_posix() if root.exists() else \
            root.relative_to(config.ROOT).as_posix()
    except ValueError:
        shown = str(root)
    broken = " (a link that points nowhere on this computer)" if root.is_symlink() and not root.exists() else ""
    return (f"The real-world data store is not built in {shown}{broken}: build it with `python -m "
            "mes.scripts.build_mimic_db` (MIMIC-IV CSV files in data/mimic-iv-2.2), or copy a store built elsewhere "
            "into that folder — the folder itself, not a link to it; see docs/DATA.md.")


# Kept for the agent layer: availability is decided by the store at run time
AVAILABLE = True


ENGINES = {"auto": "Auto — study templates, the LLM analyst when they do not fit",
           "templates": "Study templates only (fast, validated)",
           "analyst": "LLM analyst (writes and runs its own analysis code)"}


STRONG_MATCH = 1.2          # concept search score of a clear match (an exact drug class or phenotype name)


def _expressible(terms, spec, index):
    """A repair message when terms the design leaves out have a clear MIMIC-IV concept (the designer once dropped
    "cefepime", the comparator of the question), else ""."""
    used = {c.concept for c in spec.include + spec.exclude + [spec.exposure, spec.comparator] if c}
    found = []
    for term in terms or []:
        best = next((c for c in index.search(term, k=3) if c.kind in ("phe", "drugclass", "drug", "var")
                     and c.score >= STRONG_MATCH - (0.1 if c.kind == "drug" else 0)      # a medication's own name
                     and c.concept_id not in used), None)
        if best is not None:
            found.append(f"“{term}” → {best.concept_id} ({best.label}"
                         + (f", {best.n_patients:,} patients)" if best.n_patients else ")"))
    if not found:
        return ""
    return ("The design leaves out terms of the question that MIMIC-IV has: " + "; ".join(found) + ". Use each where "
            "the question means it (a second treatment as the comparator, a condition as a population criterion, a "
            "measurement as the variable) and keep the rest of the design.")


def _size_problem(spec, cohort, mapping):
    """Why a valid design cannot give an estimate on the data (None if it can), for one repair round."""
    from mes.tools.rwd.analyses import MIN_GROUP
    df = cohort.df
    small = None
    if spec.analysis == "compare" and "treated" in df:
        n1, n0 = int(df["treated"].sum()), int((1 - df["treated"]).sum())
        if min(n1, n0) < MIN_GROUP:
            small = f"only {n1} treated and {n0} comparison patients remain (at least {MIN_GROUP} per group needed)"
    elif len(df) < 30:
        small = f"only {len(df)} patients remain"
    if small is None:
        return None
    steps = "; ".join(f"{s}: {n:,}" for s, n in cohort.attrition)
    terms = "; ".join(f"'{m['term']}' → {m['label']}" + (f" ({m['patients']:,} patients)" if m.get("patients") else "")
                      for m in mapping or [])
    return (f"The design runs on MIMIC-IV but {small}. Attrition: {steps}. Term mapping: {terms}. If a population "
            "criterion only restates the population (adult ICU / hospital patients, critically ill) or matched a "
            "narrower concept than the question means, drop or replace it; widen windows if they are too strict. "
            "Keep the treatments and the outcome of the question.")


def search_rwd(plan, mode, query, query_vector=None, report=None, llm=None, spec_override=None, engine="auto",
               cancel=None):
    """
    Real-world evidence for ``plan`` (MIMIC-IV).

    ``engine``: "auto" — validated study templates, the Claude analyst when they do not fit (exploratory questions,
    no valid design) and it is available; "templates" — templates and the exploratory SQL tier only; "analyst" — the
    analyst, with the template study alongside when one fits (two independent analyses). ``cancel`` (an Event)
    stops a running analyst.
    """
    from mes.tools.rwd import analyses, analyst, designer, explorer, report as rep
    from mes.tools.rwd.compiler import Compiler
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.spec import StudySpec, validate

    report = report or (lambda msg: None)
    store = get_store()
    if not store.available:
        return [], {"available": False, "error": unavailable_reason()}
    t0 = time.time()
    index = get_index()
    question = plan.standalone_question or query
    engine = engine if engine in ENGINES else "auto"
    analyst_ok, analyst_reason = analyst.status(llm)
    timings, info, uncovered, approximation = {}, {}, [], ""
    cohorts = {}                     # cohorts built while checking designs (reused for the analysis)

    def check(spec, mapping):
        if spec.feasibility == "not_feasible" or spec.analysis in ("explore", "catalog"):
            return None
        uncovered_terms = designer.coverage(plan, spec, index)
        hint = _expressible(uncovered_terms, spec, index)
        if hint:
            return hint              # MIMIC-IV has a concept for it: the designer can still use it
        if analyst_ok and engine == "auto" and (uncovered_terms
                                                 or designer.unmatched({"mapping": mapping}, spec, index)):
            return None              # no repair can express it with the templates: the analyst takes over
        cohort = Compiler(index, store).build(spec)
        cohorts[spec.key()] = cohort
        return _size_problem(spec, cohort, mapping)

    def run_analyst():
        report("RWD: the LLM analyst explores the data and writes the analysis code")
        study = analyst.analyze(question, plan, llm, mode, store, index, report=report, cancel=cancel)
        if study["status"] == "done":
            items = rep.analyst_items(study)
        elif study["status"] == "not_feasible":
            items = [rep.not_feasible_item(StudySpec.from_dict({"analysis": "rate", "reason":
                                                                study["spec"].get("reason", "")}), study["key"])]
        else:
            items = [rep.analyst_failed_item(study)]
        study["result"] = _plain(study.get("result"))
        return items, study

    if spec_override:
        spec = StudySpec.from_dict(spec_override)
        errors, warnings = validate(spec, index)
        info = {"source": "edited by the user", "warnings": warnings, "errors": errors}
        if errors:
            raise ValueError("invalid study design: " + "; ".join(errors))
    else:
        if llm is None:
            raise ValueError("search_rwd needs the LLM to design the study")
        if engine == "analyst" and analyst_ok:
            # the analyst works independently; the template study (if one fits) runs alongside for comparison
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                job = pool.submit(run_analyst)
                try:
                    t_items, t_stats = search_rwd(plan, mode, query, query_vector, report, llm, engine="templates")
                except Exception as e:
                    logger.warning("Template study next to the analyst failed: %s", e)
                    t_items, t_stats = [], {"studies": []}
                a_items, a_study = job.result()
            keep = [(it, st) for it, st in zip(t_items, t_stats.get("studies", []))
                    if st.get("status") == "done" and (st.get("result") or {}).get("estimates")]
            items = [it for it, _ in keep] + a_items
            studies = [st for _, st in keep] + [a_study]
            return items, {"available": True, "n_found": len(items), "studies": studies,
                           "seconds": round(time.time() - t0, 2)}
        report("RWD: designing the study")
        spec, info = designer.design(question, plan, index, llm, report=report, check=check, original=query)
        # terms of the question that MIMIC-IV has no concept for, and terms the design does not represent
        unmatched = [] if spec is not None and (spec.feasibility == "not_feasible" or spec.analysis == "catalog") \
            else designer.unmatched(info, spec, index)        # (a catalog says itself what is not recorded)
        uncovered = [t for t in designer.coverage(plan, spec, index) if t not in unmatched]
        # the designer's own word that its design only approximates the question
        approximation = info.get("approximation", "") if spec is not None and spec.feasibility != "not_feasible" \
            and spec.analysis != "explore" else ""
        # only what MIMIC-IV has but the templates cannot express goes to the analyst; a limit of the data itself
        # (anchor year groups) or a design choice is reported with the template study (no analysis does better)
        routable = approximation and info.get("approximation_kind", "templates") in ("", "templates")
        if unmatched:
            info.setdefault("warnings", []).append("Not found in MIMIC-IV: " + ", ".join(unmatched))
        if uncovered:
            info.setdefault("warnings", []).append("The study design does not represent: " + ", ".join(uncovered))
        # still too few patients after the repair rounds: the templates cannot answer it on this data
        too_small = spec is not None and spec.key() in cohorts and \
            _size_problem(spec, cohorts[spec.key()], info.get("mapping")) is not None
        if spec is None or spec.analysis == "explore" or uncovered or unmatched or \
                ((too_small or routable) and engine == "auto" and analyst_ok):
            if engine == "auto" and analyst_ok:
                items, study = run_analyst()
                routed = ("MIMIC-IV has no matching concept for " + ", ".join(unmatched) if unmatched else
                          "the study templates cannot express " + ", ".join(uncovered) if uncovered else
                          "the template design leaves too few patients" if too_small else
                          "the template design only approximates the question: " + approximation if routable else
                          "no study template fits this question" if spec is not None else "no valid template design")
                study["design_info"] = {**study.get("design_info", {}), "templates": _plain(info), "routed": routed}
                usable = spec is not None and spec.analysis != "explore" and spec.feasibility != "not_feasible" \
                    and not unmatched
                if study["status"] != "failed" or not usable or (cancel is not None and cancel.is_set()):
                    return items, {"available": True, "n_found": len(items), "studies": [study],
                                   "seconds": round(time.time() - t0, 2)}
                # the custom analysis did not finish in the mode's budget: the closest validated design still answers
                report("RWD: the custom analysis did not finish; running the closest validated study design")
                failed = "; ".join((study.get("design_info") or {}).get("errors") or ["it did not finish"])
                info = {**info, "routed": routed, "analyst_failed": failed}
            elif (spec is None or unmatched) and llm is not None and not (cancel is not None and cancel.is_set()) \
                    and not (spec is not None and spec.feasibility == "not_feasible"):
                # no template fits, or a term has no concept of the study templates (a positive blood culture, the
                # commonest drugs): an exploratory aggregate query over the tables (checked SQL); the catalog after it
                report("RWD: no study template fits; writing an exploratory query")
                routed = ("MES's concepts have no entry for " + ", ".join(f"“{t}”" for t in unmatched) + ": an "
                          "exploratory query over the MIMIC-IV tables instead") if unmatched else \
                    "no study template fits this question: an exploratory query instead"
                spec = StudySpec(analysis="explore", base="icu", title=question[:160], sql_question=question)
                info = {**info, "errors": [], "design_errors": info.get("errors", []), "routed": routed,
                        "unmatched": list(unmatched)}
                unmatched = []
            if spec is None or unmatched:
                # never a study of a different population or treatment: a term without a concept stops it
                reason = (f"MIMIC-IV has no record matching {', '.join(f'“{t}”' for t in unmatched)}, so no study was "
                          "run" if unmatched else "no valid study design could be made for this question (" +
                          "; ".join(info.get("errors", []))[:300] + ")")
                item = EvidenceItem(id="rwd:design", source_type="rwd", venue=DATASET, design="Study design failed",
                                    title=f"{DATASET}: no valid study design", evidence_level=5, origin="RWD-Analyst",
                                    passages=[Passage(text=f"Real-world data: {reason}.", section="RWD feasibility")],
                                    score=0.1)
                if unmatched:
                    info = {**info, "errors": [reason[0].upper() + reason[1:]]}
                note = {} if analyst_ok or engine == "templates" else {"analyst": analyst_reason}
                # never nothing: what MIMIC-IV records about the question's terms, and what can be asked instead
                cat_items, cat_study = _catalog_study(question, plan, index, store,
                                                      "No study design fits this question in MIMIC-IV: what the data "
                                                      "records about its terms instead.")
                return [item] + cat_items, {"available": True, "n_found": len(cat_items),
                                            "studies": [{"key": "design", "spec": None, "design_info": {**info, **note},
                                                         "status": "failed"}, cat_study]}
    timings["design"] = round(time.time() - t0, 2)
    key = spec.key()
    study = {"key": key, "kind": "template", "spec": spec.to_dict(), "design_info": _plain(info), "dataset": DATASET,
             "data_built_at": store.catalog().get("built_at"), "status": "done"}
    if spec.feasibility == "not_feasible":
        study.update(status="not_feasible", result=None)
        cat_items, cat_study = _catalog_study(question, plan, index, store, "The question cannot be answered with "
                                              "MIMIC-IV as asked: what the data records about its terms.")
        return [rep.not_feasible_item(spec, key)] + cat_items, {"available": True, "n_found": len(cat_items),
                                                                "studies": [study, cat_study]}
    if spec.analysis == "catalog":                  # "what does MIMIC-IV have on …": the data catalog itself
        items, cat_study = _catalog_study(question, plan, index, store)
        return items, {"available": True, "n_found": len(items), "studies": [cat_study],
                       "seconds": round(time.time() - t0, 2)}

    t = time.time()
    if spec.analysis == "explore":
        report("RWD: writing and checking an exploratory query")
        terms = [e.get("name") for e in getattr(plan, "entities", None) or [] if isinstance(e, dict) and e.get("name")]
        res = explorer.explore(spec.sql_question or question, llm, store, report=report, terms=terms)
        if info.get("unmatched") and (res.get("tables") or {}).get("result"):
            res["warnings"].insert(0, "MES's concepts have no entry for " + ", ".join(f"“{t}”" for t in info["unmatched"])
                                   + ": the query looked for it in the MIMIC-IV tables — check its SQL.")
        study["sql"] = res.get("sql")
        timings["query"] = round(time.time() - t, 2)
        if not (res.get("tables") or {}).get("result"):
            # the exploratory query of last resort did not run either: what the data records instead
            reason = "; ".join(res.get("warnings") or ["no valid query"])[:300]
            items, cat_study = _catalog_study(question, plan, index, store, "Neither a study design nor an exploratory "
                                              f"query fits this question ({reason}): what MIMIC-IV records about its "
                                              "terms instead.")
            study.update(status="failed", result=_plain(res), design_info={**_plain(info), "errors": [reason]})
            return items, {"available": True, "n_found": len(items), "studies": [study, cat_study],
                           "seconds": round(time.time() - t0, 2)}
        items = _explore_items(spec, res, key)
    else:
        report("RWD: running the cohort query")
        cohort = cohorts.get(key) or Compiler(index, store).build(spec)
        timings["cohort"] = round(time.time() - t, 2)
        report(f"RWD: {len(cohort.df):,} {spec.base} patients after the criteria; estimating")
        t = time.time()
        res = analyses.run(spec, cohort, mode.key)
        if approximation:
            res["warnings"].insert(0, f"The study design approximates the question: {approximation}")
        if info.get("analyst_failed"):
            res["warnings"].insert(0, f"The custom analysis did not finish ({info['analyst_failed']}): the closest "
                                      "validated study design is shown instead.")
        if uncovered:
            res["warnings"].insert(0, "The study templates could not express " + ", ".join(uncovered)
                                   + ": this analysis answers a related question only.")
        coded = [c for c in spec.include + [spec.exposure, spec.comparator] if c and c.concept[:2] == "px"]
        if coded and spec.base == "icu":
            res["warnings"].append("ICD procedure codes are assigned at discharge and undercount bedside "
                                   "treatments (" + ", ".join(index.label(c.concept) for c in coded) + ").")
        timings["analysis"] = round(time.time() - t, 2)
        study["sql"] = cohort.sql
        items = rep.to_items(spec, res, key)
    try:                                # what each concept of the design means in MIMIC-IV (study panel, exports)
        from mes.tools.rwd.definitions import describe
        study["definitions"] = _plain(describe(spec.to_dict(), index))
    except Exception as e:
        logger.warning("Could not describe the concepts of the study: %s", e)
    timings["total"] = round(time.time() - t0, 2)
    study.update(result=_plain(res), timings=timings)
    report(f"RWD: {res.get('analysis')} analysis done in {timings['total']:.0f}s")
    return items, {"available": True, "n_found": len(items), "studies": [study], "seconds": timings["total"]}


def _catalog_study(question, plan, index, store, reason=""):
    """(items, study record) of a data-discovery study: what MIMIC-IV records about the question's terms."""
    from mes.tools.rwd import catalog
    t = time.time()
    terms = [e.get("name") for e in getattr(plan, "entities", None) or [] if isinstance(e, dict) and e.get("name")]
    terms = terms or [question]
    res = catalog.run(question, terms, index, store, reason)
    key = hashlib.sha256(("catalog\n" + json.dumps(terms, sort_keys=True)).encode()).hexdigest()[:16]
    spec = {"analysis": "catalog", "base": "icu", "title": res["title"], "sql_question": question}
    study = {"key": key, "kind": "template", "spec": spec, "design_info": {}, "dataset": DATASET, "status": "done",
             "data_built_at": store.catalog().get("built_at"), "result": _plain(res),
             "timings": {"total": round(time.time() - t, 2)}}
    item = EvidenceItem(id=f"rwd:{key}", source_type="rwd", venue=DATASET, title=f"{DATASET}: {res['title']}",
                        design="Data catalog (what the database records)", evidence_level=5, origin="RWD-Analyst",
                        passages=[Passage(text=catalog.text(res), section="RWD data catalog")], score=0.4,
                        meta={"study_key": key, "analysis": "catalog", "n": None, "warnings": res["warnings"],
                              "primary": None})
    return [item], study


def _explore_items(spec, res, key):
    """Evidence item of an exploratory query: the design and the query's (suppressed) result table."""
    from mes.tools.rwd import explorer, report as rep
    item = rep.to_items(spec, res, key)[0]
    table = explorer.result_text(res)
    if table:
        item.passages.insert(1, Passage(text="Query result: " + table + ".", section="RWD results"))
    return [item]


# ============================================================================ edited protocols
EDIT_KINDS = {"design": "study design", "sql": "SQL query", "code": "analysis code"}


def edit_hash(edit):
    """Identity of an SQL or code edit: a test run of the same text (the editor's ▶ Test) stands for the run."""
    text = edit.get("sql") if edit.get("kind") == "sql" else edit.get("code")
    return hashlib.sha256(f"{edit.get('kind')}\n{edit.get('study')}\n{text or ''}".encode()).hexdigest()[:16]


def editable(study, kind):
    """Why ``study`` cannot be edited as ``kind`` ("" when it can): the design of a finished template study, the
    query of an exploratory one, the code of an analyst study (also one that did not finish: its code can be fixed)."""
    analysis = (study.get("spec") or {}).get("analysis")
    if kind not in EDIT_KINDS:
        return f"unknown edit {kind!r}"
    if kind == "design":
        if study.get("kind") == "analyst":
            return "a custom analysis has no template design: edit its code"
        if study.get("status") != "done" or not study.get("spec"):
            return "only a finished study design can be edited"
        if analysis == "catalog":
            return "a data catalog has no study design: ask a question about these data instead"
    if kind == "sql" and (study.get("kind") == "analyst" or analysis != "explore"):    # (an empty query can be written)
        return "the SQL of a template study is compiled from its design: edit the design"
    if kind == "code" and (study.get("kind") != "analyst" or not study.get("code")):
        return "only a custom analysis has analysis code"
    return ""


def edited_study(study, edit, question, mode, report=None):
    """The record of ``study`` run with the user's query (``edit["kind"]`` "sql", ``edit["sql"]``) or analysis code
    ("code", ``edit["code"]``), under the same checks as the model's; ``study["edited"]`` marks it."""
    from mes.tools.rwd import analyst, explorer
    from mes.tools.rwd.concepts import get_index
    report = report or (lambda msg: None)
    store = get_store()
    if not store.available:
        raise RuntimeError(unavailable_reason())
    problem = editable(study, edit.get("kind"))
    if problem:
        raise ValueError(problem)
    t0 = time.time()
    if edit["kind"] == "code":
        record = analyst.run_edited(edit.get("code"), study, store, get_index(), mode, report=report)
        record["result"] = _plain(record.get("result"))
    else:
        report("RWD: running the edited query")
        title = (study.get("result") or {}).get("title") or (study.get("spec") or {}).get("sql_question") or question
        res = explorer.run_edited(edit.get("sql"), store, title)
        seconds = round(time.time() - t0, 2)
        record = {"key": hashlib.sha256(f"{study.get('key')}\n{res['sql']}".encode()).hexdigest()[:16],
                  "kind": study.get("kind", "template"), "spec": study.get("spec"),
                  "design_info": {**(study.get("design_info") or {}), "source": "query edited by the user"},
                  "dataset": DATASET, "data_built_at": store.catalog().get("built_at"),
                  "status": "failed" if res.get("error") else "done", "sql": res["sql"], "result": _plain(res),
                  "timings": {"query": seconds, "total": seconds}, "edited": {"kind": "sql", "from_key": study.get("key")}}
        if res.get("error"):
            record["design_info"]["errors"] = [f"the edited query did not run: {res['error']}"]
    record["edit_hash"] = edit_hash(edit)
    return record


def study_items(study):
    """Evidence items of a study record (those ``search_rwd`` gives for it)."""
    from mes.tools.rwd import report as rep
    from mes.tools.rwd.spec import StudySpec
    if study.get("kind") == "analyst":
        if study.get("status") == "done":
            return rep.analyst_items(study)
        if study.get("status") == "not_feasible":
            return [rep.not_feasible_item(StudySpec.from_dict({"analysis": "rate", "reason": study["spec"].get("reason", "")}),
                                          study["key"])]
        return [rep.analyst_failed_item(study)]
    spec = StudySpec.from_dict(study["spec"])
    res = study.get("result") or {}
    if spec.analysis == "explore":
        return _explore_items(spec, res, study["key"])
    return rep.to_items(spec, res, study["key"])


def rerun_study(study, edit, plan, mode, llm=None, report=None, tested=None):
    """
    (items, study record) of ``study`` run again with the user's ``edit``: "design" — the edited study design
    (``edit["spec"]``, run like ``spec_override``); "sql" / "code" — the edited query of an exploratory study or the
    edited code of an analyst study (``edited_study``). ``tested``: the record of the same edit run by the editor's
    test, used instead of running it again.
    """
    kind = edit.get("kind")
    if kind == "design":
        problem = editable(study, "design")
        if problem:
            raise ValueError(problem)
        items, stats = search_rwd(plan, mode, plan.query, report=report, llm=llm, spec_override=edit.get("spec"))
        if not stats.get("studies"):
            raise RuntimeError(stats.get("error") or "the edited design gave no study")
        record = stats["studies"][0]
        record["edited"] = {"kind": "design", "from_key": study.get("key")}
        return items, record
    if tested and tested.get("edit_hash") == edit_hash(edit):
        record = tested
    else:
        record = edited_study(study, edit, plan.query, mode, report)
    if record.get("status") == "failed":
        raise RuntimeError("; ".join((record.get("design_info") or {}).get("errors") or ["the edited study failed"]))
    return study_items(record), record
