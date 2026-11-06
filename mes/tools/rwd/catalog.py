"""
What MIMIC-IV records about the terms of a question (data discovery: "does MIMIC-IV have ECMO?", "which lab tests
measure kidney function?"), and the questions MES can answer with them. Also the answer of last resort: when no study
design fits a question, the user still learns what the data holds and what to ask instead. The concept index's
search with patient counts; no query is written by a model, nothing below 10 patients is shown.

    run(question, terms, index, store) -> result  ({"analysis": "catalog", "tables": {"overview", "concepts",
                                                       "suggestions"} …})
"""

import re

from mes.tools.rwd import library as L

SMALL = 10
KINDS = {"phe": "condition", "drugclass": "drug class", "drug": "medication", "var": "variable", "dx9": "ICD-9 code",
         "dx10": "ICD-10 code", "px9": "ICD-9 procedure", "px10": "ICD-10 procedure", "lab": "lab test", "item": "ICU item",
         "micro": "microbiology"}
USES = {"phe": "a condition: inclusion / exclusion, subgroup", "dx9": "a diagnosis code: inclusion / exclusion",
        "dx10": "a diagnosis code: inclusion / exclusion", "px9": "a procedure code: treatment or criterion",
        "px10": "a procedure code: treatment or criterion", "drugclass": "a treatment: comparison, use, pathway",
        "drug": "a medication: comparison, use, pathway", "lab": "a measurement: distribution, trajectory, threshold",
        "item": "an ICU measurement: distribution, trajectory, threshold",
        "var": "a cohort variable: criterion, covariate, breakdown",
        "micro": "a culture result: criterion, how often it is positive"}


# words of a question about the database itself, no concept to search ("how many patients are in MIMIC-IV?")
_DATABASE_WORDS = re.compile(r"\b(?:the|a|an|mimic(?:-?iv)?|database|dataset|data|records?|patients?|people|subjects?|"
                             r"admissions?|stays?|icu|hospital|number|numbers|of|in|total|how|many|much|count|size|"
                             r"overview|contents?|v2\.2|are|is|was|were|there|what|which|does|do|did|have|has|had|"
                             r"contain|contains|record|recorded|give|me|show|tell|about|summary)\b", re.I)


def overview(store):
    """[{what, n}]: the size of MIMIC-IV and of MES's two study populations."""
    rows = (store.catalog() or {}).get("row_counts") or {} if store is not None else {}
    out = [{"what": label, "n": rows.get(table)} for label, table in (
        ("Patients", "mimiciv_hosp.patients"), ("Hospital admissions", "mimiciv_hosp.admissions"),
        ("ICU stays", "mimiciv_icu.icustays")) if rows.get(table)]
    for key in ("icu", "hospital"):
        b = L.BASES[key]
        try:
            n = int(store.query(f"SELECT count(*) AS n FROM {b['table']} WHERE {b['where']}").n[0])
        except Exception:
            continue
        out.append({"what": f"{b['label']} — MES's {key} population", "n": n})
    return out


def _shown(n):
    return None if n is None else "<10" if 0 < n < SMALL else n


def _variable_patients(store, name):
    """Stays of the ICU population with a value of a cohort variable (with it, for a yes/no one), or None."""
    if store is None or name not in L.VARIABLES:
        return None
    kind, bases = L.VARIABLES[name][1], L.VARIABLES[name][3]
    b = L.BASES["icu" if "icu" in bases else "hospital"]
    test = f"coalesce(CAST({name} AS INTEGER), 0) = 1" if kind == "binary" else f"{name} IS NOT NULL"
    try:
        return int(store.query(f"SELECT count(*) AS n FROM {b['table']} WHERE {b['where']} AND {test}").n[0])
    except Exception:
        return None


def search(terms, index, limit=5, store=None):
    """[{term, concept, label, kind, patients, use}] for each term (its best matches with patients), [] per term when
    nothing matches; a concept is listed once, under the first term that finds it."""
    rows, seen = [], set()
    for term in [t for t in dict.fromkeys(str(t).strip() for t in terms or []) if t][:8]:
        try:
            hits = index.search(term, k=limit)
        except Exception:
            hits = []
        found = 0
        for c in hits:
            if c.concept_id in seen or c.score < 0.75:
                continue
            seen.add(c.concept_id)
            n = c.n_patients if c.n_patients is not None else index.count(c.concept_id) if c.kind != "var" else \
                _variable_patients(store, c.concept_id[4:])
            rows.append({"term": term, "concept": c.concept_id, "label": c.label, "kind": KINDS.get(c.kind, c.kind),
                         "patients": _shown(n), "use": USES.get(c.kind, "")})
            found += 1
        if not found:
            rows.append({"term": term, "concept": None, "label": "not found in MIMIC-IV", "kind": "", "patients": None,
                         "use": ""})
    return rows


def suggestions(rows, limit=4):
    """Questions MES answers with what was found: a rate for a condition, the use of a treatment, a trajectory of a
    measurement, a comparison when a treatment and a condition were both found."""
    def group(r):
        return L.VARIABLES.get(r["concept"][4:], ("", "", "", (), ""))[4] if r["concept"].startswith("var:") else ""
    found = [r for r in rows if r["concept"]]
    # a condition: a curated phenotype or a syndrome / comorbidity flag (not a treatment flag such as vasopressin_24h)
    conditions = [r for r in found if r["concept"].startswith("phe:")] + \
        [r for r in found if r["concept"].startswith("var:") and group(r) in ("syndrome", "comorbidity")
         and L.VARIABLES[r["concept"][4:]][1] == "binary"]
    drugs = [r for r in found if r["concept"].split(":")[0] in ("drugclass", "drug")]
    # a measurement with a time series: a cohort lab or vital sign, else a lab test with many patients
    from mes.tools.rwd.patterns import SERIES
    measures = [r for r in found if r["concept"].startswith("var:") and r["concept"][4:] in SERIES] + \
        [r for r in found if r["concept"].startswith("lab:") and isinstance(r.get("patients"), int) and r["patients"] >= 1000]
    out = []
    cond = conditions[0]["label"].split(" (")[0] if conditions else ""
    if drugs and cond:
        out.append(f"In ICU patients with {cond}, is {drugs[0]['label']} associated with lower 28-day mortality?")
    if cond:
        out.append(f"What is the 28-day mortality of ICU patients with {cond}?")
    if drugs:
        out.append(f"How often is {drugs[0]['label']} given in ICU patients, and when is it started?")
    if measures:
        from mes.tools.rwd.patterns import SERIES_LABELS
        m = measures[0]
        name = SERIES_LABELS.get(m["concept"][4:]) if m["concept"].startswith("var:") else m["label"].split(" (")[0]
        out.append(f"How does {name.lower() if name else 'it'} change over the first 72 hours" +
                   (f" in {cond}?" if cond else " in the ICU?"))
    if cond:
        out.append(f"What are ICU patients with {cond} like, and how do those who die differ from survivors?")
    return list(dict.fromkeys(out))[:limit]


def run(question, terms, index, store, reason=""):
    """The result of a data-discovery study: what MIMIC-IV records for each term, and questions to ask with it."""
    terms = [t for t in terms or [] if _DATABASE_WORDS.sub(" ", str(t)).strip(" -?.,()")]   # not "MIMIC-IV patients"
    rows = search(terms, index, store=store)
    catalog = store.catalog() if store is not None else {}
    found = [r for r in rows if r["concept"]]
    missing = [r["term"] for r in rows if not r["concept"]]
    res = {"analysis": "catalog", "title": "What MIMIC-IV records about " + ", ".join(dict.fromkeys(r["term"] for r in rows))
           if rows else "What MIMIC-IV records", "population": "MIMIC-IV v2.2 (all adult hospital and ICU stays)",
           "n": None, "attrition": [], "estimates": [], "warnings": [reason] if reason else [], "diagnostics": {},
           "tables": {"overview": overview(store), "concepts": rows,
                      "suggestions": [{"question": q} for q in suggestions(found)]},
           "methods": "The MIMIC-IV concept index (curated conditions and drug classes, ICD codes, medications, lab tests "
                      "and ICU chart items) searched for each term; patients with each concept anywhere in the record "
                      "(fewer than 10 not shown).", "coverage": catalog.get("coverage")}
    if missing:
        res["warnings"].append("Not found in MIMIC-IV: " + ", ".join(missing))
    return res


def text(res):
    """The passage of a data-discovery study (aggregates only)."""
    rows = (res.get("tables") or {}).get("concepts") or []
    by_term = {}
    for r in rows:
        by_term.setdefault(r["term"], []).append(r)
    parts = []
    for term, rs in by_term.items():
        found = [r for r in rs if r["concept"]]
        if not found:
            parts.append(f"“{term}”: not found in MIMIC-IV")
            continue
        parts.append(f"“{term}”: " + "; ".join(f"{r['label']} ({r['kind']}" + (f", {r['patients']:,} patients" if
                                                                             isinstance(r['patients'], int) else
                                                                             ", <10 patients" if r['patients'] == "<10"
                                                                             else "") + ")" for r in found))
    sugg = [s["question"] for s in (res.get("tables") or {}).get("suggestions") or []]
    size = "; ".join(f"{r['what']}: {r['n']:,}" for r in (res.get("tables") or {}).get("overview") or []
                     if isinstance(r.get("n"), int))
    out = (f"MIMIC-IV v2.2 (Beth Israel Deaconess Medical Center, 2008–2019) — {size}. " if size else "") + \
        ("What it records — " + " | ".join(parts) + "." if parts else "")
    out = out.strip() or "MIMIC-IV v2.2: nothing in its concept index matches the question's terms."
    if sugg:
        out += " Questions MES can answer with these data: " + " ".join(sugg)
    return out
