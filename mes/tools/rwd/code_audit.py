"""
Code check of a custom MIMIC-IV analysis (deterministic, no language model).

The analyst writes its own SQL and Python, so the codes and names that select patients, treatments and measurements
are its own choice. This module finds them in the code that produced the result — ICD codes, medication name
patterns and lists, lab / ICU item ids and names, MES variables, the tables used — and looks each one up in
MIMIC-IV: what it matches and how many patients have it.
  problems (to the result's warnings): a code, item or medication name that MIMIC-IV does not have, a pattern that
           matches nothing, a table that does not exist;
  notes (for the reader and the reviewer): a medication pattern that matches several ingredients ("%statin%" →
           atorvastatin, … and nystatin; "%epinephrine%" → epinephrine and norepinephrine).
The rows are shown with the analysis; problems and notes go to the independent review.
"""

import re

from mes.tools.rwd import library as L

ICD_ONE = re.compile(r"\bicd_code\s*(?:NOT\s+)?(?:I?LIKE|=)\s*'([^']{1,12})'", re.I)
ICD_LIST = re.compile(r"\bicd_code\s+(?:NOT\s+)?IN\s*\(([^)]*)\)", re.I)
DRUG_ONE = re.compile(r"\bdrug\)?\s*(?:NOT\s+)?(?:I?LIKE|=)\s*'([^']{2,80})'", re.I)
DRUG_LIST = re.compile(r"\bdrug\)?\s+(?:NOT\s+)?IN\s*\(([^)]*)\)", re.I)
DRUG_CALL = re.compile(r"\bdrug_names\(\s*[\"']([^\"']+)[\"']")
ITEMS = re.compile(r"\bitemid\s*(?:=\s*(\d{5,6})|(?:NOT\s+)?IN\s*\(([\d,\s]+)\))", re.I)
LABEL = re.compile(r"\blabel\)?\s*(?:I?LIKE|=)\s*'([^']{2,80})'", re.I)
TABLE = re.compile(r"\b(mimiciv_hosp|mimiciv_icu|mimiciv_derived|mes)\.([A-Za-z_][A-Za-z0-9_]*)\b")
QUOTED = re.compile(r"'([^']*)'")
MAX_ROWS = 40


def _like(pattern):
    """A SQL LIKE pattern as a regular expression (case-insensitive like ILIKE)."""
    out = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern.lower())
    return re.compile(f"^{out}$")


def _ingredients(names):
    """First word of each medication name (its ingredient, roughly), in order of appearance."""
    words = [next(iter(re.findall(r"[a-z]{4,}", n)), "") for n in names]
    return list(dict.fromkeys(w for w in words if w))


def _row(kind, written, matches="", examples="", patients=None, problem="", note=""):
    return {"kind": kind, "written": written, "matches": matches, "examples": examples, "patients": patients,
            "problem": problem, "note": note}


def _examples(names, n=4):
    names = list(names)
    return ", ".join(names[:n]) + (f" … (+{len(names) - n})" if len(names) > n else "")


def _patients(store, sql, params=None):
    try:
        return int(store.query(sql, params, timeout=20).n[0])
    except Exception:
        return None


def _drug_rows(store, index, code):
    rows = []
    names = index.drug_names.drug if hasattr(index, "drug_names") else []
    patterns = list(dict.fromkeys(DRUG_ONE.findall(code)))
    for pattern in patterns[:12]:
        rx = _like(pattern)
        hits = sorted(n for n in names if rx.match(n))
        if not hits:
            rows.append(_row("medication pattern", pattern, "0 names", problem="matches no medication name in MIMIC-IV"))
            continue
        ingredients = _ingredients(hits)
        rows.append(_row("medication pattern", pattern, f"{len(hits)} names", _examples(hits),
                         _patients(store, "SELECT count(DISTINCT subject_id) AS n FROM mes.drugs WHERE drug IN "
                                          "(SELECT unnest(?))", [hits[:400]]),
                         note=f"matches {len(ingredients)} different medications: {_examples(ingredients, 6)}"
                         if len(ingredients) > 1 else ""))
    known = set(names)
    for listed in list(dict.fromkeys(tuple(x.lower() for x in QUOTED.findall(b)) for b in DRUG_LIST.findall(code)))[:6]:
        missing = [x for x in listed if x not in known]
        rows.append(_row("medication names", _examples(listed, 3), f"{len(listed) - len(missing)} of {len(listed)} found",
                         patients=_patients(store, "SELECT count(DISTINCT subject_id) AS n FROM mes.drugs WHERE drug IN "
                                                   "(SELECT unnest(?))", [list(listed)]) if listed else None,
                         problem=f"not in MIMIC-IV: {_examples(missing, 3)}" if missing else ""))
    for concept in list(dict.fromkeys(DRUG_CALL.findall(code)))[:6]:
        cid = concept if ":" in concept else f"drug:{concept}"
        hits = index.drug_names_for(cid) if index.is_known(cid) else []
        rows.append(_row("drug class" if cid.startswith("drugclass:") else "medication", f'drug_names("{concept}")',
                         f"{len(hits)} names", _examples(hits), index.count(cid) if hits else None,
                         problem="" if hits else "unknown drug class or medication"))
    return rows


def _icd_rows(store, index, code):
    rows = []
    patterns = list(ICD_ONE.findall(code))
    for block in ICD_LIST.findall(code):
        patterns += QUOTED.findall(block)
    dictionary = index.dictionary
    codes = dictionary[dictionary.concept_id.str.match(r"^(dx|px)(9|10):")]
    keys = codes.concept_id.str.split(":", n=1).str[1].str.upper()
    for pattern in list(dict.fromkeys(p.strip().upper() for p in patterns if p.strip()))[:20]:
        rx = _like(pattern.lower())
        hit = codes[keys.str.lower().str.match(rx.pattern)]
        dx = hit[hit.concept_id.str.startswith("dx")]
        n = _patients(store, "SELECT count(DISTINCT subject_id) AS n FROM mes.diagnoses WHERE upper(icd_code) LIKE ?",
                      [pattern]) if len(dx) else None
        rows.append(_row("ICD code", pattern, f"{len(hit)} codes",
                         _examples(f"{k.split(':', 1)[1]} {lab[:40]}" for k, lab in zip(hit.concept_id, hit.label)), n,
                         problem="" if len(hit) else "no such ICD code in MIMIC-IV"))
    return rows


def _raw_items(store):
    """Every lab test and ICU item of MIMIC-IV: itemid, label (as in d_labitems / d_items)."""
    return store.query("SELECT itemid, label, 'lab' AS kind FROM mimiciv_hosp.d_labitems UNION ALL "
                       "SELECT itemid, label, 'item' AS kind FROM mimiciv_icu.d_items", timeout=20)


def _item_rows(store, index, code):
    rows = []
    ids = []
    for one, many in ITEMS.findall(code):
        ids += [one] if one else re.findall(r"\d{5,6}", many)
    names = LABEL.findall(code)
    if not ids and not names:
        return rows
    raw = _raw_items(store)
    dictionary = index.dictionary
    for itemid in list(dict.fromkeys(ids))[:20]:
        hit = dictionary[dictionary.concept_id.isin([f"lab:{itemid}", f"item:{itemid}"])]
        known = raw[raw.itemid == int(itemid)]
        label = hit.label.iloc[0] if len(hit) else (known.label.iloc[0] if len(known) else "")
        rows.append(_row("lab / ICU item", itemid, label, patients=int(hit.n_patients.iloc[0]) if len(hit) else None,
                         problem="" if label else "no such item id in MIMIC-IV"))
    for name in list(dict.fromkeys(names))[:20]:
        rx = _like(name)
        hit = raw[raw.label.fillna("").str.lower().str.match(rx.pattern)]
        rows.append(_row("lab / ICU item name", name, f"{len(hit)} items",
                         _examples(f"{lab} ({kind} {itemid})" for itemid, lab, kind in hit.itertuples(index=False)),
                         problem="" if len(hit) else "matches no lab test or ICU item name",
                         note=f"the name matches {len(hit)} items (different specimens or units?)" if len(hit) > 3 else ""))
    return rows


def _context_rows(store, code):
    rows = []
    known = {schema: set(store.tables(schema)) for schema in ("mes", "mimiciv_hosp", "mimiciv_icu", "mimiciv_derived")}
    for schema, table in sorted(set(TABLE.findall(code))):
        if table not in known.get(schema, set()):
            rows.append(_row("table", f"{schema}.{table}", problem="no such table"))
        elif schema == "mimiciv_derived":
            rows.append(_row("derived concept (mimic-code)", f"{schema}.{table}"))
    flags = [k for k, v in L.VARIABLES.items() if v[1] == "binary" and re.search(rf"\b{re.escape(k)}\b", code)]
    for k in flags[:12]:
        rows.append(_row("MES variable", k, L.VARIABLES[k][0]))
    return rows


def audit(code, index, store):
    """Rows {kind, written, matches, examples, patients, problem} for ``code`` (the analysis's cells)."""
    if not code:
        return []
    index._load_tables()
    rows = []
    for part in (_icd_rows, _drug_rows, _item_rows):
        try:
            rows += part(store, index, code)
        except Exception as e:                              # the check must never stop the analysis
            rows.append(_row("check", part.__name__.strip("_").replace("_rows", ""),
                             note=f"could not be checked ({type(e).__name__})"))
    try:
        rows += _context_rows(store, code)
    except Exception as e:
        rows.append(_row("check", "tables", note=f"could not be checked ({type(e).__name__})"))
    return rows[:MAX_ROWS]


def problems(rows):
    """The problems of an audit as sentences (for the review and the result's warnings)."""
    return [f"{r['kind']} “{r['written']}”: {r['problem']}" for r in rows if r.get("problem")]


def review_text(rows):
    """The audit as lines for the reviewer."""
    lines = []
    for r in rows:
        bits = [r["matches"], r["examples"], f"{r['patients']:,} patients" if r.get("patients") is not None else ""]
        line = f"- {r['kind']} {r['written']}: " + "; ".join(b for b in bits if b)
        lines.append(line + (f" — PROBLEM: {r['problem']}" if r.get("problem") else "")
                     + (f" — note: {r['note']}" if r.get("note") else ""))
    return "\n".join(lines) or "- (no codes, medication names or item ids in the code)"
