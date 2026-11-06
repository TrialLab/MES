"""
How a MIMIC-IV cohort compares with a trial's participants: the baseline characteristics a trial posted on
ClinicalTrials.gov (its "Total" group) beside the same characteristics of a study's cohort, with standardized
differences — how far the cohort resembles the trial's patients, so how far a result carries over between them (the
generalizability of an emulation to its trial, or of a trial to the patients of this hospital). Age, sex, race and
ethnicity, the scores, labs and vital signs MIMIC-IV records (units converted), comorbidities and organ support are
matched by name; the rest is listed as not compared. No language model; aggregates only, groups under 10 patients
are not shown.

    fetch(nct) -> {"nct", "title", "n", "entries": [...]}  (or with "error")
    run(spec_dict, nct) -> {"nct", "title", "n_trial", "n_cohort", "rows", "other", "summary"}  (or with "error")
"""

import re
import threading

import numpy as np
import pandas as pd

from mes.tools.http_utils import get_json
from mes.tools.rwd import library as L

CTGOV = "https://clinicaltrials.gov/api/v2/studies"
SMALL = 10
MEANINGFUL = 0.1                 # |SMD| from which two groups differ meaningfully (the usual balance threshold)
NCT = re.compile(r"\bNCT\d{8}\b", re.I)
_COUNT = r"10\s*\^?\s*[39]|x\s*10|\bk\s*/|thousand|\bg\s*/\s*l|/\s*nl\b"

# numeric characteristics: (pattern of the trial's title, MIMIC-IV variable, [(unit pattern, factor to MIMIC-IV's
# unit)] — None: a score, any unit)
NUMERIC = [
    (r"\bage\b", "age", [(r"year", 1.0)]),
    (r"\bsofa\b|sequential organ failure", "sofa", None),
    (r"\bsaps[\s-]*(?:ii|2)\b|simplified acute physiology", "sapsii", None),
    (r"\boasis\b", "oasis", None),
    (r"charlson", "charlson", None),
    (r"glasgow|\bgcs\b", "gcs_min", None),
    (r"lact(?:ate|ic)", "lactate_max", [(r"mmol", 1.0), (r"mg\s*/\s*dl", 1 / 9.008)]),
    (r"creatinine", "creatinine_max", [(r"mg\s*/\s*dl", 1.0), (r"mol\s*/\s*l", 1 / 88.4)]),
    (r"\bbun\b|urea nitrogen", "bun_max", [(r"mg\s*/\s*dl", 1.0)]),
    (r"bilirubin", "bilirubin_total_max", [(r"mg\s*/\s*dl", 1.0), (r"mol\s*/\s*l", 1 / 17.1)]),
    (r"platelet", "platelets_min", [(_COUNT, 1.0)]),
    (r"white (?:blood )?cell|\bwbc\b|leukocyte", "wbc_max", [(_COUNT, 1.0)]),
    (r"ha?emoglobin", "hemoglobin_min", [(r"g\s*/\s*dl", 1.0), (r"g\s*/\s*l", 0.1)]),
    (r"sodium", "sodium_min", [(r"mmol|meq", 1.0)]),
    (r"potassium", "potassium_max", [(r"mmol|meq", 1.0)]),
    (r"glucose", "glucose_max", [(r"mg\s*/\s*dl", 1.0), (r"mmol", 18.016)]),
    (r"albumin", "albumin_min", [(r"g\s*/\s*dl", 1.0), (r"g\s*/\s*l", 0.1)]),
    (r"pao2\s*[:/]\s*fio2|p\s*/\s*f ratio|pf ratio", "pao2fio2ratio_min", [(r"mm\s*hg|torr", 1.0), (r"kpa", 7.50062)]),
    (r"heart rate|pulse rate", "heart_rate_mean", [(r"beat|bpm|/\s*min|per min", 1.0)]),
    (r"mean arterial|\bmap\b", "mbp_mean", [(r"mm\s*hg", 1.0)]),
    (r"respiratory rate", "resp_rate_mean", [(r"breath|/\s*min|per min", 1.0)]),
    (r"temperature", "temperature_max", [(r"°\s*c\b|celsius|degrees? c\b", 1.0), (r"°\s*f\b|fahrenheit|degrees? f\b", "f2c")]),
    (r"\bweight\b", "weight", [(r"\bkg\b|kilogram", 1.0), (r"\blbs?\b|pound", 0.45359)]),
]
# yes / no characteristics (counts of participants): (pattern, label, MIMIC-IV variables — any of them)
FLAGS = [
    (r"diabet", "History: diabetes", ("diabetes_without_cc", "diabetes_with_cc")),
    (r"heart failure|\bchf\b", "History: congestive heart failure", ("congestive_heart_failure",)),
    (r"\bcopd\b|chronic obstructive|chronic (?:lung|pulmonary|respiratory)", "History: chronic pulmonary disease",
     ("chronic_pulmonary_disease",)),
    (r"chronic kidney|chronic renal|renal (?:disease|insufficiency)|\bckd\b|end[- ]stage (?:renal|kidney)|"
     r"chronic (?:hemo)?dialysis|\besrd\b", "History: renal disease", ("renal_disease",)),
    (r"cirrhosis|liver disease", "History: liver disease", ("mild_liver_disease", "severe_liver_disease")),
    (r"metasta", "History: metastatic solid tumor", ("metastatic_solid_tumor",)),
    (r"cancer|malignan|neoplasm|tumou?r", "History: malignancy", ("malignant_cancer", "metastatic_solid_tumor")),
    (r"myocardial infarction", "History: myocardial infarction", ("myocardial_infarct",)),
    (r"stroke|cerebrovascular", "History: cerebrovascular disease", ("cerebrovascular_disease",)),
    (r"dementia", "History: dementia", ("dementia",)),
    (r"\bhiv\b|\baids\b", "History: AIDS", ("aids",)),
    (r"peripheral (?:vascular|arterial)", "History: peripheral vascular disease", ("peripheral_vascular_disease",)),
    (r"mechanical(?:ly)? ventilat|invasive (?:mechanical )?ventilat|intubat", "Invasive mechanical ventilation",
     ("invasive_vent_24h",)),
    (r"vasopressor|norepinephrine|noradrenaline", "Vasopressor", ("vasopressor_24h",)),
    (r"renal replacement|\bdialysis\b", "Renal replacement therapy", ("rrt_24h",)),
    (r"septic shock", "Septic shock", ("septic_shock",)),
]
RACE = [(r"\bwhite\b", "White"), (r"\bblack\b|african", "Black"), (r"\basian\b", "Asian")]
YES = re.compile(r"^\s*(?:yes|present|true|1)\s*$", re.I)
NO = re.compile(r"^\s*(?:no|absent|false|0|not reported|no reported|unknown|missing|not known|none)\s*$", re.I)
OLD = re.compile(r"(?:>=|≥|>|over|older than|at least)\s*6[45]|6[45]\s*(?:years?)?\s*(?:and|or)\s*(?:over|older|above)|"
                 r"6[45]\s*\+", re.I)
NOTES = {"labs": "MIMIC-IV: the worst value of the first 24 h in the ICU; a trial measures at enrollment",
         "severity": "MIMIC-IV: the score of the first 24 h in the ICU",
         "vitals": "MIMIC-IV: over the first 24 h in the ICU", "support": "MIMIC-IV: in the first 24 h in the ICU",
         "comorbidity": "MIMIC-IV: from the admission's diagnosis codes",
         "race": "MIMIC-IV records one group per patient: Hispanic patients are not counted as White or Black"}


def _num(x):
    try:
        v = float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _denom(denoms, group):
    for d in denoms or []:
        for c in d.get("counts") or []:
            if c.get("groupId") == group:
                return _num(c.get("value"))
    return None


def parse(data):
    """The baseline characteristics of a ClinicalTrials.gov record (API v2): {nct, title, n, entries} — one entry per
    value of the trial's total group (a measure, its class and category)."""
    ps = (data or {}).get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    b = ((data or {}).get("resultsSection") or {}).get("baselineCharacteristicsModule") or {}
    groups = b.get("groups") or []
    total = next((g.get("id") for g in groups if str(g.get("title") or "").strip().lower() == "total"), None)
    if total is None and len(groups) == 1:
        total = groups[0].get("id")
    out = {"nct": ident.get("nctId", ""), "title": ident.get("briefTitle") or ident.get("officialTitle") or "",
           "n": _denom(b.get("denoms"), total) if total else None, "entries": []}
    if total is None:
        return out
    for m in b.get("measures") or []:
        param, disp = str(m.get("paramType") or "").upper(), str(m.get("dispersionType") or "").upper()
        m_n = _denom(m.get("denoms"), total)
        for c in m.get("classes") or []:
            n = _denom(c.get("denoms"), total) or m_n or out["n"]
            for cat in c.get("categories") or []:
                x = next((x for x in cat.get("measurements") or [] if x.get("groupId") == total), None)
                if x is None or _num(x.get("value")) is None:
                    continue
                out["entries"].append({"title": str(m.get("title") or ""), "class": str(c.get("title") or ""),
                                       "category": str(cat.get("title") or ""), "param": param, "disp": disp,
                                       "unit": str(m.get("unitOfMeasure") or ""), "n": n, "value": _num(x.get("value")),
                                       "spread": _num(x.get("spread")), "lo": _num(x.get("lowerLimit")),
                                       "hi": _num(x.get("upperLimit"))})
    return out


_cache, _lock = {}, threading.Lock()


def fetch(nct):
    """The trial's posted baseline (cached; a failed request is not), with "error" when there is none to compare."""
    nct = str(nct or "").strip().upper()
    if not NCT.fullmatch(nct):
        return {"nct": nct, "title": "", "error": "not an NCT number (NCT and 8 digits)"}
    with _lock:
        if nct in _cache:
            return _cache[nct]
    fields = "protocolSection.identificationModule,resultsSection.baselineCharacteristicsModule"
    try:
        data = get_json(f"{CTGOV}/{nct}", {"fields": fields}, timeout=20)
    except Exception as e:
        return {"nct": nct, "title": "", "error": f"ClinicalTrials.gov could not be reached ({type(e).__name__})"}
    if not data:
        return {"nct": nct, "title": "", "error": "ClinicalTrials.gov has no record of this trial"}
    out = parse(data)
    out["nct"] = out["nct"] or nct
    if not out["entries"] or not out["n"]:
        out["error"] = "the trial has posted no baseline characteristics (results) on ClinicalTrials.gov"
    with _lock:
        _cache[nct] = out
    return out


def clear():
    with _lock:
        _cache.clear()


# ============================================================================ matching
def _factor(unit, units):
    if units is None:
        return 1.0
    return next((f for pattern, f in units if re.search(pattern, unit, re.I)), None)


def _numeric_trial(e, factor):
    if factor == "f2c":
        conv, scale = (lambda v: None if v is None else (v - 32) * 5 / 9), 5 / 9
    else:
        conv, scale = (lambda v: None if v is None else v * factor), factor
    if e["param"] == "MEDIAN":
        iqr = e["disp"] == "INTER_QUARTILE_RANGE"
        return {"median": conv(e["value"]), "q1": conv(e["lo"]) if iqr else None, "q3": conv(e["hi"]) if iqr else None}
    sd = e["spread"] if e["disp"] == "STANDARD_DEVIATION" else \
        e["spread"] * np.sqrt(e["n"]) if e["disp"] == "STANDARD_ERROR" and e["spread"] is not None and e["n"] else None
    return {"mean": conv(e["value"]), "sd": None if sd is None else sd * scale}


def _share(e):
    if e["param"] == "NUMBER" and re.search(r"percent|%", e["unit"], re.I):
        return e["value"] / 100
    return e["value"] / e["n"] if e["n"] else None


def match(entries, base):
    """(rows, other): the trial's characteristics MIMIC-IV records — [{key, kind, label, variables, level, group,
    trial, source}], one per characteristic — and the others [{characteristic, why}] (once per measure)."""
    variables = L.variables_for(base)
    rows, other, seen, listed = [], [], set(), set()

    def add(key, row):
        if key not in seen:
            seen.add(key)
            rows.append({"key": key, **row})

    def skip(e, why):
        if e["title"] not in listed:
            listed.add(e["title"])
            other.append({"characteristic": e["title"], "why": why})

    for e in entries:
        source = " · ".join(t for t in (e["title"], e["class"], e["category"]) if t)
        title = f"{e['title']} {e['class']}".strip()
        if e["param"] in ("MEAN", "MEDIAN", "GEOMETRIC_MEAN", "LEAST_SQUARES_MEAN"):
            hit = next(((v, u) for p, v, u in NUMERIC if re.search(p, title, re.I)), None)
            if hit is None or hit[0] not in variables:
                skip(e, "not among MIMIC-IV's analysis variables" if hit is None else
                     f"not recorded for {L.BASES[base]['unit']}s")
                continue
            factor = _factor(e["unit"], hit[1])
            if factor is None:
                skip(e, f"its unit ({e['unit'] or 'none given'}) is not converted")
                continue
            add(hit[0], {"kind": "numeric", "label": L.VARIABLES[hit[0]][0], "variables": (hit[0],), "level": None,
                         "group": L.VARIABLES[hit[0]][4], "unit": L.VARIABLES[hit[0]][2],
                         "trial": _numeric_trial(e, factor), "source": source})
            continue
        p = _share(e) if e["param"] in ("COUNT_OF_PARTICIPANTS", "NUMBER", "COUNT_OF_UNITS") else None
        if p is None or not 0 <= p <= 1:
            skip(e, "not a mean, median or count of participants" if p is None else "its share cannot be read")
            continue
        cat, share = e["category"], {"p": p, "n": e["n"]}
        if re.search(r"\b(?:sex|gender)\b", e["title"], re.I):
            if re.match(r"\s*(?:female|wom[ae]n)", cat, re.I):
                add("female", {"kind": "share", "label": "Women", "variables": ("gender",), "level": "F",
                               "group": "demographics", "trial": share, "source": source})
            continue
        if re.search(r"\bage\b", e["title"], re.I):
            if OLD.search(cat or e["class"]):
                add("age65", {"kind": "share", "label": "Aged 65 or older", "variables": ("age",), "level": 65,
                              "group": "demographics", "trial": share, "source": source})
            continue
        if re.search(r"\brace\b|ethnic|hispanic|latin", e["title"], re.I) and "race_group" in variables:
            hispanic = (re.search(r"hispanic|latin", cat, re.I) and not re.search(r"\bnot\b|\bnon\b|non-", cat, re.I)) \
                or (YES.match(cat) and re.search(r"hispanic|latin", e["title"], re.I))
            level = "Hispanic" if hispanic else next((lv for pat, lv in RACE if re.search(pat, cat, re.I)), None)
            if level:
                add(f"race_{level}", {"kind": "share", "label": f"{level} (race / ethnicity)", "variables": ("race_group",),
                                      "level": level, "group": "race", "trial": share, "source": source})
            continue
        if NO.match(cat):
            continue
        name = (e["class"] or e["title"]) if (not cat or YES.match(cat)) else cat
        hit = next(((lab, vs) for pat, lab, vs in FLAGS if re.search(pat, name, re.I)), None)
        found = tuple(v for v in (hit[1] if hit else ()) if v in variables)
        if not found:
            skip(e, "not among MIMIC-IV's analysis variables" if hit is None else
                 f"not recorded for {L.BASES[base]['unit']}s")
            continue
        add("+".join(found), {"kind": "share", "label": hit[0], "variables": found, "level": 1,
                              "group": L.VARIABLES[found[0]][4], "trial": share, "source": source})
    return rows, other


# ============================================================================ the comparison
def _smd(m1, s1, m2, s2):
    pooled = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    return float((m1 - m2) / pooled) if pooled > 0 else None


def _fmt(v):
    return "—" if v is None else f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}"


def _numeric_row(r, x):
    t = r["trial"]
    center = t.get("mean", t.get("median"))
    lo, hi = x.quantile([0.005, 0.995])
    if center is None or not lo - (hi - lo) <= center <= hi + (hi - lo):
        return None, "the trial's value lies outside MIMIC-IV's range (other units?)"
    if "mean" in t:
        mm, sm = float(x.mean()), float(x.std())
        smd = _smd(mm, sm, t["mean"], t["sd"]) if t.get("sd") else ((mm - t["mean"]) / sm if sm > 0 else None)
        return {"how": "mean (SD)", "mimic": f"{_fmt(mm)} ({_fmt(sm)})", "approx": not t.get("sd"), "smd": smd,
                "trial_text": f"{_fmt(t['mean'])}" + (f" ({_fmt(t['sd'])})" if t.get("sd") else ""),
                "centers": (_fmt(mm), _fmt(t["mean"]))}, None
    q1, med, q3 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    sm = (q3 - q1) / 1.349
    st_ = (t["q3"] - t["q1"]) / 1.349 if t.get("q1") is not None and t.get("q3") is not None else None
    smd = _smd(med, sm, t["median"], st_) if st_ else ((med - t["median"]) / sm if sm > 0 else None)
    return {"how": "median (IQR)", "mimic": f"{_fmt(med)} ({_fmt(q1)}–{_fmt(q3)})", "approx": True, "smd": smd,
            "trial_text": f"{_fmt(t['median'])}" + (f" ({_fmt(t['q1'])}–{_fmt(t['q3'])})" if st_ else ""),
            "centers": (_fmt(med), _fmt(t["median"]))}, None


def _share_row(r, df):
    var = r["variables"][0]
    if var == "age":
        hit = pd.to_numeric(df["age"], errors="coerce") >= r["level"]
    elif r["level"] != 1:                        # a level of a category: sex, race / ethnicity
        hit = df[var].astype(str) == str(r["level"])
    else:
        hit = pd.concat([pd.to_numeric(df[v], errors="coerce").fillna(0) >= 1 for v in r["variables"]], axis=1).any(axis=1)
    n, k = len(df), int(hit.sum())
    p, pt = k / n, r["trial"]["p"]
    small = 0 < k < SMALL or 0 < n - k < SMALL
    pooled = np.sqrt((p * (1 - p) + pt * (1 - pt)) / 2)
    smd = None if small else float((p - pt) / pooled) if pooled > 0 else 0.0
    text = "<10 patients" if 0 < k < SMALL else "all but <10" if 0 < n - k < SMALL else f"{100 * p:.1f}%"
    return {"how": "%", "mimic": text, "approx": False, "smd": smd, "trial_text": f"{100 * pt:.1f}%",
            "centers": (text, f"{100 * pt:.1f}%")}


def compare(trial, df, base):
    """The trial's baseline beside the cohort ``df`` (one row per patient, the matched variables as columns)."""
    matched, other = match(trial.get("entries") or [], base)
    rows = []
    for r in matched:
        if any(v not in df.columns for v in r["variables"]):
            other.append({"characteristic": r["source"], "why": "not in this cohort's data"})
            continue
        if r["kind"] == "numeric":
            x = pd.to_numeric(df[r["variables"][0]], errors="coerce").dropna()
            out, why = _numeric_row(r, x) if len(x) >= SMALL else (None, "fewer than 10 MIMIC-IV values")
        else:
            out, why = _share_row(r, df), None
        if out is None:
            other.append({"characteristic": r["source"], "why": why})
            continue
        unit = r.get("unit") if r["kind"] == "numeric" and r.get("unit") not in (None, "", "points") else ""
        rows.append({"key": r["key"], "name": r["label"], "label": r["label"] + (f", {unit}" if unit else ""),
                     "unit": unit, "how": out["how"], "trial": out["trial_text"], "mimic": out["mimic"],
                     "smd": out["smd"], "approx": out["approx"], "centers": out["centers"], "source": r["source"],
                     "note": NOTES.get(r["group"], "")})
    return rows, other


def summary(res):
    """One sentence: how the cohort differs from the trial's participants (the largest differences first)."""
    rows = [r for r in res.get("rows") or [] if r.get("smd") is not None]
    head = (f"Compared with the {int(res['n_trial']):,} participants of {res['nct']}, the {int(res['n_cohort']):,} "
            "patients of this MIMIC-IV cohort")
    if not rows:
        return head + ": no characteristic could be compared."
    differ = sorted([r for r in rows if abs(r["smd"]) >= MEANINGFUL], key=lambda r: -abs(r["smd"]))
    if not differ:
        return head + f" are similar in the {len(rows)} characteristics both record (every |SMD| < {MEANINGFUL})."
    parts = [f"{r['name']} {'higher' if r['smd'] > 0 else 'lower'} ({r['centers'][0]} vs {r['centers'][1]}"
             + (f" {r['unit']}" if r["unit"] else "") + ")" for r in differ[:4]]
    return head + ": " + "; ".join(parts) + f" — {len(differ)} of {len(rows)} shared characteristics differ " \
        f"(|SMD| ≥ {MEANINGFUL})."


def cohort_values(spec_dict, variables):
    """(df, base, error): the study's cohort built again from its design, with ``variables`` from its base table."""
    from mes.tools.rwd.compiler import Compiler
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.db import get_store
    from mes.tools.rwd.spec import StudySpec, validate
    spec = StudySpec.from_dict(spec_dict)
    index, store = get_index(), get_store()
    errors, _ = validate(spec, index)
    if errors:
        return None, spec.base, "; ".join(errors)
    cohort = Compiler(index, store).build(spec)
    b = L.BASES[spec.base]
    cols = [v for v in dict.fromkeys(variables) if v in store.columns(b["table"])]
    units = cohort.df[["unit_id"]].drop_duplicates()
    if not cols:
        return units, spec.base, None
    cur = store.cursor()
    try:
        cur.register("mes_trial_units", units)
        df = cur.execute(f"SELECT b.{b['id']} AS unit_id, " + ", ".join(f"b.{c}" for c in cols) +
                         f" FROM {b['table']} b JOIN mes_trial_units u ON b.{b['id']} = u.unit_id").df()
    finally:
        try:
            cur.unregister("mes_trial_units")
        except Exception:
            pass
        cur.close()
    return df, spec.base, None


def run(spec_dict, nct):
    """The trial ``nct``'s participants beside the cohort of the study designed by ``spec_dict``."""
    trial = fetch(nct)
    out = {"nct": trial.get("nct") or str(nct), "title": trial.get("title", ""), "n_trial": trial.get("n")}
    if trial.get("error"):
        return {**out, "error": trial["error"]}
    base = spec_dict.get("base") if spec_dict.get("base") in L.BASES else "icu"
    matched, _ = match(trial["entries"], base)
    try:
        df, base, error = cohort_values(spec_dict, [v for r in matched for v in r["variables"]])
    except Exception as e:
        return {**out, "error": f"the study's cohort could not be built: {type(e).__name__}: {str(e)[:200]}"}
    if error or df is None:
        return {**out, "error": f"the study's cohort could not be built: {error}"}
    if len(df) < SMALL:
        return {**out, "error": "the study's cohort has fewer than 10 patients"}
    rows, other = compare(trial, df, base)
    res = {**out, "n_cohort": len(df), "rows": rows, "other": other}
    res["summary"] = summary(res)
    return res
