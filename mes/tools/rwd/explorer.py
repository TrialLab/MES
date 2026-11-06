"""
Free SQL tier for descriptive questions the study templates do not cover (results labelled "exploratory").

Guards, all deterministic:
- the store connection is read-only with file access locked to the store (db.py);
- the query must parse as a single SELECT (DuckDB ``json_serialize_sql``) and may only use the documented schemas;
- the output must be aggregated: no patient / admission / stay identifiers, at most MAX_ROWS rows, counts below
  MIN_CELL are suppressed;
- two independently written queries must give the same result (within 1%), otherwise the answer is marked
  low-confidence and both results are shown;
- slips of other SQL dialects are rewritten for DuckDB before a model's query runs (an ordered-set aggregate used as
  a window of a grouped query, BigQuery's APPROX_QUANTILES / TIMESTAMP_DIFF, unquoted date parts), and an alias used
  but never defined is added when the error names its only candidate table;
- other errors are sent back to the model to fix the query (twice at most).
"""

import json
import re

import numpy as np
import pandas as pd

from mes.utils import parse_json_response

MAX_ROWS = 200
MIN_CELL = 10
ID_COLUMNS = re.compile(r"^(subject_id|hadm_id|stay_id|unit_id|specimen_id|labevent_id|emar_id|pharmacy_id|poe_id)$",
                        re.I)
ALLOWED_SCHEMAS = ("mes", "mimiciv_hosp", "mimiciv_icu", "mimiciv_derived")

SCHEMA_GUIDE = """DuckDB SQL over MIMIC-IV v2.2 (adults, Beth Israel Deaconess Medical Center 2008-2019; dates shifted per
patient — never use calendar dates; anchor_year_group gives the 3-year period).
Analysis tables (use these first):
- mes.icu_stays: one row per ICU stay. Keys stay_id, hadm_id, subject_id; intime, outtime, first_icu_stay (bool:
  first ICU stay of the patient), age, gender ('M','F'), race_group, admission_type, insurance, anchor_year_group,
  careunit ('MICU','SICU','CVICU','CCU','TSICU','MICU/SICU','Neuro'); los_icu_days, los_hosp_days; scores sofa, sapsii,
  oasis, charlson, gcs_min; first-24h labs creatinine_max, bun_max, wbc_max, hemoglobin_min, platelets_min,
  bilirubin_total_max, sodium_min, potassium_max, glucose_max, albumin_min, inr_max, lactate_max, ph_min,
  pao2fio2ratio_min; vitals heart_rate_mean, mbp_mean, mbp_min, spo2_min, temperature_max, weight; 0/1 flags
  vasopressor_24h, norepinephrine_24h, invasive_vent_24h, rrt_24h, invasive_vent_any, rrt_any, septic_shock, aki_7d,
  aki_stage2_7d; sepsis3 (bool); aki_stage_48h, aki_stage_7d (0-3); outcomes death_hosp, death_icu, death_28d,
  death_90d, death_1y, icu_readmit_30d, hosp_readmit_30d (0/1), days_to_death.
  Adult patients: WHERE first_icu_stay AND age >= 18.
- mes.admissions: one row per hospital admission (hadm_id, subject_id, admittime, dischtime, first_admission (bool),
  age, gender, race_group, admission_type, insurance, anchor_year_group, ed_admission, los_hosp_days, charlson,
  n_icu_stays, any_icu, death_hosp, death_30d, death_90d, death_1y, hosp_readmit_30d).
- mes.diagnoses (subject_id, hadm_id, seq_num, icd_code without dots, icd_version 9 or 10); titles in
  mimiciv_hosp.d_icd_diagnoses (icd_code, icd_version, long_title).
- mes.drugs (subject_id, hadm_id, stay_id, source 'order'|'icu_infusion', drug lower-case name, route, starttime,
  endtime). Match names with LIKE '%name%'.
- mes.procedures (subject_id, hadm_id, stay_id, source 'icd'|'icu_procedure', code, icd_version, label, time).
Raw tables: mimiciv_hosp.labevents (subject_id, hadm_id, itemid, charttime, valuenum, valueuom; item names in
mimiciv_hosp.d_labitems: itemid, label, fluid, category; hadm_id is empty for outpatient tests),
mimiciv_icu.chartevents (subject_id, hadm_id, stay_id, itemid, charttime, valuenum; names in mimiciv_icu.d_items:
itemid, label, category), mimiciv_hosp.microbiologyevents (subject_id, hadm_id, charttime, spec_type_desc e.g.
'BLOOD CULTURE', 'URINE'; test_name; org_name — NULL when nothing grew; ab_name, interpretation 'S'/'R'/'I'),
mimiciv_hosp.prescriptions (subject_id, hadm_id, starttime, stoptime, drug, route, dose_val_rx, dose_unit_rx),
mimiciv_icu.inputevents (subject_id, hadm_id, stay_id, starttime, endtime, itemid, amount, amountuom, rate),
mimiciv_icu.procedureevents (subject_id, hadm_id, stay_id, starttime, endtime, itemid, value),
mimiciv_hosp.transfers (subject_id, hadm_id, careunit, intime, outtime, eventtype), mimiciv_hosp.services
(hadm_id, curr_service), mimiciv_hosp.patients (subject_id, gender, anchor_age, dod). Restrict raw tables to the
study population by joining mes.icu_stays (stay_id or hadm_id, and its intime for "first 24 h"). Not organisms:
org_name 'CANCELLED', 'NEGATIVE', 'NO GROWTH'. mimic-code concepts in mimiciv_derived: ventilation (stay_id, starttime,
endtime, ventilation_status 'InvasiveVent', 'Tracheostomy', 'NonInvasiveVent', 'HFNC', 'SupplementalOxygen' — one row
per episode: the hours of invasive ventilation are the summed date_diff('minute', starttime, endtime) / 60 of
'InvasiveVent' episodes), vasoactive_agent (stay_id, starttime, endtime, norepinephrine, vasopressin, epinephrine,
dopamine, phenylephrine — rates, NULL when not running), antibiotic (stay_id, hadm_id, antibiotic, route, starttime,
stoptime), rrt (stay_id, charttime, dialysis_active, dialysis_type), sofa, sepsis3, kdigo_stages, first_day_lab,
charlson, ..."""

PROMPT = """{guide}

Write ONE DuckDB SELECT query that answers the question with aggregated results only: counts, percentages, medians,
means — never one row per patient, never identifier columns, at most 50 rows. Count patients with
count(DISTINCT subject_id) when patients can have several rows. Give columns readable names; numbers stay numbers (no
'%' signs, no CAST to VARCHAR); a ranking or a breakdown is one row per group (never JSON or arrays in a cell, e.g.
ORDER BY n DESC LIMIT 20 for "the most common"). DuckDB functions:
median(x), quantile_cont(x, 0.25) for percentiles (aggregates: no OVER in a GROUP BY query), date_diff('hour', a, b).
{variant}
QUESTION: {question}
Return JSON: {{"sql": "...", "explanation": "one sentence: population and how it is measured"}}"""

VARIANT_B = ("Write the query independently and carefully, as a second analyst checking the first: think about the "
             "denominator and about patients with several stays or rows. Structure it as a CTE (WITH …) that first "
             "selects the qualifying patients or stays, then aggregate them in the outer query.")
VARIANT_C = ("Two analysts disagreed on this question. Write the query as a third analyst, as plainly as possible (no "
             "CTE): check each criterion against the schema above — which table and column record it — and the "
             "denominator.")
FIX = """The query failed: {error}
Return the corrected JSON (DuckDB: median(x), quantile_cont(x, q); every table alias defined in FROM / JOIN)."""

SQL_SCHEMA = {"type": "object", "properties": {"sql": {"type": "string"}, "explanation": {"type": "string"}},
              "required": ["sql", "explanation"]}


def check_sql(store, sql):
    """Problem with ``sql`` or None: single SELECT over the allowed schemas."""
    parsed = json.loads(store.query("SELECT json_serialize_sql(?::VARCHAR) AS j", [sql]).j[0])
    if parsed.get("error"):
        return parsed.get("error_message", "not a single SELECT statement")
    if len(parsed.get("statements", [])) != 1:
        return "exactly one SELECT statement is allowed"
    for schema in re.findall(r"\b([a-z_]+)\.[a-z_]+\b", sql.lower()):
        if schema.startswith("mimiciv") or schema == "mes":
            if schema not in ALLOWED_SCHEMAS:
                return f"schema {schema} is not available"
    if re.search(r"\b(read_csv|read_parquet|read_json|glob|copy|attach|install|load|pragma)\b", sql, re.I):
        return "file access functions are not allowed"
    return None


# ---- slips of other dialects ---------------------------------------------------------------------------------
UNITS = r"(year|quarter|month|week|day|hour|minute|second)s?"
RENAMED = (("json_object_agg", "json_group_object"), ("jsonb_object_agg", "json_group_object"),
           ("json_agg", "json_group_array"), ("jsonb_agg", "json_group_array"), ("listagg", "string_agg"),
           ("nvl", "coalesce"))
NOT_ALIAS = {"where", "group", "order", "join", "left", "right", "inner", "outer", "full", "cross", "on", "limit",
             "union", "having", "natural", "using", "window", "qualify", "except", "intersect", "positional", "asof"}


def _close(sql, i):
    """Index after the parenthesis closing the one at ``sql[i]`` (quoted text skipped), or -1."""
    depth, quote = 0, None
    for j in range(i, len(sql)):
        ch = sql[j]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "()":
            depth += 1 if ch == "(" else -1
            if depth == 0:
                return j + 1
    return -1


def _args(inner):
    """The top-level, comma-separated arguments of a call."""
    out, depth, quote, start = [], 0, None, 0
    for j, ch in enumerate(inner):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "()":
            depth += 1 if ch == "(" else -1
        elif ch == "," and depth == 0:
            out.append(inner[start:j].strip())
            start = j + 1
    return out + [inner[start:].strip()]


def _one_slip(sql):
    """``sql`` with its first slip of another dialect rewritten, or None."""
    grouped = re.search(r"\bgroup\s+by\b", sql, re.I) is not None
    for m in re.finditer(r"\b(percentile_cont|percentile_disc)\s*\(\s*([0-9.]+)\s*\)\s*within\s+group\s*\(", sql, re.I):
        end = _close(sql, m.end() - 1)
        over = re.match(r"\s*over\s*\(", sql[end:], re.I) if end > 0 else None
        stop = _close(sql, end + over.end() - 1) if over else -1
        if stop < 0:
            continue
        if grouped:              # the group's percentile, not a window over its rows (an error in every dialect)
            return sql[:end] + sql[stop:]
        order = re.match(r"\s*order\s+by\s+(.+?)(?:\s+(asc|desc))?\s*$", sql[m.end():end - 1], re.I | re.S)
        if order:
            q = float(m.group(2))
            q = 1 - q if (order.group(2) or "").lower() == "desc" else q
            name = "quantile_cont" if m.group(1).lower().endswith("cont") else "quantile_disc"
            return sql[:m.start()] + f"{name}({order.group(1)}, {q:g})" + sql[end:]
    for m in re.finditer(r"\bapprox_quantiles\s*\(", sql, re.I):          # BigQuery: APPROX_QUANTILES(x, n)[OFFSET(k)]
        end = _close(sql, m.end() - 1)
        args = _args(sql[m.end():end - 1]) if end > 0 else []
        at = re.match(r"\s*\[\s*(?:safe_)?(offset|ordinal)\s*\(\s*(\d+)\s*\)\s*\]", sql[end:], re.I) if end > 0 else None
        if len(args) == 2 and args[1].isdigit() and int(args[1]) > 0 and at:
            k = int(at.group(2)) - (1 if at.group(1).lower() == "ordinal" else 0)
            return sql[:m.start()] + f"quantile_cont({args[0]}, {min(1, max(0, k / int(args[1]))):g})" + sql[end + at.end():]
    for m in re.finditer(r"\b(datediff|date_diff|timestampdiff|timestamp_diff|datetime_diff)\s*\(", sql, re.I):
        end = _close(sql, m.end() - 1)
        args = _args(sql[m.end():end - 1]) if end > 0 else []
        if len(args) != 3:
            continue
        if re.fullmatch(UNITS, args[0], re.I):              # SQL Server / MySQL: (unit, start, end)
            unit, a, b = args
        elif re.fullmatch(UNITS, args[2], re.I) and not args[0].startswith("'"):   # BigQuery: (end, start, unit)
            b, a, unit = args
        else:
            continue
        unit = re.match(UNITS, unit, re.I).group(1).lower()
        return sql[:m.start()] + f"date_diff('{unit}', {a}, {b})" + sql[end:]
    for other, duck in RENAMED:                   # PostgreSQL / Oracle names of functions DuckDB has under another
        fixed = re.sub(rf"\b{other}\s*\(", f"{duck}(", sql, flags=re.I)
        if fixed != sql:
            return fixed
    return None


def dialect(sql):
    """A model's query with the slips of other SQL dialects rewritten for DuckDB (the same results)."""
    for _ in range(20):
        fixed = _one_slip(sql)
        if fixed is None:
            break
        sql = fixed
    return sql


def repair(sql, error):
    """``sql`` with an alias it uses but never defines added, when the error names its only candidate table (a
    mechanical slip, fixed without the model), else ``sql``."""
    m = re.search(r'Referenced table "(\w+)" not found!\s*Candidate tables: "(\w+)"(?!\s*,)', str(error or ""))
    if not m:
        return sql
    alias, table = m.groups()
    bare = [t for t in re.finditer(rf"\b(?:from|join)\s+(?:\w+\.)?{table}\b(\s+(?:as\s+)?(\w+))?", sql, re.I)
            if not t.group(2) or t.group(2).lower() in NOT_ALIAS]
    if len(bare) != 1:
        return sql
    at = bare[0].start(1) if bare[0].group(1) else bare[0].end()
    return sql[:at] + f" AS {alias}" + sql[at:]


def _run(store, sql):
    """(result, problem) of a model's query: the guards, then the query."""
    error = "no SQL returned" if not sql else check_sql(store, sql)
    if error:
        return None, error
    try:
        df = store.query(sql, timeout=45)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"
    return df, check_result(df)


def check_result(df):
    if any(ID_COLUMNS.match(str(c)) for c in df.columns):
        return "the result contains patient / admission / stay identifiers; aggregate instead"
    if len(df) > MAX_ROWS:
        return f"the result has {len(df)} rows; aggregate to at most {MAX_ROWS}"
    return None


def suppress(df):
    """Counts below MIN_CELL become "<10"."""
    out = df.copy()
    for col in out.columns:
        if re.search(r"(^n$|count|num|patients|events|_n$|^n_)", str(col), re.I) and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].apply(lambda v: f"<{MIN_CELL}" if pd.notna(v) and 0 < v < MIN_CELL else v)
    return out


def _decimals(v):
    """The decimals a column of numbers was rounded to (7: not rounded)."""
    v = v[np.isfinite(v)]
    for d in range(7):
        if np.allclose(v, np.round(v, d), rtol=0, atol=1e-9):
            return d
    return 7


def agree(a, b, tol=0.01):
    """Same shape and numbers (within ``tol`` relative) after sorting rows. Numbers one query rounded more than the
    other agree within that rounding (a median of 1.8 days and the same median cast to 2)."""
    if a is None or b is None or a.shape != b.shape:
        return False
    try:
        na, nb = a.select_dtypes("number"), b.select_dtypes("number")
        va, vb = np.sort(na.to_numpy(dtype=float), axis=0), np.sort(nb.to_numpy(dtype=float), axis=0)
    except Exception:
        return False
    if va.shape != vb.shape or va.size == 0:
        return va.shape == vb.shape and a.astype(str).sort_values(list(a.columns)).values.tolist() == \
            b.astype(str).sort_values(list(b.columns)).values.tolist()
    for j in range(va.shape[1]):
        x, y = va[:, j], vb[:, j]
        dx, dy = _decimals(x), _decimals(y)
        slack = 0.5 * 10.0 ** -min(dx, dy) + 0.5 * 10.0 ** -max(dx, dy) if dx != dy else 0.0
        if not np.allclose(x, y, rtol=tol, atol=max(1e-9, slack), equal_nan=True):
            return False
    return True


def _attempt(question, llm, store, variant, report):
    prompt = PROMPT.format(guide=SCHEMA_GUIDE, variant=variant, question=question)
    log = []
    for _ in range(3):
        data = parse_json_response(llm.invoke(prompt, json_schema=SQL_SCHEMA, reasoning=False, max_tokens=1100))
        sql = (data or {}).get("sql", "") if isinstance(data, dict) else ""
        sql = dialect(sql.strip().rstrip(";"))
        df, error = _run(store, sql)
        if error and repair(sql, error) != sql:
            log.append({"sql": sql, "error": error})
            sql = repair(sql, error)
            df, error = _run(store, sql)
        log.append({"sql": sql, "error": error})
        if not error:
            return {"sql": sql, "explanation": data.get("explanation", ""), "df": df, "log": log}
        report(f"RWD explorer: fixing the query ({error[:100]})")
        prompt += "\n\n" + json.dumps({"sql": sql}) + "\n\n" + FIX.format(error=error)
    return {"sql": log[-1]["sql"] if log else "", "df": None, "error": log[-1]["error"] if log else "", "log": log}


# words of a question's terms that say nothing about which data a query must use
_PLAIN = {"patients", "patient", "intensive", "care", "unit", "units", "hospital", "admission", "admissions", "adult",
          "adults", "mimic", "database", "records", "record", "number", "rate", "rates", "incidence", "first", "hours",
          "days", "within", "during", "after", "before", "with", "without", "total", "count", "median", "average",
          "mean", "distribution", "proportion", "percentage", "most", "common", "commonly", "given", "frequency",
          "type", "types", "group", "groups", "time", "stay", "stays", "data", "level", "levels", "value", "values"}


# how the analysis tables name what a question's words mean ("length of stay" is los_…, "mortality" a death_… flag)
_COLUMN_WORDS = {"length": ("los",), "mortality": ("death", "died", "dod"), "survival": ("death", "dod"),
                 "died": ("death",), "dying": ("death",), "deaths": ("death",), "ventilation": ("vent",),
                 "ventilated": ("vent",), "intubation": ("vent",), "kidney": ("aki", "kdigo", "creatinine", "renal"),
                 "readmission": ("readmit",), "readmitted": ("readmit",), "dialysis": ("rrt", "crrt"),
                 "vasopressors": ("vasopressor", "norepinephrine"), "women": ("gender",), "female": ("gender",),
                 "males": ("gender",), "male": ("gender",), "older": ("age",), "elderly": ("age",)}


def _same(a, b):
    norm = lambda q: re.sub(r"\s+", " ", str(q or "").strip().rstrip(";").lower())      # noqa: E731
    return bool(a) and norm(a) == norm(b)


def _without_names(sql):
    """``sql`` without the names it gives (CTEs, table and column aliases): an alias "icu_delirium" says nothing about
    what the query counts (once delirium was counted from SOFA scores)."""
    sql = str(sql or "")
    names = set(re.findall(r"\b(\w+)\s+AS\s*\(", sql, re.I))                     # WITH name AS (
    names |= set(re.findall(r"\bAS\s+\"?(\w+)\"?", sql, re.I))                     # … AS name
    names |= set(re.findall(r"\b(?:FROM|JOIN)\s+[\w.]+\s+(?!ON\b|WHERE\b|JOIN\b|LEFT\b|INNER\b|GROUP\b|ORDER\b|USING\b)"
                            r"(\w+)", sql, re.I))
    for name in sorted(names, key=len, reverse=True):
        sql = re.sub(rf"\b{re.escape(name)}\b", " ", sql)
    return sql


def missing_terms(terms, *sqls):
    """The question's terms (planner concepts) that no word of the queries refers to (aliases aside): a query may
    answer another question (once "ECMO" was counted from the vasopressor flags)."""
    text = " ".join(_without_names(q) for q in sqls).lower()
    out = []
    for term in terms or []:
        words = {w for w in re.findall(r"[a-z][a-z0-9]{3,}", str(term).lower()) if w not in _PLAIN}
        abbrev = {w.lower() for w in re.findall(r"\(([A-Za-z0-9]{2,10})\)", str(term))}
        keys = words | abbrev
        if keys and not any(k in text or (len(k) > 5 and k[:-2] in text) or any(c in text for c in _COLUMN_WORDS.get(k, ()))
                            for k in keys):
            out.append(str(term))
    return out


def explore(question, llm, store, report=None, terms=()):
    """Result dict (same shape as analyses.run) for a free descriptive question; ``terms``: the question's concepts
    (each should be referred to by the query)."""
    report = report or (lambda msg: None)
    a = _attempt(question, llm, store, "", report)
    b = _attempt(question, llm, store, VARIANT_B, report) if a.get("df") is not None else None
    res = {"analysis": "explore", "title": question, "population": "MIMIC-IV (as defined by the query)",
           "n": None, "attrition": [], "estimates": [], "tables": {}, "diagnostics": {}, "warnings": [],
           "methods": "Exploratory aggregate SQL query written by the language model and checked by MES "
                      "(single read-only SELECT, aggregated output, small counts suppressed, independent "
                      "second query compared).", "sql": a.get("sql"), "sql_check": b.get("sql") if b else None}
    if a.get("df") is None:
        res["warnings"].append(f"No valid query could be written: {a.get('error')}")
        res["diagnostics"]["queries"] = a.get("log")
        return res
    twin = b is not None and _same(a.get("sql"), b.get("sql"))
    same = None if twin else agree(a["df"], b.get("df") if b else None)
    if same is False and b and b.get("df") is not None:
        # two analysts disagree: a third decides when it agrees with one of them
        c = _attempt(question, llm, store, VARIANT_C, report)
        if c.get("df") is not None and not _same(c.get("sql"), a.get("sql")) and not _same(c.get("sql"), b.get("sql")):
            if agree(c["df"], a["df"]):
                same = True
                res["diagnostics"]["queries_agreeing"] = "2 of 3 (the first and a third)"
            elif agree(c["df"], b["df"]):
                a, b, same = b, a, True
                res["sql"], res["sql_check"] = a.get("sql"), b.get("sql")
                res["diagnostics"]["queries_agreeing"] = "2 of 3 (the check and a third)"
    res["diagnostics"].update(independent_query_agrees=same, explanation=a.get("explanation"))
    if twin:
        res["warnings"].append("The check query came out the same as the first one: the result is not independently "
                               "confirmed.")
    elif not same:
        res["warnings"].append("A second, independently written query gave a different result: treat these numbers "
                               "as low confidence.")
        if b and b.get("df") is not None:
            res["tables"]["check_result"] = suppress(b["df"]).head(MAX_ROWS).to_dict("records")
    missing = missing_terms(terms, a.get("sql"))
    if missing:
        res["diagnostics"]["terms_not_in_query"] = "; ".join(missing)
        res["warnings"].append("The query does not refer to " + ", ".join(f"“{t}”" for t in missing)
                               + ": check that it answers the question.")
    res["tables"]["result"] = suppress(a["df"]).head(MAX_ROWS).to_dict("records")
    res["tables"]["result_columns"] = [str(c) for c in a["df"].columns]
    return res


EDITED_METHODS = ("Exploratory aggregate SQL query edited by the user and checked by MES (single read-only SELECT, "
                  "aggregated output, small counts suppressed); not compared with an independently written query.")


def run_edited(sql, store, title=""):
    """Result dict (same shape as ``explore``) of a query the user edited: the same guards as the model's queries
    (one read-only SELECT over the documented schemas, aggregated output, small counts suppressed); there is no
    independent second query to compare it with. ``res["error"]`` says why it did not run."""
    sql = str(sql or "").strip().rstrip(";").strip()
    res = {"analysis": "explore", "title": title, "population": "MIMIC-IV (as defined by the query)", "n": None,
           "attrition": [], "estimates": [], "tables": {}, "diagnostics": {"independent_query": "not run (query edited "
                                                                                              "by the user)"},
           "warnings": [], "methods": EDITED_METHODS, "sql": sql, "sql_check": None}
    error = "no SQL" if not sql else check_sql(store, sql)
    df = None
    if not error:
        try:
            df = store.query(sql, timeout=60)
            error = check_result(df)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
    if error:
        res["error"] = error
        res["warnings"].append(f"The edited query did not run: {error}")
        return res
    res["tables"]["result"] = suppress(df).head(MAX_ROWS).to_dict("records")
    res["tables"]["result_columns"] = [str(c) for c in df.columns]
    return res


def result_text(res):
    """Result table as text for the evidence passage."""
    rows = res["tables"].get("result") or []
    cols = res["tables"].get("result_columns") or (list(rows[0]) if rows else [])
    lines = []
    for r in rows[:30]:
        lines.append(", ".join(f"{c} = {r[c]:.4g}" if isinstance(r.get(c), float) else f"{c} = {r.get(c)}" for c in cols))
    return "; ".join(lines)
