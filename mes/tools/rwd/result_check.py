"""
The result an analyst submits (``submit(result)`` in the sandbox): checked and normalized before it is used.

Only aggregates pass: tables without identifier columns and with at most MAX_ROWS rows, counts below MIN_CELL
suppressed, finite numbers, confidence limits that bracket their estimate. Every number shown in the answer comes
from here — it was computed by the code, not typed by the model.

Format (all keys optional except title and estimates)::

    {"title": "...", "analysis_type": "descriptive|comparative|association|prognostic|other",
     "population": "...", "time_zero": "...", "exposure": "...", "comparator": "...", "outcome": "...",
     "n": 22518, "attrition": [["Adult first ICU stays", 50920], ["with Sepsis-3", 22518]],
     "estimates": [{"label": "28-day mortality", "measure": "rate", "value": 18.4, "lo": 17.9, "hi": 19.0,
                    "n": 22518, "events": 4154, "unit": "%"}],
     "tables": {"by ICU type": DataFrame or list of dicts (rates as percentages, e.g. column "rate_%")},
     "km": {group: {"t": [...], "s": [...]}},
     "diagnostics": {"max_smd_after": 0.03}, "methods": "...", "limitations": ["..."], "warnings": ["..."]}

or, when MIMIC-IV cannot answer the question: ``{"feasible": False, "reason": "..."}``.
"""

import math
import re

MIN_CELL = 10
MAX_ROWS = 60
MAX_COLUMNS = 16
MAX_TABLES = 6
MEASURES = {"rate", "risk", "proportion", "survival", "median", "mean", "count", "RD", "RR", "OR", "HR", "IRR", "MD",
            "AUC", "slope", "correlation", "other"}
_ALIASES = {"percent": "rate", "percentage": "rate", "prevalence": "rate", "incidence": "rate", "odds ratio": "OR",
            "hazard ratio": "HR", "risk ratio": "RR", "relative risk": "RR", "risk difference": "RD",
            "mean difference": "MD", "c-statistic": "AUC", "auc": "AUC", "auroc": "AUC", "n": "count"}
PERCENT_MEASURES = {"rate", "risk", "proportion", "survival"}
ANALYSIS_TYPES = {"descriptive", "comparative", "association", "prognostic", "other"}
ID_COLUMNS = re.compile(r"^(subject_id|hadm_id|stay_id|unit_id|specimen_id|labevent_id|emar_id|pharmacy_id|poe_id|"
                        r"orderid|linkorderid|caregiver_id)$", re.I)
COUNT_COLUMN = re.compile(r"(^n$|^n_|_n$|count|patients|events|deaths|stays|admissions|^k$|^cases$)", re.I)


def _num(value, what):
    if value is None:
        return None
    if hasattr(value, "item"):          # numpy scalar
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, not {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, not {value!r}") from None
    return value if math.isfinite(value) else None


def _int(value, what):
    value = _num(value, what)
    return None if value is None else int(round(value))


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(f"{value:.6g}")
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)[:200]


def _small(n):
    return n is not None and 0 < n < MIN_CELL


def _estimate(e, i):
    if not isinstance(e, dict):
        raise ValueError(f"estimates[{i}] must be a dict with label, measure, value, lo, hi, n, events")
    label = str(e.get("label") or "").strip()
    if not label:
        raise ValueError(f"estimates[{i}] needs a label")
    measure = str(e.get("measure") or "other").strip()
    # written as an identifier too ("risk_ratio", "odds-ratio": seen in testing)
    measure = _ALIASES.get(measure.lower()) or _ALIASES.get(re.sub(r"[_\-]+", " ", measure.lower()), measure)
    if measure not in MEASURES:
        measure = next((m for m in MEASURES if m.lower() == measure.lower()), "other")
    out = {"label": label[:200], "measure": measure}
    for key in ("value", "lo", "hi", "p", "sd"):
        out[key] = _num(e.get(key), f"estimates[{i}].{key}")
    for key in ("n", "events"):
        out[key] = _int(e.get(key), f"estimates[{i}].{key}")
    if e.get("unit"):
        out["unit"] = str(e["unit"])[:30]
    v, lo, hi = out["value"], out["lo"], out["hi"]
    if v is not None and lo is not None and hi is not None and not (lo - 1e-9 <= v <= hi + 1e-9):
        raise ValueError(f"estimates[{i}] ({label}): the 95% CI ({lo}, {hi}) does not contain the estimate {v}")
    # rates are reported as percentages: a proportion that equals events / n is unambiguous and converted
    if measure in PERCENT_MEASURES and v is not None and 0 < v <= 1 and out.get("unit") != "%" \
            and out["n"] and out["events"] is not None and abs(v - out["events"] / out["n"]) < 1e-6:
        out.update(value=100 * v, lo=None if lo is None else 100 * lo, hi=None if hi is None else 100 * hi)
    if measure in PERCENT_MEASURES:
        out["unit"] = "%"
    if _small(out["n"]) or _small(out["events"]):
        out["small_cell"] = True
        out["n"] = None if _small(out["n"]) else out["n"]
        out["events"] = None if _small(out["events"]) else out["events"]
    if e.get("sensitivity"):
        out["sensitivity"] = True
    if e.get("primary"):
        out["primary"] = True
    return out


def _table(name, rows):
    if hasattr(rows, "to_dict"):        # DataFrame
        frame = rows
        if getattr(frame.index, "name", None) is not None or type(frame.index).__name__ != "RangeIndex":
            frame = frame.reset_index()
        frame.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
        rows = frame.to_dict("records")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"table '{name}' must be a DataFrame or a list of dicts")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"table '{name}' has {len(rows)} rows; aggregate it to at most {MAX_ROWS}")
    columns = list(dict.fromkeys(str(k) for r in rows for k in r))
    if any(ID_COLUMNS.match(c) for c in columns):
        raise ValueError(f"table '{name}' contains identifier columns; submit aggregates only")
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"table '{name}' has {len(columns)} columns; keep at most {MAX_COLUMNS}")
    out = []
    for r in rows:
        row = {}
        for c in columns:
            v = _scalar(r.get(c))
            if COUNT_COLUMN.search(c) and isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < MIN_CELL:
                v = f"<{MIN_CELL}"
            row[c] = v
        out.append(row)
    return out


def _km(curves):
    if not isinstance(curves, dict):
        raise ValueError("km must be a dict {group: {'t': [...], 's': [...]}} (stats.kaplan_meier returns it)")
    out = {}
    for group, c in list(curves.items())[:8]:
        t, s = list(c.get("t") or []), list(c.get("s") or [])
        if len(t) != len(s):
            raise ValueError(f"km['{group}']: t and s differ in length")
        step = max(1, len(t) // 300)
        out[str(group)] = {"t": [_scalar(x) for x in t[::step]], "s": [_scalar(x) for x in s[::step]],
                           **{k: _scalar(v) for k, v in c.items() if k not in ("t", "s") and not isinstance(v, (list, dict))}}
    return out


def normalize_result(obj):
    """Checked copy of a submitted result; raises ValueError with what to fix."""
    if not isinstance(obj, dict):
        raise ValueError("submit() takes a dict; see the result format in the instructions")
    if obj.get("feasible") is False:
        reason = str(obj.get("reason") or "").strip()
        if not reason:
            raise ValueError("say why MIMIC-IV cannot answer the question: {'feasible': False, 'reason': '...'}")
        return {"feasible": False, "reason": reason[:800], "title": str(obj.get("title") or "Not answerable")[:200],
                "estimates": []}
    title = str(obj.get("title") or "").strip()
    if not title:
        raise ValueError("the result needs a title")
    estimates = obj.get("estimates")
    if not isinstance(estimates, list) or not estimates:
        raise ValueError("the result needs a non-empty list of estimates")
    out = {"feasible": True, "title": title[:200], "estimates": [_estimate(e, i) for i, e in enumerate(estimates[:30])]}
    if not any(e["value"] is not None for e in out["estimates"]):
        raise ValueError("no estimate has a value: compute at least one number (rate, median, ratio …)")
    kind = str(obj.get("analysis_type") or "other").lower()
    out["analysis_type"] = kind if kind in ANALYSIS_TYPES else "other"
    for key in ("population", "time_zero", "exposure", "comparator", "outcome"):
        if obj.get(key):
            out[key] = str(obj[key])[:600]
    out["methods"] = str(obj.get("methods") or "")[:3000]
    n = _int(obj.get("n"), "n")
    out["n"] = None if _small(n) else n
    attrition = []
    for i, step in enumerate(obj.get("attrition") or []):
        if isinstance(step, dict):
            step = (step.get("step") or step.get("label"), step.get("n"))
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            raise ValueError(f"attrition[{i}] must be [description, number of patients]")
        count = _int(step[1], f"attrition[{i}] count")
        attrition.append([str(step[0])[:200], None if _small(count) else count])
    out["attrition"] = attrition[:20]
    tables = obj.get("tables") or {}
    if not isinstance(tables, dict):
        raise ValueError("tables must be a dict {name: DataFrame or list of dicts}")
    out["tables"] = {str(k)[:80]: _table(k, v) for k, v in list(tables.items())[:MAX_TABLES]}
    if obj.get("km"):
        out["tables"]["km"] = _km(obj["km"])
    diagnostics = obj.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics must be a dict of numbers or short texts")
    out["diagnostics"] = {str(k)[:60]: _scalar(v) for k, v in list(diagnostics.items())[:30]
                          if not isinstance(v, (list, dict, tuple, set))}
    for key in ("limitations", "warnings"):
        items = obj.get(key) or []
        if isinstance(items, str):
            items = [items]
        out[key] = [str(x)[:400] for x in items][:10]
    if any(e.get("small_cell") for e in out["estimates"]):
        out["warnings"].append(f"Counts below {MIN_CELL} are not shown (MIMIC-IV small-cell rule).")
    return out
