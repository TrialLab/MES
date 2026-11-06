"""
The criteria of a study design as rows of a table (the design editor of the study panel): one row per inclusion,
exclusion, treatment and comparator criterion — its concept, comparison, value, window and whether it is absent —
and back. Pure functions; what a table cannot mean is reported as errors, never guessed.
"""

import math
import re

ROLES = {"include": "Inclusion", "exclude": "Exclusion", "exposure": "Treatment / exposure", "comparator": "Comparator"}
ROLE_KEYS = {v: k for k, v in ROLES.items()}
OPS = ["", ">", ">=", "<", "<=", "=", "!=", "between"]
NONE = "—"                                  # "no comparison" in the table (an empty cell would read "None")
CHOICES = [NONE] + OPS[1:]
COLUMNS = ["Role", "Concept", "Comparison", "Value", "From (h)", "To (h)", "Without"]


def _value_text(c):
    v = c.get("value")
    if c.get("op") == "between" and isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]:g}–{v[1]:g}" if all(isinstance(x, (int, float)) for x in v) else f"{v[0]}–{v[1]}"
    if v is None:
        return ""
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def rows_of(spec):
    """The table of a design's criteria."""
    rows = []
    for key in ("include", "exclude", "exposure", "comparator"):
        items = spec.get(key) or []
        for c in items if isinstance(items, list) else [items]:
            if not isinstance(c, dict) or not c.get("concept"):
                continue
            window = c.get("window_h") if isinstance(c.get("window_h"), (list, tuple)) else None
            hours = lambda v: "" if v is None else f"{v:g}" if isinstance(v, (int, float)) else str(v)
            rows.append({"Role": ROLES[key], "Concept": c["concept"], "Comparison": c.get("op") or NONE,
                         "Value": _value_text(c), "From (h)": hours(window[0]) if window else "",
                         "To (h)": hours(window[1]) if window and len(window) > 1 else "", "Without": bool(c.get("negate"))})
    return rows


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _number(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def criteria_of(rows):
    """({"include", "exclude", "exposure", "comparator"}, errors) from the table's rows."""
    out = {"include": [], "exclude": [], "exposure": None, "comparator": None}
    errors = []
    for n, r in enumerate(rows or [], start=1):
        if not isinstance(r, dict) or all(_blank(r.get(k)) or r.get(k) is False for k in COLUMNS):
            continue                                          # an empty row of the table
        role = ROLE_KEYS.get(str(r.get("Role") or "").strip())
        concept = "" if _blank(r.get("Concept")) else str(r.get("Concept")).strip()
        if not role:
            errors.append(f"row {n}: choose its role (inclusion, exclusion, treatment or comparator)")
            continue
        if not concept or ":" not in concept:
            errors.append(f"row {n}: write a concept id such as var:sepsis3 or drugclass:vasopressin (find one below)")
            continue
        op = "" if _blank(r.get("Comparison")) or str(r.get("Comparison")).strip() == NONE else str(r.get("Comparison")).strip()
        if op not in OPS:
            errors.append(f"row {n}: unknown comparison {op!r}")
            continue
        raw = None if _blank(r.get("Value")) else str(r.get("Value")).strip()
        value = None
        if op == "between":
            parts = [p for p in re.split(r"\s*(?:–|—|,|\bto\b|(?<=\d)-(?=[\d.]))\s*", raw or "") if p]
            nums = [_number(p) for p in parts]
            if len(nums) != 2 or None in nums:
                errors.append(f"row {n}: “between” needs two numbers, e.g. 2–4")
                continue
            value = sorted(nums)
        elif op in (">", ">=", "<", "<="):
            value = _number(raw)
            if value is None:
                errors.append(f"row {n}: {op} needs a number")
                continue
        elif op in ("=", "!="):
            if raw is None:
                errors.append(f"row {n}: {op} needs a value")
                continue
            value = _number(raw) if _number(raw) is not None else raw
        elif raw is not None:
            errors.append(f"row {n}: a value needs a comparison (>, <, = …)")
            continue
        lo, hi = r.get("From (h)"), r.get("To (h)")
        window = None
        if not (_blank(lo) and _blank(hi)):
            a, b = _number(lo), _number(hi)
            if a is None or (not _blank(hi) and b is None):
                errors.append(f"row {n}: the window is hours from time zero, e.g. From 0 To 24 (To empty: to the end "
                              "of the stay)")
                continue
            if b is not None and b <= a:
                errors.append(f"row {n}: the window must end after it starts")
                continue
            window = [a, b]
        c = {"concept": concept, "op": op, "value": value, "window_h": window, "negate": bool(r.get("Without") is True)}
        if role in ("exposure", "comparator"):
            if out[role] is not None:
                errors.append(f"row {n}: a design has one {ROLES[role].lower()} — remove the other row")
                continue
            out[role] = c
        else:
            out[role].append(c)
    return out, errors
