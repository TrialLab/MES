"""
The cohort builder's working design (pure functions, no Streamlit): the criteria of a study design as rows — each with
its origin in the study, a switch, its comparison, window and role — the draft design they make, and the draft's flow
of patients beside the study's own, step by step (what changed, what is new, what is switched off).
"""

import copy

from mes.tools.rwd import library as L
from mes.tools.rwd.spec import DEFAULT_WINDOWS, Criterion, criterion_text

WINDOWED = ("drugclass", "drug", "px9", "px10", "lab", "item", "micro")
NUMERIC_OPS = (">", ">=", "<", "<=", "between")
OP_TEXT = {">": ">", ">=": "≥", "<": "<", "<=": "≤", "between": "between", "=": "is", "!=": "is not",
           "": "has a value"}
KIND_TEXT = {"phe": "condition · curated ICD codes", "drugclass": "drug class", "drug": "medication",
             "dx9": "ICD-9 diagnosis", "dx10": "ICD-10 diagnosis", "px9": "ICD-9 procedure", "px10": "ICD-10 procedure",
             "lab": "lab test", "item": "ICU chart item"}
# step ids of the compiler's attrition that no criterion row stands for, in the order it applies them
FIXED = {"measured": "with a measured value of the exposure", "survivors": "discharged alive",
         "landmark": "alive at the landmark", "onset": "outcome not already present",
         "new_users": "new users only", "one_treatment": "exactly one of the two treatments"}


def kind_of(concept):
    """numeric | category | binary (a cohort variable), measure (a lab test or ICU item), drug or code."""
    kind, _, key = str(concept).partition(":")
    if kind == "var":
        return L.VARIABLES.get(key, ("", "binary"))[1]
    if kind in ("lab", "item"):
        return "measure"
    return "drug" if kind in ("drugclass", "drug") else "code"


def _row(c, role, origin):
    window = c.get("window_h")
    window = [window[0], window[1]] if isinstance(window, (list, tuple)) and len(window) == 2 else None
    value = c.get("value")
    return {"uid": origin, "origin": origin, "role": role, "on": True, "concept": str(c["concept"]),
            "label": str(c.get("label") or ""), "op": str(c.get("op") or ""),
            "value": list(value) if isinstance(value, (list, tuple)) else value, "window_h": window,
            "negate": bool(c.get("negate"))}


def rows_of(spec):
    """The rows of a design: its inclusion and exclusion criteria (origins inc_0 …, exc_0 … like the compiler's
    steps), then its treatment / exposure and comparator."""
    rows = []
    for role, prefix in (("include", "inc"), ("exclude", "exc")):
        items = spec.get(role) or []
        for i, c in enumerate(items if isinstance(items, list) else []):
            if isinstance(c, dict) and c.get("concept"):
                rows.append(_row(c, role, f"{prefix}_{i}"))
    for role in ("exposure", "comparator"):
        c = spec.get(role)
        if isinstance(c, dict) and c.get("concept"):
            rows.append(_row(c, role, role))
    return rows


def new_row(concept, label, role, uid, base="icu", typical=None):
    """A criterion added by the user, with a first comparison: above the population's median for a number, the
    first level of a category, a value in the default window for a lab test."""
    kind, _, key = str(concept).partition(":")
    row = {"uid": uid, "origin": None, "role": role, "on": True, "concept": concept, "label": label or "", "op": "",
           "value": None, "window_h": list(DEFAULT_WINDOWS[role]) if kind in WINDOWED else None, "negate": False}
    if kind_of(concept) == "numeric":
        row.update(op=">=", value=typical if typical is not None else 0.0)
    return row


def criterion(row):
    return {"concept": row["concept"], "op": row.get("op") or "", "value": row.get("value"),
            "window_h": row.get("window_h"), "negate": bool(row.get("negate")), "label": ""}


def draft_of(spec, rows, grace=None, new_user=None):
    """The design the rows make (the study's other fields kept)."""
    d = copy.deepcopy(spec)
    on = [r for r in rows if r.get("on")]
    d["include"] = [criterion(r) for r in on if r["role"] == "include"]
    d["exclude"] = [criterion(r) for r in on if r["role"] == "exclude"]
    exposure = next((r for r in rows if r["role"] == "exposure"), None)
    if exposure is not None:
        d["exposure"] = criterion(exposure)
    comparator = next((r for r in rows if r["role"] == "comparator"), None)
    if comparator is not None:
        d["comparator"] = criterion(comparator) if comparator.get("on") else None
    if grace is not None:
        d["grace_h"] = float(grace)
    if new_user is not None:
        d["new_user"] = bool(new_user)
    return d


def _plain(row):
    return {k: row.get(k) for k in ("role", "concept", "op", "value", "window_h", "negate")}


def changed(row, originals):
    """The row differs from the criterion it comes from (a new row is not "changed": it is new)."""
    o = originals.get(row.get("origin"))
    return o is not None and _plain(row) != _plain(o)


def text(row):
    """The criterion of a row in words, e.g. "Highest lactate, first 24 h ≥ 4"."""
    c = Criterion.from_dict({**criterion(row), "label": row.get("label") or row["concept"]})
    return criterion_text(c)


def step_rows(rows):
    """{draft step id: row}: inc_j / exc_j of the draft are its j-th switched-on inclusion / exclusion rows."""
    out = {}
    for role, prefix in (("include", "inc"), ("exclude", "exc")):
        j = 0
        for r in rows:
            if r["role"] == role and r.get("on"):
                out[f"{prefix}_{j}"] = r
                j += 1
    return out


def beside(study_flow, draft_flow, rows, originals):
    """The steps of the draft beside the study's: [{key, label, n, was, state, role}] — base, the inclusion and
    exclusion rows in their order (a switched-off row stays in place, uncounted), then the steps the compiler adds;
    a step of the study the draft no longer has is listed as switched off. And the groups of a comparison."""
    was = {s["id"]: s["n"] for s in study_flow.get("steps") or []}
    now = {s["id"]: s for s in draft_flow.get("steps") or []}
    by_row = {r["uid"]: sid for sid, r in step_rows(rows).items()}
    out = []
    if "base" in now:
        out.append({"key": "base", "label": now["base"]["label"], "n": now["base"]["n"], "was": was.get("base"),
                    "state": "same", "role": "base"})
    for role in ("include", "exclude"):
        for r in [r for r in rows if r["role"] == role]:
            same_role = r.get("origin") is not None and originals.get(r["origin"], {}).get("role") == role
            before = was.get(r["origin"]) if same_role else None
            sid = by_row.get(r["uid"])
            if sid is None:                                  # switched off
                out.append({"key": r["uid"], "label": ("with " if role == "include" else "excluding ") + text(r),
                            "n": None, "was": before, "state": "off", "role": role})
                continue
            state = "new" if r.get("origin") is None else "changed" if changed(r, originals) else "same"
            out.append({"key": r["uid"], "label": now[sid]["label"] if sid in now else text(r),
                        "n": now[sid]["n"] if sid in now else None, "was": before, "state": state, "role": role})
    for sid in FIXED:
        if sid in now:
            out.append({"key": sid, "label": now[sid]["label"], "n": now[sid]["n"], "was": was.get(sid),
                        "state": "same" if sid in was else "new", "role": "fixed"})
        elif sid in was:
            out.append({"key": sid, "label": FIXED[sid], "n": None, "was": was[sid], "state": "off", "role": "fixed"})
    groups = None
    g, g0 = draft_flow.get("groups"), study_flow.get("groups") or {}
    if g:
        groups = {k: {"n": g.get(k), "was": g0.get(k)} for k in ("treated", "control")}
    return out, groups


def final(study_flow, draft_flow):
    """(patients in the draft's cohort, in the study's)."""
    return draft_flow.get("n"), study_flow.get("n")


def grace_options(current):
    """The grace periods the slider offers (hours), with the study's own."""
    steps = {0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168}
    try:
        steps.add(float(current))
    except (TypeError, ValueError):
        pass
    return sorted(v for v in steps if 0 <= v <= 168)
