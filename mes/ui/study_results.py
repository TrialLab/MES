"""
The "Results" of a real-world data study card (ui/web/viz_*.js) as blocks, one builder per analysis type: the risks of each
arm and forest plots of the effect estimates, survival curves, rates by subgroup, distributions, trends, the ROC
curve and calibration of a prognostic model, and charts of any aggregated table. Pure functions of the stored study
record; every chart carries its numbers as a table (the card's Chart / Table switch).
"""

import math
import re

from mes.tools.rwd import library as L
from mes.tools.rwd.report import estimate_text

RATIOS = ("RR", "OR", "HR", "IRR")
MEASURE_NAMES = {"RD": "Risk difference", "RR": "Risk ratio", "OR": "Odds ratio", "HR": "Hazard ratio", "MD": "Mean difference",
                 "IRR": "Rate ratio", "risk": "Risk", "rate": "Rate", "median": "Median", "mean": "Mean", "AUC": "AUC",
                 "slope": "Calibration slope", "survival": "Survival", "count": "Count", "proportion": "Proportion"}


def num(v):
    """A finite float, else None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def text_of(e):
    """An estimate's numbers as the answer writes them, without its label: "+6.0 percentage points (95% CI …)"."""
    full = estimate_text(e)
    label = str(e.get("label", ""))
    return full[len(label) + 2:] if full.startswith(label + ": ") else full


def short(label, outcome=""):
    """An estimate's label without the outcome it is about: "risk ratio (overlap weights)"."""
    label = str(label or "")
    if outcome and label.lower().startswith(outcome.lower() + ": "):
        label = label[len(outcome) + 2:]
    return label[:1].upper() + label[1:]


def table(columns, rows):
    """A table for the card: columns [(key, label, numeric)], rows of dicts."""
    return {"columns": [{"key": k, "label": lab, "numeric": bool(n)} for k, lab, n in columns], "rows": rows}


def compact(e):
    """An estimate as (value, interval) for a narrow column: ("+6.0 pp", "+0.5 to +11.1"), ("1.22", "1.03–1.43 · p 0.020")."""
    m, v, lo, hi, p = e.get("measure"), num(e.get("value")), num(e.get("lo")), num(e.get("hi")), num(e.get("p"))
    if v is None:
        return "—", ""
    pv = "" if p is None else " · p < 0.001" if p < 0.001 else f" · p {p:.3f}"
    if m == "RD":
        return f"{v:+.1f} pp", (f"{lo:+.1f} to {hi:+.1f}" if lo is not None and hi is not None else "")
    if m in ("risk", "rate", "proportion", "survival"):
        return f"{v:.1f}%", (f"{lo:.1f}–{hi:.1f}" if lo is not None and hi is not None else "")
    if m in RATIOS or m in ("AUC", "slope"):
        return f"{v:.2f}", (f"{lo:.2f}–{hi:.2f}{pv}" if lo is not None and hi is not None else pv.lstrip(" ·"))
    return _g(v), (f"{_g(lo)}–{_g(hi)}" if lo is not None and hi is not None else "")


def _est_rows(estimates, outcome):
    rows = []
    for e in estimates:
        value, interval = compact(e)
        rows.append({"label": short(e.get("label"), outcome), "value": num(e.get("value")), "lo": num(e.get("lo")),
                     "hi": num(e.get("hi")), "text": value, "textSub": interval, "full": text_of(e),
                     "primary": bool(e.get("primary")), "sensitivity": bool(e.get("sensitivity")),
                     "measure_label": MEASURE_NAMES.get(e.get("measure"), "Estimate")})
    return rows


def plain_name(name):
    """A treatment's name for a direction label, without its qualifier: "Hydrocortisone (systemic)" → "Hydrocortisone"."""
    name = re.sub(r"\s*\([^)]*\)", "", str(name or "")).strip()
    return name if len(name) <= 32 else name[:31].rstrip() + "…"


def _est_table(estimates):
    return table([("label", "Estimate", False), ("text", "Value", True)],
                 [{"label": e.get("label", ""), "text": text_of(e)} for e in estimates])


def forest_blocks(estimates, outcome, treated="", id_prefix="fx"):
    """Forest plots of the effect estimates: differences on a linear scale around 0, ratios on a log scale around 1."""
    blocks = []
    diffs = [e for e in estimates if e.get("measure") in ("RD", "MD") and num(e.get("value")) is not None]
    ratios = [e for e in estimates if e.get("measure") in RATIOS and (num(e.get("value")) or 0) > 0]
    direction = [f"lower with {plain_name(treated)}", f"higher with {plain_name(treated)}"] if treated else ["", ""]
    if diffs:
        unit = " pp" if all(e.get("measure") == "RD" for e in diffs) else ""
        blocks.append({"type": "chart", "id": f"{id_prefix}-diff", "title": "Absolute difference",
                       "subtitle": "percentage points, 95% CI" if unit else "95% CI",
                       "chart": {"type": "forest", "rows": _est_rows(diffs, outcome), "nul": 0, "log": False,
                                 "axis": "Risk difference (percentage points)" if unit else "Difference",
                                 "dir": direction, "table": _est_table(diffs)}})
    if ratios:
        blocks.append({"type": "chart", "id": f"{id_prefix}-ratio", "title": "Relative effect",
                       "subtitle": "ratio, 95% CI (log scale)",
                       "chart": {"type": "forest", "rows": _est_rows(ratios, outcome), "nul": 1, "log": True,
                                 "axis": "Ratio (log scale)", "dir": direction, "table": _est_table(ratios)}})
    return blocks


def km_block(res, title="Survival", x_label=None, id_="km"):
    curves = (res.get("tables") or {}).get("km") or {}
    if not isinstance(curves, dict) or not curves:
        return None
    meta = (res.get("charts") or {}).get("km") or {}
    groups_info = res.get("groups") or {}
    groups, rows = [], []
    for name, c in curves.items():
        if not isinstance(c, dict) or not c.get("t"):
            continue
        role = "treated" if name == groups_info.get("treated") else "control" if name == groups_info.get("control") else ""
        groups.append({"name": name, "role": role, "t": c.get("t"), "s": c.get("s"), "lo": c.get("lo"), "hi": c.get("hi"),
                       "n": c.get("n"), "at_risk": c.get("at_risk")})
        at = num(c.get("at_horizon"))
        rows.append({"group": name, "n": f"{int(c['n']):,}" if num(c.get("n")) is not None else "—",
                     "survival": f"{100 * at:.1f}%" if at is not None else "—"})
    if not groups:
        return None
    if any(g["role"] == "treated" for g in groups):          # the treatment first: its colour is the accent
        groups.sort(key=lambda g: {"treated": 0, "control": 1}.get(g["role"], 2))
    horizon = max([g["t"][-1] for g in groups]                 # the last time of a curve, or of its numbers at risk
                  + [num(p[0]) for g in groups for p in (g.get("at_risk") or [])
                     if isinstance(p, (list, tuple)) and len(p) == 2 and num(p[0]) is not None])
    start = meta.get("start") or ""
    caption = []
    if meta.get("weighted"):
        caption.append(f"Weighted with {meta['weighted']}.")
    if any(g.get("lo") for g in groups):
        caption.append("Shaded: pointwise 95% band (Greenwood, with the weights).")
    if (res.get("diagnostics") or {}).get("logrank_p") is not None:
        p = res["diagnostics"]["logrank_p"]
        caption.append(f"Log-rank p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}.")
    return {"type": "chart", "id": id_, "title": title, "subtitle": f"from {start}" if start else "",
            "caption": " ".join(caption),
            "chart": {"type": "km", "groups": groups, "horizon": horizon, "ordered": bool(meta.get("ordered")),
                      "x_label": x_label or (f"Days from {start}" if start else "Days"), "start": start,
                      "table": table([("group", "Group", False), ("n", "Patients", True), ("survival", "Survival at the end", True)],
                                     rows)}}


def strata_block(rows, overall=None, title="By subgroup", measure="Rate", unit="%", id_="strata"):
    """Rates (or medians) of each level of the stratification variables, one variable at a time."""
    rows = [r for r in rows or [] if isinstance(r, dict)]
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(str(r.get("variable", "")), []).append(r)
    if "median" in rows[0]:
        groups = [{"variable": v, "rows": [{"level": str(r.get("level")), "median": num(r.get("median")), "q1": num(r.get("q1")),
                                            "q3": num(r.get("q3")), "n": r.get("n")} for r in rs]} for v, rs in by.items()]
        t = table([("variable", "Subgroup", False), ("level", "Level", False), ("n", "Patients", True),
                   ("median", "Median", True), ("iqr", "IQR", True)],
                  [{"variable": r.get("variable"), "level": r.get("level"), "n": f"{r['n']:,}" if num(r.get("n")) is not None else "—",
                    "median": _g(r.get("median")), "iqr": f"{_g(r.get('q1'))}–{_g(r.get('q3'))}"} for r in rows])
        return {"type": "chart", "id": id_, "title": title, "subtitle": "median and interquartile range",
                "chart": {"type": "ranges", "groups": groups, "unit": unit, "table": t}}
    groups = [{"variable": v, "rows": [{"level": str(r.get("level")), "value": num(r.get("rate")), "lo": num(r.get("lo")),
                                        "hi": num(r.get("hi")), "n": r.get("n"), "events": r.get("events")} for r in rs]}
              for v, rs in by.items()]
    t = table([("variable", "Subgroup", False), ("level", "Level", False), ("n", "Patients", True), ("events", "Events", True),
               ("rate", f"{measure} (95% CI)", True)],
              [{"variable": r.get("variable"), "level": r.get("level"), "n": _int(r.get("n")), "events": _int(r.get("events")),
                "rate": f"{_g(r.get('rate'))}% ({_g(r.get('lo'))}–{_g(r.get('hi'))})"} for r in rows])
    return {"type": "chart", "id": id_, "title": title, "subtitle": f"{measure.lower()} with 95% CI",
            "chart": {"type": "dots", "groups": groups, "overall": overall, "unit": unit, "measure": measure,
                      "axis": f"{measure} (%)", "table": t}}


def _g(v):
    v = num(v)
    if v is None:
        return "—"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def _int(v):
    v = num(v)
    return f"{int(v):,}" if v is not None else "—"


# ============================================================================ per analysis
def compare_results(spec, res):
    blocks = []
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("",))[0]
    g = res.get("groups") or {}
    treated, control = g.get("treated", "treated"), g.get("control", "comparison")
    est = res.get("estimates") or []
    crude = {e.get("label", "").rsplit(": ", 1)[-1].replace(" (crude)", ""): e for e in est
             if e.get("measure") == "risk" and "(crude)" in e.get("label", "")}
    weighted = {e.get("label", "").split("weighted risk, ", 1)[-1]: e for e in est
                if e.get("measure") == "risk" and "weighted risk, " in e.get("label", "")}
    weighting = (res.get("diagnostics") or {}).get("weighting", "")
    if crude:
        rows = []
        for role, name in (("treated", treated), ("control", control)):
            c, w = crude.get(name), weighted.get(name)
            if not c:
                continue
            rows.append({"label": name, "role": role, "sub": f"{_int(c.get('n'))} patients · {_int(c.get('events'))} events",
                         "crude": {"value": num(c.get("value")), "lo": num(c.get("lo")), "hi": num(c.get("hi"))},
                         "weighted": num((w or {}).get("value")), "n": c.get("n"), "events": c.get("events"),
                         "text": f"{_g((w or c).get('value'))}%", "textSub": "weighted" if w else "crude"})
        if rows:
            t = table([("group", "Group", False), ("crude", "Crude risk (95% CI)", True), ("weighted", "Weighted risk", True)],
                      [{"group": r["label"], "crude": text_of(crude[r["label"]]),
                        "weighted": f"{_g(r['weighted'])}%" if r["weighted"] is not None else "—"} for r in rows])
            blocks.append({"type": "chart", "id": "arms", "title": f"Risk of {outcome[:1].lower() + outcome[1:]}" if outcome else "Risk",
                           "subtitle": "in each group: crude (95% CI) and after weighting",
                           "chart": {"type": "arms", "rows": rows, "weighting": weighting, "axis": "Risk (%)", "table": t}})
    effects = [e for e in est if e.get("measure") in ("RD", "MD") + RATIOS and not e.get("secondary")]
    blocks += forest_blocks(effects, outcome, treated)
    km = km_block(res, title="Survival after the landmark")
    if km:
        blocks.append(km)
    sub = (res.get("tables") or {}).get("subgroups") or []
    if sub:
        primary = next((e for e in est if e.get("primary") and e.get("measure") == "RD"), None)
        rows = [{"label": r.get("subgroup", ""), "n": r.get("n"), "value": num(r.get("rd")), "lo": num(r.get("lo")),
                 "hi": num(r.get("hi"))} for r in sub]
        t = table([("subgroup", "Subgroup", False), ("n", "Patients", True), ("rd", "Risk difference (95% CI)", True)],
                  [{"subgroup": r.get("subgroup"), "n": _int(r.get("n")),
                    "rd": f"{_g(r.get('rd'))} ({_g(r.get('lo'))} to {_g(r.get('hi'))})"} for r in sub])
        blocks.append({"type": "chart", "id": "subgroups", "title": "Effect by subgroup",
                       "subtitle": "risk difference (percentage points), 95% CI",
                       "chart": {"type": "subgroups", "rows": rows, "unit": "pp", "measure": "Risk difference",
                                 "overall": {"value": num(primary.get("value")), "lo": num(primary.get("lo")),
                                             "hi": num(primary.get("hi"))} if primary else None,
                                 "axis": "Risk difference (pp)",
                                 "dir": [f"lower with {plain_name(treated)}", f"higher with {plain_name(treated)}"],
                                 "table": t}})
    secondary = [e for e in est if e.get("secondary")]
    if secondary:
        blocks.append({"type": "chart", "id": "secondary", "title": "Secondary outcomes", "subtitle": "risk difference, 95% CI",
                       "chart": {"type": "forest", "rows": _est_rows(secondary, ""), "nul": 0, "axis": "Risk difference (pp)",
                                 "dir": [f"lower with {plain_name(treated)}", f"higher with {plain_name(treated)}"],
                                 "table": _est_table(secondary)}})
    medians = [e for e in est if e.get("measure") == "median"]
    if medians and not crude:
        blocks.append({"type": "table", "id": "medians", "title": f"{outcome or 'Outcome'} by group",
                       **table([("label", "Group", False), ("text", "Median", True)],
                               [{"label": short(e.get("label"), outcome), "text": text_of(e)} for e in medians])})
    return blocks


def association_results(spec, res):
    blocks = []
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("",))[0]
    est = res.get("estimates") or []
    risks = [e for e in est if e.get("measure") == "risk"]
    if risks:
        rows = [{"label": short(e.get("label"), outcome), "role": "treated" if i == 0 else "control",
                 "sub": f"{_int(e.get('n'))} patients · {_int(e.get('events'))} events",
                 "crude": {"value": num(e.get("value")), "lo": num(e.get("lo")), "hi": num(e.get("hi"))}, "weighted": None,
                 "n": e.get("n"), "events": e.get("events"), "text": f"{_g(e.get('value'))}%", "textSub": "crude"}
                for i, e in enumerate(risks)]
        blocks.append({"type": "chart", "id": "arms", "title": f"Risk of {outcome[:1].lower() + outcome[1:]}" if outcome else "Risk",
                       "subtitle": "with and without the exposure (crude, 95% CI)",
                       "chart": {"type": "arms", "rows": rows, "axis": "Risk (%)", "table": _est_table(risks)}})
    exposure = ((spec.get("exposure") or {}).get("label") or "the exposure")
    blocks += forest_blocks([e for e in est if e.get("measure") in RATIOS + ("MD",)], outcome, exposure)
    q = (res.get("tables") or {}).get("quartiles") or []
    if q:
        block = strata_block(q, title="By quartile of the exposure", id_="quartiles")
        if block:
            block["subtitle"] = "risk with 95% CI in each quarter of the patients — the shape of the association"
            blocks.append(block)
    return blocks


def _overall(res):
    e = next((x for x in res.get("estimates") or [] if x.get("primary") and x.get("measure") in ("rate", "risk")), None)
    if not e:
        return None
    return {"value": num(e.get("value")), "lo": num(e.get("lo")), "hi": num(e.get("hi")), "label": f"overall {_g(e.get('value'))}%"}


def rate_results(spec, res):
    blocks = []
    measure = "Use" if spec.get("analysis") == "utilization" else "Rate"
    block = strata_block((res.get("tables") or {}).get("strata"), _overall(res), measure=measure)
    if block:
        blocks.append(block)
    rest = [e for e in res.get("estimates") or [] if not e.get("primary")]
    if rest:
        blocks.append({"type": "table", "id": "more", "title": "Other estimates",
                       **table([("label", "Estimate", False), ("text", "Value", True)],
                               [{"label": e.get("label"), "text": text_of(e)} for e in rest])})
    return blocks


def distribution_results(spec, res):
    blocks = []
    tables = res.get("tables") or {}
    summary = tables.get("summary") or {}
    hist = tables.get("histogram") or {}
    if hist.get("edges") and hist.get("counts"):
        counts = hist["counts"]
        t = table([("bin", "Range", False), ("n", "Patients", True)],
                  [{"bin": f"{_g(a)}–{_g(b)}", "n": "<10" if 0 < k < 10 else f"{k:,}"}
                   for a, b, k in zip(hist["edges"][:-1], hist["edges"][1:], counts)])
        blocks.append({"type": "chart", "id": "hist", "title": f"Distribution of {summary.get('variable') or 'the variable'}",
                       "subtitle": "patients per bin (1st–99th percentile); shaded: interquartile range",
                       "caption": "Bins with fewer than 10 patients are drawn grey and not counted exactly.",
                       "chart": {"type": "hist", "edges": hist["edges"], "counts": counts, "label": summary.get("variable", ""),
                                 "unit": summary.get("unit", ""), "summary": summary, "table": t}})
    block = strata_block(tables.get("strata"), title="By subgroup", unit=summary.get("unit", ""))
    if block:
        blocks.append(block)
    return blocks


def survival_results(spec, res):
    blocks = []
    km = km_block(res, title="Survival")
    if km:
        blocks.append(km)
    blocks += forest_blocks([e for e in res.get("estimates") or [] if e.get("measure") in RATIOS], "",
                            (spec.get("exposure") or {}).get("label", ""))
    return blocks


def trend_results(spec, res):
    rows, ors = [], []
    for e in res.get("estimates") or []:
        m = re.search(r"(\d{4} - \d{4})$", str(e.get("label", "")))
        if m and e.get("measure") in ("rate", "median"):
            rows.append({"period": m.group(1).replace(" - ", "–"), "value": num(e.get("value")), "lo": num(e.get("lo")),
                         "hi": num(e.get("hi")), "n": e.get("n")})
        elif e.get("measure") in RATIOS:
            ors.append(e)
    blocks = []
    if rows:
        median = any(e.get("measure") == "median" for e in res.get("estimates") or [])
        t = table([("period", "Period", False), ("n", "Patients", True), ("value", "Median (IQR)" if median else "Rate (95% CI)", True)],
                  [{"period": r["period"], "n": _int(r["n"]),
                    "value": f"{_g(r['value'])}{'' if median else '%'} ({_g(r['lo'])}–{_g(r['hi'])})"} for r in rows])
        blocks.append({"type": "chart", "id": "trend", "title": "Over time",
                       "subtitle": "by 3-year anchor period (MIMIC-IV shifts dates; real years are only known to the period)",
                       "chart": {"type": "trend", "rows": rows, "unit": "" if median else "%",
                                 "measure": "median" if median else "rate", "interval": "IQR" if median else "95% CI",
                                 "axis": "Anchor year group", "table": t}})
    blocks += forest_blocks(ors, "", "")
    return blocks


def _pretty_term(label):
    """ "Odds ratio per SD: admission_type_SURGICAL SAME DAY ADMISSION" → "Admission type: Surgical same day admission"."""
    label = re.sub(r"^Odds ratio per SD: ", "", str(label or ""))
    for key, (name, kind, *_rest) in sorted(L.VARIABLES.items(), key=lambda kv: -len(kv[0])):
        if label.startswith(key + "_") and kind == "category":
            level = label[len(key) + 1:]
            return f"{name}: {level.capitalize() if level.isupper() else level}"
    return label


def prognostic_results(spec, res):
    blocks = []
    charts = res.get("charts") or {}
    pair = []
    if charts.get("roc"):
        auc = next((num(e.get("value")) for e in res.get("estimates") or [] if e.get("measure") == "AUC"), None)
        pair.append({"type": "chart", "id": "roc", "title": "Discrimination", "subtitle": "ROC curve of the cross-validated predictions",
                     "chart": {"type": "roc", "points": charts["roc"], "auc": auc,
                               "table": table([("fpr", "False positive rate", True), ("tpr", "True positive rate", True)],
                                              [{"fpr": f"{100 * p[0]:.1f}%", "tpr": f"{100 * p[1]:.1f}%"} for p in charts["roc"]])}})
    if charts.get("calibration"):
        pair.append({"type": "chart", "id": "calibration", "title": "Calibration", "subtitle": "observed vs predicted risk, by tenth",
                     "chart": {"type": "calibration", "bins": charts["calibration"],
                               "table": table([("pred", "Predicted", True), ("obs", "Observed", True), ("n", "Patients", True)],
                                              [{"pred": f"{b['pred']:.1f}%", "obs": f"{b['obs']:.1f}%", "n": f"{b['n']:,}"}
                                               for b in charts["calibration"]])}})
    if pair:
        blocks.append({"type": "grid", "blocks": pair})
    ors = [dict(e, label=_pretty_term(e.get("label"))) for e in res.get("estimates") or [] if e.get("measure") == "OR"]
    if ors:
        blocks.append({"type": "chart", "id": "predictors", "title": "Strongest predictors",
                       "subtitle": "odds ratio per standard deviation (log scale; no CIs for a penalized prediction model)",
                       "chart": {"type": "forest", "rows": _est_rows(ors, ""), "nul": 1, "log": True, "axis": "Odds ratio per SD",
                                 "table": _est_table(ors)}})
    return blocks


def _cell(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return "—" if v is None else str(v)


def table_chart(name, rows, id_):
    """Any aggregated table (an exploratory query's result, a table the analyst submitted): a chart of a numeric
    column by the first text column (choose the column on the chart), and the table itself."""
    rows = [r for r in rows or [] if isinstance(r, dict)][:60]
    if not rows:
        return None
    keys = list(dict.fromkeys(k for r in rows for k in r))
    numeric = [k for k in keys if any(num(r.get(k)) is not None and not isinstance(r.get(k), bool) for r in rows)]
    text = [k for k in keys if k not in numeric]
    t = table([(k, str(k).replace("_", " "), k in numeric) for k in keys], [{k: _cell(r.get(k)) for k in keys} for r in rows])
    if not numeric or len(rows) < 2:
        return {"type": "table", "id": id_, "title": name, **t}
    label_keys = text[:2] if text else []
    lows = {k for k in numeric if re.search(r"(^|_)(lo|lower|ci_lo|lcl)(_|%|$)", k, re.I)}
    highs = {k for k in numeric if re.search(r"(^|_)(hi|upper|ci_hi|ucl)(_|%|$)", k, re.I)}
    columns = []
    for k in numeric:
        if k in lows or k in highs:
            continue
        stem = re.sub(r"(_%|%)$", "", k)
        lo = next((x for x in lows if re.sub(r"(^|_)(lo|lower|ci_lo|lcl)", "", x) in (stem, stem + "_%", stem + "%", "_%", "%", "")
                   and len(lows) <= 2), None)
        hi = next((x for x in highs if re.sub(r"(^|_)(hi|upper|ci_hi|ucl)", "", x) in (stem, stem + "_%", stem + "%", "_%", "%", "")
                   and len(highs) <= 2), None)
        columns.append({"key": k, "label": str(k).replace("_", " "), "unit": "%" if "%" in k or "pct" in k.lower() else "",
                        "lo": lo if hi else None, "hi": hi if lo else None})
    default = next((c["key"] for c in columns if re.search(r"rate|risk|pct|%|mortality|mean|median", c["key"], re.I)), None)
    out = [{"label": " · ".join(str(r.get(k)) for k in label_keys) if label_keys else f"row {i + 1}",
            "values": {k: (num(r.get(k)) if num(r.get(k)) is not None else r.get(k)) for k in numeric}}
           for i, r in enumerate(rows)]
    return {"type": "chart", "id": id_, "title": name, "subtitle": f"{len(rows)} row{'s' if len(rows) != 1 else ''}",
            "chart": {"type": "bars", "rows": out, "columns": columns, "default": default, "table": t}}


def explore_results(spec, res):
    blocks = []
    tables = res.get("tables") or {}
    block = table_chart("Query result", tables.get("result"), "result")
    if block:
        blocks.append(block)
    check = table_chart("Independent check query", tables.get("check_result"), "check")
    if check:
        blocks.append(check)
    return blocks


def custom_results(res):
    """The analyst's submitted estimates and tables."""
    blocks = []
    est = res.get("estimates") or []
    blocks += forest_blocks([e for e in est if e.get("measure") in RATIOS + ("RD", "MD")], "", "", "ax")
    rates = [e for e in est if e.get("measure") in ("rate", "risk", "proportion", "survival") and num(e.get("value")) is not None]
    if rates:
        rows = _est_rows(rates, "")
        blocks.append({"type": "chart", "id": "ax-rates", "title": "Rates and proportions", "subtitle": "%, 95% CI where reported",
                       "chart": {"type": "forest", "rows": rows, "log": False, "zero": True, "axis": "%",
                                 "table": _est_table(rates)}})
    other = [e for e in est if e not in rates and e.get("measure") not in RATIOS + ("RD", "MD")]
    if other:
        blocks.append({"type": "table", "id": "ax-other", "title": "Other estimates",
                       **table([("label", "Estimate", False), ("text", "Value", True)],
                               [{"label": e.get("label"), "text": text_of(e)} for e in other])})
    for i, (name, rows) in enumerate((res.get("tables") or {}).items()):
        if name == "km" or not isinstance(rows, list):
            continue
        block = table_chart(str(name), rows, f"ax-t{i}")
        if block:
            blocks.append(block)
    km = km_block(res, title="Survival", id_="ax-km")
    if km:
        blocks.append(km)
    return blocks


def similar_results(spec, res):
    from mes.ui import similar_results as SR
    return SR.results(spec, res)


def _pattern(name):
    def build(spec, res):
        from mes.ui import pattern_results
        return pattern_results.RESULTS[name](spec, res)
    return build


RESULTS = {"compare": compare_results, "association": association_results, "rate": rate_results,
           "utilization": rate_results, "distribution": distribution_results, "survival": survival_results,
           "trend": trend_results, "prognostic": prognostic_results, "explore": explore_results,
           "similar": similar_results, "characterize": _pattern("characterize"), "trajectory": _pattern("trajectory"),
           "pathways": _pattern("pathways"), "top": _pattern("top")}
