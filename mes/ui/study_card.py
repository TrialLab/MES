"""
The card of a real-world data study (drawn by ui/web/viz_study.js): the header (citation, design, patients, checks), the
numbers a reader looks for first, and its tabs — Results (ui/study_results.py), Protocol (the target trial or study
protocol, a timeline, how the question's terms were matched), Cohort (the flow of patients, Table 1), Balance (a
checklist of the design, the Love plot, the overlap of the propensity scores, the diagnostics), Definitions (what each
concept means in the study's database); a custom analysis by the analyst has Results, Approach, Review, Cohort and Steps.

    card(result, study) -> spec | None

Pure functions of the stored record (no Streamlit). Old records, edited, failed and not-feasible studies all give a
card; a tab that cannot be built is left out, never the card.
"""

import logging

from mes.tools.rwd import DATASET, about, database_of
from mes.tools.rwd import library as L
from mes.tools.rwd.report import no_traceback
from mes.ui import study_results as R
from mes.ui.study_results import num, table, text_of

logger = logging.getLogger(__name__)

DESIGNS = {"compare": "Target trial emulation", "association": "Adjusted association", "rate": "Rate",
           "distribution": "Distribution", "utilization": "Treatment use", "survival": "Survival analysis",
           "trend": "Trend over time", "prognostic": "Prognostic model", "explore": "Exploratory query",
           "custom": "Custom analysis", "similar": "Patients like this", "characterize": "Characterization",
           "trajectory": "Trajectory", "pathways": "Treatment pathways", "catalog": "Data catalog",
           "top": "Commonest items"}
VERDICTS = {"sound": ("✓ independent review: no problems", "ok"), "minor_issues": ("◐ review: minor issues", "warn"),
            "major_issues": ("✕ review: major problems", "bad"), "not_reviewed": ("review not done", "warn")}


def _g(v, d=None):
    v = num(v)
    if v is None:
        return "—"
    if d is not None:
        return f"{v:.{d}f}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def _signed(v, d=1):
    v = num(v)
    return "—" if v is None else f"{v:+.{d}f}"


def _ci(e, signed=False, d=None):
    lo, hi = num(e.get("lo")), num(e.get("hi"))
    if lo is None or hi is None:
        return ""
    if signed:
        return f"95% CI {_signed(lo)} to {_signed(hi)}"
    return f"95% CI {_g(lo, d)}–{_g(hi, d)}"


def _number(result, study):
    evidence_id = study.get("evidence_id", "R")
    order = list(getattr(result, "citation_order", None) or [])
    return str(order.index(evidence_id) + 1) if evidence_id in order else str(evidence_id)


def _readable(text):
    return no_traceback(text)


# ============================================================================ key numbers
def _kpi_effect(e, tone="primary", note=""):
    m = e.get("measure")
    value = num(e.get("value"))
    if value is None:
        return None
    name = R.MEASURE_NAMES.get(m, "Estimate")
    if m == "RD":
        return {"label": name, "value": _signed(value), "unit": "pp", "ci": _ci(e, signed=True), "tone": tone,
                "note": note, "bar": {"value": value, "lo": num(e.get("lo")), "hi": num(e.get("hi")), "nul": 0,
                                      "min": -max(10, abs(num(e.get("lo")) or 0), abs(num(e.get("hi")) or 0)),
                                      "max": max(10, abs(num(e.get("lo")) or 0), abs(num(e.get("hi")) or 0))}}
    if m in R.RATIOS:
        lo, hi = num(e.get("lo")), num(e.get("hi"))
        return {"label": name, "value": f"{value:.2f}", "ci": _ci(e, d=2), "tone": tone, "note": note,
                "bar": {"value": value, "lo": lo, "hi": hi, "nul": 1, "min": 0, "max": max(2, (hi or value) * 1.1)}}
    if m in ("rate", "risk", "proportion", "survival"):
        return {"label": name if not note else note, "value": f"{value:.1f}", "unit": "%", "ci": _ci(e, d=1), "tone": tone,
                "note": f"{int(e['events']):,} of {int(e['n']):,}" if num(e.get("events")) is not None and num(e.get("n")) else "",
                "bar": {"value": value, "lo": num(e.get("lo")), "hi": num(e.get("hi")), "min": 0, "max": 100}}
    if m == "median":
        return {"label": "Median" if not note else note, "value": _g(value), "unit": e.get("unit", ""),
                "ci": f"IQR {_g(e.get('lo'))}–{_g(e.get('hi'))}" if num(e.get("lo")) is not None else "", "tone": tone,
                "note": f"{int(e['n']):,} patients" if num(e.get("n")) else ""}
    if m in ("AUC", "slope"):
        return {"label": name, "value": f"{value:.2f}", "tone": tone, "note": note}
    return {"label": name, "value": _g(value), "ci": _ci(e), "tone": tone, "note": note}


def kpis(study):
    res = study.get("result") or {}
    analysis = res.get("analysis") or (study.get("spec") or {}).get("analysis")
    if analysis == "similar":
        from mes.ui import similar_results
        return similar_results.kpis(study)
    if analysis in ("characterize", "trajectory", "pathways", "top"):
        from mes.ui import pattern_results
        return pattern_results.kpis(study)
    est = [e for e in res.get("estimates") or [] if not e.get("sensitivity")]
    out = []
    primary = next((e for e in est if e.get("primary")), est[0] if est else None)
    weighting = (res.get("diagnostics") or {}).get("weighting", "")
    if analysis == "compare" and primary:
        out.append(_kpi_effect(primary, note=weighting))
        rel = next((e for e in est if e.get("measure") == "HR"), None) or next((e for e in est if e.get("measure") == "RR"), None)
        if rel:
            out.append(_kpi_effect(rel, tone="", note=weighting))
        g = res.get("groups") or {}
        w = [e for e in est if "weighted risk, " in str(e.get("label", ""))]
        if len(w) == 2:
            out.append({"label": "Weighted risk", "value": f"{_g(w[0].get('value'), 1)}% vs {_g(w[1].get('value'), 1)}%",
                        "note": f"{g.get('treated', 'treated')} vs {g.get('control', 'comparison')}"})
        if num(g.get("n_treated")) is not None:
            out.append({"label": "Patients", "value": f"{int(g['n_treated']):,} / {int(g.get('n_control') or 0):,}",
                        "note": "treated / comparison"})
    elif analysis == "custom":
        for e in est[:4]:
            out.append(_kpi_effect(e, tone="primary" if e is primary else "", note=str(e.get("label", ""))[:80]))
    elif analysis == "prognostic":
        for m in ("AUC", "slope"):
            e = next((x for x in est if x.get("measure") == m), None)
            if e:
                out.append(_kpi_effect(e, tone="primary" if m == "AUC" else "",
                                       note="cross-validated" if m == "AUC" else "1 = well calibrated"))
        top = next((x for x in est if x.get("measure") == "OR"), None)
        if top and num(top.get("value")):
            out.append({"label": "Strongest predictor", "value": f"{top['value']:.2f}", "unit": "OR/SD",
                        "note": R._pretty_term(top.get("label"))})
    elif primary:
        out.append(_kpi_effect(primary, note="" if analysis not in ("rate", "utilization") else
                               "Overall rate" if analysis == "rate" else "Overall use"))
        for e in est:
            if e is primary or len(out) >= 4:
                continue
            if analysis in ("survival", "trend", "prognostic", "association", "utilization", "distribution"):
                out.append(_kpi_effect(e, tone="", note=R.short(e.get("label"), "")[:80]))
    n = res.get("n")
    if analysis not in ("compare",) and num(n) is not None and len(out) < 4:
        out.append({"label": "Patients", "value": f"{int(n):,}", "note": res.get("population", "")})
    return [k for k in out if k][:4]


# ============================================================================ header
def _badges(study):
    res = study.get("result") or {}
    out = []
    edited = (study.get("edited") or {}).get("kind")
    if edited:
        out.append({"text": f"✎ {edited if edited != 'design' else 'design'} edited by you", "kind": "edit"})
    if study.get("kind") == "analyst" and study.get("status") == "done":
        if edited:
            out.append({"text": "review not repeated after the edit", "kind": "warn"})
        else:
            text, kind = VERDICTS.get((study.get("review") or {}).get("verdict"), VERDICTS["not_reviewed"])
            out.append({"text": text, "kind": kind})
            ok = (study.get("reproduction") or {}).get("reproduced")
            out.append({"text": "✓ reproduced in a fresh session" if ok else "✕ not reproduced", "kind": "ok" if ok else "bad"})
    d = res.get("diagnostics") or {}
    if num(d.get("max_smd_after")) is not None:
        ok = d["max_smd_after"] < 0.1
        out.append({"text": f"{'✓' if ok else '✕'} balance: largest SMD {d['max_smd_after']:.2f} after weighting",
                    "kind": "ok" if ok else "bad"})
    if num(d.get("overlap")) is not None and d["overlap"] < 0.9:
        out.append({"text": f"⚠ overlap {100 * d['overlap']:.0f}%", "kind": "warn"})
    if d.get("independent_query_agrees") is True:
        out.append({"text": "✓ confirmed by an independent query" if not d.get("queries_agreeing") else
                    f"✓ {str(d['queries_agreeing']).split(' (')[0]} queries agree", "kind": "ok"})
    elif d.get("independent_query_agrees") is False:
        out.append({"text": "✕ the independent query disagrees", "kind": "bad"})
    warnings = [w for w in res.get("warnings") or [] if w]
    if warnings:
        out.append({"text": f"⚠ {len(warnings)} warning{'s' if len(warnings) > 1 else ''}", "kind": "warn"})
    centers = about(database_of(study))["centers"]               # what every study on its database shares
    out.append({"text": "observational" + (f" · {centers}" if centers else ""), "kind": "neutral"})
    return out


def _title(study):
    spec, res = study.get("spec") or {}, study.get("result") or {}
    analysis = res.get("analysis") or spec.get("analysis")
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("",))[0]
    population = "; ".join(c.get("label") or c.get("concept", "") for c in spec.get("include") or [] if isinstance(c, dict))
    if analysis == "compare":
        g = res.get("groups") or {}
        title = outcome or res.get("title") or "Treatment comparison"
        sub = " vs ".join(x for x in (g.get("treated"), g.get("control")) if x)
        return title, " · ".join(x for x in (sub, population) if x)
    if analysis == "custom":
        a = res.get("analyst") or {}
        return res.get("title") or "Custom analysis", " · ".join(x for x in (a.get("exposure"), a.get("outcome")) if x)[:240]
    if analysis == "similar":
        from mes.ui import similar_results
        return similar_results.title(study)
    title = res.get("title") or spec.get("title") or ""
    if ": " in title and title.split(": ", 1)[0] in ("Rate", "Distribution", "Treatment use", "Survival", "Trend",
                                                      "Comparison", "Association", "Prognostic model", "Exploratory query",
                                                      "Characteristics", "Over time", "Treatment pathways",
                                                      "Commonest"):
        title = title.split(": ", 1)[1]
        title = title[:1].upper() + title[1:]
    if not title:              # a design without a title (an edited one): what it measures
        exposure = (spec.get("exposure") or {}).get("label") if isinstance(spec.get("exposure"), dict) else ""
        variable = L.VARIABLES.get(spec.get("variable"), (spec.get("variable") or "",))[0]
        title = {"rate": outcome or exposure, "utilization": f"Use of {exposure}" if exposure else "",
                 "distribution": variable, "survival": outcome, "trend": f"{outcome or variable} over time",
                 "prognostic": f"Predictors of {outcome[:1].lower() + outcome[1:]}" if outcome else "",
                 "association": " and ".join(x for x in (exposure, outcome[:1].lower() + outcome[1:] if outcome else "") if x),
                 "explore": spec.get("sql_question") or ""}.get(analysis, "") or DESIGNS.get(analysis, "Study")
    subtitle = population or res.get("population", "")
    if population and population in title:          # the title names the population already: say who they were drawn from
        subtitle = res.get("population", "")
    return title, subtitle


# ============================================================================ protocol
def _crit(c, window=True):
    if not isinstance(c, dict):
        return ""
    from mes.tools.rwd.spec import Criterion, criterion_text
    try:
        text = criterion_text(Criterion.from_dict(c))
    except Exception:
        text = c.get("label") or c.get("concept", "")
    window = c.get("window_h") if window else None
    if isinstance(window, (list, tuple)) and len(window) == 2 and window[0] is not None:
        text += " (during the stay)" if window[1] is None else f" ({window[0]:g} to {window[1]:g} h)"
    return text


def _timeline(spec, res):
    analysis = spec.get("analysis")
    base = L.BASES.get(spec.get("base"), L.BASES["icu"])
    t0 = base["t0_label"]
    outcome = L.OUTCOMES.get(spec.get("outcome"))
    horizon = outcome[4] if outcome else None
    end = f"day {horizon}" if horizon else ("end of the stay" if outcome else "")
    if analysis == "compare":
        grace = num(spec.get("grace_h")) or 24
        segs, marks = [], []
        if spec.get("new_user", True):
            segs.append({"label": "no prior use (30 days)", "kind": "prior", "w": 1})
        segs.append({"label": f"treatment window · {grace:g} h", "kind": "exposure", "w": 1.3})
        segs.append({"label": f"follow-up · {outcome[0][:1].lower() + outcome[0][1:] if outcome else 'outcome'}", "kind": "follow",
                     "w": 3.2})
        total = sum(s["w"] for s in segs)
        at0 = (segs[0]["w"] / total) if spec.get("new_user", True) else 0
        at1 = at0 + 1.3 / total
        marks.append({"at": at0, "label": "Time zero", "sub": f"{t0}: eligibility", "align": "left" if at0 < 0.05 else ""})
        marks.append({"at": at1, "label": f"Landmark · {grace:g} h", "sub": "alive and in hospital; groups assigned"})
        marks.append({"at": 1, "label": end or "end of follow-up", "sub": "outcome assessed", "align": "right"})
        return {"segments": segs, "marks": marks}
    window = (spec.get("exposure") or {}).get("window_h") if isinstance(spec.get("exposure"), dict) else None
    if analysis == "utilization":
        lo, hi = (window or [0, 24])[:2]
        label = "the stay" if hi is None else f"{lo:g}–{hi:g} h"
        return {"segments": [{"label": f"treatment counted · {label}", "kind": "exposure", "w": 2},
                             {"label": "rest of the stay", "kind": "window", "w": 2}],
                "marks": [{"at": 0, "label": "Time zero", "sub": t0, "align": "left"},
                          {"at": 1, "label": "Discharge", "sub": "", "align": "right"}]}
    if analysis in ("rate", "survival", "association", "prognostic") and outcome:
        first = "first 24 h · characteristics" if analysis in ("association", "prognostic") else "eligibility at time zero"
        return {"segments": [{"label": first, "kind": "exposure" if analysis != "rate" else "window", "w": 1.2},
                             {"label": f"outcome window · {outcome[0][:1].lower() + outcome[0][1:]}", "kind": "follow", "w": 3}],
                "marks": [{"at": 0, "label": "Time zero", "sub": t0, "align": "left"},
                          {"at": 1, "label": end or "outcome", "sub": "", "align": "right"}]}
    if analysis == "distribution":
        return {"segments": [{"label": "first 24 h · value measured", "kind": "exposure", "w": 1.5},
                             {"label": "rest of the stay", "kind": "window", "w": 2.5}],
                "marks": [{"at": 0, "label": "Time zero", "sub": t0, "align": "left"}]}
    return None


def protocol_blocks(study):
    spec, res = study.get("spec") or {}, study.get("result") or {}
    analysis = spec.get("analysis")
    if not spec or analysis in (None, "custom"):
        return []
    base = L.BASES.get(spec.get("base"), L.BASES["icu"])
    rows = []
    eligibility = [base["label"]] + [f"with {_crit(c)}" for c in spec.get("include") or []] \
        + [f"excluding {_crit(c)}" for c in spec.get("exclude") or []]
    rows.append(["Eligibility", "; ".join(x for x in eligibility if x)])
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("",))[0]
    if analysis == "compare":
        grace = num(spec.get("grace_h")) or 24
        treat = _crit(spec.get("exposure"), window=False)
        comp = _crit(spec.get("comparator"), window=False) if spec.get("comparator") else f"no {treat}"
        rows += [["Treatment strategies", f"Start {treat} within {grace:g} h of {base['t0_label']} — vs — "
                                         + (f"start {comp} within {grace:g} h" if spec.get("comparator") else
                                            f"do not start it within {grace:g} h")],
                 ["Assignment", "Observational: emulated with a propensity score on the baseline covariates "
                                f"({(res.get('diagnostics') or {}).get('weighting') or 'weighting'})"],
                 ["Time zero", f"{base['t0_label'][:1].upper()}{base['t0_label'][1:]} (eligibility); groups assigned at the "
                               f"landmark, {grace:g} h"],
                 ["Follow-up", "From the landmark to the outcome or its horizon; "
                               + ("new users only (no use in the 30 days before)" if spec.get("new_user", True) else
                                  "prevalent users allowed")],
                 ["Outcome", outcome or "—"],
                 ["Causal contrast", "Observational analogue of the effect of starting the treatment within the grace "
                                     "period" + (" among patients for whom either strategy was plausible (overlap weights)"
                                                 if (res.get("diagnostics") or {}).get("weighting") == "overlap weights" else "")]]
    else:
        if analysis == "similar":
            from mes.ui import similar_results
            rows += similar_results.protocol_rows(spec)
        rows.append(["Time zero", base["t0_label"][:1].upper() + base["t0_label"][1:]])
        if spec.get("exposure"):
            rows.append(["Treatment option" if analysis == "similar" else "Exposure", _crit(spec["exposure"])])
        if spec.get("comparator"):
            rows.append(["Comparator", _crit(spec["comparator"])])
        if outcome:
            rows.append(["Outcome", outcome])
        if spec.get("variable"):
            rows.append(["Variable", L.VARIABLES.get(spec["variable"], (spec["variable"],))[0]])
        if spec.get("sql_question"):
            rows.append(["Question", spec["sql_question"]])
    if spec.get("stratify"):
        rows.append(["Broken down by", ", ".join(L.VARIABLES.get(v, (v,))[0] for v in spec["stratify"])])
    if spec.get("subgroups"):
        rows.append(["Subgroups", ", ".join(L.VARIABLES.get(v, (v,))[0] for v in spec["subgroups"])])
    if analysis in ("compare", "association", "prognostic") and spec.get("covariates"):  # (similar: its Table 1 only)
        rows.append(["Adjusted for", f"{len(spec['covariates'])} baseline covariates: "
                                     + ", ".join(L.VARIABLES.get(v, (v,))[0] for v in spec["covariates"])])
    if res.get("methods"):
        rows.append(["Analysis", res["methods"]])
    blocks = []
    tl = _timeline(spec, res)
    if tl:
        blocks.append({"type": "chart", "id": "timeline", "title": "Timeline", "subtitle": "schematic, not to scale",
                       "chart": {"type": "timeline", **tl}})
    blocks.append({"type": "kv", "id": "protocol", "title": "Target trial protocol" if analysis == "compare" else "Study protocol",
                   "rows": rows})
    info = study.get("design_info") or {}
    mapping = info.get("mapping") or []
    if mapping:
        blocks.append({"type": "chips", "id": "mapping", "title": "How the question's terms were matched",
                       "items": [{"text": f"“{m.get('term', '')}” → {m.get('label', '')}",
                                  "detail": " · ".join(x for x in (m.get("role", ""), f"{m['patients']:,} patients"
                                                                   if num(m.get("patients")) is not None else "") if x)}
                                 for m in mapping if isinstance(m, dict)]})
    notes = [n for n in (info.get("notes") or []) + ([info["approximation"]] if info.get("approximation") else []) if n]
    if notes:
        blocks.append({"type": "notes", "tone": "info", "title": "Design notes", "items": [_readable(n) for n in notes]})
    return blocks


# ============================================================================ cohort
def cohort_blocks(study):
    res = study.get("result") or {}
    steps = [s for s in res.get("attrition") or [] if isinstance(s, (list, tuple)) and len(s) >= 2]
    blocks = []
    g = res.get("groups") or {}
    split = None
    if steps and str(steps[-1][0]).startswith("treated: ") and num(g.get("n_treated")) is not None:
        steps = steps[:-1]
        split = [{"label": g.get("treated", "treated"), "n": g.get("n_treated"), "role": "treated",
                  "sub": f"{int(g['events_treated']):,} events" if num(g.get("events_treated")) is not None else ""},
                 {"label": g.get("control", "comparison"), "n": g.get("n_control"), "role": "control",
                  "sub": f"{int(g['events_control']):,} events" if num(g.get("events_control")) is not None else ""}]
    if steps:
        blocks.append({"type": "chart", "id": "flow", "title": "Flow of patients", "subtitle": "each criterion in turn",
                       "chart": {"type": "flow", "steps": [[str(s[0]), num(s[1])] for s in steps], "split": split,
                                 "table": table([("step", "Step", False), ("n", "Patients", True)],
                                                [{"step": str(s[0]), "n": f"{int(s[1]):,}" if num(s[1]) is not None else "<10"}
                                                 for s in steps])}})
    t1 = res.get("table1") or {}
    if t1.get("rows") and res.get("analysis") != "characterize":     # a characterization shows it in its Results
        blocks.append(table1_block(t1))
    return blocks


def table1_block(t1):
    groups = t1.get("groups") or ["treated", "comparison"]
    n = t1.get("n") or [None, None]
    head = lambda i: f"{groups[i]} (n = {int(n[i]):,})" if i < len(n) and num(n[i]) is not None else groups[i]
    columns = [{"key": "label", "label": "Characteristic"},
               {"key": "treated_raw", "label": head(0), "numeric": True, "view": "raw"},
               {"key": "control_raw", "label": head(1), "numeric": True, "view": "raw"},
               {"key": "smd_raw", "label": "SMD", "numeric": True, "view": "raw", "smd": True}]
    weighted = bool(t1.get("weighted"))
    if weighted:
        columns += [{"key": "treated_weighted", "label": f"{groups[0]} (weighted)", "numeric": True, "view": "weighted"},
                    {"key": "control_weighted", "label": f"{groups[1]} (weighted)", "numeric": True, "view": "weighted"},
                    {"key": "smd_weighted", "label": "SMD", "numeric": True, "view": "weighted", "smd": True}]
    rows = []
    for r in t1["rows"]:
        if r.get("heading"):
            rows.append({"label": r.get("label", ""), "__heading": True})
            continue
        label = r.get("label", "")
        if r.get("unit") and r.get("kind") == "numeric":
            label += f", {r['unit']} — mean (SD)"
        elif r.get("kind") == "numeric":
            label += " — mean (SD)"
        rows.append({**{k: r.get(k) for k in ("treated_raw", "control_raw", "treated_weighted", "control_weighted")},
                     "smd_raw": num(r.get("smd_raw")), "smd_weighted": num(r.get("smd_weighted")),
                     "label": label, "__level": bool(r.get("level"))})
    block = {"type": "table", "id": "table1", "title": "Baseline characteristics (Table 1)",
             "subtitle": "n (%) or mean (SD); SMD: standardized mean difference, red from 0.1; cells under 10 patients "
                         "are not shown", "columns": columns, "rows": rows, "smd": True, "max_height": 520}
    if weighted:
        block["toggle"] = {"label": "Weighting", "default": "raw",
                           "options": [{"key": "raw", "label": "Before weighting"}, {"key": "weighted", "label": "After weighting"}]}
    return block


# ============================================================================ balance and diagnostics
def _cov_label(name):
    name = str(name or "")
    if name.endswith("__missing"):
        return f"{_cov_label(name[:-9])} (missing)"
    if name in L.VARIABLES:
        return L.VARIABLES[name][0]
    for key in sorted(L.VARIABLES, key=len, reverse=True):
        if name.startswith(key + "_") and L.VARIABLES[key][1] == "category":
            return f"{L.VARIABLES[key][0]}: {name[len(key) + 1:]}"
    return name.replace("_", " ")


def scorecard(study):
    """A checklist of the design and its diagnostics: met, caution, not met — or a note."""
    spec, res = study.get("spec") or {}, study.get("result") or {}
    d = res.get("diagnostics") or {}
    analysis = spec.get("analysis")
    items = []
    if analysis == "compare":
        grace = num(spec.get("grace_h")) or 24
        items.append({"label": "Time zero aligned with treatment assignment", "status": "ok",
                      "detail": f"Treatment is the first dose within {grace:g} h; both groups are followed from the "
                                f"landmark ({grace:g} h), so no immortal time."})
        items.append({"label": "New users only", "status": "ok" if spec.get("new_user", True) else "warn",
                      "detail": "Patients treated in the 30 days before time zero are excluded." if spec.get("new_user", True)
                      else "Prevalent users are allowed: earlier treatment can bias the comparison."})
        items.append({"label": "Active comparator", "status": "ok" if spec.get("comparator") else "info",
                      "detail": f"Compared with {_crit(spec['comparator'])}." if spec.get("comparator") else
                      "Compared with patients who did not start the treatment: confounding by indication is harder to "
                      "remove than with an active comparator."})
        if num(d.get("max_smd_after")) is not None:
            ok = d["max_smd_after"] < 0.1
            items.append({"label": "Covariate balance after weighting", "status": "ok" if ok else "bad",
                          "detail": f"Largest SMD {d.get('max_smd_before', 0):.2f} before and {d['max_smd_after']:.2f} after "
                                    f"{d.get('weighting', 'weighting')} (threshold 0.10)."})
        if num(d.get("overlap")) is not None:
            items.append({"label": "Overlap of the propensity scores", "status": "ok" if d["overlap"] >= 0.9 else "warn",
                          "detail": f"{100 * d['overlap']:.0f}% of patients within the common support"
                                    + (f"; propensity model c-statistic {d['ps_auc']:.2f}" if num(d.get("ps_auc")) is not None else "")
                                    + "."})
        g = res.get("groups") or {}
        if num(d.get("ess_treated")) is not None and num(g.get("n_treated")):
            share = min(d["ess_treated"] / g["n_treated"], (d.get("ess_control") or 0) / max(1, g.get("n_control") or 1))
            items.append({"label": "Effective sample size", "status": "ok" if share >= 0.5 else "warn",
                          "detail": f"{d['ess_treated']:,.0f} of {int(g['n_treated']):,} treated and "
                                    f"{(d.get('ess_control') or 0):,.0f} of {int(g.get('n_control') or 0):,} comparison "
                                    "patients after weighting."})
        if num(d.get("events_treated")) is not None:
            low = min(d["events_treated"], d.get("events_control") or 0)
            items.append({"label": "Outcome events", "status": "ok" if low >= 20 else "warn" if low >= 10 else "bad",
                          "detail": f"{int(d['events_treated']):,} in the treated and {int(d.get('events_control') or 0):,} "
                                    "in the comparison group."})
        sens = [e for e in res.get("estimates") or [] if e.get("sensitivity") and num(e.get("value")) is not None]
        primary = next((e for e in res.get("estimates") or [] if e.get("primary")), None)
        if sens and primary and num(primary.get("value")) is not None:
            side = lambda e: (num(e["value"]) - (1 if e.get("measure") in R.RATIOS else 0))
            same = all((side(e) >= 0) == (side(primary) >= 0) for e in sens)
            items.append({"label": "Sensitivity analyses agree in direction", "status": "ok" if same else "warn",
                          "detail": "; ".join(f"{R.short(e.get('label'), '')}: {text_of(e)}" for e in sens)[:400]})
    if analysis == "association" and num(d.get("events_per_parameter")) is not None:
        epp = d["events_per_parameter"]
        items.append({"label": "Events per model parameter", "status": "ok" if epp >= 10 else "warn",
                      "detail": f"{epp:g} outcome events per parameter of the adjusted model (10 or more is the usual rule)."})
    if num(d.get("e_value")) is not None:
        ci = num(d.get("e_value_ci"))
        items.append({"label": "Robustness to unmeasured confounding (E-value)", "status": "info",
                      "detail": f"An unmeasured confounder would need a risk ratio of at least {d['e_value']:.2f} with both the "
                                f"exposure and the outcome to explain the estimate away"
                                + (" — the confidence interval already includes no effect." if ci is not None and ci <= 1.0
                                   else f" ({ci:.2f} for the confidence limit)." if ci is not None else ".")})
    if items:                                      # what every study on this database shares
        db = database_of(study)
        a = about(db)
        items.append({"label": "Observational data" + (f" from {a['centers']}" if a["centers"] else ""), "status": "info",
                      "detail": db + (f" ({a['where']})" if a["where"] else "") + ": residual confounding cannot be "
                                "excluded" + (f"; {a['limits']}" if a["limits"] else "") + "."})
    return items


DIAGNOSTICS = (("weighting", "Weighting"), ("max_smd_before", "Largest SMD before weighting"),
               ("max_smd_after", "Largest SMD after weighting"), ("overlap", "Share within common support"),
               ("ps_auc", "Propensity model c-statistic"), ("truncated_share", "Share with truncated weights"),
               ("ess_treated", "Effective sample (treated)"), ("ess_control", "Effective sample (comparison)"),
               ("events_treated", "Events (treated)"), ("events_control", "Events (comparison)"),
               ("e_value", "E-value (estimate)"), ("e_value_ci", "E-value (CI limit)"),
               ("events_per_parameter", "Events per model parameter"), ("logrank_p", "Log-rank p"),
               ("independent_query_agrees", "Independent query agrees"), ("queries_agreeing", "Queries that agree"),
               ("independent_query", "Independent check query"), ("terms_not_in_query", "Terms the query does not use"))


def balance_blocks(study):
    res = study.get("result") or {}
    d = res.get("diagnostics") or {}
    blocks = []
    items = scorecard(study)
    if items:
        blocks.append({"type": "scorecard", "id": "checks", "title": "Design and diagnostics checklist", "items": items})
    balance = (res.get("tables") or {}).get("balance") or []
    g = res.get("groups") or {}
    if balance and isinstance(balance, list):
        rows = [{"label": _cov_label(b.get("covariate")), "before": num(b.get("smd_before")),
                 "after": num(b.get("smd_after", b.get("smd"))), "mean_t": num(b.get("mean_treated")),
                 "mean_c": num(b.get("mean_control"))} for b in balance if isinstance(b, dict)]
        t = table([("label", "Covariate", False), ("before", "SMD before", True), ("after", "SMD after", True)],
                  [{"label": r["label"], "before": f"{r['before']:.3f}" if r["before"] is not None else "—",
                    "after": f"{r['after']:.3f}" if r["after"] is not None else "—"} for r in rows])
        blocks.append({"type": "chart", "id": "love", "title": "Covariate balance (Love plot)",
                       "subtitle": "|standardized mean difference| of each covariate, before and after weighting",
                       "chart": {"type": "love", "rows": rows, "threshold": 0.1, "weighting": d.get("weighting", ""),
                                 "groups": [g.get("treated", "treated"), g.get("control", "comparison")], "table": t}})
    ps = (res.get("charts") or {}).get("ps")
    names = [g.get("treated", "treated"), g.get("control", "comparison")]
    if isinstance(ps, dict) and ps.get("q") and ps.get("treated") and ps.get("control"):
        blocks.append({"type": "chart", "id": "overlap", "title": "Overlap of the propensity scores",
                       "subtitle": "distribution of the score in each group (shape only, from its percentiles)",
                       "caption": f"{100 * d['overlap']:.0f}% of patients within the common support." if num(d.get("overlap")) is not None else "",
                       "chart": {"type": "overlap", "q": ps["q"], "treated": ps["treated"], "control": ps["control"], "names": names}})
    elif isinstance(d.get("ps_treated"), list) and isinstance(d.get("ps_control"), list) and len(d["ps_treated"]) == 3:
        blocks.append({"type": "chart", "id": "overlap", "title": "Overlap of the propensity scores",
                       "subtitle": "5th, 50th and 95th percentile of the score in each group",
                       "chart": {"type": "overlap", "names": names,
                                 "ranges": [{"name": names[i], "p5": v[0], "p50": v[1], "p95": v[2]}
                                            for i, v in enumerate((d["ps_treated"], d["ps_control"]))]}})
    rows = []
    for key, label in DIAGNOSTICS:
        v = d.get(key)
        if v is None or isinstance(v, (list, dict)):
            continue
        text = (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}") if isinstance(v, float) else f"{v:,}" if isinstance(v, int) \
            and not isinstance(v, bool) else str(v)
        rows.append({"check": label, "value": text})
    if rows:
        blocks.append({"type": "table", "id": "diagnostics", "title": "Diagnostics",
                       **table([("check", "Check", False), ("value", "Value", True)], rows)})
    missing = {k: v for k, v in (d.get("covariate_missing") or {}).items() if num(v)}
    if missing:
        top = sorted(missing.items(), key=lambda kv: -kv[1])[:12]
        blocks.append({"type": "chart", "id": "missing", "title": "Missing covariate values",
                       "subtitle": "share of patients; imputed with the median plus a missing indicator",
                       "chart": {"type": "bars", "columns": [{"key": "share", "label": "Missing", "unit": "%"}],
                                 "rows": [{"label": _cov_label(k), "values": {"share": round(100 * v, 1)}} for k, v in top],
                                 "table": table([("label", "Covariate", False), ("share", "Missing", True)],
                                                [{"label": _cov_label(k), "share": f"{100 * v:.1f}%"} for k, v in top])}})
    return blocks


def definition_blocks(study):
    items = study.get("definitions")
    if not items:
        try:                       # older records: what the library itself says (no counts, no medication names)
            from mes.tools.rwd.definitions import describe
            items = describe(study.get("spec") or {}, None)
        except Exception:
            items = []
    items = [d for d in items or [] if isinstance(d, dict)]
    if not items:
        return []
    return [{"type": "defs", "id": "defs", "title": f"What each concept means in {database_of(study)}",
             "database": database_of(study),
             "subtitle": "code sets, medication names and sources of the criteria, the treatment and the outcome",
             "items": items}]


# ============================================================================ the analyst
def analyst_tabs(study):
    res = study.get("result") or {}
    a = res.get("analyst") or {}
    tabs = []
    if study.get("status") == "done":
        tabs.append({"id": "results", "label": "Results", "blocks": R.custom_results(res)})
        rows = [[k, v] for k, v in (("Question type", a.get("analysis_type")), ("Population", res.get("population")),
                                    ("Time zero", a.get("time_zero")), ("Exposure", a.get("exposure")),
                                    ("Comparator", a.get("comparator")), ("Outcome", a.get("outcome")),
                                    ("Methods", res.get("methods"))) if v]
        approach = [{"type": "kv", "id": "approach", "title": "Approach", "rows": [[k, _readable(str(v))] for k, v in rows]}] if rows else []
        if a.get("limitations"):
            approach.append({"type": "notes", "tone": "info", "title": "Limitations stated by the analyst",
                             "items": [_readable(x) for x in a["limitations"]]})
        if study.get("finish"):
            approach.append({"type": "text", "title": "The analyst's summary", "text": _readable(study["finish"])[:1500]})
        tabs.append({"id": "approach", "label": "Approach", "blocks": approach})
    review = []
    for i, rv in enumerate(study.get("reviews") or [], start=1):
        text, kind = VERDICTS.get(rv.get("verdict"), VERDICTS["not_reviewed"])
        status = {"ok": "ok", "warn": "warn", "bad": "bad"}.get(kind, "info")
        items = [{"label": f"Independent review{f' (round {i})' if len(study.get('reviews')) > 1 else ''}: "
                           f"{text.split(': ', 1)[-1]}", "status": status, "detail": _readable(rv.get("summary", ""))[:900]}]
        for issue in rv.get("issues") or []:
            sev = str(issue.get("severity") or "").lower()
            items.append({"label": _readable(issue.get("problem", ""))[:300], "status": "bad" if "major" in sev else "warn",
                          "detail": f"{sev or 'issue'} → {_readable(issue.get('fix', ''))[:400]}"})
        review.append({"type": "scorecard", "id": f"review{i}", "title": f"Review round {i}", "items": items})
    repro = study.get("reproduction") or {}
    if repro:
        review.append({"type": "notes", "tone": "info" if repro.get("reproduced") else "warn",
                       "title": "Reproduced in a fresh session" if repro.get("reproduced") else "Not reproduced",
                       "items": [_readable(repro.get("detail", "")) or "—"]})
    audit = study.get("code_audit") or []
    if audit:
        db = database_of(study)
        review.append({"type": "table", "id": "audit", "title": f"Codes and names in the code, looked up in {db}",
                       "subtitle": f"by a program, not by the model: ⚠ not in {db}; ⓘ worth a look",
                       **table([("written", "In the code", False), ("found", f"In {db}", False), ("patients", "Patients", True),
                                ("check", "Check", False)],
                               [{"written": f"{r.get('kind')}: {r.get('written')}",
                                 "found": " — ".join(x for x in (r.get("matches"), r.get("examples")) if x),
                                 "patients": f"{r['patients']:,}" if num(r.get("patients")) is not None else "",
                                 "check": f"⚠ {r['problem']}" if r.get("problem") else f"ⓘ {r['note']}" if r.get("note") else "✓"}
                                for r in audit])})
    if review:
        tabs.append({"id": "review", "label": "Review", "blocks": review})
    cohort = cohort_blocks(study)
    if cohort:
        tabs.append({"id": "cohort", "label": "Cohort", "blocks": cohort})
    steps = study.get("steps") or []
    if steps:
        cells = iter(study.get("cells") or [])
        rows = []
        for s in steps:
            ok = True
            if s.get("action") == "python":
                cell = next(cells, None) or {}
                ok = bool(cell.get("ok", True))
            rows.append({"action": s.get("action"), "seconds": num(s.get("seconds")), "plan": _readable(s.get("plan") or "")[:300],
                         "ok": ok})
        tabs.append({"id": "steps", "label": "Steps", "count": len(rows),
                     "blocks": [{"type": "chart", "id": "steps", "title": "The analyst's steps",
                                 "subtitle": "each turn's action and time; the code and output of each are below the card",
                                 "chart": {"type": "steps", "rows": rows}}]})
    return tabs


# ============================================================================ the card
def older_note(analysis, res):
    """A note under the results of a study stored before results carried the card's data (``res["card_data"]``):
    what asking the question again adds. Nothing for a study that has them or would not get them."""
    if res.get("card_data") or not res.get("estimates"):
        return []
    charts, tables = res.get("charts") or {}, res.get("tables") or {}
    continuous = any("(per SD" in str(e.get("label")) for e in res["estimates"])
    wanted = {"compare": ["table1", "ps", "km"], "association": [] if continuous else ["table1"], "survival": ["km"],
              "prognostic": ["roc"]}.get(analysis) or []
    names = {"table1": "the baseline table (Table 1)", "ps": "the distributions of the propensity scores",
             "km": "the confidence bands and numbers at risk of the survival curves", "roc": "the ROC and calibration curves"}
    parts = [names[k] for k in wanted if not (res.get("table1") if k == "table1" else charts.get(k))
             and (k != "km" or tables.get("km"))]
    if not parts:
        return []
    added = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return [{"type": "notes", "tone": "info", "title": "Computed by an earlier version of MES",
             "items": [f"Ask the question again (Re-run under the last answer) to add {added}."]}]


def catalog_blocks(study):
    res = study.get("result") or {}
    rows = [r for r in (res.get("tables") or {}).get("concepts") or [] if isinstance(r, dict)]
    out, last = [], None
    for r in rows:
        if r.get("term") != last:
            last = r.get("term")
            out.append({"label": f"“{last}”", "__heading": True})
        out.append({"label": r.get("label"), "kind": r.get("kind") or "—", "patients": f"{r['patients']:,}" if isinstance(
            r.get("patients"), int) else r.get("patients") or "—", "use": r.get("use") or "",
            "concept": r.get("concept") or "", "__level": True})
    blocks = []
    size = [r for r in (res.get("tables") or {}).get("overview") or [] if isinstance(r, dict) and num(r.get("n")) is not None]
    if size:
        blocks.append({"type": "kv", "id": "overview", "title": f"{study.get('dataset') or DATASET} at a glance",
                       "subtitle": about(database_of(study))["where"],
                       "rows": [[str(r.get("what")), f"{int(r['n']):,}"] for r in size]})
    if out:
        blocks.append({"type": "table", "id": "concepts", "title": f"What {database_of(study)} records",
                       "subtitle": "each term's matches in the concept index, with the patients who have them anywhere in "
                                   "the record (fewer than 10 not shown)",
                       "columns": [{"key": "label", "label": "Concept"}, {"key": "kind", "label": "Kind"},
                                   {"key": "patients", "label": "Patients", "numeric": True},
                                   {"key": "use", "label": "Used in a study as"}, {"key": "concept", "label": "Id"}],
                       "rows": out, "max_height": 560})
    sugg = [x.get("question") for x in (res.get("tables") or {}).get("suggestions") or [] if isinstance(x, dict)]
    if sugg:
        blocks.append({"type": "chips", "id": "ask", "title": "Questions MES can answer with these data",
                       "items": [{"text": q} for q in sugg if q]})
    if res.get("coverage"):
        blocks.append({"type": "text", "title": "The data", "text": str(res["coverage"])})
    return blocks


def catalog_kpis(study):
    rows = [r for r in ((study.get("result") or {}).get("tables") or {}).get("concepts") or [] if isinstance(r, dict)]
    found = [r for r in rows if r.get("concept")]
    missing = list(dict.fromkeys(r.get("term") for r in rows if not r.get("concept")))
    top = max((r for r in found if isinstance(r.get("patients"), int)), key=lambda r: r["patients"], default=None)
    out = [{"label": "Concepts found", "value": f"{len(found):,}", "tone": "primary",
            "note": f"for {len(dict.fromkeys(r.get('term') for r in rows))} terms"}]
    if top:
        out.append({"label": "Most patients", "value": f"{top['patients']:,}", "note": str(top.get("label"))[:60]})
    out.append({"label": "Not found", "value": f"{len(missing)}", "note": ", ".join(str(m) for m in missing)[:80] or "—"})
    return out


def _safe(build, *args):
    """The blocks of one tab; a tab that cannot be built from a damaged record is left out, not the card."""
    try:
        return build(*args) or []
    except Exception:
        logger.warning("A tab of a real-world data study card could not be built", exc_info=True)
        return []


_DICTS = {"spec": None, "result": None, "design_info": None, "timings": None, "review": None, "reproduction": None,
          "edited": None}
_LISTS = ("definitions", "reviews", "code_audit", "steps", "cells")
_RESULT_DICTS = ("tables", "diagnostics", "groups", "charts", "table1", "analyst")
_RESULT_LISTS = ("estimates", "warnings", "attrition")


def _shape(study):
    """A copy of a stored record with the containers the card reads in their shape: a damaged or foreign record (a
    list where a dict belongs, a number for a list) becomes an empty container instead of breaking the card."""
    rows = lambda v: [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
    out = dict(study)
    for key in _DICTS:
        out[key] = study.get(key) if isinstance(study.get(key), dict) else ({} if key != "edited" else None)
    for key in _LISTS:
        out[key] = rows(study.get(key))
    res = dict(out["result"])
    for key in _RESULT_DICTS:
        res[key] = res.get(key) if isinstance(res.get(key), dict) else {}
    res["estimates"] = rows(res.get("estimates"))
    res["warnings"] = [w for w in res.get("warnings") or [] if isinstance(w, str)] if isinstance(res.get("warnings"), list) else []
    res["attrition"] = [a for a in res.get("attrition") or [] if isinstance(a, (list, tuple)) and len(a) >= 2] \
        if isinstance(res.get("attrition"), list) else []
    out["result"] = res
    spec = dict(out["spec"])
    for key in ("include", "exclude"):
        spec[key] = rows(spec.get(key))
    for key in ("exposure", "comparator"):
        spec[key] = spec.get(key) if isinstance(spec.get(key), dict) else None
    for key in ("stratify", "subgroups", "covariates"):
        spec[key] = [v for v in spec.get(key) or [] if isinstance(v, str)] if isinstance(spec.get(key), list) else []
    for key in ("analysis", "base", "outcome", "variable", "title", "sql_question", "reason"):   # names used as keys
        spec[key] = spec.get(key) if isinstance(spec.get(key), str) else ""
    res["analysis"] = res.get("analysis") if isinstance(res.get("analysis"), str) else ""
    out["spec"] = spec
    out["kind"] = study.get("kind") if isinstance(study.get("kind"), str) else ""
    info = dict(out["design_info"])
    info["mapping"] = rows(info.get("mapping"))
    out["design_info"] = info
    return out


def card(result, study):
    """The spec of the study card, or None for a study that has nothing to show (its notes are drawn by the page)."""
    try:
        return _card(result, _shape(study) if isinstance(study, dict) else {})
    except Exception:
        logger.exception("Could not build the card of a real-world data study")
        return None


def _card(result, study):
    spec, res = study.get("spec") or {}, study.get("result") or {}
    status = study.get("status")
    analysis = res.get("analysis") or spec.get("analysis") or ("custom" if study.get("kind") == "analyst" else "")
    try:
        title, subtitle = _title(study)
    except Exception:
        title, subtitle = DESIGNS.get(analysis, "Study"), ""
    base = L.BASES.get(spec.get("base"), L.BASES["icu"])
    out = {"cite": _number(result, study), "dataset": study.get("dataset") or DATASET,
           "design": DESIGNS.get(analysis, "Study"), "n": num(res.get("n")),
           "unit": f"{base['unit']}s" if analysis != "custom" else "patients",
           "seconds": (lambda t: t if t is not None and t >= 0.1 else None)(num((study.get("timings") or {}).get("total"))),
           "title": title, "subtitle": subtitle,
           "badges": [], "kpis": [], "tabs": []}
    if status != "done":
        return None
    out["badges"], out["kpis"] = _safe(_badges, study), _safe(kpis, study)
    warnings = [_readable(w) for w in res.get("warnings") or [] if w]
    if study.get("kind") == "analyst":
        out["tabs"] = _safe(analyst_tabs, study)
    elif analysis == "catalog":                    # what the database records about the question's terms
        out["kpis"] = _safe(catalog_kpis, study)
        out["tabs"] = [{"id": "results", "label": f"What {database_of(study)} has",
                        "blocks": _safe(catalog_blocks, study)}]
        out["unit"] = "concepts"
        out["n"] = len([r for r in (res.get("tables") or {}).get("concepts") or [] if isinstance(r, dict)
                        and r.get("concept")])
    else:
        results = _safe(R.RESULTS.get(analysis, lambda s, r: []), spec, res)
        if warnings:
            results = [{"type": "notes", "tone": "warn", "title": f"⚠ {len(warnings)} warning{'s' if len(warnings) > 1 else ''}",
                        "items": warnings}] + results
        if res.get("methods") and analysis != "compare":
            results.append({"type": "text", "text": str(res["methods"])})
        results += _safe(older_note, analysis, res)
        extra = []
        if analysis == "similar":
            from mes.ui import similar_results
            extra = [{"id": "profile", "label": "Profile", "blocks": _safe(similar_results.profile_blocks, study)},
                     {"id": "map", "label": "Map", "blocks": _safe(similar_results.map_blocks, study)}]
        out["tabs"] = [{"id": "results", "label": "Results", "blocks": results}] + extra + [
                       {"id": "protocol", "label": "Protocol", "blocks": _safe(protocol_blocks, study)},
                       {"id": "cohort", "label": "Cohort", "blocks": _safe(cohort_blocks, study)},
                       {"id": "balance", "label": "Balance & checks" if analysis == "compare" else "Checks",
                        "blocks": _safe(balance_blocks, study)},
                       {"id": "definitions", "label": "Definitions", "blocks": _safe(definition_blocks, study)}]
    if study.get("kind") == "analyst" and warnings and out["tabs"]:
        out["tabs"][0]["blocks"] = [{"type": "notes", "tone": "warn", "items": warnings,
                                     "title": f"⚠ {len(warnings)} warning{'s' if len(warnings) > 1 else ''}"}] \
            + out["tabs"][0]["blocks"]
    return out
