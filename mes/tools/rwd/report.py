"""
RWD results → MES evidence items (citation prefix R) and the study record shown in the UI.

The passages state every number the synthesis may cite (n, events, estimate, 95% CI, p) exactly as it should be
written, so the answer's numbers are checked against them like any other source.
"""

import re

from mes.tools.rwd import DATASET
from mes.tools.rwd import library as L
from mes.tools.rwd.spec import criterion_text
from mes.tools.result_schema import EvidenceItem, Passage

TRACEBACK = re.compile(r"Traceback \(most recent call last\):?[ \t]*(?:\n[ \t]+[^\n]*)*\n?[ \t]*([^\n]*)")


def no_traceback(text):
    """``text`` with each Python traceback reduced to its error line (a failed analysis cell leaves one in warnings
    and reproduction notes; readers and the synthesis need the error, not the frames)."""
    text = str(text or "")
    if "Traceback (most recent call last)" not in text:
        return text
    return TRACEBACK.sub(lambda m: m.group(1).strip()[:300] or "an error", text)
DESIGNS = {"compare": "Target trial emulation (observational, propensity-score weighting)", "association": "Adjusted association (observational)",
           "rate": "Descriptive cohort analysis", "distribution": "Descriptive cohort analysis",
           "utilization": "Treatment utilization analysis", "survival": "Survival analysis (Kaplan-Meier)",
           "trend": "Trend analysis", "prognostic": "Prognostic model", "explore": "Exploratory aggregate query",
           "similar": "Patients like this (similar-patient cohort, observational)",
           "characterize": "Cohort characterization (descriptive)", "trajectory": "Trajectory over time (descriptive)",
           "pathways": "Treatment pathways (descriptive)", "top": "Commonest medications, diagnoses or procedures "
                                                                  "(descriptive)"}


def pattern_text(spec, res):
    """What a characterization, trajectory or pathway found, in words (aggregates only, small cells left out)."""
    tables = res.get("tables") or {}
    if spec.analysis == "characterize":
        diffs = tables.get("differences") or []
        g = res.get("groups") or {}
        if diffs:
            return (f"Characteristics, {g.get('treated', 'group 1')} vs {g.get('control', 'group 2')} (mean (SD) or n (%), "
                    "standardized mean difference): " + "; ".join(f"{d['label']} {d['a']} vs {d['b']} (SMD {d['smd']:+.2f})"
                                                                   for d in diffs[:12] if d.get("smd") is not None) + ".")
        rows = [r for r in (res.get("table1") or {}).get("rows") or [] if not r.get("heading")][:16]
        if rows:
            return ("Characteristics of the cohort (mean (SD) or n (%)): "
                    + "; ".join(f"{r['label']} {r.get('treated_raw')}" for r in rows) + ".")
        return ""
    if spec.analysis == "trajectory":
        chart = (res.get("charts") or {}).get("trajectory") or {}
        parts = []
        for grp in chart.get("groups") or []:
            pts = grp.get("points") or []
            step = max(1, len(pts) // 12 + (len(pts) % 12 > 0))
            parts.append(f"{grp['label']}: " + "; ".join(f"{p['t0']:g}–{p['t1']:g} h {_fmt(p['median'])} ({_fmt(p['q1'])}–"
                                                         f"{_fmt(p['q3'])}), n = {_n(p['n'])}" for p in pts[::step]))
        unit = f" ({chart['unit']})" if chart.get("unit") else ""
        return (f"Median{unit} and IQR of {chart.get('variable', 'the measurement')} per interval after "
                f"{chart.get('t0_label', 'time zero')} — " + " | ".join(parts) + ".") if parts else ""
    if spec.analysis == "pathways":
        rows = [r for r in tables.get("treatments") or [] if r.get("given") is not None]
        bits = [f"{r['treatment']} received by {r['given']:.1f}% (95% CI {r['lo']:.1f}–{r['hi']:.1f})"
                + (f", first in {r['first']:.1f}%" if r.get("first") is not None else "")
                + (f", first dose at a median {r['median_h']:g} h (IQR {r['q1_h']:g}–{r['q3_h']:g})" if r.get("median_h")
                   is not None else "") for r in rows]
        seqs = [f"{q['sequence']} {q['share']:.1f}%" for q in (tables.get("sequences") or [])[:8]]
        text = f"Within {spec.horizon_h:g} h: " + "; ".join(bits) + "." if bits else ""
        return (text + (" Commonest sequences (by the first dose): " + "; ".join(seqs) + "." if seqs else "")).strip()
    return ""


def profile_text(spec):
    """The described patient in words: "age 68; sex M; septic shock (required); highest lactate, first 24 h 6 …"."""
    bits = []
    profile = spec.profile if hasattr(spec, "profile") else (spec or {}).get("profile")
    for f in profile if isinstance(profile, list) else []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("concept") or "")[4:]
        label = L.VARIABLES.get(name, (name,))[0]
        value = f.get("value")
        kind = L.VARIABLES.get(name, ("", "numeric"))[1]
        if kind == "binary":
            text = label if value in (1, True, "1", "yes", "true") else f"no {label[:1].lower() + label[1:]}"
        else:
            text = f"{label} {value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else f"{label} {value}"
        bits.append(text + (" (required)" if f.get("must") else ""))
    return "; ".join(bits)


def _n(v):
    return "<10" if v is None else f"{int(v):,}"          # (None: a count under 10, not shown)


def _fmt(v, digits=2):
    if v is None:
        return "NR"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}" if abs(v) < 10 else f"{v:.1f}"


def estimate_text(e):
    """One estimate as text, e.g. "risk ratio (IPTW): 0.84 (95% CI 0.71–0.99), p = 0.04"."""
    m, v, lo, hi = e.get("measure"), e.get("value"), e.get("lo"), e.get("hi")
    if v is None:
        return f"{e['label']}: not estimable" + (" (fewer than 10 patients)" if e.get("small_cell") else "")
    unit = f" {e['unit']}" if e.get("unit") and e.get("unit") != "%" else ""
    if m in ("risk", "rate", "survival", "proportion"):
        text = f"{v:.1f}%"
        if lo is not None and hi is not None:
            text += f" (95% CI {lo:.1f}–{hi:.1f})"
        if e.get("events") is not None and e.get("n"):
            text += f"; {_n(e['events'])} of {_n(e['n'])}"
    elif m == "RD":
        text = f"{v:+.1f} percentage points"
        if lo is not None and hi is not None:
            text += f" (95% CI {lo:+.1f} to {hi:+.1f})"
    elif m in ("median",):
        text = _fmt(v) + unit + (f" (IQR {_fmt(lo)}–{_fmt(hi)})" if lo is not None and hi is not None else "")
        if e.get("n"):
            text += f"; n = {_n(e['n'])}"
    elif m == "mean":
        text = _fmt(v) + unit + (f" (SD {_fmt(e['sd'])})" if e.get("sd") is not None else "")
        if lo is not None and hi is not None:
            text += f" (95% CI {_fmt(lo)}–{_fmt(hi)})"
    elif m == "count":
        text = _n(v) + unit
    elif m in ("AUC", "slope"):
        text = f"{v:.2f}"
    else:   # ratios and differences
        text = f"{v:.2f}{unit}"
        if lo is not None and hi is not None:
            text += f" (95% CI {lo:.2f}–{hi:.2f})"
        if e.get("p") is not None:
            p = e["p"]
            text += ", p < 0.001" if p < 0.001 else f", p = {p:.3f}"
    return f"{e['label']}: {text}"


def _design_text(spec, res):
    parts = [f"{DESIGNS.get(spec.analysis, spec.analysis)} in {DATASET} ({res['population'].lower()})."]
    if spec.analysis == "similar":
        d = res.get("diagnostics") or {}
        parts.append(f"The described patient: {profile_text(spec)}. The {_n(d.get('k') or res.get('n') or 0)} stays most "
                     f"similar to this patient were selected among {_n(d.get('n_pool') or 0)} by a weighted distance over "
                     f"{d.get('features', len(spec.profile))} characteristics of the first 24 h.")
    steps = res.get("attrition") or []
    if steps:
        parts.append("Cohort: " + "; ".join(f"{s} → {_n(n)}" for s, n in steps[1:]) + "." if len(steps) > 1
                     else f"Cohort: {_n(steps[0][1])}.")
    if spec.exposure is not None:
        comparator = criterion_text(spec.comparator) if spec.comparator else "patients without it"
        role = "Treatment" if spec.analysis == "compare" else "Exposure"
        parts.append(f"{role}: {criterion_text(spec.exposure)} vs {comparator}.")
    if spec.outcome:
        parts.append(f"Outcome: {L.OUTCOMES[spec.outcome][0]}.")
    if res.get("methods"):
        parts.append(res["methods"])
    return " ".join(parts)


def _diagnostics_text(res):
    d = res.get("diagnostics") or {}
    bits = []
    if d.get("max_smd_after") is not None:
        bits.append(f"largest standardized mean difference {d['max_smd_before']:.2f} before and "
                    f"{d['max_smd_after']:.2f} after weighting")
    if d.get("overlap") is not None:
        bits.append(f"{100 * d['overlap']:.0f}% of patients within the common support of the propensity score")
    if d.get("ps_auc") is not None:
        bits.append(f"propensity model c-statistic {d['ps_auc']:.2f}")
    if d.get("e_value") is not None:
        bits.append(f"E-value {d['e_value']:.2f} (CI limit {d['e_value_ci']:.2f})")
    if d.get("logrank_p") is not None:
        bits.append(f"log-rank p = {d['logrank_p']:.3g}")
    if d.get("median_similarity") is not None:
        bits.append(f"median similarity of the similar patients {d['median_similarity']:.2f} (lowest "
                    f"{d['min_similarity']:.2f}; 1 = identical)")
    if d.get("risk_auc") is not None:
        bits.append(f"risk model c-statistic {d['risk_auc']:.2f} (cross-validated)")
    text = ("Diagnostics: " + "; ".join(bits) + ".") if bits else ""
    if res.get("warnings"):
        text += " Warnings: " + " ".join(no_traceback(w) for w in res["warnings"])
    return text.strip()


def _table_text(res):
    rows = []
    for key in ("strata", "quartiles"):
        for r in res.get("tables", {}).get(key, [])[:16]:
            if "rate" in r:
                rows.append(f"{r['variable']} {r['level']}: {r['rate']:.1f}% ({_n(r['events'])} of {_n(r['n'])})")
            elif "median" in r:
                rows.append(f"{r['variable']} {r['level']}: median {_fmt(r['median'])} (IQR {_fmt(r['q1'])}–"
                            f"{_fmt(r['q3'])}), n = {_n(r['n'])}")
    for r in res.get("tables", {}).get("subgroups", [])[:8]:
        rows.append(f"Subgroup {r['subgroup']} (n = {_n(r['n'])}): risk difference {r['rd']:+.1f} percentage points "
                    f"(95% CI {r['lo']:+.1f} to {r['hi']:+.1f})")
    return ("By subgroup: " + "; ".join(rows) + ".") if rows else ""


def similar_text(res):
    """The similar patients beside the population (characteristics, what they received): aggregates only."""
    tables = res.get("tables") or {}
    profile = [f"{r['label']} {r['similar']} vs {r['cohort']}" for r in (tables.get("profile") or [])[:12]
               if r.get("similar") and r.get("cohort") and r.get("similar") != "<10"]
    given = [f"{r['label']} {r['similar']:.1f}% (95% CI {r['lo']:.1f}–{r['hi']:.1f}) vs {r['cohort']:.1f}%"
             for r in (tables.get("treatments") or [])[:10] if r.get("similar") is not None
             and r.get("lo") is not None and r.get("cohort") is not None]
    parts = []
    if profile:
        parts.append("The similar patients beside the whole population (median (IQR) or %): " + "; ".join(profile) + ".")
    if given:
        parts.append("Received in the first 24 h by the similar patients vs the whole population: " + "; ".join(given) + ".")
    return " ".join(parts)


def to_items(spec, res, study_key):
    """Evidence items for a finished analysis: the main result (and a cohort description)."""
    estimates = res.get("estimates") or []
    main = [e for e in estimates if not e.get("sensitivity")]
    sens = [e for e in estimates if e.get("sensitivity")]
    title = spec.title or f"{DESIGNS.get(spec.analysis)}: {L.OUTCOMES.get(spec.outcome, ('',))[0]}"
    passages = [Passage(text=_design_text(spec, res), section="RWD design")]
    if main:
        passages.append(Passage(text="Results: " + "; ".join(estimate_text(e) for e in main) + ".",
                                section="RWD results"))
    table = _table_text(res)
    if table:
        passages.append(Passage(text=table, section="RWD results by subgroup"))
    similar = similar_text(res) if spec.analysis == "similar" else ""
    if similar:
        passages.append(Passage(text=similar, section="RWD similar patients"))
    pattern = pattern_text(spec, res) if spec.analysis in ("characterize", "trajectory", "pathways") else ""
    if spec.analysis == "top":
        from mes.tools.rwd import ranking
        pattern = ranking.text(spec, res)
    if pattern:
        passages.append(Passage(text=pattern, section="RWD results"))
    if sens:
        passages.append(Passage(text="Sensitivity analyses: " + "; ".join(estimate_text(e) for e in sens) + ".",
                                section="RWD sensitivity"))
    diag = _diagnostics_text(res)
    if diag:
        passages.append(Passage(text=diag, section="RWD diagnostics"))
    item = EvidenceItem(
        id=f"rwd:{study_key}", source_type="rwd", title=f"{DATASET} analysis: {title}", venue=DATASET,
        design=DESIGNS.get(spec.analysis, "Observational analysis"), evidence_level=3, origin="RWD-Analyst",
        passages=passages, score=1.0,
        meta={"study_key": study_key, "analysis": spec.analysis, "n": res.get("n"),
              "warnings": res.get("warnings", []), "primary": next((e for e in estimates if e.get("primary")), None)})
    return [item]


def not_feasible_item(spec, study_key):
    reason = spec.reason or "the question cannot be answered with MIMIC-IV"
    return EvidenceItem(
        id=f"rwd:{study_key}", source_type="rwd", title=f"{DATASET}: question not answerable with this data",
        venue=DATASET, design="Feasibility assessment", evidence_level=5, origin="RWD-Analyst", score=0.5,
        passages=[Passage(text=f"Real-world data feasibility: not feasible in {DATASET} — {reason}",
                          section="RWD feasibility")],
        meta={"study_key": study_key, "analysis": "not_feasible", "warnings": [reason]})


# ============================================================================ analyst studies
ANALYST_DESIGN = "Custom analysis (code written by the model; observational)"
VERDICTS = {"sound": "no problems found", "minor_issues": "minor issues only", "major_issues": "major problems found",
            "not_reviewed": "not reviewed"}


def _rows_text(name, rows, limit=16):
    out = []
    for r in rows[:limit]:
        out.append(", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in r.items()))
    more = f" (+{len(rows) - limit} more rows)" if len(rows) > limit else ""
    return f"{name}: " + "; ".join(out) + more


def analyst_items(study):
    """Evidence item of a finished analyst study (citation prefix R); every number comes from submit()."""
    res = study["result"]
    a = res.get("analyst") or {}
    review, repro = study.get("review") or {}, study.get("reproduction") or {}
    edited = (study.get("edited") or {}).get("kind") == "code"
    if edited:
        design = [f"{ANALYST_DESIGN} in {DATASET}: code written by the MES analyst ({study.get('llm', '')}) and edited "
                  "by the user, run as one script in a fresh sandboxed session; the edited code was not reviewed "
                  "independently."]
    else:
        design = [f"{ANALYST_DESIGN} in {DATASET}, written and run by the MES analyst ({study.get('llm', '')}) in a "
                  f"sandboxed session; independent review: {VERDICTS.get(review.get('verdict'), 'not reviewed')}; "
                  f"reproduced in a fresh session: {'yes' if repro.get('reproduced') else 'no'}."]
    for key, label in (("population", "Population"), ("time_zero", "Time zero"), ("exposure", "Exposure"),
                       ("comparator", "Comparator"), ("outcome", "Outcome")):
        value = res.get(key) if key == "population" else a.get(key)
        if value:
            design.append(f"{label}: {value}.")
    steps = res.get("attrition") or []
    if steps:
        design.append("Cohort: " + "; ".join(f"{s} → {_n(n) if n is not None else '<10'}" for s, n in steps) + ".")
    if res.get("methods"):
        design.append("Methods: " + res["methods"])
    passages = [Passage(text=" ".join(design), section="RWD design")]
    estimates = res.get("estimates") or []
    main = [e for e in estimates if not e.get("sensitivity")]
    sens = [e for e in estimates if e.get("sensitivity")]
    if main:
        passages.append(Passage(text="Results: " + "; ".join(estimate_text(e) for e in main) + ".",
                                section="RWD results"))
    tables = [(name, rows) for name, rows in (res.get("tables") or {}).items() if name != "km" and rows
              and isinstance(rows, list)]
    # result tables first: a long covariate-balance table must not push them out of the passage
    tables.sort(key=lambda t: bool(re.search(r"smd|balance|covariate", t[0] + " ".join(t[1][0]), re.I)))
    if tables:
        passages.append(Passage(text=" | ".join(_rows_text(name, rows) for name, rows in tables)[:5000],
                                section="RWD results by subgroup"))
    if sens:
        passages.append(Passage(text="Sensitivity analyses: " + "; ".join(estimate_text(e) for e in sens) + ".",
                                section="RWD sensitivity"))
    notes = []
    d = res.get("diagnostics") or {}
    if d:
        notes.append("Diagnostics: " + "; ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                                 for k, v in list(d.items())[:12]) + ".")
    if review.get("summary"):
        notes.append(f"Review: {no_traceback(review['summary'])}")
    if a.get("limitations"):
        notes.append("Limitations stated by the analyst: " + " ".join(a["limitations"]))
    if res.get("warnings"):
        notes.append("Warnings: " + " ".join(no_traceback(w) for w in res["warnings"]))
    notes.append("Exploratory: the analysis code was written for this question by the model"
                 + (" and edited by the user" if edited else "") + "; check the code tab.")
    passages.append(Passage(text=" ".join(notes), section="RWD diagnostics"))
    item = EvidenceItem(
        id=f"rwd:{study['key']}", source_type="rwd", title=f"{DATASET} custom analysis: {res['title']}",
        venue=DATASET, design=ANALYST_DESIGN, evidence_level=4, origin="RWD-Analyst (LLM analyst)",
        passages=passages, score=0.95,
        meta={"study_key": study["key"], "analysis": "custom", "n": res.get("n"), "warnings": res.get("warnings", []),
              "primary": next((e for e in estimates if e.get("primary")), estimates[0] if estimates else None),
              "review": review.get("verdict"), "reproduced": repro.get("reproduced"), "edited": edited})
    return [item]


def analyst_failed_item(study):
    reason = no_traceback("; ".join((study.get("design_info") or {}).get("errors", [])) or "no result")
    return EvidenceItem(
        id=f"rwd:{study['key']}", source_type="rwd", title=f"{DATASET}: the custom analysis did not finish",
        venue=DATASET, design="Custom analysis (not completed)", evidence_level=5, origin="RWD-Analyst", score=0.1,
        passages=[Passage(text=f"Real-world data: the MES analyst could not complete an analysis of {DATASET} for this "
                               f"question ({reason}).", section="RWD feasibility")],
        meta={"study_key": study["key"], "analysis": "failed", "warnings": [reason]})
