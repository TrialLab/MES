"""
RWD analysis types (deterministic). Each takes the validated spec and the compiled cohort and returns a result:

    {"analysis", "title", "n", "attrition", "estimates": [...], "tables": {...}, "diagnostics": {...},
     "warnings": [...], "methods": "..."}

Estimates are only reported when the groups are large enough (MIN_GROUP patients, MIN_EVENTS events);
otherwise the analysis falls back to describing the cohort and says why.
"""

import math

import numpy as np
import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd import stats as S
from mes.tools.rwd.spec import criterion_text

MIN_GROUP = 50
MIN_EVENTS = 10
BOOT = {"osler": 500, "sackett": 1000, "snow": 1000}


def _kinds(columns):
    return {c: L.VARIABLES[c][1] for c in columns if c in L.VARIABLES}


def _pct(p):
    return None if p is None else round(100 * p, 1)


def _strata(series, name):
    """Grouping of a stratification variable: categories as is, numbers in quartiles."""
    kind = L.VARIABLES.get(name, ("", "numeric"))[1]
    if kind == "numeric":
        s = pd.to_numeric(series, errors="coerce")
        if s.nunique() > 6:
            q = pd.qcut(s, 4, duplicates="drop")
            names = {iv: f"{max(iv.left, s.min()):.3g}–{iv.right:.3g}" for iv in q.cat.categories}
            return q.map(names).astype(str).replace("nan", "missing")
        return s.astype("Int64").astype(str).replace("<NA>", "missing")
    if kind == "binary":
        return series.fillna(0).astype(float).map({1.0: "yes", 0.0: "no"})
    return series.fillna("missing").astype(str)


def _rate_rows(df, y, by):
    rows = []
    for level, part in df.groupby(_strata(df[by], by), observed=True):
        k, n = int(part[y].fillna(0).astype(float).sum()), len(part)
        p, lo, hi = S.wilson(k, n)
        rows.append({"variable": by, "level": str(level), "n": n, "events": k, "rate": _pct(p),
                     "lo": _pct(lo), "hi": _pct(hi)})
    return rows


def _outcome(name):
    label, kind, _bases, col, horizon = L.OUTCOMES[name]
    return label, kind, col, horizon


def _time_to_event(df, horizon, offset_days=0.0):
    """Time from the start of follow-up (days, capped at the horizon) and death within the horizon."""
    days = df["days_to_death"].astype(float) - offset_days
    event = (df["days_to_death"].notna() & (df["days_to_death"] <= horizon)).astype(int).values
    time = np.where(event == 1, days, horizon - offset_days)
    return np.clip(time, 1e-3, horizon - offset_days), event


# Version of the study card's data in a result (res["table1"], res["charts"] below): a stored result without it was
# computed before they existed, and the card says how to add them
CARD_DATA = 1


def base_result(spec, cohort):
    return {"analysis": spec.analysis, "title": spec.title, "population": L.BASES[spec.base]["label"],
            "n": len(cohort.df), "attrition": cohort.attrition, "estimates": [], "tables": {},
            "diagnostics": {}, "warnings": [], "methods": "", "card_data": CARD_DATA}


# ============================================================================ Table 1 and chart data
# What the study panel draws besides the estimates: res["table1"] (baseline characteristics by group) and
# res["charts"] (the band and numbers at risk of survival curves, the propensity score distributions, the ROC curve
# and calibration of a prognostic model). Aggregates only, counts below 10 not shown; never in the evidence passages.
def _num_text(v):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def _count_text(k, n):
    if not n:
        return "—"
    return "<10" if 0 < k < S.SMALL_CELL else f"{k:,} ({100 * k / n:.1f}%)"


def _smd_of(x, t, w):
    """Standardized mean difference of one column (missing values left out), weighted with ``w``."""
    ok = ~np.isnan(x)
    a, b = ok & (t == 1), ok & (t == 0)
    if not a.any() or not b.any() or w[a].sum() <= 0 or w[b].sum() <= 0:
        return None
    m1, m0 = np.average(x[a], weights=w[a]), np.average(x[b], weights=w[b])
    v1, v0 = np.average((x[a] - m1) ** 2, weights=w[a]), np.average((x[b] - m0) ** 2, weights=w[b])
    pooled = math.sqrt((v1 + v0) / 2)
    return round(float((m1 - m0) / pooled), 3) if pooled > 0 else 0.0


def baseline_table(df, covariates, t, w=None, names=("treated", "comparison")):
    """Table 1: each covariate by group — n (%) of binary and category variables (the six commonest levels), mean
    (SD) of numeric ones — with standardized mean differences; with ``w`` also the weighted proportions / means."""
    t = np.asarray(t).astype(int)
    weights = {"raw": np.ones(len(t))}
    if w is not None:
        weights["weighted"] = np.asarray(w, float)
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    rows = []

    def add(label, x, kind, level=False, unit="", variable=""):
        x = np.asarray(x, float)
        row = {"variable": variable, "label": label, "kind": kind, "level": level, "unit": unit,
               "missing": round(float(np.isnan(x).mean()), 4) if len(x) else 0.0}
        for name, wt in weights.items():
            for side, g in (("treated", 1), ("control", 0)):
                m = (t == g) & ~np.isnan(x)
                if not m.any() or wt[m].sum() <= 0:
                    row[f"{side}_{name}"] = "—"
                    continue
                mean = float(np.average(x[m], weights=wt[m]))
                if kind == "numeric":
                    sd = math.sqrt(float(np.average((x[m] - mean) ** 2, weights=wt[m])))
                    row[f"{side}_{name}"] = f"{_num_text(mean)} ({_num_text(sd)})"
                elif name == "raw":
                    row[f"{side}_{name}"] = _count_text(int(round(x[m].sum())), int(m.sum()))
                else:
                    row[f"{side}_{name}"] = f"{100 * mean:.1f}%"
            row[f"smd_{name}"] = _smd_of(x, t, wt)
        rows.append(row)

    for v in covariates:
        if v not in df or v not in L.VARIABLES:
            continue
        label, kind, unit = L.VARIABLES[v][:3]
        if kind == "numeric":
            add(label, pd.to_numeric(df[v], errors="coerce").astype(float).values, "numeric", unit=unit, variable=v)
        elif kind == "binary":
            add(label, pd.to_numeric(df[v].astype(float), errors="coerce").fillna(0).values, "binary", variable=v)
        else:
            levels = df[v].fillna("Missing").astype(str)
            rows.append({"variable": v, "label": label, "kind": "category", "heading": True})
            for level in levels.value_counts().index[:6]:
                add(level, (levels == level).astype(float).values, "binary", level=True, variable=v)
    return {"groups": list(names), "n": [n1, n0], "weighted": w is not None, "rows": rows}


def _charts(res, key, value):
    """A chart's data for the study panel; a chart that cannot be computed is left out, never the study."""
    try:
        data = value() if callable(value) else value
    except Exception:
        return
    if data:
        res.setdefault("charts", {})[key] = data


# ============================================================================ compare (target trial emulation)
def run_compare(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    t = df["treated"].values.astype(int)
    n1, n0 = int(t.sum()), int((1 - t).sum())
    exposure = criterion_text(spec.exposure)
    comparator = criterion_text(spec.comparator) if spec.comparator else f"no {spec.exposure.label}"
    res["groups"] = {"treated": exposure, "control": comparator, "n_treated": n1, "n_control": n0}
    if min(n1, n0) < MIN_GROUP:
        res["warnings"].append(f"Too few patients for a comparison ({n1} treated, {n0} comparison; at least "
                               f"{MIN_GROUP} per group needed).")
        return res
    X, missing = S.design_matrix(df, spec.covariates, _kinds(spec.covariates))
    ps, auc = S.propensity(X, t)
    w, winfo = S.iptw(ps, t)
    before, after = S.smd(X, t), S.smd(X, t, w)
    weighting, iptw_max = "IPTW", float(after.smd.abs().max())
    if iptw_max >= 0.1:            # extreme weights / imperfect model: overlap weights balance the covariates exactly
        w_ow, oinfo = S.overlap_weights(ps, t)
        after_ow = S.smd(X, t, w_ow)
        if float(after_ow.smd.abs().max()) < iptw_max:
            w, after, weighting = w_ow, after_ow, "overlap weights"
            winfo.update(oinfo)
            res["warnings"].append(
                f"Inverse probability weighting left residual imbalance (largest SMD {iptw_max:.2f}); the estimates "
                "use overlap weights instead, which balance all covariates and describe the effect among patients "
                "for whom either treatment was plausible (clinical equipoise).")
    balance = before.rename(columns={"smd": "smd_before"}).assign(smd_after=after["smd"].values)
    res["tables"]["balance"] = balance.round(3).to_dict("records")
    try:
        res["table1"] = baseline_table(df, spec.covariates, t, w, (exposure, comparator))
    except Exception:                  # Table 1 is a view of the cohort: the estimates do not depend on it
        pass
    _charts(res, "ps", lambda: S.ps_distribution(ps, t))
    max_before, max_after = float(balance.smd_before.abs().max()), float(balance.smd_after.abs().max())
    label, kind, col, horizon = _outcome(spec.outcome)
    y = pd.to_numeric(df[col], errors="coerce")
    diag = {"ps_auc": auc, "max_smd_before": max_before, "max_smd_after": max_after, "weighting": weighting,
            "iptw_max_smd_after": iptw_max, "balance_ok": max_after < 0.1, "covariate_missing": missing, **winfo}
    if kind == "binary":
        y = y.fillna(0).astype(int).values
        e1, e0 = int(y[t == 1].sum()), int(y[t == 0].sum())
        diag.update(events_treated=e1, events_control=e0)
        res["groups"].update(events_treated=e1, events_control=e0)
        if min(e1, e0) < MIN_EVENTS:
            res["warnings"].append(f"Too few outcome events for a stable estimate ({e1} vs {e0}).")
        p1, lo1, hi1 = S.wilson(e1, n1)
        p0, lo0, hi0 = S.wilson(e0, n0)
        res["estimates"] += [
            {"label": f"{label}: {exposure} (crude)", "measure": "risk", "value": _pct(p1), "lo": _pct(lo1),
             "hi": _pct(hi1), "n": n1, "events": e1},
            {"label": f"{label}: {comparator} (crude)", "measure": "risk", "value": _pct(p0), "lo": _pct(lo0),
             "hi": _pct(hi0), "n": n0, "events": e0}]
        wr = S.weighted_risks(y, t, w, n_boot=BOOT.get(mode, 200))
        res["estimates"] += [
            {"label": f"{label}: weighted risk, {exposure}", "measure": "risk", "value": _pct(wr["risk_treated"])},
            {"label": f"{label}: weighted risk, {comparator}", "measure": "risk", "value": _pct(wr["risk_control"])},
            {"label": f"{label}: risk difference ({weighting})", "measure": "RD", "value": _pct(wr["rd"]),
             "lo": _pct(wr["rd_ci"][0]), "hi": _pct(wr["rd_ci"][1]), "primary": True},
            {"label": f"{label}: risk ratio ({weighting})", "measure": "RR", "value": wr["rr"], "lo": wr["rr_ci"][0],
             "hi": wr["rr_ci"][1]}]
        orw = S.weighted_or(y, t, w)
        res["estimates"].append({"label": f"{label}: odds ratio ({weighting})", "measure": "OR", **orw})
        ev, ev_ci = S.e_value(wr["rr"], *wr["rr_ci"], measure="RR")
        diag.update(e_value=ev, e_value_ci=ev_ci)
        if horizon:
            time, event = _time_to_event(df, horizon, offset_days=spec.grace_h / 24)
            hr = S.cox(time, event, t, w=w)
            res["estimates"].append({"label": f"{label}: hazard ratio ({weighting}, Cox)", "measure": "HR", **hr})
            curves, _p = S.kaplan_meier(time, event, np.where(t == 1, exposure, comparator), w=w,
                                        horizon=horizon - spec.grace_h / 24, detail=True)
            res["tables"]["km"] = curves
            res.setdefault("charts", {})["km"] = {"start": f"the landmark, {spec.grace_h:g} h after "
                                                           f"{L.BASES[spec.base]['t0_label']}",
                                                  "offset_days": round(spec.grace_h / 24, 4), "weighted": weighting}
        if mode in ("sackett", "snow"):
            dr = S.weighted_or(y, t, w, X)
            res["estimates"].append({"label": f"{label}: odds ratio (doubly robust: {weighting} + covariates)",
                                     "measure": "OR", **dr, "sensitivity": True})
            crude = S.weighted_or(y, t, np.ones(len(t)))
            res["estimates"].append({"label": f"{label}: odds ratio (crude)", "measure": "OR", **crude,
                                     "sensitivity": True})
            pairs = S.match_1to1(ps, t)
            if len(pairs) >= MIN_GROUP:
                idx = np.array(pairs).ravel()
                ym, tm = y[idx], t[idx]
                pm1, pm0 = ym[tm == 1].mean(), ym[tm == 0].mean()
                res["estimates"].append({"label": f"{label}: risk difference (1:1 propensity matching, "
                                                  f"{len(pairs)} pairs)", "measure": "RD", "value": _pct(pm1 - pm0),
                                         "sensitivity": True})
    else:
        y = y.astype(float)
        md = S.mean_difference(y.fillna(y.median()), t, w)
        res["estimates"] += [
            {"label": f"{label}: median, {exposure}", "measure": "median", "value": float(y[t == 1].median())},
            {"label": f"{label}: median, {comparator}", "measure": "median", "value": float(y[t == 0].median())},
            {"label": f"{label}: mean difference ({weighting})", "measure": "MD", **md, "primary": True}]
    for extra in spec.secondary_outcomes:
        if extra == spec.outcome or extra not in L.OUTCOMES or L.OUTCOMES[extra][1] != "binary":
            continue
        lab2, _k, col2, _h = _outcome(extra)
        y2 = pd.to_numeric(df[col2], errors="coerce").fillna(0).astype(int).values
        wr2 = S.weighted_risks(y2, t, w, n_boot=300)
        res["estimates"].append({"label": f"{lab2}: risk difference ({weighting})", "measure": "RD", "value": _pct(wr2["rd"]),
                                 "lo": _pct(wr2["rd_ci"][0]), "hi": _pct(wr2["rd_ci"][1]), "secondary": True})
    if spec.subgroups and kind == "binary":
        rows = []
        for name in spec.subgroups[:3]:
            groups = _strata(df[name], name)
            for level in pd.unique(groups):
                m = (groups == level).values
                if m.sum() < 2 * MIN_GROUP or t[m].sum() < 20 or (1 - t[m]).sum() < 20:
                    continue
                wr3 = S.weighted_risks(y[m], t[m], w[m], n_boot=300)
                rows.append({"subgroup": f"{L.VARIABLES[name][0]}: {level}", "n": int(m.sum()),
                             "rd": _pct(wr3["rd"]), "lo": _pct(wr3["rd_ci"][0]), "hi": _pct(wr3["rd_ci"][1])})
        res["tables"]["subgroups"] = rows
    if not diag["balance_ok"]:
        res["warnings"].append(f"Residual imbalance after weighting (largest SMD {max_after:.2f} > 0.10).")
    if diag["overlap"] < 0.9:
        res["warnings"].append(f"Limited overlap of propensity scores ({100 * diag['overlap']:.0f}% within common "
                               "support): the groups differ strongly.")
    if "exposure_hours" in df:
        h = S.summarize(df.loc[df["treated"] == 1, "exposure_hours"])
        diag["exposure_start_hours"] = {k: h.get(k) for k in ("median", "q1", "q3")}
    res["diagnostics"] = diag
    res["methods"] = (f"Target trial emulation in {L.BASES[spec.base]['label'].lower()}. Time zero: "
                      f"{L.BASES[spec.base]['t0_label']}; treatment assigned by the first dose within "
                      f"{spec.grace_h:g} h (grace period), follow-up from the landmark at {spec.grace_h:g} h among "
                      f"patients alive and in hospital"
                      + ("; prior users excluded" if spec.new_user else "")
                      + (". Inverse probability of treatment weighting (stabilized, propensity scores truncated at the "
                         "1st/99th percentiles)" if weighting == "IPTW" else
                         f". Overlap weighting (inverse probability weighting left a largest SMD of {iptw_max:.2f})")
                      + f" with a propensity model (logistic, splines for continuous covariates) on "
                      f"{len(spec.covariates)} baseline covariates "
                      f"({', '.join(L.VARIABLES[c][0] for c in spec.covariates[:8])}"
                      f"{', …' if len(spec.covariates) > 8 else ''}); missing values imputed with the median plus a "
                      f"missing indicator. Weighted risks with bootstrap CIs, weighted logistic regression with robust "
                      f"SEs" + (", weighted Cox model" if kind == "binary" and horizon else "") + ".")
    return res


# ============================================================================ association
def run_association(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    label, kind, col, horizon = _outcome(spec.outcome)
    exposure_name = criterion_text(spec.exposure)
    continuous = "exposure_value" in df and not spec.exposure.op
    covs = [c for c in spec.covariates if c != spec.exposure.concept.partition(":")[2]]
    X, missing = S.design_matrix(df, covs, _kinds(covs))
    y = pd.to_numeric(df[col], errors="coerce")
    if continuous:
        x = pd.to_numeric(df["exposure_value"], errors="coerce")
        keep = x.notna().values
        res["warnings"] += [f"{int((~keep).sum()):,} patients without a value of {spec.exposure.label} excluded."] \
            if (~keep).any() else []
        df, X, y, x = df[keep], X[keep], y[keep], x[keep]
        sd = float(x.std()) or 1.0
        xs = (x - x.mean()) / sd
        exp_label = f"{spec.exposure.label} (per SD = {sd:.3g} {L.VARIABLES.get(spec.exposure.concept[4:], ('', '', ''))[2]})"
    else:
        xs = df["exposed"].fillna(False).astype(int)
        exp_label = exposure_name
        try:
            res["table1"] = baseline_table(df, covs, xs.values, None, (exposure_name, f"not {exposure_name}"))
        except Exception:
            pass
    res["n"] = len(df)
    if kind == "binary":
        y = y.fillna(0).astype(int)
        events = int(y.sum())
        if events < MIN_EVENTS:
            res["warnings"].append(f"Only {events} outcome events: no adjusted estimate.")
            return res
        if not continuous:
            for name, m in ((exposure_name, xs == 1), (f"not {exposure_name}", xs == 0)):
                k, n = int(y[m].sum()), int(m.sum())
                p, lo, hi = S.wilson(k, n)
                res["estimates"].append({"label": f"{label}: {name}", "measure": "risk", "value": _pct(p),
                                         "lo": _pct(lo), "hi": _pct(hi), "n": n, "events": k})
        design = pd.concat([pd.Series(xs.values, name="exposure", index=X.index), X], axis=1)
        adj = S.effect_from_glm(S._glm(y.values, S._standardize(design).assign(exposure=xs.values)), "exposure")
        crude = S.effect_from_glm(S._glm(y.values, pd.DataFrame({"exposure": xs.values})), "exposure")
        res["estimates"] += [{"label": f"{label}: odds ratio for {exp_label} (adjusted)", "measure": "OR", **adj,
                              "primary": True},
                             {"label": f"{label}: odds ratio for {exp_label} (crude)", "measure": "OR", **crude}]
        if continuous:
            q = pd.qcut(x, 4, labels=False, duplicates="drop")
            rows = []
            for level in sorted(pd.unique(q.dropna())):
                m = (q == level).values
                k, n = int(y[m].sum()), int(m.sum())
                p, lo, hi = S.wilson(k, n)
                rows.append({"variable": spec.exposure.label, "level": f"Q{int(level) + 1} "
                             f"({x[m].min():.3g}–{x[m].max():.3g})", "n": n, "events": k, "rate": _pct(p),
                             "lo": _pct(lo), "hi": _pct(hi)})
            res["tables"]["quartiles"] = rows
        ev, ev_ci = S.e_value(adj["value"], adj["lo"], adj["hi"], measure="OR", rare=y.mean() < 0.15)
        res["diagnostics"].update(e_value=ev, e_value_ci=ev_ci, events=events,
                                  events_per_parameter=round(events / max(1, design.shape[1]), 1))
        if horizon and mode in ("sackett", "snow"):
            time, event = _time_to_event(df, horizon)
            hr = S.cox(time, event, xs.values, X=X.reset_index(drop=True), name="exposure")
            res["estimates"].append({"label": f"{label}: hazard ratio for {exp_label} (adjusted Cox)",
                                     "measure": "HR", **hr})
    else:
        md = S.mean_difference(y.fillna(y.median()), xs.values, X=X.reset_index(drop=True))
        res["estimates"].append({"label": f"{label}: adjusted difference for {exp_label}", "measure": "MD", **md,
                                 "primary": True})
    res["diagnostics"]["covariate_missing"] = missing
    res["methods"] = (f"Multivariable {'logistic' if kind == 'binary' else 'linear'} regression of {label.lower()} "
                      f"on {exp_label} adjusted for {len(covs)} covariates "
                      f"({', '.join(L.VARIABLES[c][0] for c in covs[:8])}{', …' if len(covs) > 8 else ''}); robust "
                      "standard errors; missing covariates imputed with the median plus a missing indicator.")
    return res


# ============================================================================ descriptive analyses
def _measured(spec, df):
    """Column whose rate is described: the outcome, else the exposure / condition flag."""
    if spec.outcome:
        label, _kind, col, _h = _outcome(spec.outcome)
        return label, col
    return criterion_text(spec.exposure), "exposed"


def _window_text(spec, window):
    if window[1] is None:
        return "during the " + ("ICU stay" if spec.base == "icu" else "hospital stay")
    return f"between {window[0]:g} and {window[1]:g} h after {L.BASES[spec.base]['t0_label']}"


def run_rate(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    label, col = _measured(spec, df)
    if spec.outcome and spec.exposure is not None and "exposed" in df:
        # an outcome rate with an exposure: among the exposed patients (never an exposure silently ignored)
        df = df[df["exposed"].fillna(False).astype(bool)]
        name = criterion_text(spec.exposure)
        res["attrition"] = list(res["attrition"]) + [(f"with {name}", len(df))]
        res["n"] = len(df)
        label = f"{label} among patients with {name}"
    values = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    k, n = int(values.sum()), len(df)
    p, lo, hi = S.wilson(k, n)
    res["estimates"].append({"label": f"{label}", "measure": "rate", "value": _pct(p), "lo": _pct(lo),
                             "hi": _pct(hi), "n": n, "events": k, "primary": True})
    rows = []
    for name in spec.stratify[:3]:
        rows += _rate_rows(df.assign(_y=values), "_y", name)
    for r in rows:
        r["variable"] = L.VARIABLES[r["variable"]][0]
    res["tables"]["strata"] = rows
    res["methods"] = f"Proportion of {L.BASES[spec.base]['unit']}s with {label.lower()}, Wilson 95% CI."
    return res


def run_distribution(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    if spec.variable in L.VARIABLES:
        col, label, unit = spec.variable, L.VARIABLES[spec.variable][0], L.VARIABLES[spec.variable][2]
    else:                                    # a lab test / ICU item / treatment duration, named by the compiler
        col, label, unit = "variable_value", getattr(cohort, "labels", {}).get("variable_value") or spec.variable, ""
        if str(spec.variable).startswith("derived:"):
            unit = L.DERIVED.get(spec.variable.split(":", 1)[1], ("", ""))[1]
    summary = S.summarize(df[col])
    res["tables"]["summary"] = {"variable": label, "unit": unit, **summary}
    if summary.get("n"):
        res["estimates"] += [
            {"label": f"{label}: median (IQR) {unit}".strip(), "measure": "median", "value": summary["median"],
             "lo": summary["q1"], "hi": summary["q3"], "n": summary["n"], "primary": True},
            {"label": f"{label}: mean (SD) {unit}".strip(), "measure": "mean", "value": summary["mean"],
             "sd": summary["sd"]}]
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        lo, hi = v.quantile([0.01, 0.99])
        counts, edges = np.histogram(v.clip(lo, hi), bins=20)
        res["tables"]["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
    rows = []
    for name in spec.stratify[:2]:
        for level, part in df.groupby(_strata(df[name], name), observed=True):
            s = S.summarize(part[col])
            if s.get("n"):
                rows.append({"variable": L.VARIABLES[name][0], "level": str(level), "n": s["n"],
                             "median": s["median"], "q1": s["q1"], "q3": s["q3"]})
    res["tables"]["strata"] = rows
    res["methods"] = f"Distribution of {label.lower()} (median, IQR, percentiles); missing values not imputed."
    return res


def run_utilization(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    label = criterion_text(spec.exposure)
    exposed = df["exposed"].fillna(False).astype(int)
    k, n = int(exposed.sum()), len(df)
    p, lo, hi = S.wilson(k, n)
    window = spec.exposure.window_h or [0, 24]
    res["estimates"].append({"label": f"Received {label} {_window_text(spec, window)}", "measure": "rate",
                             "value": _pct(p),
                             "lo": _pct(lo), "hi": _pct(hi), "n": n, "events": k, "primary": True})
    coded = spec.exposure.concept.split(":")[0] in ("px9", "px10")     # ICD procedures carry a day, not a time
    if "exposure_hours" in df and k and not coded:
        h = S.summarize(df.loc[exposed == 1, "exposure_hours"])
        res["estimates"].append({"label": f"Hours to the first dose of {label} (median, IQR)", "measure": "median",
                                 "value": h.get("median"), "lo": h.get("q1"), "hi": h.get("q3"), "n": h.get("n")})
    rows = []
    for name in spec.stratify[:3]:
        rows += _rate_rows(df.assign(_y=exposed), "_y", name)
    for r in rows:
        r["variable"] = L.VARIABLES[r["variable"]][0]
    res["tables"]["strata"] = rows
    res["methods"] = (f"Proportion of {L.BASES[spec.base]['unit']}s with {label} (an ICD procedure code of the "
                      "admission; the code has a date, not a time), Wilson 95% CI.") if coded else \
        (f"Proportion of {L.BASES[spec.base]['unit']}s with at least one order / infusion of {label} in the window, "
         "Wilson 95% CI; time to the first dose among users.")
    return res


def run_survival(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    label, _kind, _col, horizon = _outcome(spec.outcome)
    horizon = horizon or 365
    if "exposure_value" in df:              # continuous exposure: curves by quartile
        df = df[df["exposure_value"].notna()]
        res["n"] = len(df)
        groups = _strata(df["exposure_value"], spec.exposure.concept[4:]).map(
            lambda level: f"{spec.exposure.label or criterion_text(spec.exposure)} {level}").values
    elif spec.exposure is not None:
        groups = np.where(df["exposed"].fillna(False).astype(bool), criterion_text(spec.exposure),
                          f"not {criterion_text(spec.exposure)}")
    elif spec.stratify:
        groups = _strata(df[spec.stratify[0]], spec.stratify[0]).values
    else:
        groups = np.array(["all"] * len(df))
    time, event = _time_to_event(df, horizon)
    curves, p = S.kaplan_meier(time, event, groups, horizon=horizon, detail=True)
    res["tables"]["km"] = curves
    # curves of quartiles (a numeric exposure or stratum) are ordered: the panel draws them on one hue
    ordered = "exposure_value" in df or (spec.exposure is None and bool(spec.stratify)
                                         and L.VARIABLES.get(spec.stratify[0], ("", ""))[1] == "numeric")
    res.setdefault("charts", {})["km"] = {"start": L.BASES[spec.base]["t0_label"], "offset_days": 0,
                                          "ordered": bool(ordered)}
    for g, c in curves.items():
        res["estimates"].append({"label": f"Survival at {horizon} days: {g}", "measure": "survival",
                                 "value": _pct(c["at_horizon"]), "n": c["n"]})
    if spec.exposure is not None and "exposed" in df and len(curves) == 2:
        hr = S.cox(time, event, df["exposed"].fillna(False).astype(int).values)
        res["estimates"].append({"label": f"Hazard ratio, {criterion_text(spec.exposure)} (unadjusted Cox)",
                                 "measure": "HR",
                                 **hr, "primary": True})
    res["diagnostics"]["logrank_p"] = p
    res["methods"] = f"Kaplan-Meier survival up to {horizon} days from {L.BASES[spec.base]['t0_label']}; log-rank test."
    return res


def run_trend(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df.copy()
    df = df[df["anchor_year_group"].map(df["anchor_year_group"].value_counts()) >= 100]
    periods = sorted(df["anchor_year_group"].dropna().unique())
    if spec.variable:
        col = spec.variable if spec.variable in L.VARIABLES else "variable_value"
        label = L.VARIABLES.get(spec.variable, (spec.variable,))[0]
        for period in periods:
            s = S.summarize(df.loc[df.anchor_year_group == period, col])
            res["estimates"].append({"label": f"{label}: median in {period}", "measure": "median",
                                     "value": s.get("median"), "lo": s.get("q1"), "hi": s.get("q3"), "n": s.get("n")})
        return res
    label, col = _measured(spec, df)
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for period in periods:
        m = (df.anchor_year_group == period).values
        p, lo, hi = S.wilson(int(y[m].sum()), int(m.sum()))
        res["estimates"].append({"label": f"{label}: {period}", "measure": "rate", "value": _pct(p), "lo": _pct(lo),
                                 "hi": _pct(hi), "n": int(m.sum()), "events": int(y[m].sum())})
    if len(periods) > 2 and y.sum() >= MIN_EVENTS:
        index = df["anchor_year_group"].map({p: i for i, p in enumerate(periods)})
        trend = S.effect_from_glm(S._glm(y.values, pd.DataFrame({"period": index.values})), "period")
        res["estimates"].append({"label": f"{label}: odds ratio per 3-year period (trend)", "measure": "OR", **trend,
                                 "primary": True})
    res["methods"] = "Rates per anchor year group (MIMIC dates are shifted; only 3-year groups are available); " \
                     "logistic regression on the period index for the trend."
    return res


def run_prognostic(spec, cohort, mode):
    res = base_result(spec, cohort)
    df = cohort.df
    label, kind, col, _h = _outcome(spec.outcome)
    if kind != "binary":
        res["warnings"].append("Prognostic models are available for binary outcomes only.")
        return res
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if y.sum() < MIN_EVENTS * 2:
        res["warnings"].append(f"Only {int(y.sum())} events: no model.")
        return res
    X, missing = S.design_matrix(df, spec.covariates, _kinds(spec.covariates))
    model = S.prognostic_model(X, y.values)
    top = sorted(model["odds_ratios_per_sd"], key=lambda r: -abs(np.log(r["value"])))[:10]
    res["estimates"].append({"label": f"{label}: cross-validated AUC", "measure": "AUC", "value": model["auc"],
                             "primary": True})
    res["estimates"].append({"label": "Calibration slope (cross-validated)", "measure": "slope",
                             "value": model["calibration_slope"]})
    for r in top:
        name = r["term"].replace("__missing", " (missing)")
        res["estimates"].append({"label": f"Odds ratio per SD: {L.VARIABLES.get(name, (name,))[0]}",
                                 "measure": "OR", **{k: r[k] for k in ("value", "lo", "hi", "p")}})
    for key in ("roc", "calibration"):
        _charts(res, key, model.get(key))
    res["diagnostics"]["covariate_missing"] = missing
    res["methods"] = f"Multivariable logistic model with {X.shape[1]} terms; 5-fold cross-validated AUC and " \
                     "calibration slope; odds ratios per standard deviation."
    return res


# ============================================================================ patients like this
def run_similar(spec, cohort, mode, live=False):
    """The stays most similar to a described patient (tools/rwd/similar.py): their outcome beside the population's,
    what they received and their outcome by treatment (crude), the risk a model of the population predicts for the
    patient, their characteristics beside the patient's, and a map of where the patient sits. ``live``: the patient
    explorer's preview — without the model's cross-validation, the map and Table 1."""
    from mes.tools.rwd import similar as SIM
    res = base_result(spec, cohort)
    res["attrition"] = list(cohort.attrition)
    df = cohort.df.reset_index(drop=True)
    label, kind, col, _horizon = _outcome(spec.outcome)
    feats = SIM.features(spec.profile, spec.base, set(df.columns))
    if not feats:
        res["warnings"].append("The patient is described by no characteristic of the data: no similar patients can be "
                               "found.")
        return res
    sc = SIM.scales(df, feats)
    must = SIM.must_mask(df, feats, sc)
    required = [f["label"] for f in feats if f["must"]]
    if required:
        res["attrition"].append((f"matching the patient on {', '.join(required)}", int(must.sum())))
    pool = df[must].reset_index(drop=True)
    if len(pool) < SIM.MIN_K:
        res["warnings"].append(f"Only {len(pool)} stays match the patient's required characteristics (at least "
                               f"{SIM.MIN_K} are needed).")
        return res
    k = min(int(spec.k), len(pool))
    if k < int(spec.k):
        res["warnings"].append(f"Only {len(pool):,} stays could be compared: all of them are used.")
    dist = SIM.distance(pool, feats, sc)
    order = SIM.nearest(dist, k)
    near = pool.iloc[order]
    in_near = np.zeros(len(pool), bool)
    in_near[order] = True
    res["attrition"].append((f"the {k:,} stays most similar to the patient", k))
    res["n"] = k
    similarity = 1 - dist[order]
    res["diagnostics"].update(k=k, n_pool=len(pool), n_cohort=len(df), features=len(feats),
                              median_similarity=round(float(np.median(similarity)), 3),
                              min_similarity=round(float(similarity.min()), 3))
    if kind == "binary":
        res["estimates"].append({"label": f"{label}: the {k:,} most similar patients", "measure": "rate",
                                 **SIM._rate(near[col]), "primary": True})
        res["estimates"].append({"label": f"{label}: all {len(pool):,} patients of the population", "measure": "rate",
                                 **SIM._rate(pool[col])})
    else:
        for name, group, primary in ((f"the {k:,} most similar patients", near, True),
                                     (f"all {len(pool):,} patients of the population", pool, False)):
            d = S.summarize(group[col])
            if d.get("n", 0) >= S.SMALL_CELL:
                res["estimates"].append({"label": f"{label}: {name}", "measure": "median", "value": d["median"],
                                         "lo": d["q1"], "hi": d["q3"], "n": d["n"], "unit": "days", "primary": primary})
    groups = []
    if "exposed" in near and spec.exposure is not None:
        exposed = SIM.as_float(near["exposed"]) == 1
        name_a = criterion_text(spec.exposure)
        if spec.comparator is not None and "comparator" in near:
            other = (SIM.as_float(near["comparator"]) == 1) & ~exposed
            name_b = criterion_text(spec.comparator)
        else:
            other, name_b = ~exposed, f"without {name_a}"
        for name, mask, role in ((name_a, exposed, "treated"), (name_b, other, "control")):
            counts = SIM._rate(near.loc[mask, col]) if kind == "binary" else {"n": int(mask.sum())}
            groups.append({"label": name, "role": role, **counts})
        a, b = groups
        if kind == "binary" and min(a["n"], b["n"]) >= S.SMALL_CELL and a["events"] is not None \
                and b["events"] is not None:
            rd, lo, hi = SIM.newcombe(a["events"], a["n"], b["events"], b["n"])
            res["estimates"].append({"label": f"{label}: {name_a} vs {name_b.removeprefix('without ')} among the similar "
                                              "patients (crude risk difference)" if spec.comparator is not None else
                                              f"{label}: with vs without {name_a} among the similar patients (crude "
                                              "risk difference)",
                                     "measure": "RD", "value": rd, "lo": lo, "hi": hi})
            for g in groups:
                res["estimates"].append({"label": f"{label}: similar patients {'' if g['label'].startswith('without') else 'with '}"
                                                  f"{g['label']}", "measure": "rate",
                                         **{k_: g[k_] for k_ in ("n", "events", "value", "lo", "hi", "small_cell")}})
        res["warnings"].append("The outcomes of similar patients by treatment are crude: the treatment was not assigned "
                               "at random (confounding by indication). An effect estimate needs a target trial "
                               "emulation.")
    res["tables"]["by_treatment"] = groups
    if kind == "binary":
        risk = SIM.risk_model(pool, feats, pd.to_numeric(pool[col], errors="coerce").fillna(0).astype(int),
                              validate=not live)
        if risk is not None:
            res["estimates"].append({"label": f"{label}: risk predicted for the patient (logistic model of the "
                                              f"population on {len(feats)} characteristics)", "measure": "risk",
                                     "value": risk["value"], "lo": risk["lo"], "hi": risk["hi"]})
            res["diagnostics"]["risk_auc"] = risk.get("auc")
    res["tables"]["profile"] = SIM.profile_rows(pool, near, feats)
    design_drugs = [c.concept for c in (spec.exposure, spec.comparator) if c is not None]
    treatments = [(v, L.VARIABLES[v][0]) for v in L.SIMILAR_TREATMENTS if v in pool and not SIM.same_drug(v, design_drugs)]
    if "exposed" in pool and spec.exposure is not None:
        treatments.append(("exposed", criterion_text(spec.exposure)))
    if "comparator" in pool and spec.comparator is not None:
        treatments.append(("comparator", criterion_text(spec.comparator)))
    res["tables"]["treatments"] = SIM.treatment_rows(pool, near, treatments)
    if not live:
        _charts(res, "map", lambda: SIM.map_of(pool, feats, sc, in_near))
        covs = list(dict.fromkeys([f["name"] for f in feats] + [c for c in spec.covariates if c in pool]))
        try:
            res["table1"] = baseline_table(pool, covs, in_near.astype(int), None, ("similar patients", "other patients"))
        except Exception:
            pass
    res["methods"] = (f"The {k:,} stays most similar to the patient among {len(pool):,}: weighted distance over "
                      f"{len(feats)} characteristics (numbers in robust standard deviations of the population, capped "
                      "at 3; yes/no and categories by agreement; a missing value counts half)"
                      + (f"; required to match: {', '.join(required)}" if required else "") + ". Proportions with "
                      "Wilson 95% CIs; crude differences with Newcombe's interval; the patient's risk from a logistic "
                      "model of the population on the same characteristics (95% CI of the prediction).")
    return res


def _patterns(name):
    def run(spec, cohort, mode):
        from mes.tools.rwd import patterns
        return getattr(patterns, f"run_{name}")(spec, cohort, mode)
    return run


RUNNERS = {"compare": run_compare, "association": run_association, "rate": run_rate,
           "distribution": run_distribution, "utilization": run_utilization, "survival": run_survival,
           "trend": run_trend, "prognostic": run_prognostic, "similar": run_similar,
           "characterize": _patterns("characterize"), "trajectory": _patterns("trajectory"),
           "pathways": _patterns("pathways"), "top": lambda spec, cohort, mode: _ranking(spec, cohort, mode)}


def _ranking(spec, cohort, mode):
    from mes.tools.rwd import ranking
    return ranking.run_top(spec, cohort, mode)


RATIOS = {"OR", "RR", "HR", "IRR"}


def _computed(e):
    """The estimate has finite numbers (a model that did not converge gives nan / inf or a ratio of 0)."""
    for key in ("value", "lo", "hi"):
        try:
            v = float(e[key]) if e.get(key) is not None else None
        except (TypeError, ValueError):
            continue                                   # e.g. a suppressed count "<10"
        if v is not None and (not np.isfinite(v) or (e.get("measure") in RATIOS and v <= 0)):
            return False
    return True


def few(spec, cohort):
    """The result of a cohort of fewer than 10 patients: how the criteria narrowed it, nothing more (small numbers
    could identify patients)."""
    res = base_result(spec, cohort)
    res["n"] = None if len(cohort.df) else 0
    res["attrition"] = [(label, None if 0 < int(n) < S.SMALL_CELL else int(n)) for label, n in cohort.attrition]
    res["warnings"].append("Fewer than 10 patients meet these criteria: no result is shown (small numbers could "
                           "identify patients). Widen the criteria — the flow of patients shows which one removes the most.")
    return res


def run(spec, cohort, mode="osler"):
    df = getattr(cohort, "df", None)
    # (patients like this need at least 30 candidates, and say so themselves)
    if df is not None and len(df) < S.SMALL_CELL and spec.analysis not in ("explore", "catalog", "similar"):
        return few(spec, cohort)
    res = RUNNERS[spec.analysis](spec, cohort, mode)
    failed = [e for e in res.get("estimates") or [] if not _computed(e)]
    if failed:
        res["estimates"] = [e for e in res["estimates"] if _computed(e)]
        res.setdefault("warnings", []).append("Not shown because the model could not be estimated on these data: "
                                              + "; ".join(e.get("label", "") for e in failed)[:400])
    return res
