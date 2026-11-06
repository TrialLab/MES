"""
Statistics for RWD analyses (deterministic, tested against known answers in tests/test_rwd.py).

Propensity scores and stabilized, trimmed inverse probability of treatment weights; weighted balance (SMD);
weighted risks with risk difference / ratio and odds ratio (robust CI); weighted Cox and Kaplan-Meier;
adjusted logistic / Cox / linear models; Wilson intervals; E-values; 1:1 propensity matching; prognostic model
discrimination. Missing covariates: median imputation plus a missing indicator.
"""

import math
import warnings

import numpy as np
import pandas as pd

Z = 1.959963984540054


# ============================================================================ basics
def wilson(k, n):
    """Proportion with Wilson 95% interval."""
    if not n:
        return None, None, None
    p = min(1.0, max(0.0, k / n))          # (a count is never above its denominator)
    denom = 1 + Z ** 2 / n
    centre = (p + Z ** 2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    v = s.dropna()
    if v.empty:
        return {"n": 0, "missing": int(s.isna().sum())}
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(v)), "missing": int(s.isna().sum()), "mean": float(v.mean()), "sd": float(v.std()),
            "median": float(q[0.5]), "q1": float(q[0.25]), "q3": float(q[0.75]), "p5": float(q[0.05]),
            "p95": float(q[0.95]), "min": float(v.min()), "max": float(v.max())}


def _first(x):
    """The values of a (values, info) pair from another function of this module, when the pair is passed on whole."""
    if isinstance(x, tuple) and len(x) == 2 and not np.isscalar(x[0]):
        return x[0]
    return x


def _kind(s):
    values = set(pd.unique(s.dropna()))
    if values <= {0, 1, True, False}:
        return "binary"
    return "numeric" if pd.api.types.is_numeric_dtype(s) else "category"


def design_matrix(df, columns, kinds=None):
    """Numeric model matrix: numeric → median-imputed (+ missing indicator), binary → 0/1, category → one-hot.
    ``kinds``: {column: kind}, a list in the order of ``columns``, or None (inferred from the values)."""
    if kinds is None:
        kinds = {c: _kind(df[c]) for c in columns if c in df}
    elif isinstance(kinds, (list, tuple)):
        kinds = dict(zip(columns, kinds))
    elif isinstance(kinds, str):
        kinds = {c: kinds for c in columns}
    parts, info = [], {}
    for col in columns:
        if col not in df:
            continue
        kind = kinds.get(col, "numeric")
        s = df[col]
        info[col] = round(float(s.isna().mean()), 4)
        if kind == "category":
            levels = s.fillna("Missing").astype(str)
            top = levels.value_counts().index[:8]
            levels = levels.where(levels.isin(top), "Other")
            dummies = pd.get_dummies(levels, prefix=col, drop_first=True, dtype=float)
            parts.append(dummies)
        elif kind == "binary":
            parts.append(pd.to_numeric(s.astype(float), errors="coerce").fillna(0).rename(col).to_frame())
        else:
            v = pd.to_numeric(s, errors="coerce").astype(float)
            if v.isna().any():
                parts.append(v.isna().astype(float).rename(f"{col}__missing").to_frame())
            parts.append(v.fillna(v.median() if v.notna().any() else 0).rename(col).to_frame())
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    # near-constant indicators (< 0.5% or > 99.5%) cause separation and unstable fits
    binary = [c for c in X.columns if set(pd.unique(X[c])) <= {0.0, 1.0}]
    rare = [c for c in binary if not 0.005 <= X[c].mean() <= 0.995]
    X = X.drop(columns=rare)
    X = X.loc[:, X.std(ddof=0) > 0]
    return X, info


def _standardize(X):
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    return (X - mu) / sd


# ============================================================================ propensity / weights
def _ps_features(X):
    """Standardized covariates plus cubic-spline terms of the continuous ones (age, scores and labs rarely act
    linearly on the log-odds of treatment)."""
    from sklearn.preprocessing import SplineTransformer
    Z = _standardize(X).values
    continuous = [c for c in X.columns if X[c].nunique() > 10]
    if not continuous:
        return Z
    splines = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False).fit_transform(
        X[continuous].values)
    return np.hstack([Z, splines])


def propensity(X, t):
    """Propensity scores and their AUC: (ps, auc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X, t = _first(X), np.asarray(t)
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float))
    Xs = _ps_features(X) if X.shape[1] else np.zeros((len(t), 1))
    model = LogisticRegression(max_iter=5000, C=10.0)
    model.fit(Xs, t)
    ps = model.predict_proba(Xs)[:, 1]
    auc = float(roc_auc_score(t, ps)) if 0 < t.sum() < len(t) else None
    return ps, auc


def iptw(ps, t, trim=(0.01, 0.99)):
    """Stabilized ATE weights with propensity scores truncated at the given quantiles."""
    ps, t = np.asarray(_first(ps), float), np.asarray(t)
    lo, hi = np.quantile(ps, trim)
    p = np.clip(ps, max(lo, 1e-6), min(hi, 1 - 1e-6))
    pt = t.mean()
    w = np.where(t == 1, pt / p, (1 - pt) / (1 - p))
    info = {"truncated_share": float(np.mean((ps < lo) | (ps > hi))),
            "ess_treated": float(w[t == 1].sum() ** 2 / (w[t == 1] ** 2).sum()),
            "ess_control": float(w[t == 0].sum() ** 2 / (w[t == 0] ** 2).sum()),
            "ps_treated": [float(np.quantile(ps[t == 1], q)) for q in (0.05, 0.5, 0.95)],
            "ps_control": [float(np.quantile(ps[t == 0], q)) for q in (0.05, 0.5, 0.95)]}
    # common support: share of each group inside the other group's 1st–99th percentile range of the score
    lo1, hi1 = np.quantile(ps[t == 1], [0.01, 0.99])
    lo0, hi0 = np.quantile(ps[t == 0], [0.01, 0.99])
    info["overlap"] = float(min(np.mean((ps[t == 1] >= lo0) & (ps[t == 1] <= hi0)),
                                np.mean((ps[t == 0] >= lo1) & (ps[t == 0] <= hi1))))
    return w, info


PS_QUANTILES = [round(q, 3) for q in np.linspace(0.025, 0.975, 39)]


def ps_distribution(ps, t):
    """The propensity score of each group as quantiles (2.5 % to 97.5 %, in steps of 2.5 %), for an overlap plot:
    the shape of both distributions without any count of patients."""
    ps, t = np.asarray(_first(ps), float), np.asarray(t)
    if not (t == 1).any() or not (t == 0).any():
        return None
    return {"q": PS_QUANTILES, "treated": [float(v) for v in np.quantile(ps[t == 1], PS_QUANTILES)],
            "control": [float(v) for v in np.quantile(ps[t == 0], PS_QUANTILES)]}


def overlap_weights(ps, t):
    """Overlap weights (treated 1 − e, comparison e). With a logistic propensity model they balance the means of
    its covariates exactly and never explode; the estimand is the effect among patients for whom either treatment
    is plausible (clinical equipoise; Li, Morgan & Zaslavsky, JASA 2018)."""
    ps, t = np.asarray(_first(ps), float), np.asarray(t)
    w = np.where(t == 1, 1 - ps, ps)
    return w, {"ess_treated": float(w[t == 1].sum() ** 2 / (w[t == 1] ** 2).sum()),
               "ess_control": float(w[t == 0].sum() ** 2 / (w[t == 0] ** 2).sum())}


def smd(X, t, w=None):
    """Standardized mean differences (unweighted and, with ``w``, weighted)."""
    X, t = _first(X), np.asarray(t)
    w = np.ones(len(t)) if w is None else np.asarray(_first(w), float)
    rows = []
    for col in X.columns:
        x = X[col].values.astype(float)
        m1, m0 = np.average(x[t == 1], weights=w[t == 1]), np.average(x[t == 0], weights=w[t == 0])
        v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1])
        v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
        pooled = math.sqrt((v1 + v0) / 2) or 1.0
        rows.append((col, m1, m0, (m1 - m0) / pooled))
    return pd.DataFrame(rows, columns=["covariate", "mean_treated", "mean_control", "smd"])


# ============================================================================ effects
def weighted_risks(y, t, w, n_boot=1000, seed=0):
    """Weighted risk in each group, risk difference and ratio with bootstrap 95% CIs (weights fixed; fewer
    resamples for very large samples)."""
    y, t, w = np.asarray(y, float), np.asarray(t), np.asarray(_first(w), float)
    treated = t == 1
    # per row: weighted events and weights of each group; a resample's risks are ratios of column sums
    parts = np.column_stack([w * y * treated, w * treated, w * y * ~treated, w * ~treated])
    if not (parts[:, 1].sum() > 0 and parts[:, 3].sum() > 0):
        raise ValueError("weighted_risks needs treated and untreated rows with positive weights")
    n_boot = min(n_boot, max(200, 100_000_000 // max(len(y), 1)))

    def risks(sums):
        with np.errstate(divide="ignore", invalid="ignore"):
            r1, r0 = sums[..., 0] / sums[..., 1], sums[..., 2] / sums[..., 3]
            return r1, r0, r1 - r0, np.where(r0 > 0, r1 / r0, np.nan)

    base = risks(parts.sum(0))
    rng = np.random.default_rng(seed)
    n, step = len(y), max(1, 1_000_000 // max(len(y), 1))
    sums = np.concatenate([parts[rng.integers(0, n, (min(step, n_boot - i), n))].sum(1)
                           for i in range(0, n_boot, step)])
    boots = np.column_stack(risks(sums))
    ci = lambda j: (float(np.nanquantile(boots[:, j], 0.025)), float(np.nanquantile(boots[:, j], 0.975)))
    return {"risk_treated": float(base[0]), "risk_control": float(base[1]),
            "rd": float(base[2]), "rd_ci": ci(2), "rr": float(base[3]), "rr_ci": ci(3)}


TERMS = ("treated", "exposure", "lp", "period")      # the effect a model is fitted for: never dropped


def _independent(X, keep=TERMS, tol=1e-7):
    """The columns of ``X`` (with a constant first) that are not linear combinations of earlier ones: a covariate set
    with a redundant column (two indicators of the same missing values, a category nobody has) otherwise leaves the
    robust standard errors undefined. Columns in ``keep`` come first."""
    cols = [c for c in X.columns if c in keep] + [c for c in X.columns if c not in keep]
    basis, chosen = [], []
    for c in ["const"] + cols:
        v = np.ones(len(X)) if c == "const" else X[c].to_numpy(float)
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            continue
        r = v.copy()
        for q in basis:
            r -= (q @ r) * q
        if np.linalg.norm(r) > tol * norm:
            basis.append(r / np.linalg.norm(r))
            chosen.append(c)
    return [c for c in cols if c in chosen]


def _glm(y, X, w=None, family="binomial"):
    import statsmodels.api as sm
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Gaussian()
    X = pd.DataFrame(X).reset_index(drop=True)
    X = X[_independent(X)]
    model = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=fam,
                   var_weights=None if w is None else np.asarray(w, float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(cov_type="HC0")


def effect_from_glm(fit, term, exp=True):
    est, se = float(fit.params[term]), float(fit.bse[term])
    lo, hi = est - Z * se, est + Z * se
    f = (lambda v: math.exp(min(v, 700))) if exp else (lambda v: v)
    return {"value": f(est), "lo": f(lo), "hi": f(hi), "p": float(fit.pvalues[term])}


def weighted_or(y, t, w, X=None):
    """Odds ratio of treatment (weighted logistic regression, robust SE); doubly robust when ``X`` is given."""
    design = pd.DataFrame({"treated": np.asarray(t, float)})
    X, w = _first(X), _first(w)
    if X is not None and X.shape[1]:
        design = pd.concat([design, _standardize(X).reset_index(drop=True)], axis=1)
    return effect_from_glm(_glm(np.asarray(y, float), design, w), "treated")


def mean_difference(y, t, w=None, X=None):
    design = pd.DataFrame({"treated": np.asarray(t, float)})
    X, w = _first(X), _first(w)
    if X is not None and X.shape[1]:
        design = pd.concat([design, _standardize(X).reset_index(drop=True)], axis=1)
    return effect_from_glm(_glm(np.asarray(y, float), design, w, family="gaussian"), "treated", exp=False)


def cox(time, event, x, w=None, X=None, name="treated"):
    """Hazard ratio of ``x`` (Cox, robust SE; weighted with ``w``; adjusted for ``X``)."""
    from lifelines import CoxPHFitter
    data = pd.DataFrame({"T": np.asarray(time, float), "E": np.asarray(event, int), name: np.asarray(x, float)})
    X, w = _first(X), _first(w)
    if X is not None and X.shape[1]:
        data = pd.concat([data, _standardize(X).reset_index(drop=True)], axis=1)
    if w is not None:
        data["w"] = np.asarray(w, float)
    model = CoxPHFitter(penalizer=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, duration_col="T", event_col="E", weights_col="w" if w is not None else None, robust=True)
    row = model.summary.loc[name]
    return {"value": float(row["exp(coef)"]), "lo": float(row["exp(coef) lower 95%"]),
            "hi": float(row["exp(coef) upper 95%"]), "p": float(row["p"])}


SMALL_CELL = 10          # counts below this are not shown (patients at risk, Table 1 cells)


def risk_times(horizon):
    """The times of a numbers-at-risk table (days): 0, steps of a round length (≤ 5 of them), and the horizon."""
    if not horizon or horizon <= 0:
        return [0.0]
    step = next((s for s in (1, 2, 5, 7, 10, 14, 30, 60, 90, 180, 365) if horizon / s <= 5), horizon / 4)
    times = [float(k * step) for k in range(int(horizon // step) + 1)]
    if horizon - times[-1] >= step / 2:
        times.append(float(horizon))
    return [round(t, 4) for t in times]


def kaplan_meier(time, event, groups, w=None, horizon=None, detail=False):
    """Survival curves per group: {group: {"t": [...], "s": [...], "at_horizon": s}} and the log-rank p.
    ``detail``: each curve also gets its pointwise 95% band ("lo", "hi"; Greenwood, with the weights) and the
    patients at risk at ``risk_times(horizon)`` ("at_risk": [[day, n]], n None when below 10)."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    time, event, groups = np.asarray(time, float), np.asarray(event, int), np.asarray(groups)
    w = _first(w)
    curves = {}
    warnings.filterwarnings("ignore", message=".*weights are not integers.*")
    for g in pd.unique(groups):
        m = groups == g
        kmf = KaplanMeierFitter()
        kmf.fit(time[m], event[m], weights=None if w is None else np.asarray(w)[m])
        sf = kmf.survival_function_.iloc[:, 0]
        step = max(1, len(sf) // 120)
        curves[str(g)] = {"t": [float(v) for v in sf.index[::step]], "s": [float(v) for v in sf.values[::step]],
                          "at_horizon": float(kmf.predict(horizon)) if horizon else None, "n": int(m.sum())}
        if detail:
            try:
                band = kmf.confidence_interval_survival_function_.iloc[::step]
                curves[str(g)]["lo"] = [float(v) for v in band.iloc[:, 0].values]
                curves[str(g)]["hi"] = [float(v) for v in band.iloc[:, 1].values]
            except Exception:              # a band is a detail: the curve stands without it
                pass
            tg = time[m]
            curves[str(g)]["at_risk"] = [[t, (lambda n: n if n >= SMALL_CELL else None)(int((tg >= t - 1e-9).sum()))]
                                         for t in risk_times(horizon or float(tg.max(initial=0)))]
    p = None
    if len(curves) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(multivariate_logrank_test(time, groups, event).p_value)
    return curves, p


def e_value(value, lo, hi, measure="RR", rare=True):
    """E-value for the estimate and for the confidence limit closest to the null (VanderWeele & Ding)."""
    def to_rr(v):
        if v is None or v <= 0 or (isinstance(v, float) and math.isnan(v)):
            return None
        if measure == "OR" and not rare:
            return math.sqrt(v)
        if measure == "HR" and not rare:
            return (1 - 0.5 ** math.sqrt(v)) / (1 - 0.5 ** math.sqrt(1 / v))
        return v

    def ev(rr):
        if rr is None:
            return None
        rr = 1 / rr if rr < 1 else rr
        return rr + math.sqrt(rr * (rr - 1))

    point = to_rr(value)
    if point is None:
        return None, None
    limit = lo if point >= 1 else hi
    limit = to_rr(limit)
    crosses = limit is None or (point >= 1 and limit <= 1) or (point < 1 and limit >= 1)
    return ev(point), (1.0 if crosses else ev(limit))


def match_1to1(ps, t, caliper_sd=0.2, seed=0):
    """Greedy nearest-neighbour 1:1 matching on the logit of the propensity score (no replacement)."""
    ps, t = np.asarray(_first(ps), float), np.asarray(t)
    logit = np.log(np.clip(ps, 1e-6, 1 - 1e-6) / (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
    caliper = caliper_sd * logit.std()
    treated = np.where(t == 1)[0]
    controls = np.where(t == 0)[0]
    order = np.random.default_rng(seed).permutation(treated)
    available = np.ones(len(controls), bool)
    ctrl_logit = logit[controls]
    pairs = []
    for i in order:
        d = np.abs(ctrl_logit - logit[i])
        d[~available] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            available[j] = False
            pairs.append((i, controls[j]))
    return pairs


def prognostic_model(X, y, folds=5, seed=0):
    """Logistic model of ``y`` on ``X``: cross-validated AUC, calibration slope, ORs per SD."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    Xs = _standardize(_first(X))
    y = np.asarray(y, int)
    pred = np.zeros(len(y))
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xs, y):
        m = LogisticRegression(max_iter=3000).fit(Xs.values[train], y[train])
        pred[test] = m.predict_proba(Xs.values[test])[:, 1]
    auc = float(roc_auc_score(y, pred))
    lp = np.log(np.clip(pred, 1e-6, 1 - 1e-6) / (1 - np.clip(pred, 1e-6, 1 - 1e-6)))
    slope = effect_from_glm(_glm(y, pd.DataFrame({"lp": lp})), "lp", exp=False)["value"]
    # odds ratios from the (L2-penalized) model fitted on all data; no CIs for a prediction model
    full = LogisticRegression(max_iter=3000).fit(Xs.values, y)
    ors = [{"term": c, "value": float(np.exp(b)), "lo": None, "hi": None, "p": None}
           for c, b in zip(Xs.columns, full.coef_[0])]
    return {"auc": auc, "calibration_slope": slope, "odds_ratios_per_sd": ors, **discrimination(y, pred)}


def discrimination(y, pred, points=60, bins=10):
    """The ROC curve ({"roc": [[false positive rate, true positive rate]], at most ``points``}) and the calibration
    of predicted risks ({"calibration": [{"pred", "obs", "n"}]} per tenth of the predictions, in %; left out when a
    tenth has fewer than 10 patients) of cross-validated predictions."""
    from sklearn.metrics import roc_curve
    y, pred = np.asarray(y, int), np.asarray(pred, float)
    out = {}
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, pred)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), points)).round().astype(int))
        out["roc"] = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep]
    groups = pd.qcut(pd.Series(pred), bins, labels=False, duplicates="drop").values
    rows = [{"pred": round(100 * float(pred[groups == g].mean()), 2), "obs": round(100 * float(y[groups == g].mean()), 2),
             "n": int((groups == g).sum())} for g in sorted(pd.unique(groups[~pd.isna(groups)]))]
    if len(rows) >= 3 and all(r["n"] >= SMALL_CELL for r in rows):
        out["calibration"] = rows
    return out
