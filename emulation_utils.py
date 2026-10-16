"""
Statistical tools used by the RWD-Analyst for target trial emulation:
covariate balancing (propensity score matching, inverse probability of treatment
weighting), standardized mean differences, survival models (Cox, Kaplan-Meier,
parametric AFT, random survival forest, doubly robust estimation),
clone-censor-weight analysis for time-dependent treatment initiation,
subgroup analysis, sample size and E-values.
"""

import math

import lifelines
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from causallib.estimation import IPW, PropensityFeatureStandardization, Standardization
from causallib.evaluation import evaluate
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================ balance
def check_balance_after_matching(X, all_vars, treatment_col, weights=None):
    """Absolute standardized mean difference (SMD) of each variable between treatment groups."""
    smd = {}
    for var in all_vars:
        groups = []
        for g in (1, 0):
            mask = X[treatment_col] == g
            x = X.loc[mask, var].astype(float)
            w = np.ones(len(x)) if weights is None else X.loc[mask, weights].values
            mean = np.average(x, weights=w)
            var_ = np.average((x - mean) ** 2, weights=w)
            groups.append((mean, var_))
        pooled = np.sqrt((groups[0][1] + groups[1][1]) / 2)
        smd[var] = abs(groups[0][0] - groups[1][0]) / pooled if pooled > 0 else 0.0
    return pd.Series(smd)


def calculate_propensity_scores(X, treatment_col, covariate_list):
    model = LogisticRegression(solver="liblinear")
    model.fit(X[covariate_list], X[treatment_col])
    return pd.Series(model.predict_proba(X[covariate_list])[:, 1], index=X.index, name="ps")


def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def match_pairs(distances, treated_idx, control_idx, N=1, caliper=None):
    """Greedy nearest-neighbour matching without replacement (N controls per treated patient)."""
    matched_pairs = []
    matches_per_treated = dict.fromkeys(treated_idx, 0)
    unmatchable = set()
    available_controls = np.ones(len(control_idx), dtype=bool)
    for _ in range(N):
        for i, treated in enumerate(treated_idx):
            if treated in unmatchable or matches_per_treated[treated] >= N:
                continue
            dist_row = distances[i, available_controls]
            if dist_row.size == 0:
                break
            local_min_idx = np.argmin(dist_row)
            if caliper is not None and dist_row[local_min_idx] > caliper:
                unmatchable.add(treated)
                continue
            original_pos = np.where(available_controls)[0][local_min_idx]
            matched_pairs.append((treated, control_idx[original_pos]))
            matches_per_treated[treated] += 1
            available_controls[original_pos] = False
    return matched_pairs


def matching_propensity(X, covariates, treatment_col, N=1, use_caliper=False):
    treated_idx = X[X[treatment_col] == 1].index
    control_idx = X[X[treatment_col] == 0].index
    ps = calculate_propensity_scores(X, treatment_col, covariates)
    distances = scipy.spatial.distance.cdist(ps.loc[treated_idx].values.reshape(-1, 1),
                                             ps.loc[control_idx].values.reshape(-1, 1))
    caliper = mad(distances.flatten()) if use_caliper else None
    return match_pairs(distances, treated_idx, control_idx, N=N, caliper=caliper), X.assign(ps=ps)


def perform_balancing_method(df, continuous_variables, categorical_variables, treatment_col,
                             use_caliper=True, matching_ratio="1:1"):
    """One round of propensity score matching with the given matching ratio."""
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = df.copy()
    if continuous_variables:
        X[continuous_variables] = scaler.fit_transform(X[continuous_variables])
    all_vars = continuous_variables + categorical_variables
    N = int(matching_ratio.split(":")[-1])
    matched_pairs, X_ps = matching_propensity(X, all_vars, treatment_col, N=N, use_caliper=use_caliper)
    if not matched_pairs:
        return df, None
    matched_indices = [idx for pair in matched_pairs for idx in pair]
    return X_ps.loc[matched_indices].reset_index(drop=True), scaler


def perform_psm(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment,
                list_of_outcome, outcome_definition, break_flag, results):
    """Propensity score matching, trying matching ratios 1:1 to 1:4 until covariates are balanced (SMD < 0.1)."""
    ps_info = "Propensity Score Matching Summary:"
    all_vars = continuous_covariates + categorical_covariates
    tolerance = np.ceil(len(all_vars) * 0.02)
    X, scaler, balances, unbalanced = pd.DataFrame(), None, pd.Series(dtype=float), all_vars
    for ratio in ["1:1", "1:2", "1:3", "1:4"]:
        temp_X, temp_scaler = perform_balancing_method(df, continuous_covariates, categorical_covariates,
                                                       list_of_treatment[0], matching_ratio=ratio)
        if temp_scaler is None or temp_X.empty:
            continue
        X, scaler = temp_X, temp_scaler
        balances = check_balance_after_matching(X, all_vars, list_of_treatment[0])
        unbalanced = [c for c in balances[balances > 0.1].index if c not in ignore_covariates]
        if len(unbalanced) <= tolerance:
            ps_info += f"\nMatching ratio: {ratio}"
            break

    if X.empty:
        return df, ps_info + "\nNo matched pairs could be formed.", results, True, None
    results["Balances after performing balancing"] = balances
    ps_info += f"\n\nCurrently used covariates: {', '.join(all_vars)}"
    ps_info += f"\nNumber of matched patients: {len(X)}"
    ps_info += f"\nNumber of unbalanced covariates (SMD > 0.1): {len(unbalanced)}"
    ps_info += "\nAll covariates are balanced." if not unbalanced else f"\nUnbalanced covariates: {', '.join(unbalanced)}"
    if len(unbalanced) > tolerance:
        return X, ps_info, results, True, scaler

    incidence_treated = X[X[list_of_treatment[0]] == 1][list_of_outcome[0]].mean()
    incidence_control = X[X[list_of_treatment[0]] == 0][list_of_outcome[0]].mean()
    ps_info += f"\nIncidence in the treated group: {incidence_treated:.4f}"
    ps_info += f"\nIncidence in the control group: {incidence_control:.4f}"
    ps_info += f"\nAverage Treatment Effect for {outcome_definition[0]}: {incidence_treated - incidence_control:.4f}"
    return X, ps_info, results, break_flag, scaler


def perform_iptw(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment,
                 list_of_outcome, outcome_definition, break_flag, results):
    """Inverse probability of treatment weighting with balance assessment (weighted SMD)."""
    df = df.copy()
    iptw_info = "Inverse Probability of Treatment Weighting Summary:"
    covariates = categorical_covariates + continuous_covariates
    X, a, y = df[covariates], df[list_of_treatment[0]], df[list_of_outcome[0]]
    ipw = IPW(LogisticRegression(solver="liblinear"))
    ipw.fit(X, a)
    df["iptw_weight"] = ipw.compute_weights(X, a)

    results_ipw = evaluate(ipw, X, a, y)
    balances = results_ipw.evaluated_metrics.covariate_balance["weighted"]
    results["Balances after performing balancing"] = balances
    unbalanced = [c for c in balances[balances > 0.1].index if c not in ignore_covariates]
    iptw_info += f"\n\nCurrently used covariates: {', '.join(covariates)}"
    iptw_info += f"\nNumber of unbalanced covariates (SMD > 0.1): {len(unbalanced)}"
    iptw_info += "\nAll covariates are balanced." if not unbalanced else f"\nUnbalanced covariates: {', '.join(unbalanced)}"
    if len(unbalanced) > np.ceil(len(covariates) * 0.02):
        return df, iptw_info, results, True

    outcomes = ipw.estimate_population_outcome(X, a, y)
    ate = ipw.estimate_effect(outcomes.loc[1], outcomes.loc[0])
    iptw_info += f"\nIncidence in the treated group: {outcomes.loc[1]:.4f}"
    iptw_info += f"\nIncidence in the control group: {outcomes.loc[0]:.4f}"
    iptw_info += f"\nAverage Treatment Effect for {outcome_definition[0]}: {ate['diff']:.4f}"
    return df, iptw_info, results, break_flag


# ============================================================================ data cleaning
def clean_outliers(data):
    """Set values outside the 1st-99th percentiles to missing."""
    lower, upper = data.quantile(0.01), data.quantile(0.99)
    return data.where((data >= lower) & (data <= upper))


def clean_data(list_of_covariates, list_of_treatment, list_of_outcome, list_of_duration, identifier_columns, data,
               extra_columns=None):
    """Select columns, type covariates, remove outliers, impute (mean/mode) and one-hot encode categoricals."""
    data = data.copy()
    keep = list_of_covariates + list_of_treatment + list_of_outcome + list_of_duration + identifier_columns + (extra_columns or [])
    data = data[[c for c in dict.fromkeys(keep) if c in data.columns]]

    categorical, continuous = [], []
    for covariate in list_of_covariates:
        if data[covariate].dtype == "object":
            numeric_col = pd.to_numeric(data[covariate], errors="coerce")
            if numeric_col.notna().sum() > 0.8 * data[covariate].notna().sum():
                data[covariate] = numeric_col
                continuous.append(covariate)
            else:
                categorical.append(covariate)
        else:
            continuous.append(covariate)

    if continuous:
        data[continuous] = clean_outliers(data[continuous])
    imputation = {col: data[col].mean() for col in continuous}
    imputation.update({col: data[col].mode().iloc[0] for col in categorical if data[col].notna().any()})
    data = data.fillna(imputation)

    data = pd.get_dummies(data, columns=categorical, prefix=categorical, drop_first=True)
    dummies = [c for c in data.columns if any(c.startswith(cat + "_") for cat in categorical)]
    for col in list_of_treatment + list_of_outcome:
        data[col] = data[col].astype(float)
    for col in data.columns:
        if data[col].dtype == "bool":
            data[col] = data[col].astype(int)
    # Drop covariates without variation.
    continuous = [c for c in continuous if data[c].nunique() > 1]
    dummies = [c for c in dummies if data[c].nunique() > 1]
    return data, dummies, continuous, list_of_treatment, list_of_outcome, list_of_duration


# ============================================================================ survival models
def calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates,
                            continuous_covariates):
    """Cox proportional hazards model of the outcome on treatment and covariates (robust variance)."""
    cph = CoxPHFitter(penalizer=0.1)
    features = categorical_covariates + continuous_covariates + list_of_treatment + list_of_outcome + list_of_duration
    if "iptw_weight" in X.columns:
        features.append("iptw_weight")
    data_cox = X[features].dropna(axis=1, how="all")
    data_cox = data_cox.loc[:, data_cox.nunique() > 1]
    kwargs = {"weights_col": "iptw_weight"} if "iptw_weight" in data_cox.columns else {}
    cph.fit(data_cox, duration_col=list_of_duration[0], event_col=list_of_outcome[0], robust=True, **kwargs)
    return cph, data_cox


def run_cox_model(X, list_of_treatment, list_of_outcome, list_of_duration, outcome_definition, categorical_covariates,
                  continuous_covariates):
    cph, data_cox = calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration,
                                            categorical_covariates, continuous_covariates)
    row = cph.summary.loc[list_of_treatment[0]]
    try:
        ph = proportional_hazard_test(cph, data_cox, time_transform="rank").summary
        ph_str = (f"Proportional hazards test (scaled Schoenfeld residuals): p = {ph.loc[list_of_treatment[0], 'p']:.4f} "
                  f"for treatment; {int((ph['p'] < 0.05).sum())} of {len(ph)} terms with p < 0.05.")
    except Exception:
        ph_str = ""
    cph_info = (
        "Cox Proportional Hazards Model Summary:"
        f"\nHazard Ratio for {outcome_definition[0]}: {row['exp(coef)']:.4f}"
        f"\n95% Confidence Interval: ({row['exp(coef) lower 95%']:.4f}, {row['exp(coef) upper 95%']:.4f})"
        f"\np-value: {row['p']:.4f}\n{ph_str}\n"
    )
    primary = {"effect_measure": "HR", "effect_size": float(row["exp(coef)"]),
               "ci_lower": float(row["exp(coef) lower 95%"]), "ci_upper": float(row["exp(coef) upper 95%"]),
               "p_value": float(row["p"])}
    return cph.summary.to_dict(), cph_info, primary


def run_kaplan_meier(df, list_of_treatment, list_of_outcome, list_of_duration):
    treatment_col, outcome_col, duration_col = list_of_treatment[0], list_of_outcome[0], list_of_duration[0]
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    survival_functions = {}
    for group in sorted(df[treatment_col].unique()):
        mask = df[treatment_col] == group
        kmf.fit(df.loc[mask, duration_col], event_observed=df.loc[mask, outcome_col],
                weights=df.loc[mask, "iptw_weight"] if "iptw_weight" in df.columns else None,
                label=f"Group {int(group)}")
        kmf.plot_survival_function(ax=ax)
        survival_functions[str(group)] = kmf.survival_function_
    ax.set_title("Kaplan-Meier Survival Curves")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival Probability")
    plt.close(fig)
    return survival_functions, fig


def run_parametric_model(balanced_df, list_of_treatment, list_of_outcome, list_of_duration, outcome_definition,
                         covariates=None):
    """Select the best univariate parametric model by AIC and fit the corresponding AFT model."""
    duration_col, event_col, treatment_col = list_of_duration[0], list_of_outcome[0], list_of_treatment[0]
    cols = (covariates or []) + [treatment_col, event_col, duration_col]
    if "iptw_weight" in balanced_df.columns:
        cols.append("iptw_weight")
    df = balanced_df[cols].copy()
    df[duration_col] = df[duration_col].clip(lower=1e-3)
    fitters = {"Weibull": lifelines.WeibullFitter(), "LogNormal": lifelines.LogNormalFitter(),
               "LogLogistic": lifelines.LogLogisticFitter()}
    aics = {name: fitter.fit(df[duration_col], df[event_col]).AIC_ for name, fitter in fitters.items()}
    best = min(aics, key=aics.get)
    aft, prefix = {"Weibull": (lifelines.WeibullAFTFitter(), "lambda_"),
                   "LogNormal": (lifelines.LogNormalAFTFitter(), "mu_"),
                   "LogLogistic": (lifelines.LogLogisticAFTFitter(), "alpha_")}[best]
    kwargs = {"weights_col": "iptw_weight"} if "iptw_weight" in df.columns else {}
    aft.fit(df, duration_col=duration_col, event_col=event_col, **kwargs)
    row = aft.summary.loc[(prefix, treatment_col)]
    info = ("Parametric Survival Models Summary:"
            f"\nUsing Model: {best} AFT (selected by AIC)"
            f"\nTime ratio from {best} AFT for {outcome_definition[0]}: {row['exp(coef)']:.4f}"
            f"\n95% Confidence Interval: ({row['exp(coef) lower 95%']:.4f}, {row['exp(coef) upper 95%']:.4f})"
            f"\np-value: {row['p']:.4f}\n")
    return aft.summary, info


def _bootstrap_hazard_ratio(rsf, X_test, treatment_col, time_point, n_bootstrap=100, ci_level=0.95, seed=20):
    """Ratio of mean cumulative hazards (treated vs control) at ``time_point`` with bootstrap CI over the test set."""
    rng = np.random.default_rng(seed)
    chf_all = rsf.predict_cumulative_hazard_function(X_test)
    time_point = min(time_point, min(fn.x[-1] for fn in chf_all))
    chf = np.array([fn(time_point) for fn in chf_all])
    treated = (X_test[treatment_col] == 1).values
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(chf), len(chf))
        t, c = chf[idx][treated[idx]], chf[idx][~treated[idx]]
        if len(t) and len(c) and c.mean() > 0:
            boot.append(t.mean() / c.mean())
    observed = chf[treated].mean() / chf[~treated].mean()
    lower, upper = np.percentile(boot, [(1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100])
    return observed, (lower, upper)


def run_survival_forest(balanced_df, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates,
                        continuous_covariates, outcome_definition):
    info_str = "Random Survival Forest Results:"
    try:
        treatment_col = list_of_treatment[0]
        Xt = balanced_df[categorical_covariates + continuous_covariates + [treatment_col]]
        y = np.array(list(zip(balanced_df[list_of_outcome[0]].astype(bool), balanced_df[list_of_duration[0]])),
                     dtype=[("event", bool), ("time", float)])
        X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=20)
        rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=10, min_samples_leaf=15, n_jobs=-1,
                                   random_state=20)
        rsf.fit(X_train, y_train)
        info_str += f"\nC-Index (test set): {rsf.score(X_test, y_test):.4f}"
        time_point = float(np.median(y["time"]))
        hr, ci = _bootstrap_hazard_ratio(rsf, X_test, treatment_col, time_point)
        info_str += (f"\nRandom Survival Forest cumulative hazard ratio at {time_point:.1f} days for "
                     f"{outcome_definition[0]}: {hr:.3f}\n95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})\n")
    except Exception as e:
        info_str += f"\nRandom Survival Forest could not be run. Error: {e}\n"
    return info_str


def _bootstrap_ate(X, a, y, estimator, indices):
    X_b, a_b, y_b = X.iloc[indices], a.iloc[indices], y.iloc[indices]
    estimator.fit(X_b, a_b, y_b)
    outcome = estimator.estimate_population_outcome(X_b, a_b)
    return outcome[1] - outcome[0]


def run_doubly_robust(df, list_of_treatment, list_of_outcome, categorical_covariates, continuous_covariates,
                      outcome_definition, n_bootstrap=200, seed=42):
    """Doubly robust (propensity-feature standardization) ATE with bootstrap CI."""
    dr = PropensityFeatureStandardization(Standardization(LinearRegression()), IPW(LogisticRegression(solver="liblinear")))
    X = df[continuous_covariates + categorical_covariates]
    a, y = df[list_of_treatment[0]], df[list_of_outcome[0]]
    rng = np.random.default_rng(seed)
    samples = [rng.choice(len(X), size=len(X), replace=True) for _ in range(n_bootstrap)]
    estimates = Parallel(n_jobs=-1)(delayed(_bootstrap_ate)(X, a, y, dr, s) for s in samples)
    ci = np.percentile(estimates, [2.5, 97.5])
    return (f"Doubly Robust Estimation Summary:\nAverage Treatment Effect (risk difference) for {outcome_definition[0]}: "
            f"{np.mean(estimates):.3f}\n95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})\n")


# ============================================================================ clone-censor-weight
def is_time_dependent_treatment(df, treatment_col, treatment_time_col, min_delay_days=0.25, min_fraction=0.1):
    """Treatment initiation is time dependent if a relevant share of treated patients start after time zero."""
    if treatment_time_col is None or treatment_time_col not in df.columns:
        return False
    times = df.loc[df[treatment_col] == 1, treatment_time_col].dropna()
    return len(times) > 0 and (times > min_delay_days).mean() >= min_fraction


def _fit_probability(X, y, at_risk):
    """P(y = 1 | X) from a logistic regression fitted on rows that are at risk; returns probabilities for all rows."""
    if at_risk.sum() == 0 or y[at_risk].min() == y[at_risk].max():
        return np.full(len(X), y[at_risk].mean() if at_risk.any() else 0.0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X[at_risk], y[at_risk])
    return model.predict_proba(X)[:, 1]


def clone_censor_weight(df, id_col, duration_col, event_col, treatment_time_col, covariates, grace_period,
                        n_grace_intervals=4, truncation_quantile=None):
    """
    Clone-censor-weight (CCW) data construction for the strategies
    "initiate treatment within the grace period" (arm 1) versus "do not initiate
    treatment within the grace period" (arm 0).

    Every patient is cloned into both arms at time zero, so both arms share the same
    baseline covariate distribution. A clone is artificially censored when its observed
    history deviates from the assigned strategy:
      * arm 1 clones are censored at the end of the grace period if treatment has not
        started by then (and the patient is still under follow-up);
      * arm 0 clones are censored at treatment initiation if it occurs within the grace period.
    Events or administrative censoring during the grace period before any deviation are
    kept in both arms. Informative artificial censoring is corrected with stabilized
    inverse probability of censoring weights (IPCW) estimated from baseline covariates
    (logistic models for treatment initiation; optional truncation at
    ``truncation_quantile``). Follow-up within the grace period is split into intervals
    so that weights can change over time.

    Returns a long-format DataFrame with one row per clone-interval:
    [id_col, 'arm', 'start', 'stop', 'event', 'weight'] + covariates.
    """
    G = float(grace_period)
    base = df[[id_col, duration_col, event_col, treatment_time_col] + covariates].copy()
    T = base[duration_col].clip(lower=1e-3).values
    E = base[event_col].values.astype(int)
    A = base[treatment_time_col].values
    initiated_in_grace = ~np.isnan(A) & (A <= G) & (A < T)
    X = base[covariates].reset_index(drop=True)
    X = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
    grid = np.linspace(G / n_grace_intervals, G, n_grace_intervals)

    # ---- arm 1: censored at G if not initiated by G while still under follow-up.
    # Weight after G = P(initiated by G) / P(initiated by G | X) among clones still at risk at G.
    censored1 = ~initiated_in_grace & (T > G)
    end1 = np.where(censored1, G, T)
    event1 = np.where(censored1, 0, E)
    at_risk_G = T > G
    p_uncens1 = _fit_probability(X, initiated_in_grace.astype(int), at_risk_G)
    marginal1 = initiated_in_grace[at_risk_G].mean() if at_risk_G.any() else 1.0
    w1_after = marginal1 / np.clip(p_uncens1, 1e-3, 1)

    # ---- arm 0: censored at initiation within G.
    # Discrete-time initiation hazards h_k(X) are estimated per grace-period interval with logistic
    # regression among clones at risk; the weight in interval k is prod_j<=k (1 - h_j) / (1 - h_j(X)).
    end0 = np.where(initiated_in_grace, A, T)
    event0 = np.where(initiated_in_grace, 0, E)
    surv0 = np.ones((len(grid), len(T)))
    marginal0 = np.ones(len(grid))
    s_cond, s_marg, t_prev = np.ones(len(T)), 1.0, 0.0
    for k, t_k in enumerate(grid):
        not_started = np.isnan(A) | (A > t_prev) | (k == 0)
        at_risk = (T > t_prev) & not_started
        starts_here = initiated_in_grace & (A <= t_k) & ((A > t_prev) | (k == 0))
        h = _fit_probability(X, starts_here.astype(int), at_risk)
        s_cond = s_cond * (1 - np.clip(h, 0, 0.999))
        s_marg = s_marg * (1 - (starts_here[at_risk].mean() if at_risk.any() else 0.0))
        surv0[k], marginal0[k], t_prev = s_cond, s_marg, t_k

    ids = base[id_col].values
    pieces = []

    def add(mask, arm, start, stop, event, weight):
        pieces.append(pd.DataFrame({id_col: ids[mask], "arm": arm, "start": np.broadcast_to(start, len(T))[mask],
                                    "stop": stop[mask], "event": event[mask],
                                    "weight": np.broadcast_to(weight, len(T))[mask]}))

    # arm 1: weight 1 during the grace period, IPCW after G for clones that remained uncensored
    within = end1 <= G
    add(np.ones(len(T), bool), 1, 0.0, np.minimum(end1, G), np.where(within, event1, 0), 1.0)
    add(~within, 1, G, end1, event1, w1_after)
    # arm 0: piecewise weights over the grace-period grid, constant after G
    t_prev = 0.0
    for k, t_k in enumerate(grid):
        in_interval = end0 > t_prev
        add(in_interval, 0, t_prev, np.minimum(end0, t_k), np.where(end0 <= t_k, event0, 0),
            marginal0[k] / np.maximum(surv0[k], 1e-3))
        t_prev = t_k
    add(end0 > G, 0, G, end0, event0, marginal0[-1] / np.maximum(surv0[-1], 1e-3))

    long = pd.concat(pieces, ignore_index=True)
    long = long[long["stop"] > long["start"]].reset_index(drop=True)
    if truncation_quantile is not None:
        long["weight"] = long["weight"].clip(upper=long["weight"].quantile(truncation_quantile))
    long = long.merge(base[[id_col] + covariates], on=id_col, how="left")

    long.attrs["diagnostics"] = {
        "n_patients": len(T),
        "grace_period_days": G,
        "initiated_within_grace": int(initiated_in_grace.sum()),
        "arm1_artificially_censored": int(censored1.sum()),
        "arm0_artificially_censored": int(initiated_in_grace.sum()),
        "weight_mean": float(long["weight"].mean()),
        "weight_max": float(long["weight"].max()),
    }
    return long


def run_ccw_analysis(clones, id_col, outcome_definition, follow_up_end):
    """Weighted Cox model (cluster-robust variance) and weighted Kaplan-Meier risks on CCW data."""
    cph = CoxPHFitter()
    cph.fit(clones[[id_col, "arm", "start", "stop", "event", "weight"]], duration_col="stop", event_col="event",
            entry_col="start", weights_col="weight", cluster_col=id_col, robust=True)
    row = cph.summary.loc["arm"]
    risks = {}
    for arm in (1, 0):
        sub = clones[clones["arm"] == arm]
        km = KaplanMeierFitter().fit(sub["stop"], sub["event"], entry=sub["start"], weights=sub["weight"])
        risks[arm] = 1 - float(km.survival_function_at_times(follow_up_end).values[0])
    d = clones.attrs.get("diagnostics", {})
    info = (
        "Clone-Censor-Weight Analysis (immortal-time bias adjustment):"
        f"\nStrategies: initiate treatment within {d.get('grace_period_days')} days of admission vs. do not initiate within this grace period."
        f"\nPatients cloned: {d.get('n_patients')}; initiated within grace period: {d.get('initiated_within_grace')}."
        f"\nArtificially censored clones: treatment arm {d.get('arm1_artificially_censored')}, "
        f"no-treatment arm {d.get('arm0_artificially_censored')}."
        f"\nStabilized IPCW: mean {d.get('weight_mean', float('nan')):.3f}, maximum {d.get('weight_max', float('nan')):.3f}."
        f"\nHazard Ratio for {outcome_definition[0]}: {row['exp(coef)']:.4f}"
        f"\n95% Confidence Interval: ({row['exp(coef) lower 95%']:.4f}, {row['exp(coef) upper 95%']:.4f})"
        f"\np-value: {row['p']:.4f}"
        f"\nWeighted cumulative incidence at day {follow_up_end:g}: treatment strategy {risks[1]:.4f}, "
        f"no-treatment strategy {risks[0]:.4f} (risk difference {risks[1] - risks[0]:.4f})\n"
    )
    primary = {"effect_measure": "HR", "effect_size": float(row["exp(coef)"]),
               "ci_lower": float(row["exp(coef) lower 95%"]), "ci_upper": float(row["exp(coef) upper 95%"]),
               "p_value": float(row["p"]), "risk_treated": risks[1], "risk_control": risks[0]}
    return info, primary


# ============================================================================ subgroup analysis
def run_subgroup_cox(data, exposure_col, duration_col, event_col, subgroup_col, threshold, weights_col=None,
                     entry_col=None, cluster_col=None):
    """Hazard ratios of the exposure below / at-or-above ``threshold`` of ``subgroup_col`` (Cox model with interaction)."""
    d = pd.DataFrame({
        "T": data[duration_col], "E": data[event_col], "x": data[exposure_col].astype(float),
        "g": (data[subgroup_col] >= threshold).astype(int),
    })
    d["x_g"] = d["x"] * d["g"]
    kwargs = {"robust": True}
    for name, col in (("weights_col", weights_col), ("entry_col", entry_col), ("cluster_col", cluster_col)):
        if col:
            d[col] = data[col].values
            kwargs[name] = col
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(d, duration_col="T", event_col="E", **kwargs)
    b, cov = cph.params_, cph.variance_matrix_
    se_a = np.sqrt(cov.loc["x", "x"])
    se_b = np.sqrt(cov.loc["x", "x"] + cov.loc["x_g", "x_g"] + 2 * cov.loc["x", "x_g"])
    est_a, est_b = b["x"], b["x"] + b["x_g"]
    p = lambda est, se: 2 * (1 - scipy.stats.norm.cdf(abs(est / se)))
    # Number of patients per subgroup (clone-interval rows are counted once per patient).
    ids = data[cluster_col].values if cluster_col else np.arange(len(d))
    n_a, n_b = len(set(ids[d["g"].values == 0])), len(set(ids[d["g"].values == 1]))
    return (
        f"Subgroup analysis by {subgroup_col} (threshold {threshold:g}):"
        f"\n  {subgroup_col} < {threshold:g}: HR {np.exp(est_a):.4f} (95% CI {np.exp(est_a - 1.96 * se_a):.4f}-"
        f"{np.exp(est_a + 1.96 * se_a):.4f}), p = {p(est_a, se_a):.4f}, n = {n_a}"
        f"\n  {subgroup_col} >= {threshold:g}: HR {np.exp(est_b):.4f} (95% CI {np.exp(est_b - 1.96 * se_b):.4f}-"
        f"{np.exp(est_b + 1.96 * se_b):.4f}), p = {p(est_b, se_b):.4f}, n = {n_b}"
        f"\n  Interaction p-value: {cph.summary.loc['x_g', 'p']:.4f}\n"
    )


# ============================================================================ sample size and E-value
def compute_cox_sample_size(df, treatment_col, outcome_col, HR, alpha=0.05, power=0.8):
    """Total sample size for a two-group Cox model (Schoenfeld, Biometrics 1983;39:499-503)."""
    q1 = df[treatment_col].mean()
    q0 = 1 - q1
    event_rate = df[outcome_col].mean()
    z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    z_beta = scipy.stats.norm.ppf(power)
    required_events = (z_alpha + z_beta) ** 2 / (np.log(HR) ** 2 * q0 * q1)
    return math.ceil(required_events / event_rate), {"required_events": math.ceil(required_events),
                                                    "event_rate": event_rate, "treated_fraction": q1}


def calculate_evalues(df, outcome_col, hr):
    """E-value for a hazard ratio (VanderWeele & Ding 2017); common outcomes use the HR-to-RR approximation."""
    incidence = df[outcome_col].mean()
    is_rare = incidence <= 0.15
    rr = hr if is_rare else (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr))
    if rr < 1:
        rr = 1 / rr
    e_value = 1.0 if rr == 1 else rr + np.sqrt(rr * (rr - 1))
    return {"outcome_incidence": incidence, "is_rare_outcome": is_rare, "e_value": float(e_value)}
