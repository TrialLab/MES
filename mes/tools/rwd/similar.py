"""
Patients like this one (the "green button" question): the stays of a population most similar to a described patient,
their outcomes beside the population's, the treatments they received and their outcomes by treatment, the risk a
model of the population predicts for the patient, and where the patient sits among all stays (a map of aggregates).

Similarity is a weighted Gower-type distance over the patient's characteristics: a number by its distance in robust
standard deviations of the population (IQR / 1.349), capped at 3; yes/no and categories by agreement; a value the
stay does not have counts half. Characteristics marked "must match" filter instead (a category or yes/no exactly, a
number within its tolerance). Aggregates only: at least 30 similar patients, groups under 10 not shown.

    profile [{concept, value, weight, must, tolerance}] → features(...) → distance(...) → the k nearest
"""

import math
import threading

import numpy as np
import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd import stats as S

MIN_K, MAX_K, DEFAULT_K = 30, 5000, 200
MISSING = 0.5               # the distance of a characteristic the stay has no value for
CAP = 3.0                   # robust standard deviations beyond which a number counts as fully different
SMALL = S.SMALL_CELL


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _yes(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v) if v in (0, 1) else None
    text = str(v or "").strip().lower()
    return True if text in ("1", "yes", "true", "y", "present") else False if text in ("0", "no", "false", "n", "absent") \
        else None


def as_float(x):
    """A column as floats: yes/no flags (booleans, also with missing values) as 1 / 0, missing as NaN."""
    x = pd.Series(x)
    if pd.api.types.is_bool_dtype(x.dtype):
        return x.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    if x.dtype == object:
        x = x.map(lambda t: float(t) if isinstance(t, (bool, np.bool_)) else t)
    return pd.to_numeric(x, errors="coerce").to_numpy(float)


def features(profile, base, columns=None):
    """The usable characteristics of a profile: [{name, kind, value, weight, must, tolerance, label, unit}] — cohort
    variables of the population with a value of their kind (the last one of a name wins)."""
    variables = L.variables_for(base)
    out = {}
    for f in profile or []:
        if not isinstance(f, dict):
            continue
        concept = str(f.get("concept") or "")
        name = concept[4:] if concept.startswith("var:") else concept
        if name not in variables or (columns is not None and name not in columns):
            continue
        kind = variables[name][1]
        value = _num(f.get("value")) if kind == "numeric" else _yes(f.get("value")) if kind == "binary" else \
            (str(f.get("value")).strip() or None) if f.get("value") is not None else None
        if value is None:
            continue
        weight = _num(f.get("weight"))
        out[name] = {"name": name, "kind": kind, "value": value, "label": variables[name][0], "unit": variables[name][2],
                     "weight": min(10.0, max(0.1, weight if weight is not None else 1.0)), "must": bool(f.get("must")),
                     "tolerance": _num(f.get("tolerance"))}
    return list(out.values())


def scales(df, feats):
    """The robust standard deviation of each numeric characteristic in the population (1 when it has none)."""
    out = {}
    for f in feats:
        if f["kind"] != "numeric":
            continue
        v = pd.to_numeric(df[f["name"]], errors="coerce").dropna()
        s = (v.quantile(.75) - v.quantile(.25)) / 1.349 if len(v) else 0.0
        if not s or not np.isfinite(s):
            s = float(v.std()) if len(v) > 1 else 0.0
        out[f["name"]] = float(s) if s and np.isfinite(s) and s > 0 else 1.0
    return out


def parts(df, feats, sc):
    """{name: distance of every stay on the characteristic} (0 same … 1 far; NaN where the stay has no value)."""
    out = {}
    for f in feats:
        x = df[f["name"]]
        if f["kind"] == "numeric":
            v = pd.to_numeric(x, errors="coerce").to_numpy(float)
            out[f["name"]] = np.minimum(np.abs(v - f["value"]) / sc[f["name"]], CAP) / CAP
        elif f["kind"] == "binary":
            v = as_float(x)
            out[f["name"]] = np.where(np.isnan(v), np.nan, (np.round(v) != (1.0 if f["value"] else 0.0)).astype(float))
        else:
            missing = x.isna().to_numpy()
            out[f["name"]] = np.where(missing, np.nan, (x.astype(str).to_numpy() != f["value"]).astype(float))
    return out


def distance(df, feats, sc=None):
    """The weighted distance (0 … 1) of every stay to the patient over the soft characteristics."""
    soft = [f for f in feats if not f["must"]] or feats
    sc = sc if sc is not None else scales(df, feats)
    p = parts(df, soft, sc)
    w = np.array([f["weight"] for f in soft], float)
    P = np.column_stack([p[f["name"]] for f in soft]) if soft else np.zeros((len(df), 0))
    P = np.where(np.isnan(P), MISSING, P)
    return np.sqrt((P ** 2 * w).sum(1) / w.sum()) if len(soft) else np.zeros(len(df))


def must_mask(df, feats, sc=None):
    """The stays that match every "must match" characteristic (a number within its tolerance — by default half a
    robust standard deviation — a yes/no or category exactly; no value: no match)."""
    mask = np.ones(len(df), bool)
    sc = sc if sc is not None else scales(df, feats)
    for f in feats:
        if not f["must"]:
            continue
        if f["kind"] == "numeric":
            v = pd.to_numeric(df[f["name"]], errors="coerce").to_numpy(float)
            tol = f["tolerance"] if f["tolerance"] is not None else 0.5 * sc[f["name"]]
            mask &= np.abs(v - f["value"]) <= tol
        else:
            d = parts(df, [f], sc)[f["name"]]
            mask &= d == 0
    return mask


def nearest(dist, k):
    """Positions of the ``k`` smallest distances (ties in the order of the stays)."""
    return np.argsort(np.asarray(dist, float), kind="stable")[:max(0, int(k))]


def _rate(y):
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().astype(int)
    n, k = len(y), int(y.sum())
    if n < SMALL:
        return {"n": n, "events": None, "value": None, "lo": None, "hi": None, "small_cell": True}
    p, lo, hi = S.wilson(k, n)
    small = 0 < k < SMALL
    return {"n": n, "events": None if small else k, "value": round(100 * p, 2), "lo": round(100 * lo, 2),
            "hi": round(100 * hi, 2), "small_cell": small}


def newcombe(k1, n1, k0, n0):
    """Difference of two proportions (pp) with Newcombe's hybrid score interval."""
    p1, l1, u1 = S.wilson(k1, n1)
    p0, l0, u0 = S.wilson(k0, n0)
    d = p1 - p0
    lo = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return round(100 * d, 2), round(100 * lo, 2), round(100 * hi, 2)


def describe(series, kind):
    """A characteristic of a group: median (IQR) of a number, % yes, or the most common category with its %."""
    s = series.dropna()
    n = len(s)
    if n < SMALL:
        return {"n": n, "text": "<10" if n else "—"}
    if kind == "numeric":
        v = pd.to_numeric(s, errors="coerce").dropna()
        q = v.quantile([.25, .5, .75])
        a, b, c = numbers([q[.5], q[.25], q[.75]])
        return {"n": n, "median": float(q[.5]), "q1": float(q[.25]), "q3": float(q[.75]), "text": f"{a} ({b}–{c})"}
    if kind == "binary":
        k = int(np.nansum(np.round(as_float(s))))
        pct = 100 * k / n
        return {"n": n, "pct": round(pct, 1), "text": "<10" if 0 < k < SMALL else f"{pct:.0f}%"}
    counts = s.astype(str).value_counts()
    top, k = counts.index[0], int(counts.iloc[0])
    return {"n": n, "top": top, "pct": round(100 * k / n, 1),
            "text": f"{top} {100 * k / n:.0f}%" if k >= SMALL else "<10"}


def numbers(values):
    """Numbers written alike: whole numbers without decimals, else one decimal (two under 1)."""
    values = [float(v) for v in values]
    if all(abs(v - round(v)) < 1e-9 for v in values):
        return [f"{v:,.0f}" for v in values]
    d = 1 if max(abs(v) for v in values) >= 1 else 2
    return [f"{v:,.{d}f}" for v in values]


def share(series, kind, value):
    """% of a group with the patient's value of a yes/no or category characteristic (None under 10)."""
    s = series.dropna()
    if len(s) < SMALL:
        return None
    if kind == "binary":
        hit = np.round(as_float(s)) == (1.0 if value else 0.0)
    else:
        hit = s.astype(str) == str(value)
    k = int(hit.sum())
    return None if 0 < k < SMALL else round(100 * k / len(s), 1)


# ============================================================================ the analysis
def risk_model(df, feats, y, validate=True):
    """The risk the population's logistic model (the patient's characteristics as predictors) predicts for the
    patient, with its 95% CI, and the model's cross-validated c-statistic. None when it cannot be fitted."""
    import statsmodels.api as sm
    names = [f["name"] for f in feats]
    patient = pd.DataFrame([{f["name"]: (float(f["value"]) if f["kind"] == "binary" else f["value"]) for f in feats}])
    data = pd.concat([df[names].reset_index(drop=True), patient], ignore_index=True)
    kinds = {f["name"]: f["kind"] for f in feats}
    for f in feats:
        if f["kind"] == "binary":
            data[f["name"]] = as_float(data[f["name"]])
    X, _ = S.design_matrix(data, names, kinds)
    y = np.asarray(y, int)
    if X.shape[1] == 0 or y.sum() < 2 * SMALL or y.sum() > len(y) - 2 * SMALL:
        return None
    try:
        fit = S._glm(y, X.iloc[:-1])
        cols = [c for c in fit.model.exog_names if c != "const"]
        row = sm.add_constant(X.iloc[[-1]][cols], has_constant="add")
        frame = fit.get_prediction(row).summary_frame(alpha=0.05)
        out = {"value": round(100 * float(frame["mean"].iloc[0]), 2), "lo": round(100 * float(frame["mean_ci_lower"].iloc[0]), 2),
               "hi": round(100 * float(frame["mean_ci_upper"].iloc[0]), 2)}
    except Exception:
        return None
    out["auc"] = None
    if validate:
        try:
            out["auc"] = round(S.prognostic_model(X.iloc[:-1], y)["auc"], 3)
        except Exception:
            pass
    return out


# ============================================================================ the patient explorer's preview
_cohorts = {}
_lock = threading.Lock()
MAX_COHORTS = 4


def preview(spec_dict, index=None, store=None, timeout=60):
    """(result, errors) of a "patients like this" design as the patient explorer shows it while the patient is
    changed: the cohort is built once for the characteristics it needs (values, weights and k only rank it again)."""
    import json
    from mes.tools.rwd import analyses
    from mes.tools.rwd.compiler import Compiler
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.db import get_store
    from mes.tools.rwd.spec import StudySpec, validate as check
    store = store or get_store()
    if not store.available:
        return None, ["the MIMIC-IV store is not built on this computer"]
    index = index or get_index()
    try:
        spec = StudySpec.from_dict(spec_dict)
        errors, _warnings = check(spec, index)
    except Exception as e:
        return None, [f"not a study design: {type(e).__name__}: {e}"]
    if errors:
        return None, list(errors)
    shape = spec.to_dict()
    shape["profile"] = sorted(str(f.get("concept")) for f in spec.profile)
    shape.pop("k", None)
    key = (str(store.root), store.catalog().get("built_at"), json.dumps(shape, sort_keys=True, default=str))
    with _lock:
        cohort = _cohorts.get(key)
    if cohort is None:
        try:
            cohort = Compiler(index, store).build(spec, timeout=timeout)
        except Exception as e:
            return None, [f"the cohort did not run: {type(e).__name__}: {e}"]
        with _lock:
            _cohorts[key] = cohort
            while len(_cohorts) > MAX_COHORTS:
                _cohorts.pop(next(iter(_cohorts)))
    try:
        res = analyses.run_similar(spec, cohort, "osler", live=True)
    except Exception as e:
        return None, [f"the similar patients could not be found: {type(e).__name__}: {e}"]
    return res, []


def _standardized(df, feats, sc):
    """The characteristics as numbers on one scale (robust SDs; yes/no and categories as 0 / 1 for agreeing with the
    patient), missing at the population's median, and the patient's own row."""
    cols, patient = [], []
    for f in feats:
        x = df[f["name"]]
        if f["kind"] == "numeric":
            v = pd.to_numeric(x, errors="coerce").to_numpy(float)
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            cols.append((np.where(np.isnan(v), med, v) - med) / sc[f["name"]])
            patient.append((f["value"] - med) / sc[f["name"]])
        else:
            d = parts(df, [f], sc)[f["name"]]
            agree = 1 - np.where(np.isnan(d), 1.0, d)
            mu = float(agree.mean())
            cols.append(agree - mu)
            patient.append(1 - mu)
    return np.column_stack(cols), np.array(patient)


def map_of(df, feats, sc, near_mask, max_points=20000, seed=0):
    """Where the patient sits among the stays: the first two principal components of the characteristics, the stays
    as hexagons of at least 10, the similar stays as an ellipse (their mean ± 2 SD) and the patient as a point."""
    if len(feats) < 2 or len(df) < 3 * SMALL:
        return None
    Z, z0 = _standardized(df, feats, sc)
    rng = np.random.default_rng(seed)
    sample = Z if len(Z) <= max_points else Z[rng.choice(len(Z), max_points, replace=False)]
    mu = sample.mean(0)
    try:
        _u, _s, vt = np.linalg.svd(sample - mu, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    comps = vt[:2]
    if comps.shape[0] < 2:
        return None
    for i in range(2):                     # each axis grows with its strongest characteristic (the sign is arbitrary)
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] = -comps[i]
    xy, p = (Z - mu) @ comps.T, (z0 - mu) @ comps.T
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    lo, hi = np.minimum(lo, p) - 0.3, np.maximum(hi, p) + 0.3
    radius = max(hi[0] - lo[0], hi[1] - lo[1]) / 26
    # pointy hexagons: rows ``radius * 1.5`` apart, every other row shifted by half a width
    w, hgt = math.sqrt(3) * radius, 1.5 * radius
    r_idx = np.round((xy[:, 1] - lo[1]) / hgt).astype(int)
    c_idx = np.round((xy[:, 0] - lo[0] - (r_idx % 2) * w / 2) / w).astype(int)
    keep = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
    cells = pd.DataFrame({"r": r_idx[keep], "c": c_idx[keep], "near": np.asarray(near_mask, bool)[keep]})
    counts = cells.groupby(["r", "c"]).agg(n=("near", "size"), k=("near", "sum")).reset_index()
    counts = counts[counts.n >= SMALL]
    hexes = [{"x": round(float(lo[0] + c * w + (r % 2) * w / 2), 4), "y": round(float(lo[1] + r * hgt), 4), "n": int(n),
              "k": int(k) if k >= SMALL else None} for r, c, n, k in counts.itertuples(index=False)]
    near = xy[np.asarray(near_mask, bool)]
    ellipse = None
    if len(near) >= SMALL:
        cov = np.cov(near.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-9, None)
        angle = math.degrees(math.atan2(vecs[1, -1], vecs[0, -1]))
        ellipse = {"cx": round(float(near[:, 0].mean()), 4), "cy": round(float(near[:, 1].mean()), 4),
                   "rx": round(2 * math.sqrt(vals[-1]), 4), "ry": round(2 * math.sqrt(vals[0]), 4), "angle": round(angle, 2)}

    from mes.tools.rwd.patient import SHORT

    def axis(v):                 # "higher creatinine, higher lactate": what grows along the axis
        top = [i for i in np.argsort(-np.abs(v))[:2] if abs(v[i]) >= 0.25 * np.abs(v).max()]
        name = lambda f: SHORT.get(f["name"]) or f["label"].split(",")[0].lower()
        return ", ".join((("higher " if v[i] > 0 else "lower ") + name(feats[i])) if feats[i]["kind"] == "numeric" else
                         f"{name(feats[i])} {'as' if v[i] > 0 else 'unlike'} the patient" for i in top)
    return {"hexes": hexes, "radius": round(float(radius), 4), "ellipse": ellipse,
            "patient": {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4)},
            "extent": [round(float(lo[0]), 4), round(float(hi[0]), 4), round(float(lo[1]), 4), round(float(hi[1]), 4)],
            "axes": [axis(comps[0]), axis(comps[1])], "n": int(keep.sum())}


def profile_rows(df, near, feats):
    """The patient beside the similar stays and the whole population, characteristic by characteristic."""
    rows = []
    for f in feats:
        a, b = describe(near[f["name"]], f["kind"]), describe(df[f["name"]], f["kind"])
        row = {"variable": f["name"], "label": f["label"], "kind": f["kind"], "unit": f["unit"], "must": f["must"],
               "weight": f["weight"], "patient": f["value"] if f["kind"] != "binary" else ("yes" if f["value"] else "no"),
               "similar": a.get("text"), "cohort": b.get("text"),
               "missing_similar": round(100 * float(near[f["name"]].isna().mean()), 1)}
        if f["kind"] == "numeric":
            v = pd.to_numeric(df[f["name"]], errors="coerce").dropna()
            if len(v) >= SMALL:
                row["range"] = [float(v.quantile(.02)), float(v.quantile(.98))]
            row.update({k: a.get(k) for k in ("median", "q1", "q3")})
            row["cohort_iqr"] = [b.get("q1"), b.get("median"), b.get("q3")] if b.get("median") is not None else None
        else:
            row["share_similar"], row["share_cohort"] = share(near[f["name"]], f["kind"], f["value"]), \
                share(df[f["name"]], f["kind"], f["value"])
        rows.append(row)
    return rows


# a flag of what was given in the first 24 h, and the drug names it stands for (left out when the design's own treatment
# or comparator names the drug: the design's definition is the one its groups use)
FLAG_DRUGS = {"norepinephrine_24h": ("norepinephrine", "noradrenaline"), "vasopressin_24h": ("vasopressin",)}


def same_drug(flag, concepts):
    """The flag stands for a drug one of ``concepts`` (drugclass: / drug: ids) names."""
    names = FLAG_DRUGS.get(flag, ())
    return any(n in str(c).lower() for c in concepts for n in names)


def treatment_rows(df, near, columns):
    """What the similar stays received (in the first 24 h) beside the population: [{variable, label, similar,
    cohort}] in %, with the similar stays' 95% CI; under 10 patients not shown."""
    rows = []
    for name, label in columns:
        if name not in df:
            continue
        k_near = int(np.nansum(np.round(as_float(near[name]))))
        k_all = int(np.nansum(np.round(as_float(df[name]))))
        p, lo, hi = S.wilson(k_near, len(near)) if len(near) else (None, None, None)
        rows.append({"variable": name, "label": label, "n": len(near), "events": k_near if k_near >= SMALL else None,
                     "similar": None if 0 < k_near < SMALL or p is None else round(100 * p, 1),
                     "lo": None if 0 < k_near < SMALL or lo is None else round(100 * lo, 1),
                     "hi": None if 0 < k_near < SMALL or hi is None else round(100 * hi, 1),
                     "cohort": None if 0 < k_all < SMALL or not len(df) else round(100 * k_all / len(df), 1)})
    return rows
