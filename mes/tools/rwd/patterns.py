"""
Descriptive RWD questions beyond comparisons: what patients are like (characterize: Table 1 of a cohort, or of two
groups — who died and who survived, with and without a treatment or condition), how a measurement evolves after time
zero (trajectory: the median and IQR of a lab test or vital sign in intervals, by group) and in which order treatments
are started (pathways: first, second and third line, the time to the first dose). Aggregates only: intervals,
groups and sequences under 10 patients are not shown.

    run_characterize / run_trajectory / run_pathways (spec, cohort, mode) -> result      (analyses.RUNNERS)
"""

import numpy as np
import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd import stats as S
from mes.tools.rwd.spec import criterion_text

SMALL = S.SMALL_CELL

# the time series behind a first-24-h variable: (concept, conversion) — vital signs from several items are pooled
SERIES = {"lactate_max": [("lab:50813", None)], "creatinine_max": [("lab:50912", None)], "bun_max": [("lab:51006", None)],
          "wbc_max": [("lab:51301", None)], "hemoglobin_min": [("lab:51222", None)],
          "platelets_min": [("lab:51265", None)], "bilirubin_total_max": [("lab:50885", None)],
          "sodium_min": [("lab:50983", None)], "sodium_max": [("lab:50983", None)], "potassium_max": [("lab:50971", None)],
          "glucose_max": [("lab:50931", None)], "albumin_min": [("lab:50862", None)], "inr_max": [("lab:51237", None)],
          "bicarbonate_min": [("lab:50882", None)], "ph_min": [("lab:50820", None)],
          "heart_rate_mean": [("item:220045", None)], "resp_rate_mean": [("item:220210", None)],
          "spo2_min": [("item:220277", None)],
          "mbp_mean": [("item:220052", None), ("item:220181", None), ("item:225312", None)],
          "mbp_min": [("item:220052", None), ("item:220181", None), ("item:225312", None)],
          "temperature_max": [("item:223762", None), ("item:223761", "f2c")]}
SERIES_LABELS = {"lactate_max": "Lactate", "creatinine_max": "Creatinine", "bun_max": "BUN", "wbc_max": "WBC",
                 "hemoglobin_min": "Hemoglobin", "platelets_min": "Platelets", "bilirubin_total_max": "Total bilirubin",
                 "sodium_min": "Sodium", "sodium_max": "Sodium", "potassium_max": "Potassium", "glucose_max": "Glucose",
                 "albumin_min": "Albumin", "inr_max": "INR", "bicarbonate_min": "Bicarbonate", "ph_min": "pH",
                 "heart_rate_mean": "Heart rate", "resp_rate_mean": "Respiratory rate", "spo2_min": "SpO2",
                 "mbp_mean": "Mean arterial pressure", "mbp_min": "Mean arterial pressure",
                 "temperature_max": "Temperature"}


# outcomes as the name of a group: "with readmission", "without AKI"
SHORT_OUTCOMES = {"aki_7d": "AKI", "aki_stage2_7d": "AKI stage ≥ 2", "rrt_any": "RRT", "invasive_vent_any": "ventilation",
                  "icu_readmit_30d": "ICU readmission", "hosp_readmit_30d": "readmission"}


def series_of(variable):
    """[(concept, conversion)] of the measurement a trajectory follows (a lab:/item: concept, or the time series of a
    first-24-h variable); [] when there is none."""
    v = str(variable or "")
    if v.startswith(("lab:", "item:")):
        return [(v, None)]
    return list(SERIES.get(v.removeprefix("var:"), []))


# ============================================================================ groups
def groups_of(spec, df):
    """(t, names, how) of the two groups a description compares — by the outcome, the exposure or a yes/no or
    category characteristic (its two commonest levels) — or (None, [name], "") for one cohort. ``t``: 1 / 0 per
    stay, NaN for stays in neither group."""
    from mes.tools.rwd.similar import as_float
    how = spec.group_by
    if how == "outcome" and spec.outcome in L.OUTCOMES and L.OUTCOMES[spec.outcome][1] == "binary":
        label = L.OUTCOMES[spec.outcome][0]
        y = pd.to_numeric(df[L.OUTCOMES[spec.outcome][3]], errors="coerce").fillna(0).to_numpy(float)
        short = SHORT_OUTCOMES.get(spec.outcome, label[:1].lower() + label[1:])
        names = ["died", "survived"] if spec.outcome.startswith("death") else [f"with {short}", f"without {short}"]
        return y, names, "outcome"
    if how == "exposure" and spec.exposure is not None and "exposed" in df:
        name = criterion_text(spec.exposure)
        return (as_float(df["exposed"]) == 1).astype(float), [name, f"without {name}"], "exposure"
    if how in L.VARIABLES and how in df:
        label, kind = L.VARIABLES[how][0], L.VARIABLES[how][1]
        if kind == "binary":
            x = as_float(df[how])
            return np.where(np.isnan(x), np.nan, (x == 1).astype(float)), [label, f"no {label[:1].lower() + label[1:]}"], how
        if kind == "category":
            levels = df[how].dropna().astype(str).value_counts().index[:2]
            if len(levels) == 2:
                s = df[how].astype(str)
                return np.where(s == levels[0], 1.0, np.where(s == levels[1], 0.0, np.nan)), \
                    [f"{label}: {levels[0]}", f"{label}: {levels[1]}"], how
    return None, ["all patients"], ""


def _one_group_table(df, covariates):
    """Table 1 of one cohort (n (%) or mean (SD)), in the shape of ``baseline_table``."""
    from mes.tools.rwd.analyses import baseline_table
    t1 = baseline_table(df, covariates, np.ones(len(df)), None, ("all patients", ""))
    for r in t1["rows"]:
        for k in ("control_raw", "smd_raw"):
            r.pop(k, None)
    t1.update(groups=["all patients"], n=[len(df)], single=True)
    return t1


def run_characterize(spec, cohort, mode):
    """What patients are like: Table 1 of the cohort, or of two groups with their standardized differences."""
    from mes.tools.rwd.analyses import base_result, baseline_table
    res = base_result(spec, cohort)
    df = cohort.df.reset_index(drop=True)
    covs = [c for c in spec.covariates if c in df]
    t, names, how = groups_of(spec, df)
    if spec.outcome in L.OUTCOMES and L.OUTCOMES[spec.outcome][1] == "binary" and how != "outcome":
        y = pd.to_numeric(df[L.OUTCOMES[spec.outcome][3]], errors="coerce").fillna(0).astype(int)
        p, lo, hi = S.wilson(int(y.sum()), len(y)) if len(y) else (None, None, None)
        if p is not None and len(y) >= SMALL:
            res["estimates"].append({"label": f"{L.OUTCOMES[spec.outcome][0]}: all patients", "measure": "rate",
                                     "value": round(100 * p, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
                                     "n": len(y), "events": int(y.sum()) if y.sum() >= SMALL or not y.sum() else None,
                                     "primary": t is None})
    if t is None:
        res["table1"] = _one_group_table(df, covs)
        res["estimates"].insert(0, {"label": "Patients", "measure": "count", "value": len(df), "primary": not res["estimates"]})
        res["methods"] = "Characteristics of the cohort: n (%) or mean (SD); categories by their six commonest levels."
        return res
    keep = ~np.isnan(t)
    sub, tt = df[keep].reset_index(drop=True), t[keep].astype(int)
    n1, n0 = int(tt.sum()), int((1 - tt).sum())
    if min(n1, n0) < SMALL:
        res["warnings"].append(f"A group has fewer than {SMALL} patients: the characteristics are not compared.")
        res["table1"] = _one_group_table(df, covs)
        return res
    res["table1"] = baseline_table(sub, covs, tt, None, tuple(names))
    res["groups"] = {"treated": names[0], "control": names[1], "n_treated": n1, "n_control": n0, "by": how}
    if how == "outcome":
        p, lo, hi = S.wilson(n1, n1 + n0)
        res["estimates"].append({"label": f"{L.OUTCOMES[spec.outcome][0]}: all patients", "measure": "rate",
                                 "value": round(100 * p, 2), "lo": round(100 * lo, 2), "hi": round(100 * hi, 2),
                                 "n": n1 + n0, "events": n1, "primary": True})
    for name, n in zip(names, (n1, n0)):
        res["estimates"].append({"label": f"Patients: {name}", "measure": "count", "value": n})
    diffs = sorted([r for r in res["table1"]["rows"] if not r.get("heading") and r.get("smd_raw") is not None],
                   key=lambda r: -abs(r["smd_raw"]))
    res["tables"]["differences"] = [{"variable": r["variable"], "label": (f"{L.VARIABLES.get(r['variable'], ('',))[0]}: "
                                                                          f"{r['label']}" if r.get("level") else r["label"]),
                                     "smd": r["smd_raw"], "a": r.get("treated_raw"), "b": r.get("control_raw")}
                                    for r in diffs[:20]]
    res["methods"] = (f"Characteristics of {names[0]} vs {names[1]}: n (%) or mean (SD), standardized mean differences "
                      "(|SMD| ≥ 0.1: a meaningful difference); categories by their six commonest levels.")
    return res


# ============================================================================ trajectory
def measurements(store, cohort, series, horizon_h, bin_h, base):
    """The mean value of every stay in every interval: DataFrame [unit_id, bin, v] (aggregated in DuckDB; the stays
    of the cohort registered as a table of the query)."""
    units = cohort.df[["unit_id", "hadm_id", "t0"]].copy()
    last = max(0, int(np.ceil(float(horizon_h) / float(bin_h))) - 1)       # the index of the last interval
    parts = []
    cur = store.cursor()
    try:
        cur.register("mes_units", units)
        for concept, conversion in series:
            kind, _, item = concept.partition(":")
            table = "mimiciv_hosp.labevents" if kind == "lab" else "mimiciv_icu.chartevents"
            link = "m.stay_id = u.unit_id" if kind == "item" and base == "icu" else "m.hadm_id = u.hadm_id"
            value = "(m.valuenum - 32) * 5 / 9" if conversion == "f2c" else "m.valuenum"
            # the hours elapsed (date_diff counts minute boundaries: 47 h 59 m 45 s would be 48 h, a bin past the end)
            sql = (f"SELECT u.unit_id, least(floor(epoch(m.charttime - u.t0) / 3600.0 / {float(bin_h)!r}), {last}) AS bin, "
                   f"avg({value}) AS v FROM mes_units u JOIN {table} m ON {link} AND m.itemid = {int(item)} "
                   f"AND m.charttime >= u.t0 AND m.charttime < u.t0 + INTERVAL ({float(horizon_h)!r}) HOUR "
                   f"AND m.valuenum IS NOT NULL GROUP BY 1, 2")
            parts.append(cur.execute(sql).df())
    finally:
        try:
            cur.unregister("mes_units")
        except Exception:
            pass
        cur.close()
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["unit_id", "bin", "v"])
    if len(df) and len(series) > 1:              # one value per stay and interval when items are pooled
        df = df.groupby(["unit_id", "bin"], as_index=False)["v"].mean()
    if len(df) > 100:                             # implausible entries (typos, units) at either end
        lo, hi = df["v"].quantile([0.005, 0.995])
        df = df[(df["v"] >= lo) & (df["v"] <= hi)]
    return df


def run_trajectory(spec, cohort, mode):
    """How a lab test or vital sign evolves after time zero: per interval the median (IQR) of the stays' values, by
    group (the outcome, the exposure, a characteristic) or for all."""
    from mes.tools.rwd.analyses import base_result
    from mes.tools.rwd.db import get_store
    res = base_result(spec, cohort)
    df = cohort.df.reset_index(drop=True)
    series = series_of(spec.variable)
    horizon, step = float(spec.horizon_h or 72), float(spec.bin_h or 6)
    if not series:
        res["warnings"].append("The variable has no time series in MIMIC-IV (a lab test or a vital sign is needed).")
        return res
    m = measurements(getattr(cohort, "store", None) or get_store(), cohort, series, horizon, step, spec.base)
    t, names, how = groups_of(spec, df)
    group = pd.Series(np.where(np.isnan(t), -1, t).astype(int) if t is not None else np.zeros(len(df), int),
                      index=df["unit_id"].values)
    m = m.assign(g=m["unit_id"].map(group)).dropna(subset=["g"])
    m = m[m["g"] >= 0]
    labels = {1: names[0], 0: names[1]} if t is not None else {0: names[0]}
    variable = str(spec.variable).removeprefix("var:")
    unit = L.VARIABLES[variable][2] if variable in L.VARIABLES else ""
    name = SERIES_LABELS.get(variable) or variable
    if variable not in L.VARIABLES:
        try:
            from mes.tools.rwd.concepts import get_index
            name = get_index().label(spec.variable)
        except Exception:
            pass
    groups, rows = [], []
    for g in sorted(labels, reverse=True):
        part = m[m["g"] == g]
        points = []
        for b in sorted(part["bin"].dropna().unique()):
            v = part.loc[part["bin"] == b, "v"]
            if len(v) < SMALL:
                continue
            q = v.quantile([.25, .5, .75])
            points.append({"t0": round(float(b) * step, 2), "t1": round(min((float(b) + 1) * step, horizon), 2),
                           "n": int(len(v)), "median": round(float(q[.5]), 3), "q1": round(float(q[.25]), 3),
                           "q3": round(float(q[.75]), 3)})
            rows.append({"group": labels[g], **points[-1]})
        groups.append({"label": labels[g], "role": ("treated" if g == 1 else "control") if t is not None else "",
                       "points": points, "patients": int(part["unit_id"].nunique())})
    measured = int(m["unit_id"].nunique())
    res["n"] = int(len(df))
    if measured < SMALL or not any(g["points"] for g in groups):
        res["warnings"].append(f"Fewer than {SMALL} stays have {name.lower()} measured in the window.")
        return res
    res["tables"]["trajectory"] = rows
    res.setdefault("charts", {})["trajectory"] = {"variable": name, "unit": unit, "bin_h": step, "horizon_h": horizon,
                                                  "groups": groups, "t0_label": L.BASES[spec.base]["t0_label"]}
    for g in groups:
        if not g["points"]:
            continue
        for p, first in ((g["points"][0], True), (g["points"][-1], False)):
            res["estimates"].append({"label": f"{name}, {g['label']}, {p['t0']:g}–{p['t1']:g} h", "measure": "median",
                                     "value": p["median"], "lo": p["q1"], "hi": p["q3"], "n": p["n"], "unit": unit,
                                     "group": g["label"], "interval": f"{p['t0']:g}–{p['t1']:g} h",
                                     "primary": first and g is groups[0]})
    res["diagnostics"].update(measured=measured, coverage=round(measured / max(1, len(df)), 3))
    res["methods"] = (f"{name}: the mean of each stay's values in {step:g}-h intervals from {L.BASES[spec.base]['t0_label']} "
                      f"to {horizon:g} h, then the median and IQR across stays; intervals with fewer than {SMALL} stays "
                      f"are not shown. {measured:,} of {len(df):,} stays have a value.")
    return res


# ============================================================================ pathways
def sequences(starts, names, depth=3):
    """The order in which each stay received the treatments (by the first dose; the same hour: the order of the
    design): a tuple of names per stay, () when it received none."""
    out = []
    for row in starts:
        given = [(t, i) for i, t in enumerate(row) if np.isfinite(t)]
        out.append(tuple(names[i] for _t, i in sorted(given))[:depth])
    return out


def sankey(seqs, depth=3, other="other"):
    """Links of the treatment lines: [{"from": (level, name), "to": (level + 1, name), "n"}] between consecutive lines
    ("none" after the last treatment), links under 10 stays merged into "other" and dropped when still under 10."""
    links = {}
    for seq in seqs:
        path = list(seq) + ["none"] * (depth - len(seq)) if seq else ["none"] * depth
        for level in range(depth - 1):
            if path[level] == "none":
                break
            key = ((level + 1, path[level]), (level + 2, path[level + 1]))
            links[key] = links.get(key, 0) + 1
    merged = {}
    for (a, b), n in links.items():
        if n < SMALL:
            b = (b[0], other) if b[1] != "none" else b
        merged[(a, b)] = merged.get((a, b), 0) + n
    return [{"from": list(a), "to": list(b), "n": n} for (a, b), n in sorted(merged.items(), key=lambda x: -x[1])
            if n >= SMALL]


def run_pathways(spec, cohort, mode):
    """In which order the treatments are started after time zero (within the horizon): who received each, the first
    line, the commonest sequences and the time to the first dose — first, second and third line as a flow."""
    from mes.tools.rwd.analyses import base_result
    res = base_result(spec, cohort)
    df = cohort.df.reset_index(drop=True)
    names = [criterion_text(c) for c in spec.treatments]
    cols = [f"start_{i}" for i in range(len(names))]
    if len(names) < 2 or not all(c in df for c in cols):
        res["warnings"].append("A treatment pathway needs at least two treatments.")
        return res
    starts = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in cols])
    n = len(df)
    res["n"] = n
    if n < SMALL:
        res["warnings"].append(f"Fewer than {SMALL} patients.")
        return res
    seqs = sequences(starts, names)
    rows = []
    for i, name in enumerate(names):
        got = np.isfinite(starts[:, i])
        k = int(got.sum())
        p, lo, hi = S.wilson(k, n)
        first = sum(1 for s in seqs if s and s[0] == name)
        times = starts[got, i]
        row = {"treatment": name, "n": n, "events": k if k >= SMALL else None, "given": None if 0 < k < SMALL else
               round(100 * p, 1), "lo": None if 0 < k < SMALL else round(100 * lo, 1),
               "hi": None if 0 < k < SMALL else round(100 * hi, 1),
               "first": None if 0 < first < SMALL else round(100 * first / n, 1)}
        if k >= SMALL:
            q = np.quantile(times, [.25, .5, .75])
            row.update(median_h=round(float(q[1]), 1), q1_h=round(float(q[0]), 1), q3_h=round(float(q[2]), 1))
            res["estimates"].append({"label": f"{name}: received within {spec.horizon_h:g} h", "measure": "rate",
                                     "value": row["given"], "lo": row["lo"], "hi": row["hi"], "n": n, "events": k,
                                     "primary": i == 0})
            res["estimates"].append({"label": f"{name}: hours to the first dose", "measure": "median",
                                     "value": row["median_h"], "lo": row["q1_h"], "hi": row["q3_h"], "n": k, "unit": "h"})
        rows.append(row)
    counts = pd.Series([" → ".join(s) if s else "none of them" for s in seqs]).value_counts()
    common = [{"sequence": seq, "n": int(k), "share": round(100 * k / n, 1)} for seq, k in counts.items() if k >= SMALL][:12]
    shown = sum(r["n"] for r in common)
    if n - shown >= SMALL:
        common.append({"sequence": "other sequences", "n": n - shown, "share": round(100 * (n - shown) / n, 1)})
    res["tables"].update(treatments=rows, sequences=common)
    res.setdefault("charts", {})["sankey"] = {"links": sankey(seqs), "names": names, "n": n,
                                              "horizon_h": float(spec.horizon_h or 72)}
    none = sum(1 for s in seqs if not s)
    res["diagnostics"].update(none=none if none >= SMALL or not none else None, treated_any=n - none)
    res["methods"] = (f"Each patient's treatments in the order of their first dose within {spec.horizon_h:g} h of "
                      f"{L.BASES[spec.base]['t0_label']} (the same hour: in the order of the design); proportions with "
                      f"Wilson 95% CIs; sequences and links of fewer than {SMALL} patients are not shown.")
    return res
