"""
Generate a synthetic dataset that follows the MIMIC-IV table layout.

MIMIC-IV cannot be redistributed, so the demo runs on fully synthetic records
produced here from parametric distributions (no real patient data is used).
The generated tables mirror the MIMIC-IV schema used by the RWD-Analyst:

    hosp/patients.csv.gz, hosp/admissions.csv.gz, hosp/diagnoses_icd.csv.gz,
    hosp/prescriptions.csv.gz, icu/icustays.csv.gz,
    icu_first_day_covariates.csv.gz   (first-day ICU features, one row per stay)

Data-generating process for the demo question (metformin in ICU patients with
type 2 diabetes):
  * Metformin is (re)started during the index admission in a subset of patients
    with type 2 diabetes. Initiation is more likely with better renal function,
    lower lactate and lower SOFA (confounding by indication), and part of the
    initiations happen 0.5-2 days after admission (a source of immortal-time bias).
  * After initiation the mortality hazard is multiplied by TRUE_HR_METFORMIN.
  * The counterfactual 30- and 90-day mortality under "initiate within the grace period"
    and "no initiation" is simulated for the demo cohort and saved as ground_truth.json.

Usage:
    python generate_synthetic_mimic.py --out Data/synthetic_mimic --n-patients 30000 --seed 2026
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

TRUE_HR_METFORMIN = 0.80
GRACE_PERIOD_DAYS = 2.0

# condition -> (ICD-10 codes without dots, base prevalence)
CONDITIONS = {
    "t2dm": (["E119", "E1122", "E1165"], 0.30),
    "hypertension": (["I10"], 0.45),
    "ckd": (["N183", "N184", "N186"], 0.12),
    "heart_failure": (["I509"], 0.16),
    "sepsis": (["A419"], 0.22),
    "pneumonia": (["J189"], 0.14),
    "aki": (["N179"], 0.18),
    "copd": (["J449"], 0.10),
    "atrial_fibrillation": (["I4891"], 0.14),
    "myocardial_infarction": (["I214"], 0.06),
    "stroke": (["I639"], 0.05),
    "cirrhosis": (["K7460"], 0.04),
    "lung_cancer": (["C3490"], 0.05),
    "dementia": (["F0390"], 0.05),
}


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _weibull_inverse(h, scale, shape, lin):
    """Time t at which the cumulative hazard (t/scale)^shape * exp(lin) equals h."""
    return scale * (h / np.exp(lin)) ** (1.0 / shape)


def generate(n_patients=30000, seed=2026):
    rng = np.random.default_rng(seed)
    n = n_patients
    subject_id = 10_000_001 + np.arange(n)
    hadm_id = 20_000_001 + np.arange(n)

    # ---------------- Demographics ----------------
    age = np.clip(rng.normal(64, 15, n), 18, 91).round().astype(int)
    gender = np.where(rng.random(n) < 0.55, "M", "F")
    anchor_year = rng.integers(2110, 2190, n)
    year_groups = np.array(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"])
    admittime = (pd.DatetimeIndex(pd.to_datetime(pd.Series(anchor_year).astype(str) + "-01-01"))
                 + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
                 + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="m"))

    # ---------------- Conditions ----------------
    age_z = (age - 64) / 15
    cond = {}
    for name, (_, prev) in CONDITIONS.items():
        logit = np.log(prev / (1 - prev)) + 0.3 * age_z
        if name == "ckd":
            logit = logit + 0.9 * cond["t2dm"] + 0.5 * cond["hypertension"]
        if name == "heart_failure":
            logit = logit + 0.5 * cond["hypertension"] + 0.3 * cond["t2dm"]
        if name == "aki":
            logit = logit + 1.0 * cond["sepsis"] + 0.6 * cond["ckd"]
        cond[name] = (rng.random(n) < _logistic(logit)).astype(int)
    septic_shock = cond["sepsis"] * (rng.random(n) < 0.35)

    # ---------------- ICU stay and first-day severity ----------------
    has_icu = rng.random(n) < 0.85
    severity = (rng.normal(0, 1, n) + 0.8 * cond["sepsis"] + 1.2 * septic_shock + 0.3 * cond["heart_failure"]
                + 0.4 * cond["ckd"] + 0.4 * cond["aki"] + 0.2 * age_z)

    def lognormal(mu, sigma):
        return np.exp(mu + sigma * rng.normal(0, 1, n))

    cov = pd.DataFrame({"subject_id": subject_id, "hadm_id": hadm_id})
    cov["age"] = age
    cov["gender"] = gender
    cov["sofa"] = np.clip(np.round(2.5 + 2.2 * severity + rng.normal(0, 1, n)), 0, 20).astype(int)
    parts = rng.dirichlet(np.ones(6), n) * cov["sofa"].values[:, None]
    for i, comp in enumerate(["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]):
        cov[comp] = np.clip(np.round(parts[:, i]), 0, 4).astype(int)
    cov["lactate"] = lognormal(np.log(1.5) + 0.30 * severity + 0.5 * septic_shock, 0.35).round(1)
    cov["creatinine"] = lognormal(np.log(0.95) + 0.65 * cond["ckd"] + 0.20 * severity + 0.3 * cond["aki"], 0.3).round(2)
    cov["bun"] = (14 * cov["creatinine"] ** 0.8 * lognormal(0, 0.2)).round(0)
    cov["heart_rate"] = (86 + 9 * severity + 6 * cond["sepsis"] + rng.normal(0, 10, n)).round(0)
    cov["sbp"] = (122 - 9 * severity - 14 * septic_shock + rng.normal(0, 14, n)).round(0)
    cov["dbp"] = (66 - 4 * severity + rng.normal(0, 9, n)).round(0)
    cov["mbp"] = ((cov["sbp"] + 2 * cov["dbp"]) / 3).round(0)
    cov["resp_rate"] = (18 + 2.5 * severity + rng.normal(0, 3, n)).round(0)
    cov["temperature"] = (36.9 + 0.5 * cond["sepsis"] + rng.normal(0, 0.5, n)).round(1)
    cov["spo2"] = np.clip(97 - 1.2 * severity + rng.normal(0, 1.5, n), 75, 100).round(0)
    cov["glucose"] = (118 + 55 * cond["t2dm"] + 12 * severity + rng.normal(0, 25, n)).round(0)
    cov["wbc"] = np.clip(9 + 4 * cond["sepsis"] + 1.5 * severity + rng.normal(0, 3, n), 0.5, None).round(1)
    cov["platelets"] = np.clip(230 - 30 * severity + rng.normal(0, 60, n), 10, None).round(0)
    cov["hemoglobin"] = np.clip(11.6 - 0.5 * severity + rng.normal(0, 1.6, n), 5, 18).round(1)
    cov["hematocrit"] = (3 * cov["hemoglobin"] + rng.normal(0, 1.5, n)).round(1)
    cov["sodium"] = (139 + rng.normal(0, 4, n)).round(0)
    cov["potassium"] = (4.1 + 0.4 * cond["ckd"] + rng.normal(0, 0.5, n)).round(1)
    cov["chloride"] = (103 + rng.normal(0, 5, n)).round(0)
    cov["bicarbonate"] = (24 - 1.8 * severity + rng.normal(0, 3, n)).round(0)
    cov["aniongap"] = (cov["sodium"] - cov["chloride"] - cov["bicarbonate"]).round(0)
    cov["calcium"] = (8.6 + rng.normal(0, 0.6, n)).round(1)
    cov["albumin"] = np.clip(3.3 - 0.3 * severity + rng.normal(0, 0.5, n), 1.2, 5).round(1)
    cov["ph"] = (7.38 - 0.035 * severity + rng.normal(0, 0.05, n)).round(2)
    cov["pco2"] = (40 + rng.normal(0, 7, n)).round(0)
    cov["po2"] = np.clip(105 - 12 * severity + rng.normal(0, 30, n), 40, None).round(0)
    cov["pao2fio2ratio"] = np.clip(300 - 55 * severity + rng.normal(0, 70, n), 50, 600).round(0)
    cov["baseexcess"] = (-1.8 * severity + rng.normal(0, 3, n)).round(0)
    cov["bilirubin_total"] = lognormal(np.log(0.7) + 0.2 * severity + 1.0 * cond["cirrhosis"], 0.5).round(1)
    cov["alt"] = lognormal(np.log(28) + 0.2 * severity, 0.6).round(0)
    cov["ast"] = lognormal(np.log(32) + 0.25 * severity, 0.6).round(0)
    cov["inr"] = np.clip(1.15 + 0.15 * severity + 0.6 * cond["cirrhosis"] + rng.normal(0, 0.2, n), 0.8, None).round(2)
    cov["pt"] = (11.5 * cov["inr"]).round(1)
    cov["ptt"] = (31 + 3 * severity + rng.normal(0, 6, n)).round(1)
    cov["gcs"] = np.clip(np.round(14.5 - 1.5 * severity + rng.normal(0, 1.5, n)), 3, 15).astype(int)
    cov["urineoutput"] = np.clip(1800 - 350 * severity - 300 * cond["ckd"] + rng.normal(0, 450, n), 0, None).round(0)

    missing_rates = {"lactate": 0.25, "ph": 0.35, "pco2": 0.35, "po2": 0.35, "pao2fio2ratio": 0.4,
                     "baseexcess": 0.35, "albumin": 0.5, "bilirubin_total": 0.4, "alt": 0.4, "ast": 0.4,
                     "inr": 0.2, "pt": 0.2, "ptt": 0.25}
    for col in cov.columns:
        if col in ("subject_id", "hadm_id", "age", "gender", "sofa"):
            continue
        rate = missing_rates.get(col, 0.03)
        cov.loc[rng.random(n) < rate, col] = np.nan

    # ---------------- Hospital course and metformin initiation ----------------
    los_hosp = lognormal(np.log(6.5) + 0.25 * severity, 0.55)
    creat_obs = cov["creatinine"].fillna(1.0).values
    lact_obs = cov["lactate"].fillna(1.5).values
    p_init = _logistic(0.1 - 0.9 * np.log(creat_obs) - 0.4 * (lact_obs > 2) - 0.08 * (cov["sofa"].values - 3)
                       - 0.01 * (age - 64))
    intends = (cond["t2dm"] == 1) & has_icu & (rng.random(n) < p_init)
    delayed = rng.random(n) < 0.45
    planned_start = np.where(delayed, rng.uniform(0.5, GRACE_PERIOD_DAYS, n), rng.uniform(0.0, 0.4, n))

    # Mortality: Weibull baseline hazard (shape < 1: higher early risk), effect applied from initiation onward.
    shape, scale = 0.75, 1700.0
    lin = (0.035 * (age - 64) + 0.28 * severity + 0.35 * np.log(creat_obs) + 0.25 * np.log(lact_obs)
           + 0.35 * cond["ckd"] + 0.3 * cond["heart_failure"] + 0.5 * cond["lung_cancer"] + 0.5 * cond["cirrhosis"]
           + 0.3 * cond["dementia"])
    e = rng.exponential(1.0, n)
    death_untreated = _weibull_inverse(e, scale, shape, lin)

    def death_if_started(start):
        """Death time if metformin is started at ``start`` (only if the patient is alive at that time)."""
        alive = start < death_untreated
        h_start = np.where(alive, (start / scale) ** shape * np.exp(lin), 0.0)
        treated_death = _weibull_inverse(h_start + (e - h_start) / TRUE_HR_METFORMIN, scale, shape, lin)
        return np.where(alive, treated_death, death_untreated)

    initiated = intends & (planned_start < death_untreated) & (planned_start < los_hosp)
    treatment_start = np.where(initiated, planned_start, np.nan)
    death_time = np.where(initiated, death_if_started(planned_start), death_untreated)

    # Ground truth for the demo cohort (type 2 diabetes with an ICU stay): counterfactual mortality if every
    # patient started metformin at their planned time within the grace period vs. never started it.
    target = (cond["t2dm"] == 1) & has_icu
    t_treat, t_none = death_if_started(planned_start)[target], death_untreated[target]
    ground_truth = {
        "conditional_hazard_ratio_after_initiation": TRUE_HR_METFORMIN,
        "grace_period_days": GRACE_PERIOD_DAYS,
        "cohort": "type 2 diabetes with an ICU stay during the index admission",
    }
    for horizon in (30, 90):
        stacked = pd.DataFrame({"T": np.minimum(np.r_[t_treat, t_none], horizon),
                                "E": (np.r_[t_treat, t_none] <= horizon).astype(int),
                                "arm": np.r_[np.ones(len(t_treat)), np.zeros(len(t_none))]})
        ground_truth[f"{horizon}day"] = {
            "counterfactual_mortality_initiate_within_grace": round(float((t_treat <= horizon).mean()), 4),
            "counterfactual_mortality_no_initiation": round(float((t_none <= horizon).mean()), 4),
            "marginal_hazard_ratio": round(float(np.exp(CoxPHFitter().fit(stacked, "T", "E").params_["arm"])), 4),
        }

    died_in_hospital = death_time <= los_hosp
    dischtime_days = np.where(died_in_hospital, death_time, los_hosp)
    dischtime = admittime + pd.to_timedelta(dischtime_days * 24 * 60, unit="m")
    death_dt = admittime + pd.to_timedelta(np.minimum(death_time, 3650) * 24 * 60, unit="m")
    # MIMIC-IV records out-of-hospital deaths up to one year after discharge.
    death_recorded = died_in_hospital | (death_time <= dischtime_days + 365)

    # ---------------- patients ----------------
    patients = pd.DataFrame({
        "subject_id": subject_id,
        "gender": gender,
        "anchor_age": age,
        "anchor_year": anchor_year,
        "anchor_year_group": rng.choice(year_groups, n),
        "dod": np.where(death_recorded, death_dt.strftime("%Y-%m-%d"), None),
    })

    # ---------------- admissions (index + readmissions) ----------------
    admissions = pd.DataFrame({
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "admittime": admittime.strftime("%Y-%m-%d %H:%M:%S"),
        "dischtime": dischtime.strftime("%Y-%m-%d %H:%M:%S"),
        "deathtime": np.where(died_in_hospital, dischtime.strftime("%Y-%m-%d %H:%M:%S"), None),
        "admission_type": rng.choice(["EW EMER.", "URGENT", "ELECTIVE", "OBSERVATION ADMIT"], n, p=[0.6, 0.2, 0.12, 0.08]),
        "admission_location": rng.choice(["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL", "PHYSICIAN REFERRAL"], n, p=[0.65, 0.2, 0.15]),
        "discharge_location": np.where(died_in_hospital, "DIED",
                                       rng.choice(["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB"], n)),
        "insurance": rng.choice(["Medicare", "Medicaid", "Other"], n, p=[0.5, 0.15, 0.35]),
        "hospital_expire_flag": died_in_hospital.astype(int),
    })
    p_readmit = _logistic(-2.2 + 0.5 * cond["heart_failure"] + 0.4 * cond["ckd"] + 0.3 * cond["copd"])
    readmit = (~died_in_hospital) & (rng.random(n) < p_readmit)
    gap_days = rng.exponential(60, n) + 1
    readmit &= dischtime_days + gap_days < death_time
    ridx = np.where(readmit)[0]
    re_admit = dischtime[ridx] + pd.to_timedelta(gap_days[ridx] * 24 * 60, unit="m")
    re_disch = re_admit + pd.to_timedelta(lognormal(np.log(4), 0.5)[ridx] * 24 * 60, unit="m")
    readmissions = pd.DataFrame({
        "subject_id": subject_id[ridx],
        "hadm_id": 29_000_001 + np.arange(len(ridx)),
        "admittime": re_admit.strftime("%Y-%m-%d %H:%M:%S"),
        "dischtime": re_disch.strftime("%Y-%m-%d %H:%M:%S"),
        "deathtime": None,
        "admission_type": "EW EMER.",
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "HOME",
        "insurance": admissions["insurance"].values[ridx],
        "hospital_expire_flag": 0,
    })
    admissions = pd.concat([admissions, readmissions], ignore_index=True)

    # ---------------- diagnoses_icd ----------------
    diag_rows = []
    for name, (codes, _) in CONDITIONS.items():
        idx = np.where(cond[name] == 1)[0]
        chosen = rng.choice(codes, len(idx))
        diag_rows.append(pd.DataFrame({"subject_id": subject_id[idx], "hadm_id": hadm_id[idx], "icd_code": chosen}))
    shock_idx = np.where(septic_shock == 1)[0]
    diag_rows.append(pd.DataFrame({"subject_id": subject_id[shock_idx], "hadm_id": hadm_id[shock_idx], "icd_code": "R6521"}))
    diagnoses = pd.concat(diag_rows, ignore_index=True).sort_values(["hadm_id"]).reset_index(drop=True)
    diagnoses["seq_num"] = diagnoses.groupby("hadm_id").cumcount() + 1
    diagnoses["icd_version"] = 10
    diagnoses = diagnoses[["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]]

    # ---------------- prescriptions ----------------
    rx_rows = []

    def add_rx(mask, drug, start_days=None, dose="", unit="", route="PO"):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        start = start_days[idx] if start_days is not None else rng.uniform(0, 1.0, len(idx))
        start = np.minimum(start, dischtime_days[idx] * 0.95)
        stop = np.minimum(start + rng.uniform(1, 6, len(idx)), dischtime_days[idx])
        rx_rows.append(pd.DataFrame({
            "subject_id": subject_id[idx], "hadm_id": hadm_id[idx],
            "starttime": (admittime[idx] + pd.to_timedelta(start * 24 * 60, unit="m")).strftime("%Y-%m-%d %H:%M:%S"),
            "stoptime": (admittime[idx] + pd.to_timedelta(stop * 24 * 60, unit="m")).strftime("%Y-%m-%d %H:%M:%S"),
            "drug_type": "MAIN", "drug": drug, "dose_val_rx": dose, "dose_unit_rx": unit, "route": route,
        }))

    add_rx(initiated, "MetFORMIN (Glucophage)", start_days=np.nan_to_num(treatment_start), dose="500", unit="mg")
    add_rx((cond["t2dm"] == 1) & (rng.random(n) < 0.7), "Insulin", dose="2-10", unit="UNIT", route="SC")
    add_rx((septic_shock == 1) & (rng.random(n) < 0.9), "Norepinephrine", dose="0.05", unit="mcg/kg/min", route="IV")
    add_rx((cond["sepsis"] == 1) & (rng.random(n) < 0.75), "Vancomycin", dose="1000", unit="mg", route="IV")
    add_rx((cond["sepsis"] == 1) & (rng.random(n) < 0.6), "Piperacillin-Tazobactam", dose="4.5", unit="g", route="IV")
    add_rx((septic_shock == 1) & (rng.random(n) < 0.4), "Hydrocortisone Na Succ.", dose="50", unit="mg", route="IV")
    add_rx((cond["heart_failure"] == 1) & (rng.random(n) < 0.7), "Furosemide", dose="40", unit="mg", route="IV")
    add_rx(rng.random(n) < 0.55, "Heparin", dose="5000", unit="UNIT", route="SC")
    add_rx(rng.random(n) < 0.3, "Aspirin", dose="81", unit="mg")
    add_rx(rng.random(n) < 0.35, "Atorvastatin", dose="40", unit="mg")
    add_rx(rng.random(n) < 0.25, "Metoprolol Tartrate", dose="25", unit="mg")
    add_rx((cond["atrial_fibrillation"] == 1) & (rng.random(n) < 0.15), "Esmolol", dose="50", unit="mcg/kg/min", route="IV")
    add_rx((cond["ckd"] == 0) & (cond["hypertension"] == 1) & (rng.random(n) < 0.3), "Lisinopril", dose="10", unit="mg")
    prescriptions = pd.concat(rx_rows, ignore_index=True).sort_values(["subject_id", "starttime"]).reset_index(drop=True)
    prescriptions.insert(2, "pharmacy_id", 40_000_001 + np.arange(len(prescriptions)))

    # ---------------- icustays and first-day covariates ----------------
    icu_idx = np.where(has_icu)[0]
    intime = admittime[icu_idx] + pd.to_timedelta(rng.uniform(0, 0.4, len(icu_idx)) * 24 * 60, unit="m")
    icu_los = np.minimum(lognormal(np.log(2.5) + 0.3 * severity, 0.6)[icu_idx], dischtime_days[icu_idx])
    outtime = intime + pd.to_timedelta(icu_los * 24 * 60, unit="m")
    stay_id = 30_000_001 + np.arange(len(icu_idx))
    icustays = pd.DataFrame({
        "subject_id": subject_id[icu_idx], "hadm_id": hadm_id[icu_idx], "stay_id": stay_id,
        "first_careunit": rng.choice(["Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)",
                                      "Cardiac Vascular Intensive Care Unit (CVICU)"], len(icu_idx)),
        "intime": intime.strftime("%Y-%m-%d %H:%M:%S"), "outtime": outtime.strftime("%Y-%m-%d %H:%M:%S"),
        "los": icu_los.round(3),
    })
    covariates = cov.iloc[icu_idx].copy()
    covariates.insert(0, "stay_id", stay_id)

    return {
        "hosp/patients.csv.gz": patients,
        "hosp/admissions.csv.gz": admissions,
        "hosp/diagnoses_icd.csv.gz": diagnoses,
        "hosp/prescriptions.csv.gz": prescriptions,
        "icu/icustays.csv.gz": icustays,
        "icu_first_day_covariates.csv.gz": covariates,
    }, ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="Data/synthetic_mimic")
    parser.add_argument("--n-patients", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    tables, ground_truth = generate(args.n_patients, args.seed)
    for rel_path, df in tables.items():
        path = os.path.join(args.out, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        print(f"{rel_path}: {len(df)} rows")
    with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    main()
