"""
MES analysis layer, concept dictionary and catalog of the RWD store (run by scripts/build_mimic_db.py).

Built on top of the raw MIMIC-IV tables and the mimic-code concepts (``mimiciv_derived``), so severity scores,
Sepsis-3, KDIGO AKI, ventilation and vasoactive agents follow the community definitions:

    mes.icu_stays      one row per ICU stay: demographics, first-day severity / labs / vitals / treatments,
                       Sepsis-3, AKI, outcomes (death in ICU / hospital / 28 / 90 / 365 days, LOS, readmission)
    mes.admissions     one row per hospital admission: demographics, Charlson, outcomes
    mes.diagnoses      ICD-9 and ICD-10 diagnoses per admission
    mes.drugs          medication orders (prescriptions) and ICU infusions (inputevents), lower-case names
    mes.drug_names     distinct medication names with patient counts
    mes.procedures     ICD procedures and ICU procedure events
"""

import json
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np

from mes.tools.rwd.db import Store

RACE_GROUP = """CASE
    WHEN {c} ILIKE 'WHITE%' OR {c} = 'PORTUGUESE' THEN 'White'
    WHEN {c} ILIKE 'BLACK%' THEN 'Black'
    WHEN {c} ILIKE 'HISPANIC%' OR {c} = 'SOUTH AMERICAN' THEN 'Hispanic'
    WHEN {c} ILIKE 'ASIAN%' THEN 'Asian'
    WHEN {c} IN ('UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER') OR {c} IS NULL THEN 'Unknown'
    ELSE 'Other' END"""

CAREUNIT_GROUP = """CASE
    WHEN {c} = 'Medical Intensive Care Unit (MICU)' THEN 'MICU'
    WHEN {c} = 'Medical/Surgical Intensive Care Unit (MICU/SICU)' THEN 'MICU/SICU'
    WHEN {c} = 'Surgical Intensive Care Unit (SICU)' THEN 'SICU'
    WHEN {c} = 'Trauma SICU (TSICU)' THEN 'TSICU'
    WHEN {c} = 'Cardiac Vascular Intensive Care Unit (CVICU)' THEN 'CVICU'
    WHEN {c} = 'Coronary Care Unit (CCU)' THEN 'CCU'
    WHEN {c} ILIKE 'Neuro%' THEN 'Neuro'
    ELSE 'Other' END"""

# Date-only deaths (state records) are placed at noon of that day
DEATH_TIME = """CASE WHEN adm.deathtime IS NOT NULL THEN adm.deathtime
    WHEN pat.dod IS NOT NULL THEN CAST(pat.dod AS TIMESTAMP) + INTERVAL 12 HOUR END"""

CHARLSON_FLAGS = ["myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
                  "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease", "rheumatic_disease",
                  "peptic_ulcer_disease", "mild_liver_disease", "diabetes_without_cc", "diabetes_with_cc", "paraplegia",
                  "renal_disease", "malignant_cancer", "severe_liver_disease", "metastatic_solid_tumor", "aids"]

ICU_STAYS = f"""
WITH vaso AS (
    SELECT v.stay_id,
        max(CASE WHEN coalesce(v.norepinephrine, v.epinephrine, v.dopamine, v.vasopressin, v.phenylephrine) IS NOT NULL
                 THEN 1 ELSE 0 END) AS vasopressor_24h,
        max(CASE WHEN v.norepinephrine IS NOT NULL THEN 1 ELSE 0 END) AS norepinephrine_24h,
        max(CASE WHEN v.epinephrine IS NOT NULL THEN 1 ELSE 0 END) AS epinephrine_24h,
        max(CASE WHEN v.dopamine IS NOT NULL THEN 1 ELSE 0 END) AS dopamine_24h,
        max(CASE WHEN v.vasopressin IS NOT NULL THEN 1 ELSE 0 END) AS vasopressin_24h,
        max(CASE WHEN v.phenylephrine IS NOT NULL THEN 1 ELSE 0 END) AS phenylephrine_24h,
        max(CASE WHEN v.dobutamine IS NOT NULL OR v.milrinone IS NOT NULL THEN 1 ELSE 0 END) AS inotrope_24h
    FROM mimiciv_derived.vasoactive_agent v JOIN mimiciv_icu.icustays ie USING (stay_id)
    WHERE v.starttime < ie.intime + INTERVAL 24 HOUR AND v.endtime > ie.intime
    GROUP BY v.stay_id),
vent AS (
    SELECT v.stay_id,
        max(CASE WHEN v.starttime < ie.intime + INTERVAL 24 HOUR AND v.endtime > ie.intime THEN 1 ELSE 0 END)
            AS invasive_vent_24h,
        1 AS invasive_vent_any
    FROM mimiciv_derived.ventilation v JOIN mimiciv_icu.icustays ie USING (stay_id)
    WHERE v.ventilation_status IN ('InvasiveVent', 'Tracheostomy')
    GROUP BY v.stay_id),
aki AS (
    SELECT k.stay_id,
        max(CASE WHEN k.charttime <= ie.intime + INTERVAL 48 HOUR THEN k.aki_stage END) AS aki_stage_48h,
        max(CASE WHEN k.charttime <= ie.intime + INTERVAL 7 DAY THEN k.aki_stage END) AS aki_stage_7d
    FROM mimiciv_derived.kdigo_stages k JOIN mimiciv_icu.icustays ie USING (stay_id)
    WHERE k.charttime >= ie.intime
    GROUP BY k.stay_id),
sep AS (
    SELECT stay_id, bool_or(sepsis3) AS sepsis3, min(suspected_infection_time) AS suspected_infection_time
    FROM mimiciv_derived.sepsis3 GROUP BY stay_id),
rrt_any AS (
    SELECT stay_id, max(CASE WHEN dialysis_active = 1 THEN 1 ELSE 0 END) AS rrt_any
    FROM mimiciv_derived.rrt GROUP BY stay_id),
stays AS (
    SELECT stay_id, lead(intime) OVER (PARTITION BY subject_id ORDER BY intime) AS next_icu_intime,
           row_number() OVER (PARTITION BY subject_id ORDER BY intime) AS icu_stay_number
    FROM mimiciv_icu.icustays),
nxt_adm AS (
    SELECT hadm_id, lead(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime
    FROM mimiciv_hosp.admissions),
base AS (
    SELECT
        ie.subject_id, ie.hadm_id, ie.stay_id, ie.intime, ie.outtime, ie.first_careunit,
        {CAREUNIT_GROUP.format(c="ie.first_careunit")} AS careunit,
        adm.admittime, adm.dischtime,
        det.admission_age AS age, pat.gender, {RACE_GROUP.format(c="adm.race")} AS race_group,
        adm.admission_type, adm.insurance, adm.marital_status, adm.language, pat.anchor_year_group,
        stays.icu_stay_number = 1 AS first_icu_stay, det.first_icu_stay AS first_icu_stay_of_admission,
        date_diff('second', ie.intime, ie.outtime) / 86400.0 AS los_icu_days,
        date_diff('second', adm.admittime, adm.dischtime) / 86400.0 AS los_hosp_days,
        date_diff('second', adm.admittime, ie.intime) / 3600.0 AS hours_admit_to_icu,
        {DEATH_TIME} AS death_time,
        adm.hospital_expire_flag AS death_hosp,
        stays.next_icu_intime, nxt_adm.next_admittime,
        fs.sofa, fs.respiration AS sofa_respiration, fs.coagulation AS sofa_coagulation, fs.liver AS sofa_liver,
        fs.cardiovascular AS sofa_cardiovascular, fs.cns AS sofa_cns, fs.renal AS sofa_renal,
        sp.sapsii, oa.oasis,
        ch.charlson_comorbidity_index AS charlson, {", ".join(f"ch.{c}" for c in CHARLSON_FLAGS)},
        lab.creatinine_max, lab.creatinine_min, lab.bun_max, lab.wbc_max, lab.wbc_min, lab.hemoglobin_min,
        lab.platelets_min, lab.bilirubin_total_max, lab.sodium_min, lab.sodium_max, lab.potassium_max,
        lab.glucose_max, lab.albumin_min, lab.inr_max, lab.bicarbonate_min, lab.aniongap_max,
        bg.lactate_max, bg.ph_min, bg.pao2fio2ratio_min,
        vs.heart_rate_mean, vs.heart_rate_max, vs.mbp_mean, vs.mbp_min, vs.sbp_min, vs.resp_rate_mean,
        vs.spo2_min, vs.temperature_max, vs.temperature_min,
        gcs.gcs_min, wt.weight, uo.urineoutput AS urine_output_24h,
        coalesce(rrt.dialysis_active, 0) AS rrt_24h, coalesce(rrt_any.rrt_any, 0) AS rrt_any,
        coalesce(vaso.vasopressor_24h, 0) AS vasopressor_24h,
        coalesce(vaso.norepinephrine_24h, 0) AS norepinephrine_24h,
        coalesce(vaso.epinephrine_24h, 0) AS epinephrine_24h, coalesce(vaso.dopamine_24h, 0) AS dopamine_24h,
        coalesce(vaso.vasopressin_24h, 0) AS vasopressin_24h,
        coalesce(vaso.phenylephrine_24h, 0) AS phenylephrine_24h,
        coalesce(vaso.inotrope_24h, 0) AS inotrope_24h,
        coalesce(vent.invasive_vent_24h, 0) AS invasive_vent_24h, coalesce(vent.invasive_vent_any, 0)
            AS invasive_vent_any,
        aki.aki_stage_48h, aki.aki_stage_7d,
        coalesce(sep.sepsis3, false) AS sepsis3, sep.suspected_infection_time
    FROM mimiciv_icu.icustays ie
    JOIN mimiciv_hosp.admissions adm ON adm.hadm_id = ie.hadm_id
    JOIN mimiciv_hosp.patients pat ON pat.subject_id = ie.subject_id
    LEFT JOIN mimiciv_derived.icustay_detail det ON det.stay_id = ie.stay_id
    LEFT JOIN stays ON stays.stay_id = ie.stay_id
    LEFT JOIN nxt_adm ON nxt_adm.hadm_id = ie.hadm_id
    LEFT JOIN mimiciv_derived.first_day_sofa fs ON fs.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.sapsii sp ON sp.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.oasis oa ON oa.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.charlson ch ON ch.hadm_id = ie.hadm_id
    LEFT JOIN mimiciv_derived.first_day_lab lab ON lab.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_bg bg ON bg.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_vitalsign vs ON vs.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_gcs gcs ON gcs.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_weight wt ON wt.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_urine_output uo ON uo.stay_id = ie.stay_id
    LEFT JOIN mimiciv_derived.first_day_rrt rrt ON rrt.stay_id = ie.stay_id
    LEFT JOIN rrt_any ON rrt_any.stay_id = ie.stay_id
    LEFT JOIN vaso ON vaso.stay_id = ie.stay_id
    LEFT JOIN vent ON vent.stay_id = ie.stay_id
    LEFT JOIN aki ON aki.stay_id = ie.stay_id
    LEFT JOIN sep ON sep.stay_id = ie.stay_id)
SELECT *,
    -- DuckDB's greatest() ignores NULL: only patients who died get a time to death
    CASE WHEN death_time IS NOT NULL THEN greatest(date_diff('second', intime, death_time) / 86400.0, 0) END
        AS days_to_death,
    CASE WHEN death_time IS NOT NULL AND death_time <= outtime + INTERVAL 6 HOUR THEN 1 ELSE 0 END AS death_icu,
    CASE WHEN death_time IS NOT NULL AND death_time <= intime + INTERVAL 28 DAY THEN 1 ELSE 0 END AS death_28d,
    CASE WHEN death_time IS NOT NULL AND death_time <= intime + INTERVAL 90 DAY THEN 1 ELSE 0 END AS death_90d,
    CASE WHEN death_time IS NOT NULL AND death_time <= intime + INTERVAL 365 DAY THEN 1 ELSE 0 END AS death_1y,
    CASE WHEN death_hosp = 0 AND next_icu_intime IS NOT NULL AND next_icu_intime <= outtime + INTERVAL 30 DAY
         THEN 1 ELSE 0 END AS icu_readmit_30d,
    CASE WHEN death_hosp = 0 AND next_admittime IS NOT NULL AND next_admittime <= dischtime + INTERVAL 30 DAY
         THEN 1 ELSE 0 END AS hosp_readmit_30d,
    CASE WHEN aki_stage_7d >= 1 THEN 1 ELSE 0 END AS aki_7d,
    CASE WHEN aki_stage_7d >= 2 THEN 1 ELSE 0 END AS aki_stage2_7d,
    CASE WHEN sepsis3 AND vasopressor_24h = 1 AND lactate_max > 2 THEN 1 ELSE 0 END AS septic_shock
FROM base
"""

ADMISSIONS = f"""
WITH nxt AS (
    SELECT hadm_id, lead(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime,
           row_number() OVER (PARTITION BY subject_id ORDER BY admittime) AS admission_number
    FROM mimiciv_hosp.admissions),
icu AS (SELECT hadm_id, count(*) AS n_icu_stays, min(intime) AS first_icu_intime FROM mimiciv_icu.icustays GROUP BY 1),
base AS (
    SELECT adm.subject_id, adm.hadm_id, adm.admittime, adm.dischtime, ag.age, pat.gender,
        {RACE_GROUP.format(c="adm.race")} AS race_group, adm.admission_type, adm.admission_location,
        adm.discharge_location, adm.insurance, adm.marital_status, adm.language, pat.anchor_year_group,
        nxt.admission_number = 1 AS first_admission, adm.edregtime IS NOT NULL AS ed_admission,
        date_diff('second', adm.admittime, adm.dischtime) / 86400.0 AS los_hosp_days,
        adm.hospital_expire_flag AS death_hosp, {DEATH_TIME} AS death_time, nxt.next_admittime,
        coalesce(icu.n_icu_stays, 0) AS n_icu_stays, icu.first_icu_intime,
        ch.charlson_comorbidity_index AS charlson, {", ".join(f"ch.{c}" for c in CHARLSON_FLAGS)}
    FROM mimiciv_hosp.admissions adm
    JOIN mimiciv_hosp.patients pat ON pat.subject_id = adm.subject_id
    LEFT JOIN mimiciv_derived.age ag ON ag.hadm_id = adm.hadm_id
    LEFT JOIN nxt ON nxt.hadm_id = adm.hadm_id
    LEFT JOIN icu ON icu.hadm_id = adm.hadm_id
    LEFT JOIN mimiciv_derived.charlson ch ON ch.hadm_id = adm.hadm_id)
SELECT *,
    CASE WHEN n_icu_stays > 0 THEN 1 ELSE 0 END AS any_icu,
    CASE WHEN death_time IS NOT NULL THEN greatest(date_diff('second', admittime, death_time) / 86400.0, 0) END
        AS days_to_death,
    CASE WHEN death_time IS NOT NULL AND death_time <= admittime + INTERVAL 30 DAY THEN 1 ELSE 0 END AS death_30d,
    CASE WHEN death_time IS NOT NULL AND death_time <= admittime + INTERVAL 90 DAY THEN 1 ELSE 0 END AS death_90d,
    CASE WHEN death_time IS NOT NULL AND death_time <= admittime + INTERVAL 365 DAY THEN 1 ELSE 0 END AS death_1y,
    CASE WHEN death_hosp = 0 AND next_admittime IS NOT NULL AND next_admittime <= dischtime + INTERVAL 30 DAY
         THEN 1 ELSE 0 END AS hosp_readmit_30d
FROM base
"""

DIAGNOSES = """
SELECT d.subject_id, d.hadm_id, d.seq_num, upper(trim(d.icd_code)) AS icd_code, d.icd_version
FROM mimiciv_hosp.diagnoses_icd d WHERE d.icd_code IS NOT NULL
"""

DRUGS = """
SELECT subject_id, hadm_id, CAST(NULL AS INTEGER) AS stay_id, 'order' AS source, lower(trim(drug)) AS drug,
       lower(route) AS route, starttime, stoptime AS endtime
FROM mimiciv_hosp.prescriptions WHERE drug IS NOT NULL AND starttime IS NOT NULL
UNION ALL
SELECT ie.subject_id, ie.hadm_id, ie.stay_id, 'icu_infusion' AS source, lower(trim(di.label)) AS drug, 'iv' AS route,
       ie.starttime, ie.endtime
FROM mimiciv_icu.inputevents ie JOIN mimiciv_icu.d_items di ON di.itemid = ie.itemid
"""

PROCEDURES = """
SELECT p.subject_id, p.hadm_id, CAST(NULL AS INTEGER) AS stay_id, 'icd' AS source, upper(trim(p.icd_code)) AS code,
       p.icd_version, d.long_title AS label, CAST(p.chartdate AS TIMESTAMP) AS time
FROM mimiciv_hosp.procedures_icd p
LEFT JOIN mimiciv_hosp.d_icd_procedures d ON d.icd_code = p.icd_code AND d.icd_version = p.icd_version
UNION ALL
SELECT pe.subject_id, pe.hadm_id, pe.stay_id, 'icu_procedure' AS source, CAST(pe.itemid AS VARCHAR) AS code,
       CAST(NULL AS SMALLINT) AS icd_version, di.label, pe.starttime AS time
FROM mimiciv_icu.procedureevents pe JOIN mimiciv_icu.d_items di ON di.itemid = pe.itemid
"""

MES_TABLES = {"icu_stays": ICU_STAYS, "admissions": ADMISSIONS, "diagnoses": DIAGNOSES, "drugs": DRUGS,
              "procedures": PROCEDURES}


def _say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _quote(path):
    return str(path).replace("'", "''")


def _copy(con, sql, target, force):
    target = Path(target)
    if target.exists() and not force:
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({sql}) TO '{_quote(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(target)
    return con.execute(f"SELECT count(*) FROM read_parquet('{_quote(target)}')").fetchone()[0]


def _refresh(con, store, schema, name):
    path = store.root / {"mes": "mes"}.get(schema, schema) / f"{name}.parquet"
    con.execute(f"CREATE OR REPLACE VIEW {schema}.{name} AS SELECT * FROM read_parquet('{_quote(path)}')")


def build_mes_layer(out, force=False):
    store = Store(out)
    con = store._open(lock=False)
    status = {}
    for name, sql in MES_TABLES.items():
        t = time.time()
        status[name] = _copy(con, sql, store.root / "mes" / f"{name}.parquet", force)
        _refresh(con, store, "mes", name)
        _say(f"mes.{name}: {status[name]} ({time.time() - t:.0f}s)")
    status["drug_names"] = _copy(con, "SELECT drug, count(DISTINCT subject_id) AS n_patients, "
                                      "string_agg(DISTINCT source, ',') AS sources FROM mes.drugs GROUP BY drug",
                                 store.root / "mes" / "drug_names.parquet", force)
    con.close()
    return status


# ============================================================================ dictionary
DICTIONARY = """
SELECT 'diagnosis' AS kind, 'dx' || CAST(d.icd_version AS VARCHAR) || ':' || d.icd_code AS concept_id,
       d.long_title AS label, d.icd_code AS code, CAST(d.icd_version AS VARCHAR) AS vocabulary, n.n_patients
FROM mimiciv_hosp.d_icd_diagnoses d
JOIN (SELECT icd_code, icd_version, count(DISTINCT subject_id) AS n_patients FROM mes.diagnoses GROUP BY 1, 2) n
  ON n.icd_code = upper(trim(d.icd_code)) AND n.icd_version = d.icd_version
UNION ALL
SELECT 'procedure', 'px' || CAST(d.icd_version AS VARCHAR) || ':' || d.icd_code, d.long_title, d.icd_code,
       CAST(d.icd_version AS VARCHAR), n.n_patients
FROM mimiciv_hosp.d_icd_procedures d
JOIN (SELECT code, icd_version, count(DISTINCT subject_id) AS n_patients FROM mes.procedures WHERE source = 'icd'
      GROUP BY 1, 2) n ON n.code = upper(trim(d.icd_code)) AND n.icd_version = d.icd_version
UNION ALL
SELECT 'lab', 'lab:' || CAST(l.itemid AS VARCHAR), l.label || ' (' || coalesce(l.fluid, '') || ', ' ||
       coalesce(l.category, '') || ')', CAST(l.itemid AS VARCHAR), 'd_labitems', n.n_patients
FROM mimiciv_hosp.d_labitems l
JOIN (SELECT itemid, approx_count_distinct(subject_id) AS n_patients FROM mimiciv_hosp.labevents GROUP BY 1) n
  ON n.itemid = l.itemid
UNION ALL
SELECT 'icu_item', 'item:' || CAST(i.itemid AS VARCHAR), i.label || ' (' || coalesce(i.category, '') || ')',
       CAST(i.itemid AS VARCHAR), i.linksto, n.n_patients
FROM mimiciv_icu.d_items i
JOIN (SELECT itemid, approx_count_distinct(subject_id) AS n_patients FROM (
        SELECT itemid, subject_id FROM mimiciv_icu.chartevents UNION ALL
        SELECT itemid, subject_id FROM mimiciv_icu.procedureevents UNION ALL
        SELECT itemid, subject_id FROM mimiciv_icu.outputevents UNION ALL
        SELECT itemid, subject_id FROM mimiciv_icu.datetimeevents) GROUP BY 1) n ON n.itemid = i.itemid
UNION ALL
SELECT 'drug', 'drug:' || drug, drug, drug, sources, n_patients FROM mes.drug_names WHERE n_patients >= 5
"""


def build_dictionary(out, force=False):
    store = Store(out)
    target = store.root / "dictionary.parquet"
    vectors = store.root / "dictionary_vectors.npy"
    if target.exists() and vectors.exists() and not force:
        return "kept"
    con = store._open(lock=False)
    t = time.time()
    df = con.execute(DICTIONARY).df()
    con.close()
    df = df[df["n_patients"] > 0].sort_values("n_patients", ascending=False).reset_index(drop=True)
    _say(f"dictionary: {len(df):,} concepts in {time.time() - t:.0f}s; embedding labels")
    from mes.tools.ranking_tools import encode
    vecs = encode(df["label"].fillna("").tolist()).astype(np.float16)
    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)
    np.save(str(vectors) + ".tmp.npy", vecs)
    Path(str(vectors) + ".tmp.npy").replace(vectors)
    return {"concepts": len(df), "by_kind": df["kind"].value_counts().to_dict(), "seconds": round(time.time() - t)}


# ============================================================================ catalog
def build_catalog(out, expected):
    store = Store(out)
    con = store._open(lock=False)
    counts = {}
    for schema in ("mimiciv_hosp", "mimiciv_icu", "mimiciv_derived", "mes"):
        for name in store.tables(schema):
            counts[f"{schema}.{name}"] = con.execute(f"SELECT count(*) FROM {schema}.{name}").fetchone()[0]
    checks = []
    for table, n in expected.items():
        schema = "mimiciv_icu" if table == "icustays" else "mimiciv_hosp"
        got = counts.get(f"{schema}.{table}")
        checks.append({"check": f"{schema}.{table} rows", "expected": n, "observed": got, "ok": got == n})
    stats = con.execute("""
        SELECT count(*) AS adult_first_icu_stays, avg(death_hosp) AS hosp_mortality, avg(death_28d) AS mortality_28d,
               avg(CAST(sepsis3 AS INTEGER)) AS sepsis3_rate, median(age) AS median_age, avg(sofa) AS mean_sofa
        FROM mes.icu_stays WHERE first_icu_stay AND age >= 18""").df().iloc[0].to_dict()
    adm = con.execute("SELECT count(*) AS adult_first_admissions, avg(death_hosp) AS hosp_mortality "
                      "FROM mes.admissions WHERE first_admission AND age >= 18").df().iloc[0].to_dict()
    years = con.execute("SELECT anchor_year_group, count(*) AS n FROM mimiciv_hosp.patients GROUP BY 1 ORDER BY 1"
                        ).df().set_index("anchor_year_group")["n"].to_dict()
    con.close()
    plausible = 0.05 <= stats["hosp_mortality"] <= 0.20 and 0.10 <= stats["sepsis3_rate"] <= 0.60
    checks.append({"check": "ICU in-hospital mortality and Sepsis-3 prevalence in a plausible range",
                   "observed": {k: round(float(stats[k]), 3) for k in ("hosp_mortality", "sepsis3_rate")},
                   "ok": bool(plausible)})
    source = (Path(__file__).parent / "vendor" / "mimic_code" / "SOURCE.md").read_text()
    commit = next((line.split(":", 1)[1].strip() for line in source.splitlines() if line.startswith("- Commit")), "")
    catalog = {
        "dataset": "MIMIC-IV v2.2 (hosp + icu modules)",
        "built_at": datetime.now().isoformat(timespec="seconds"), "duckdb": duckdb.__version__,
        "mimic_code_commit": commit, "row_counts": counts, "checks": checks,
        "ok": all(c["ok"] for c in checks),
        "summary": {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in {**stats, **{
            "adult_first_admissions": adm["adult_first_admissions"],
            "admission_hosp_mortality": adm["hosp_mortality"]}}.items()},
        "anchor_year_groups": {k: int(v) for k, v in years.items()},
        "coverage": ("Beth Israel Deaconess Medical Center (Boston, USA): adult hospital admissions and ICU stays, "
                     "anchor years 2008-2022. Dates are shifted per patient (no calendar dates; trends only by "
                     "3-year anchor_year_group). Deaths: in-hospital time of death, and out-of-hospital date of death "
                     "from state records up to one year after the last discharge. Medications are inpatient orders "
                     "and ICU infusions only; no outpatient prescriptions, no clinical notes, no imaging."),
    }
    path = store.root / "catalog.json"
    path.write_text(json.dumps(catalog, indent=1, default=str))
    if not catalog["ok"]:
        raise SystemExit("RWD store validation failed: " + json.dumps([c for c in checks if not c["ok"]], default=str))
    return {"ok": True, "checks": checks}
