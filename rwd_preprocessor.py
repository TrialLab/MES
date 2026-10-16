"""
RWD preprocessor for MIMIC-IV.

Translates a free-text clinical question into an analysis-ready cohort:
  1. an LLM extracts the analysis type, population (mapped to ICD-10 codes),
     age limits, outcome, treatment and comparator from the query;
  2. patients are selected from the MIMIC-IV hosp/icu tables;
  3. first-day ICU covariates are attached from a pre-computed covariate table
     (one row per ICU stay, e.g. built from the MIMIC-IV derived concepts);
  4. the outcome (time-to-event or continuous) and the treatment indicator with
     its initiation time are derived for the index admission.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import parse_json_response

logger = logging.getLogger(__name__)

# ICD-10 code prefixes of the demo population; other concepts are mapped to ICD-10 codes by the LLM.
CONDITION_ICD_CODES = {
    "type 2 diabetes mellitus": ["E11"],
    "type 2 diabetes": ["E11"],
    "t2dm": ["E11"],
}

# Descriptions of the first-day ICU covariates (used to give the LLM context).
MIMIC_COLUMN_INFO = {
    "subject_id": "Unique patient identifier.",
    "stay_id": "Unique ICU stay identifier.",
    "hadm_id": "Hospital admission identifier.",
    "age": "Patient's age in years.",
    "anchor_age": "Patient age at anchor year.",
    "gender": "Patient's biological sex (M/F).",
    "lactate": "Lactate level (mmol/L). Elevated indicates tissue hypoxia.",
    "ph": "Blood pH level. Normal range 7.35-7.45.",
    "po2": "Partial pressure of oxygen (mmHg).",
    "pco2": "Partial pressure of carbon dioxide (mmHg).",
    "pao2fio2ratio": "PaO2/FiO2 ratio (mmHg). Indicator of respiratory function.",
    "baseexcess": "Base excess (mmol/L). Metabolic acid-base status.",
    "bicarbonate": "Bicarbonate level (mmol/L).",
    "hematocrit": "Hematocrit (%).",
    "hemoglobin": "Hemoglobin level (g/dL).",
    "temperature": "Body temperature (Celsius).",
    "chloride": "Chloride level (mEq/L).",
    "calcium": "Calcium level (mg/dL).",
    "glucose": "Blood glucose level (mg/dL).",
    "potassium": "Potassium level (mEq/L).",
    "sodium": "Sodium level (mEq/L).",
    "gcs": "Glasgow Coma Scale total score (3-15).",
    "platelets": "Platelet count (x10^3/uL).",
    "wbc": "White blood cell count (x10^3/uL).",
    "albumin": "Albumin level (g/dL).",
    "aniongap": "Anion gap (mEq/L).",
    "bun": "Blood urea nitrogen (mg/dL). Kidney function marker.",
    "creatinine": "Serum creatinine (mg/dL). Kidney function marker.",
    "inr": "International normalized ratio.",
    "pt": "Prothrombin time (seconds).",
    "ptt": "Partial thromboplastin time (seconds).",
    "alt": "Alanine aminotransferase (U/L).",
    "ast": "Aspartate aminotransferase (U/L).",
    "bilirubin_total": "Total bilirubin (mg/dL).",
    "heart_rate": "Heart rate (beats per minute).",
    "sbp": "Systolic blood pressure (mmHg).",
    "dbp": "Diastolic blood pressure (mmHg).",
    "mbp": "Mean arterial blood pressure (mmHg).",
    "resp_rate": "Respiratory rate (breaths per minute).",
    "spo2": "Peripheral oxygen saturation (%).",
    "urineoutput": "Urine output over the first ICU day (mL).",
    "sofa": "Sequential Organ Failure Assessment score.",
    "respiration": "SOFA respiration component (0-4).",
    "coagulation": "SOFA coagulation component (0-4).",
    "liver": "SOFA liver component (0-4).",
    "cardiovascular": "SOFA cardiovascular component (0-4).",
    "cns": "SOFA central nervous system component (0-4).",
    "renal": "SOFA renal component (0-4).",
    "outcome": "Outcome indicator (1 = event occurred, 0 = censored/no event), or the continuous outcome value.",
    "time_to_event": "Time from admission to the outcome event or censoring (days).",
    "treatment": "Treatment indicator (1 = treated during the index admission, 0 = comparator).",
    "treatment_time": "Time from admission to treatment initiation (days); missing if not treated.",
}

NO_COMPARATOR_TERMS = ("standard care", "usual care", "no ", "none", "placebo", "without", "not ")
# Care settings are handled by the ICU-stay criterion, not by diagnosis codes.
SETTING_TERMS = ("icu", "intensive care", "critical care", "critically ill", "hospitaliz", "admission", "inpatient")
# Generic words removed from treatment names before matching prescriptions (e.g., "metformin initiation").
GENERIC_TREATMENT_WORDS = r"\b(initiation|initiating|initiated|use|users?|therapy|treatment|exposure|administration|prescription)\b"


def _drug_name(name):
    return " ".join(re.sub(GENERIC_TREATMENT_WORDS, " ", name.lower()).split()) or name.lower().strip()


class AnalysisType(Enum):
    TTE = "time_to_event"
    BINARY_CLASSIFICATION = "binary_classification"
    CONTINUOUS_PREDICTION = "continuous_prediction"
    DESCRIPTIVE = "descriptive"
    CLUSTERING = "clustering"
    PATIENT_SIMILARITY = "patient_similarity"


@dataclass
class EligibilityCriteria:
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    gender: Optional[List[str]] = None
    diagnoses_include: List[List[str]] = field(default_factory=list)  # one code list per required condition
    diagnoses_exclude: List[str] = field(default_factory=list)
    icu_stay_required: bool = False


@dataclass
class OutcomeSpecification:
    outcome_type: str  # mortality, in_hospital_mortality, readmission, los
    follow_up_end: Optional[int] = None  # days


@dataclass
class TreatmentSpecification:
    treatment_medications: List[str]
    control_medications: Optional[List[str]] = None  # None: comparator is "not treated"
    treatment_window_days: Optional[float] = None


class LLMConceptExtractor:
    """Extracts the structured components of an RWD query with a single LLM call."""

    def __init__(self, llm):
        self.llm = llm

    def extract_all_from_query(self, query: str) -> dict:
        default = {
            "analysis_type": None, "inclusion_concepts": [], "exclusion_concepts": [],
            "age_min": None, "age_max": None, "outcome_type": None, "outcome_follow_up_days": None,
            "treatment_name": None, "control_name": None, "treatment_window_days": None,
        }
        if self.llm is None:
            return default
        prompt = f"""You are a medical data analyst. Parse this clinical research query into structured components.

Query: "{query}"

Return a valid JSON object with these exact fields:
{{
  "analysis_type": "<one of: time_to_event, binary_classification, continuous_prediction, descriptive, clustering, patient_similarity>",
  "inclusion_concepts": ["<diseases/conditions the study population must have, using full medical names; not care settings such as ICU>"],
  "exclusion_concepts": ["<diseases/conditions to exclude from the population>"],
  "age_min": <minimum patient age as integer, or null>,
  "age_max": <maximum patient age as integer, or null>,
  "outcome_type": "<one of: mortality, in_hospital_mortality, readmission, los, or null if no specific outcome>",
  "outcome_follow_up_days": <follow-up window in days as integer, e.g. 30 for 30-day mortality, or null>,
  "treatment_name": "<name of the primary drug being evaluated (e.g. "metformin", not "metformin initiation"), or null>",
  "control_name": "<comparator drug, or 'standard care' when the comparator is no treatment, or null>",
  "treatment_window_days": <time window after admission within which treatment must be initiated, in days (e.g. 1 for 'within 24 hours'), or null>
}}

Guidelines:
- analysis_type: "time_to_event" for survival/comparative effectiveness; "binary_classification" for prediction/AUC
- inclusion_concepts: conditions defining who is in the study (e.g. "heart failure", "sepsis")
- outcome_type: "mortality" for death within a follow-up window, "in_hospital_mortality" for death before discharge,
  "readmission" for re-hospitalization, "los" for length of stay, null otherwise
- treatment_name: the specific drug/intervention under study (NOT the disease), null if no specific treatment

Return ONLY the JSON object, no other text."""
        result = parse_json_response(self.llm.invoke(prompt))
        if not isinstance(result, dict):
            logger.warning("Query component extraction failed; falling back to rule-based parsing.")
            return default
        merged = {**default, **result}
        for key in ("inclusion_concepts", "exclusion_concepts"):
            if not isinstance(merged.get(key), list):
                merged[key] = []
        return merged

    def generate_icd_codes_for_concept(self, concept: str) -> List[str]:
        """Map a medical concept to ICD-10 code prefixes (curated codes first, then the LLM)."""
        concept_lower = concept.lower().strip()
        if concept_lower in CONDITION_ICD_CODES:
            return CONDITION_ICD_CODES[concept_lower]
        if self.llm is not None:
            prompt = f"""Generate ICD-10 codes for the medical concept: "{concept}"

Codes are matched as prefixes, so use a 3-character category (e.g. "N18") when the whole category applies
and more specific codes (e.g. "R65.21") otherwise.
Return a JSON array of ICD code strings, e.g. ["N18"]
Return ONLY the JSON array, no additional text."""
            codes = parse_json_response(self.llm.invoke(prompt))
            if isinstance(codes, list) and codes:
                return [str(c) for c in codes]
        logger.warning("No ICD codes found for concept '%s'.", concept)
        return []


class QueryParser:
    """Parse a free-text query into structured specifications."""

    def __init__(self, llm=None):
        self.llm_extractor = LLMConceptExtractor(llm)

    def parse_query(self, query: str):
        query_lower = query.lower()
        llm_result = self.llm_extractor.extract_all_from_query(query)
        analysis_type = self._infer_analysis_type(query_lower, llm_result)
        eligibility = self._extract_eligibility_criteria(query_lower, llm_result)
        outcome = self._extract_outcome_specification(query_lower, llm_result)
        treatment = self._extract_treatment_specification(query_lower, llm_result)
        return analysis_type, eligibility, outcome, treatment

    @staticmethod
    def _infer_analysis_type(query, llm_result):
        values = {t.value: t for t in AnalysisType}
        if llm_result.get("analysis_type") in values:
            return values[llm_result["analysis_type"]]
        if any(t in query for t in ["compare", " vs", "versus", "treated with", "effect", "survival", "hazard"]):
            return AnalysisType.TTE
        if any(t in query for t in ["predict", "auc", "risk score"]):
            return AnalysisType.BINARY_CLASSIFICATION
        if any(t in query for t in ["cluster", "phenotype", "subtype"]):
            return AnalysisType.CLUSTERING
        if any(t in query for t in ["similar", "match"]):
            return AnalysisType.PATIENT_SIMILARITY
        return AnalysisType.DESCRIPTIVE

    def _extract_eligibility_criteria(self, query, llm_result):
        criteria = EligibilityCriteria()
        if llm_result.get("age_min") is not None:
            criteria.age_min = float(llm_result["age_min"])
        if llm_result.get("age_max") is not None:
            criteria.age_max = float(llm_result["age_max"])
        if criteria.age_min is None:
            m = re.search(r"age[d]?\s*(?:>=?|>)\s*(\d+)", query)
            if m:
                criteria.age_min = float(m.group(1))
        if criteria.age_max is None:
            m = re.search(r"age[d]?\s*(?:<=?|<)\s*(\d+)", query)
            if m:
                criteria.age_max = float(m.group(1))

        concepts_in = llm_result.get("inclusion_concepts") or [c for c in CONDITION_ICD_CODES if c in query]
        concepts_in = [c for c in concepts_in if not any(t in c.lower() for t in SETTING_TERMS)]
        # A patient must have every inclusion condition (any code of each condition).
        for concept in concepts_in:
            codes = self.llm_extractor.generate_icd_codes_for_concept(concept)
            if codes:
                criteria.diagnoses_include.append(codes)
        for concept in llm_result.get("exclusion_concepts", []):
            criteria.diagnoses_exclude.extend(self.llm_extractor.generate_icd_codes_for_concept(concept))

        if re.search(r"\bicu\b|intensive care|critically ill", query):
            criteria.icu_stay_required = True
        return criteria

    @staticmethod
    def _extract_outcome_specification(query, llm_result):
        supported = {"mortality", "in_hospital_mortality", "readmission", "los"}
        outcome_type = llm_result.get("outcome_type")
        follow_up = llm_result.get("outcome_follow_up_days")
        if outcome_type not in supported:
            outcome_type = None
            if "in-hospital mortality" in query or "hospital mortality" in query:
                outcome_type = "in_hospital_mortality"
            elif any(t in query for t in ["mortality", "death", "survival"]):
                outcome_type = "mortality"
            elif "readmission" in query:
                outcome_type = "readmission"
            elif "length of stay" in query:
                outcome_type = "los"
        if outcome_type is None:
            return None
        if follow_up is None and outcome_type in ("mortality", "readmission"):
            day_match = re.search(r"(\d+)[\s-]day", query)
            follow_up = int(day_match.group(1)) if day_match else 30
        return OutcomeSpecification(outcome_type=outcome_type, follow_up_end=int(follow_up) if follow_up else None)

    @staticmethod
    def _extract_treatment_specification(query, llm_result):
        treatment_name = (llm_result.get("treatment_name") or "").strip()
        control_name = (llm_result.get("control_name") or "").strip()
        if not treatment_name:
            m = re.search(r"treated?\s+with\s+([a-z0-9\-]+)", query) or re.search(r"of\s+([a-z0-9\-]+)\s+(?:on|in|for)\b", query)
            if not m:
                return None
            treatment_name = m.group(1)
        treatment_name = _drug_name(treatment_name)
        control_meds = None
        # The comparator is "not treated" unless an active drug other than the treatment is named.
        untreated = any(term in f"{control_name.lower()} " for term in NO_COMPARATOR_TERMS) or \
            treatment_name in control_name.lower()
        if control_name and not untreated:
            control_meds = [_drug_name(control_name)]
        window = llm_result.get("treatment_window_days")
        return TreatmentSpecification(
            treatment_medications=[treatment_name],
            control_medications=control_meds,
            treatment_window_days=float(window) if window else None,
        )


class MIMICPreprocessor:
    """Builds analysis-ready cohorts from MIMIC-IV (or MIMIC-IV-formatted) tables."""

    def __init__(self, mimic_path: str, covariate_csv_path: str, llm=None, col_info: Optional[dict] = None,
                 data_source: str = "MIMIC-IV"):
        self.mimic_path = mimic_path
        self.covariate_csv_path = covariate_csv_path
        self.data_source = data_source
        self.llm = llm
        self.query_parser = QueryParser(llm=llm)
        self.col_info = dict(MIMIC_COLUMN_INFO if col_info is None else col_info)
        self.patients_df = self.admissions_df = self.diagnoses_df = None
        self.prescriptions_df = self.icustays_df = self.covariates_df = None

    def load_data(self):
        logger.info("Loading MIMIC-IV tables from %s", self.mimic_path)
        read = lambda rel, **kw: pd.read_csv(os.path.join(self.mimic_path, rel), **kw)
        self.patients_df = read("hosp/patients.csv.gz")
        self.admissions_df = read("hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
        self.diagnoses_df = read("hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
        self.prescriptions_df = read("hosp/prescriptions.csv.gz", usecols=["subject_id", "hadm_id", "starttime", "drug"],
                                     parse_dates=["starttime"], low_memory=False)
        self.icustays_df = read("icu/icustays.csv.gz", parse_dates=["intime"])
        self.covariates_df = pd.read_csv(self.covariate_csv_path)
        # Index admission: the first hospital admission of each patient.
        self.index_admissions = (self.admissions_df.sort_values("admittime")
                                 .groupby("subject_id", as_index=False).first())
        logger.info("Loaded %d patients and %d admissions.", len(self.patients_df), len(self.admissions_df))

    # ------------------------------------------------------------------ main entry point
    def process_query(self, query: str, max_patient_num: int = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        analysis_type, eligibility, outcome_spec, treatment_spec = self.query_parser.parse_query(query)
        logger.info("Analysis type: %s | outcome: %s | treatment: %s", analysis_type.value, outcome_spec, treatment_spec)

        cohort = self._select_patients(eligibility)
        cohort = cohort.merge(self.index_admissions[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                                                     "hospital_expire_flag"]], on="subject_id", how="inner")
        logger.info("Selected %d patients meeting eligibility criteria.", len(cohort))

        df = self._attach_covariates(cohort)
        covariate_cols = [c for c in self.covariates_df.columns
                          if c not in ("subject_id", "hadm_id", "stay_id") and c in df.columns]

        outcome_col = time_col = None
        if outcome_spec is not None:
            df = df.merge(self._extract_outcome(cohort, outcome_spec), on="subject_id", how="inner")
            outcome_col = "outcome"
            time_col = "time_to_event" if "time_to_event" in df.columns else None

        treatment_col = None
        if treatment_spec is not None:
            treatment_df = self._extract_treatment(cohort, treatment_spec)
            df = df.merge(treatment_df, on="subject_id", how="inner")
            treatment_col = "treatment"

        df = df.drop(columns=["admittime", "dischtime", "deathtime", "hospital_expire_flag"], errors="ignore")
        if max_patient_num is not None and len(df) > max_patient_num:
            df = df.sample(n=max_patient_num, random_state=42)
        df = df.reset_index(drop=True)

        metadata = {
            "data_source": self.data_source,
            "time_zero": "admission time of the index (first) hospital admission",
            "covariate_description": "first-day ICU measurements of the first ICU stay of the index admission",
            "analysis_type": analysis_type.value,
            "col_id": "subject_id",
            "col_patient_id": "subject_id",
            "col_outcome": outcome_col,
            "col_treatment": treatment_col,
            "col_treatment_time": "treatment_time" if treatment_col else None,
            "col_time_to_event": time_col,
            "cols_covariates": covariate_cols,
            "col_info": {c: self.col_info.get(c, c.replace("_", " ").capitalize() + ".") for c in df.columns},
            "query": query,
            "n_patients": len(df),
            "eligibility_criteria": {
                "age_min": eligibility.age_min, "age_max": eligibility.age_max,
                "diagnoses_include": eligibility.diagnoses_include, "diagnoses_exclude": eligibility.diagnoses_exclude,
                "icu_stay_required": eligibility.icu_stay_required,
            },
            "outcome_definition": {"type": outcome_spec.outcome_type, "follow_up_days": outcome_spec.follow_up_end}
            if outcome_spec else None,
            "treatment_definition": {
                "treatment": treatment_spec.treatment_medications,
                "control": treatment_spec.control_medications or "not treated during the index admission",
                "treatment_window_days": treatment_spec.treatment_window_days,
            } if treatment_spec else None,
        }
        logger.info("Final dataframe: %d patients x %d columns.", df.shape[0], df.shape[1])
        return df, metadata

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clean_codes(codes):
        return [re.sub(r"[^a-z0-9]", "", str(code).lower()) for code in codes if str(code).strip()]

    def _patients_with_codes(self, codes):
        clean = self._clean_codes(codes)
        if not clean:
            return set()
        df_codes = self.diagnoses_df["icd_code"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        pattern = "^(?:" + "|".join(map(re.escape, clean)) + ")"
        return set(self.diagnoses_df.loc[df_codes.str.contains(pattern, regex=True, na=False), "subject_id"])

    def _select_patients(self, criteria: EligibilityCriteria) -> pd.DataFrame:
        cohort = self.patients_df
        if criteria.age_min is not None:
            cohort = cohort[cohort["anchor_age"] >= criteria.age_min]
        if criteria.age_max is not None:
            cohort = cohort[cohort["anchor_age"] <= criteria.age_max]
        if criteria.gender:
            cohort = cohort[cohort["gender"].isin(criteria.gender)]
        for codes in criteria.diagnoses_include:
            cohort = cohort[cohort["subject_id"].isin(self._patients_with_codes(codes))]
        if criteria.diagnoses_exclude:
            cohort = cohort[~cohort["subject_id"].isin(self._patients_with_codes(criteria.diagnoses_exclude))]
        if criteria.icu_stay_required:
            icu_index = self.icustays_df.merge(self.index_admissions[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"])
            cohort = cohort[cohort["subject_id"].isin(icu_index["subject_id"])]
        return cohort[["subject_id", "anchor_age", "gender"]].copy()

    def _attach_covariates(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Attach first-day covariates of the first ICU stay of the index admission (numeric columns and gender)."""
        cov = self.covariates_df.merge(cohort[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"], how="inner")
        if "stay_id" in cov.columns:
            first_stays = (self.icustays_df.sort_values("intime").groupby("hadm_id", as_index=False).first()["stay_id"])
            cov = cov[cov["stay_id"].isin(first_stays)]
        cov = cov.drop_duplicates(subset="subject_id")
        keep = [c for c in cov.columns if c in ("subject_id", "stay_id", "gender") or pd.api.types.is_numeric_dtype(cov[c])]
        cov = cov[keep].drop(columns=["hadm_id"], errors="ignore")
        if "gender" in cov.columns:
            cohort = cohort.drop(columns=["gender"])
        return cohort.merge(cov, on="subject_id", how="left")

    def _extract_outcome(self, cohort: pd.DataFrame, spec: OutcomeSpecification) -> pd.DataFrame:
        out = cohort[["subject_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"]].copy()
        if spec.outcome_type == "in_hospital_mortality":
            out["outcome"] = out["hospital_expire_flag"].astype(int)
            out["time_to_event"] = (out["dischtime"] - out["admittime"]).dt.total_seconds() / 86400
        elif spec.outcome_type == "mortality":
            follow_up = spec.follow_up_end or 30
            # In-hospital deaths have an exact time (deathtime); other deaths only a date (dod), taken as midday.
            dod = pd.to_datetime(out["subject_id"].map(self.patients_df.set_index("subject_id")["dod"]))
            death = out["deathtime"].where(out["deathtime"].notna(), dod + pd.Timedelta(hours=12))
            days_to_death = (death - out["admittime"]).dt.total_seconds() / 86400
            days_to_death = days_to_death.mask(days_to_death < 0, 0.01)
            died = days_to_death.notna() & (days_to_death <= follow_up)
            out["outcome"] = died.astype(int)
            out["time_to_event"] = np.where(died, days_to_death, float(follow_up))
        elif spec.outcome_type == "readmission":
            follow_up = spec.follow_up_end or 30
            later = self.admissions_df.merge(out[["subject_id", "dischtime"]], on="subject_id", suffixes=("", "_index"))
            later = later[later["admittime"] > later["dischtime_index"]]
            first_readmit = later.groupby("subject_id")["admittime"].min()
            gap = (out["subject_id"].map(first_readmit) - out["dischtime"]).dt.total_seconds() / 86400
            readmitted = gap.notna() & (gap <= follow_up)
            out["outcome"] = readmitted.astype(int)
            out["time_to_event"] = np.where(readmitted, gap, float(follow_up))
            # Patients who died in hospital cannot be readmitted.
            out = out[out["hospital_expire_flag"] == 0]
        elif spec.outcome_type == "los":
            out["outcome"] = (out["dischtime"] - out["admittime"]).dt.total_seconds() / 86400
        else:
            raise ValueError(f"Outcome type '{spec.outcome_type}' is not supported by the MIMIC-IV preprocessor.")
        cols = ["subject_id", "outcome"] + (["time_to_event"] if "time_to_event" in out.columns else [])
        return out[cols]

    def _first_start(self, cohort: pd.DataFrame, drugs: List[str]) -> pd.Series:
        """Days from admission to the first prescription of any of ``drugs`` during the index admission."""
        pattern = "|".join(map(re.escape, drugs))
        rx = self.prescriptions_df[self.prescriptions_df["drug"].str.contains(pattern, case=False, na=False)]
        rx = rx.merge(cohort[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="inner")
        rx["days"] = ((rx["starttime"] - rx["admittime"]).dt.total_seconds() / 86400).clip(lower=0)
        return rx.groupby("subject_id")["days"].min()

    def _extract_treatment(self, cohort: pd.DataFrame, spec: TreatmentSpecification) -> pd.DataFrame:
        start = self._first_start(cohort, spec.treatment_medications)
        out = cohort[["subject_id"]].copy()
        out["treatment_time"] = out["subject_id"].map(start)
        out["treatment"] = out["treatment_time"].notna().astype(int)
        if spec.control_medications:
            # Active comparator: keep patients exposed to exactly one of the two strategies.
            control_start = out["subject_id"].map(self._first_start(cohort, spec.control_medications))
            exposed_control = control_start.notna()
            keep = (out["treatment"] == 1) ^ exposed_control
            out = out[keep]
        return out


TASK_TYPE_REQUIREMENTS = {
    "Causal / TTE": {"required": ["outcome", "treatment", "population", "timeframe"],
                     "description": "Treatment effect on an outcome with time-to-event analysis"},
    "Prediction / Prognosis": {"required": ["outcome", "population", "predictors"],
                               "description": "Predictive model for an outcome using specified features"},
    "Descriptive / Cohort Characterization": {"required": ["population"],
                                              "description": "Describe characteristics of a patient cohort"},
    "Clustering / Phenotyping": {"required": ["population", "features"],
                                 "description": "Identify patient subtypes based on features"},
    "Similarity / Patient Matching": {"required": ["population", "matching_features"],
                                      "description": "Find similar patients based on characteristics"},
}


def process_and_complete_query(query: str, llm) -> dict:
    """
    Check whether a query specifies all components required for an RWD analysis and,
    where reasonable medical defaults exist (e.g., timeframe, comparator), complete it.
    """
    task_types_desc = "\n".join(
        f"{i + 1}. **{name}**: {info['description']}\n   - Required components: {', '.join(info['required'])}"
        for i, (name, info) in enumerate(TASK_TYPE_REQUIREMENTS.items())
    )
    prompt = f"""You are an expert in real-world data (RWD) analysis query formulation.

Your task: Analyze the user's query and determine if it is complete enough to run an actual RWD analysis.
Each component must be SPECIFICALLY and PRECISELY defined, not just mentioned as a generic keyword.

=== SUPPORTED TASK TYPES ===
{task_types_desc}

=== USER QUERY ===
"{query}"

=== COMPONENT SPECIFICITY ===
- TREATMENT must name a specific intervention (e.g., "norepinephrine"), not "treatment" or "therapy".
- OUTCOME must name a measurable endpoint (e.g., "30-day mortality", "hospital readmission", "length of stay").
- POPULATION must define a patient group with a disease/condition or clinical setting.
- TIMEFRAME must specify a concrete period (e.g., "30-day", "in-hospital").

=== COMPLETION RULES ===
1. Components that CAN be completed with reasonable defaults: timeframe ("30-day" for mortality), population setting
   ("hospitalized patients" or "ICU patients" if a disease is given), comparator ("standard care" if the treatment is specific),
   predictors ("demographics, vital signs, lab values").
2. Components that CANNOT be completed: the treatment, the disease of interest, the diagnostic test.
3. ALWAYS preserve all information from the original query.

Return a JSON object with this exact structure:
{{
    "identified_type": "<one of the task type names above>",
    "missing_components": ["<required components missing or too vague in the original query>"],
    "completed_query": "<the completed query, or the original query if it cannot be completed>",
    "completed_missing": ["<components still missing after completion>"],
    "completion_reasoning": "<explanation>"
}}

Return ONLY valid JSON, no markdown or extra text."""
    analysis = parse_json_response(llm.invoke(prompt)) if llm is not None else None
    if not isinstance(analysis, dict):
        return {"completed_query": query, "query_type": "Unknown", "missing_info": [], "missing_after_completion": [],
                "analysis": None, "method": "unchanged"}
    return {
        "completed_query": analysis.get("completed_query") or query,
        "query_type": analysis.get("identified_type", "Unknown"),
        "missing_info": analysis.get("missing_components", []),
        "missing_after_completion": analysis.get("completed_missing", []),
        "analysis": analysis,
        "method": "llm",
    }
