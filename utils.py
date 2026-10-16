"""Shared helpers: robust parsing of LLM output and the ClinicalTrials.gov client."""

import json
import re

import pandas as pd
import requests


def parse_json_response(text, default=None):
    """
    Parse a JSON object or array from an LLM response.

    Handles markdown code fences (```json ... ```), leading/trailing prose and
    responses that are already Python objects. Returns ``default`` on failure.
    """
    if isinstance(text, (dict, list)):
        return text
    if text is None:
        return default
    text = str(text).strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Fall back to the outermost {...} or [...] block.
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start, end = candidate.find(open_char), candidate.rfind(close_char)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return default


def to_float(value):
    """Convert numbers or numeric strings (e.g., '0.82', '0.82 (95% CI...)') to float; None otherwise."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def flatten_dict(d, parent_key="", sep="_"):
    """Recursively flatten a nested dictionary (lists of strings are joined)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ", ".join(map(str, v)) if all(isinstance(i, str) for i in v) else str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"

CTGOV_COLUMNS = {
    "protocolSection_identificationModule_nctId": "NCT ID",
    "protocolSection_identificationModule_briefTitle": "Brief Title",
    "protocolSection_identificationModule_officialTitle": "Official Title",
    "protocolSection_statusModule_overallStatus": "Overall Status",
    "protocolSection_statusModule_whyStopped": "Reason Stopped",
    "protocolSection_descriptionModule_briefSummary": "Brief Summary",
    "protocolSection_conditionsModule_conditions": "Conditions",
    "protocolSection_designModule_studyType": "Study Type",
    "protocolSection_designModule_phases": "Study Phases",
    "protocolSection_designModule_designInfo_allocation": "Allocation Type",
    "protocolSection_designModule_enrollmentInfo_count": "Enrollment Count",
    "protocolSection_armsInterventionsModule_interventions": "Interventions",
    "protocolSection_outcomesModule_primaryOutcomes": "Primary Outcomes",
    "protocolSection_outcomesModule_secondaryOutcomes": "Secondary Outcomes",
    "protocolSection_eligibilityModule_eligibilityCriteria": "Eligibility Criteria",
    "hasResults": "Has Results",
}


def get_clinical_trials_data(essex_disease_str, essex_trt_str, other_keywords_str=None, outcome=None, page_size=1000):
    """
    Query the ClinicalTrials.gov v2 API (Essie expression syntax) and return a
    DataFrame with one row per trial plus a 'string' column summarizing the record.
    """
    params = {"query.cond": essex_disease_str, "query.intr": essex_trt_str, "format": "json",
              "pageSize": page_size, "sort": "@relevance"}
    if other_keywords_str:
        params["query.term"] = other_keywords_str
    if outcome:
        params["query.outc"] = outcome
    response = requests.get(CTGOV_URL, params=params, timeout=60)
    response.raise_for_status()
    studies = response.json().get("studies", [])

    df = pd.DataFrame([flatten_dict(study) for study in studies])
    if df.empty:
        return pd.DataFrame(columns=list(CTGOV_COLUMNS.values()) + ["string"])
    df = df.rename(columns=CTGOV_COLUMNS)
    df = df[[col for col in CTGOV_COLUMNS.values() if col in df.columns]]
    df["string"] = df.apply(
        lambda row: "".join(f"{col}: {row[col]}\n" for col in df.columns if pd.notna(row[col])), axis=1
    )
    return df
