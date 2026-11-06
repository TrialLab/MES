"""
What each concept of a study design means in MIMIC-IV: the code sets of curated conditions, the
ingredients and matched medication names of drug classes, the source of derived variables and outcomes, and the
patients each concept has. Shown in the study panel ("Definitions") so a reader can check every criterion the way a
phenotype library is checked; it also travels with the study record and its exports.

    describe(spec_dict, index) -> [{"role", "concept", "label", "kind", "detail", "codes", "medications", ...}]
"""

from mes.tools.rwd import library as L
from mes.tools.rwd.concepts import KIND_LABEL

ROLE_LABELS = {"include": "Inclusion", "exclude": "Exclusion", "exposure": "Treatment / exposure",
               "comparator": "Comparator", "outcome": "Outcome", "variable": "Variable", "stratify": "Breakdown",
               "subgroups": "Subgroup", "patient": "Patient characteristic"}

_DEATH = ("time of death from the admission record, else the date of death (patients.dod: hospital and state "
          "records, up to one year after the last discharge) at noon")
OUTCOME_SOURCES = {
    "death_hosp": "Died before hospital discharge (admissions.hospital_expire_flag).",
    "death_icu": "Died in the ICU: the time of death no later than 6 h after ICU discharge.",
    "death_28d": f"Died within 28 days of ICU admission — {_DEATH}.",
    "death_30d": f"Died within 30 days of hospital admission — {_DEATH}.",
    "death_90d": f"Died within 90 days of time zero — {_DEATH}.",
    "death_1y": f"Died within 1 year of time zero — {_DEATH}.",
    "aki_7d": "Highest KDIGO stage ≥ 1 within 7 days of ICU admission (mimic-code kdigo_stages: creatinine and urine "
              "output criteria).",
    "aki_stage2_7d": "Highest KDIGO stage ≥ 2 within 7 days of ICU admission (mimic-code kdigo_stages).",
    "rrt_any": "Renal replacement therapy active at any time of the ICU stay (mimic-code rrt).",
    "invasive_vent_any": "Invasive ventilation or tracheostomy at any time of the ICU stay (mimic-code ventilation).",
    "icu_readmit_30d": "Another ICU admission within 30 days of ICU discharge, among patients discharged alive.",
    "hosp_readmit_30d": "Another hospital admission within 30 days of discharge, among patients discharged alive.",
    "los_icu_days": "ICU discharge minus ICU admission, in days.",
    "los_hosp_days": "Hospital discharge minus hospital admission, in days.",
}
VARIABLE_SOURCES = {
    "sepsis3": "mimic-code sepsis3: suspected infection (an antibiotic with a body-fluid culture close in time) and "
               "a SOFA score ≥ 2 from 48 h before to 24 h after it.",
    "septic_shock": "Sepsis-3 and a vasopressor in the first 24 h of the ICU stay and a highest lactate above "
                    "2 mmol/L in the first 24 h.",
    "sofa": "mimic-code first_day_sofa: SOFA score of the first 24 h of the ICU stay.",
    "sapsii": "mimic-code sapsii: SAPS II of the first 24 h.",
    "oasis": "mimic-code oasis: OASIS of the first 24 h.",
    "charlson": "mimic-code charlson: Charlson comorbidity index from the admission's ICD codes.",
    "aki_stage_48h": "Highest KDIGO stage in the first 48 h of the ICU stay (mimic-code kdigo_stages).",
    "vasopressor_24h": "Any vasoactive agent (norepinephrine, epinephrine, dopamine, vasopressin, phenylephrine) "
                       "running in the first 24 h (mimic-code vasoactive_agent).",
    "norepinephrine_24h": "Norepinephrine running in the first 24 h (mimic-code vasoactive_agent).",
    "vasopressin_24h": "Vasopressin running in the first 24 h (mimic-code vasoactive_agent).",
    "invasive_vent_24h": "Invasive ventilation or tracheostomy in the first 24 h (mimic-code ventilation).",
    "rrt_24h": "Renal replacement therapy active in the first 24 h (mimic-code rrt).",
    "careunit": "The first ICU care unit of the stay, grouped (MICU, SICU, CVICU, CCU, TSICU, Neuro …).",
    "anchor_year_group": "MIMIC-IV shifts dates per patient: only the 3-year group of the real year is known.",
    "race_group": "Recorded race / ethnicity grouped as White, Black, Hispanic, Asian, Other, Unknown.",
}
_GROUP_SOURCES = {"labs": "mimic-code first_day_lab / first_day_bg: value of the first 24 h of the ICU stay.",
                  "vitals": "mimic-code first_day_vitalsign: value of the first 24 h of the ICU stay.",
                  "comorbidity": "mimic-code charlson: from the ICD codes of the admission.",
                  "demographics": "mimic-code icustay_detail / admissions.",
                  "utilization": "From the admission and ICU stay times."}


def _window(c, role):
    window = c.get("window_h")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        return ""
    lo, hi = window
    if hi is None:
        return f"from {lo:g} h to the end of the stay"
    return f"{lo:g} to {hi:g} h from time zero"


def _comparison(c):
    op, value = c.get("op") or "", c.get("value")
    if not op:
        return ""
    if op == "between" and isinstance(value, (list, tuple)) and len(value) == 2:
        return f"between {value[0]:g} and {value[1]:g}"
    shown = f"{value:g}" if isinstance(value, (int, float)) else str(value)
    return f"{op.replace('>=', '≥').replace('<=', '≤')} {shown}"


def _concept(concept, role, index, criterion=None, limit=12):
    kind, _, key = str(concept).partition(":")
    out = {"role": ROLE_LABELS.get(role, role), "concept": concept, "kind": KIND_LABEL.get(kind, kind),
           "label": "", "detail": "", "codes": {}, "medications": [], "n_medications": 0, "patients": None}
    try:
        out["label"] = index.label(concept) if index is not None else ""
    except Exception:
        out["label"] = ""
    if not out["label"] or out["label"] == concept:     # without the store: the design's own label, or the library's
        library = {"phe": L.PHENOTYPES, "drugclass": L.DRUG_CLASSES, "var": L.VARIABLES, "micro": L.MICRO}.get(kind, {})
        out["label"] = (criterion or {}).get("label") or (library[key][0] if key in library else concept)
    if kind == "phe" and key in L.PHENOTYPES:
        label, icd10, icd9, synonyms = L.PHENOTYPES[key]
        out["codes"] = {"ICD-10": list(icd10), "ICD-9": list(icd9)}
        out["detail"] = "Any diagnosis code of the admission starting with one of these codes ('_' stands for one " \
                        "character); codes are assigned at discharge."
    elif kind == "drugclass" and key in L.DRUG_CLASSES:
        label, ingredients, excluded, synonyms = L.DRUG_CLASSES[key]
        out["codes"] = {"Ingredients": list(ingredients)}
        if excluded:
            out["codes"]["Excluded names"] = list(excluded)
        out["detail"] = "Orders, administrations and ICU infusions whose name starts a word with an ingredient" + (
            "; systemic routes only (no topical, inhaled or local routes)" if L.systemic_only(key) else "") + "."
    elif kind == "drug":
        out["codes"] = {"Name contains": [key]}
        out["detail"] = "Orders, administrations and ICU infusions whose name contains the text."
    elif kind == "var" and key in L.VARIABLES:
        label, vkind, unit, _bases, group = L.VARIABLES[key]
        out["detail"] = VARIABLE_SOURCES.get(key) or _GROUP_SOURCES.get(group, "")
        if unit:
            out["unit"] = unit
    elif kind in ("dx9", "dx10", "px9", "px10"):
        out["codes"] = {f"ICD-{kind[2:]} {'diagnosis' if kind[:2] == 'dx' else 'procedure'}": [key + "*"]}
        out["detail"] = "This code and its sub-codes; ICD codes are assigned at discharge."
    elif kind in ("lab", "item"):
        out["codes"] = {"labevents itemid" if kind == "lab" else "chartevents itemid": [key]}
        out["detail"] = "A measured value of this item in the window."
    elif kind == "micro" and key in L.MICRO:
        out["codes"] = {"Specimen": list(L.MICRO[key][1]) or ["any specimen"]}
        out["detail"] = ("A culture of the specimen that grew an organism in the window (mimiciv_hosp.microbiologyevents; "
                         "cancelled tests and cultures with no growth are not positive; possible contaminants such as "
                         "coagulase-negative staphylococci are not removed).")
    if kind in ("drugclass", "drug") and index is not None:
        try:
            names = index.drug_names_for(concept)
            out["n_medications"] = len(names)
            out["medications"] = names[:limit]
        except Exception:
            pass
    if index is not None and kind != "var":
        try:
            out["patients"] = index.count(concept)
        except Exception:
            pass
    if criterion:
        bits = [b for b in (_comparison(criterion), _window(criterion, role)) if b]
        if criterion.get("negate"):
            bits.insert(0, "absent")
        out["rule"] = " · ".join(bits)
    return out


def describe(spec, index=None, limit=12):
    """The definitions of the concepts and outcome of a study design (a ``StudySpec.to_dict()``), in the order of the
    protocol: inclusion, exclusion, treatment, comparator, outcome, the analysed variable. ``index``: the concept index
    (labels, medication names, patient counts); without it only what the library itself says."""
    spec = spec or {}
    rows = []
    for role in ("include", "exclude"):
        for c in spec.get(role) or []:
            if isinstance(c, dict) and c.get("concept"):
                rows.append(_concept(c["concept"], role, index, c, limit))
    for role in ("exposure", "comparator"):
        c = spec.get(role)
        if isinstance(c, dict) and c.get("concept"):
            rows.append(_concept(c["concept"], role, index, c, limit))
    outcome = spec.get("outcome")
    if outcome in L.OUTCOMES:
        rows.append({"role": ROLE_LABELS["outcome"], "concept": f"outcome:{outcome}", "kind": "outcome",
                     "label": L.OUTCOMES[outcome][0], "detail": OUTCOME_SOURCES.get(outcome, ""), "codes": {},
                     "medications": [], "n_medications": 0, "patients": None})
    variable = spec.get("variable")
    if variable:
        rows.append(_concept(variable if ":" in variable else f"var:{variable}", "variable", index, None, limit))
    for f in spec.get("profile") or []:          # the described patient of a "patients like this" design
        if isinstance(f, dict) and str(f.get("concept") or "").startswith("var:"):
            row = _concept(f["concept"], "patient", index, None, limit)
            value = f.get("value")
            row["rule"] = ("value " + (f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool)
                                       else str(value))) + (f" · weight {f['weight']:g}" if isinstance(f.get("weight"), (int, float))
                                                            and f.get("weight") != 1 else "")
            rows.append(row)
    return rows
