"""
Building blocks of RWD study designs: populations, variables, outcomes, covariate sets, curated phenotypes
(ICD-9 + ICD-10 code sets) and drug classes (ingredient names). The study designer can only choose from these
and from the concept dictionary, so it cannot invent codes or columns.

Codes are stored without dots, as in MIMIC-IV ("I4891", "42731"); a code set entry matches by prefix and may use
SQL LIKE wildcards ("250_0" = ICD-9 250.x0, type 2 diabetes).
"""

BASES = {
    "icu": {"label": "Adult ICU patients (first ICU stay of each patient)", "table": "mes.icu_stays", "id": "stay_id",
            "unit": "ICU stay", "where": "first_icu_stay AND age >= 18", "t0": "intime", "t0_label": "ICU admission"},
    "hospital": {"label": "Adult hospital admissions (first admission of each patient)", "table": "mes.admissions",
                 "id": "hadm_id", "unit": "admission", "where": "first_admission AND age >= 18", "t0": "admittime",
                 "t0_label": "hospital admission"},
}

BOTH, ICU = ("icu", "hospital"), ("icu",)

# name: (label, kind, unit, bases, group)
VARIABLES = {
    "age": ("Age", "numeric", "years", BOTH, "demographics"),
    "gender": ("Sex", "category", "", BOTH, "demographics"),
    "race_group": ("Race / ethnicity", "category", "", BOTH, "demographics"),
    "admission_type": ("Admission type", "category", "", BOTH, "demographics"),
    "insurance": ("Insurance", "category", "", BOTH, "demographics"),
    "anchor_year_group": ("Period (anchor year group)", "category", "", BOTH, "demographics"),
    "careunit": ("ICU type", "category", "", ICU, "demographics"),
    "ed_admission": ("Admitted through the emergency department", "binary", "", ("hospital",), "demographics"),
    "charlson": ("Charlson comorbidity index", "numeric", "points", BOTH, "comorbidity"),
    "myocardial_infarct": ("History: myocardial infarction", "binary", "", BOTH, "comorbidity"),
    "congestive_heart_failure": ("History: congestive heart failure", "binary", "", BOTH, "comorbidity"),
    "peripheral_vascular_disease": ("History: peripheral vascular disease", "binary", "", BOTH, "comorbidity"),
    "cerebrovascular_disease": ("History: cerebrovascular disease", "binary", "", BOTH, "comorbidity"),
    "dementia": ("History: dementia", "binary", "", BOTH, "comorbidity"),
    "chronic_pulmonary_disease": ("History: chronic pulmonary disease", "binary", "", BOTH, "comorbidity"),
    "rheumatic_disease": ("History: rheumatic disease", "binary", "", BOTH, "comorbidity"),
    "peptic_ulcer_disease": ("History: peptic ulcer disease", "binary", "", BOTH, "comorbidity"),
    "mild_liver_disease": ("History: mild liver disease", "binary", "", BOTH, "comorbidity"),
    "diabetes_without_cc": ("History: diabetes without complications", "binary", "", BOTH, "comorbidity"),
    "diabetes_with_cc": ("History: diabetes with complications", "binary", "", BOTH, "comorbidity"),
    "paraplegia": ("History: hemiplegia / paraplegia", "binary", "", BOTH, "comorbidity"),
    "renal_disease": ("History: renal disease", "binary", "", BOTH, "comorbidity"),
    "malignant_cancer": ("History: malignancy", "binary", "", BOTH, "comorbidity"),
    "severe_liver_disease": ("History: severe liver disease", "binary", "", BOTH, "comorbidity"),
    "metastatic_solid_tumor": ("History: metastatic solid tumor", "binary", "", BOTH, "comorbidity"),
    "aids": ("History: AIDS", "binary", "", BOTH, "comorbidity"),
    "sofa": ("SOFA score, first 24 h", "numeric", "points", ICU, "severity"),
    "sapsii": ("SAPS II, first 24 h", "numeric", "points", ICU, "severity"),
    "oasis": ("OASIS, first 24 h", "numeric", "points", ICU, "severity"),
    "gcs_min": ("Lowest GCS, first 24 h", "numeric", "points", ICU, "severity"),
    "sepsis3": ("Sepsis-3 (suspected infection + SOFA ≥ 2)", "binary", "", ICU, "syndrome"),
    "septic_shock": ("Septic shock (Sepsis-3 + vasopressor + lactate > 2 mmol/L, first 24 h)", "binary", "", ICU,
                     "syndrome"),
    "aki_stage_48h": ("Highest KDIGO AKI stage, first 48 h", "numeric", "stage", ICU, "syndrome"),
    "creatinine_max": ("Highest creatinine, first 24 h", "numeric", "mg/dL", ICU, "labs"),
    "bun_max": ("Highest BUN, first 24 h", "numeric", "mg/dL", ICU, "labs"),
    "wbc_max": ("Highest WBC, first 24 h", "numeric", "K/uL", ICU, "labs"),
    "hemoglobin_min": ("Lowest hemoglobin, first 24 h", "numeric", "g/dL", ICU, "labs"),
    "platelets_min": ("Lowest platelets, first 24 h", "numeric", "K/uL", ICU, "labs"),
    "bilirubin_total_max": ("Highest total bilirubin, first 24 h", "numeric", "mg/dL", ICU, "labs"),
    "sodium_min": ("Lowest sodium, first 24 h", "numeric", "mEq/L", ICU, "labs"),
    "sodium_max": ("Highest sodium, first 24 h", "numeric", "mEq/L", ICU, "labs"),
    "potassium_max": ("Highest potassium, first 24 h", "numeric", "mEq/L", ICU, "labs"),
    "glucose_max": ("Highest glucose, first 24 h", "numeric", "mg/dL", ICU, "labs"),
    "albumin_min": ("Lowest albumin, first 24 h", "numeric", "g/dL", ICU, "labs"),
    "inr_max": ("Highest INR, first 24 h", "numeric", "", ICU, "labs"),
    "bicarbonate_min": ("Lowest bicarbonate, first 24 h", "numeric", "mEq/L", ICU, "labs"),
    "lactate_max": ("Highest lactate, first 24 h", "numeric", "mmol/L", ICU, "labs"),
    "ph_min": ("Lowest pH, first 24 h", "numeric", "", ICU, "labs"),
    "pao2fio2ratio_min": ("Lowest PaO2/FiO2, first 24 h", "numeric", "mmHg", ICU, "labs"),
    "heart_rate_mean": ("Mean heart rate, first 24 h", "numeric", "/min", ICU, "vitals"),
    "mbp_mean": ("Mean arterial pressure (mean), first 24 h", "numeric", "mmHg", ICU, "vitals"),
    "mbp_min": ("Lowest mean arterial pressure, first 24 h", "numeric", "mmHg", ICU, "vitals"),
    "resp_rate_mean": ("Mean respiratory rate, first 24 h", "numeric", "/min", ICU, "vitals"),
    "spo2_min": ("Lowest SpO2, first 24 h", "numeric", "%", ICU, "vitals"),
    "temperature_max": ("Highest temperature, first 24 h", "numeric", "°C", ICU, "vitals"),
    "weight": ("Weight, first 24 h", "numeric", "kg", ICU, "vitals"),
    "urine_output_24h": ("Urine output, first 24 h", "numeric", "mL", ICU, "vitals"),
    "vasopressor_24h": ("Vasopressor in the first 24 h", "binary", "", ICU, "support"),
    "norepinephrine_24h": ("Norepinephrine in the first 24 h", "binary", "", ICU, "support"),
    "vasopressin_24h": ("Vasopressin in the first 24 h", "binary", "", ICU, "support"),
    "invasive_vent_24h": ("Invasive mechanical ventilation in the first 24 h", "binary", "", ICU, "support"),
    "rrt_24h": ("Renal replacement therapy in the first 24 h", "binary", "", ICU, "support"),
    "los_icu_days": ("ICU length of stay", "numeric", "days", ICU, "utilization"),
    "los_hosp_days": ("Hospital length of stay", "numeric", "days", BOTH, "utilization"),
    "hours_admit_to_icu": ("Hours from hospital admission to ICU admission", "numeric", "h", ICU, "utilization"),
    "n_icu_stays": ("Number of ICU stays in the admission", "numeric", "", ("hospital",), "utilization"),
    "any_icu": ("ICU admission during the hospital stay", "binary", "", ("hospital",), "utilization"),
}

# Other names of variables, for the concept search and for checking that a design represents a question's terms
# (Claude's planner once described the SOFA score as "severity of organ dysfunction")
VARIABLE_SYNONYMS = {
    "sofa": ["Sequential Organ Failure Assessment", "organ dysfunction", "organ failure"],
    "sapsii": ["Simplified Acute Physiology Score", "severity of illness", "illness severity"],
    "oasis": ["Oxford Acute Severity of Illness Score", "severity of illness"],
    "gcs_min": ["Glasgow Coma Scale", "level of consciousness"],
    "charlson": ["comorbidity", "comorbidities", "comorbidity burden"],
    "lactate_max": ["lactate", "lactic acid", "hyperlactatemia"],
    "creatinine_max": ["creatinine", "kidney function"],
    "anchor_year_group": ["calendar year", "year", "period", "time trend", "era"],
    "careunit": ["ICU type", "type of ICU", "care unit"],
    "age": ["older age", "elderly"],
    "gender": ["sex", "male", "female", "men", "women"],
}

# key: (label, kind, bases, column, horizon_days); kind binary | continuous; death outcomes also support time-to-event
OUTCOMES = {
    "death_hosp": ("In-hospital death", "binary", BOTH, "death_hosp", None),
    "death_icu": ("Death in the ICU", "binary", ICU, "death_icu", None),
    "death_28d": ("Death within 28 days of ICU admission", "binary", ICU, "death_28d", 28),
    "death_30d": ("Death within 30 days of hospital admission", "binary", ("hospital",), "death_30d", 30),
    "death_90d": ("Death within 90 days", "binary", BOTH, "death_90d", 90),
    "death_1y": ("Death within 1 year", "binary", BOTH, "death_1y", 365),
    "aki_7d": ("Acute kidney injury (KDIGO stage ≥ 1) within 7 days of ICU admission", "binary", ICU, "aki_7d", None),
    "aki_stage2_7d": ("AKI KDIGO stage ≥ 2 within 7 days of ICU admission", "binary", ICU, "aki_stage2_7d", None),
    "rrt_any": ("Renal replacement therapy during the ICU stay", "binary", ICU, "rrt_any", None),
    "invasive_vent_any": ("Invasive mechanical ventilation during the ICU stay", "binary", ICU, "invasive_vent_any",
                          None),
    "icu_readmit_30d": ("ICU readmission within 30 days (survivors of the stay)", "binary", ICU, "icu_readmit_30d",
                        None),
    "hosp_readmit_30d": ("Hospital readmission within 30 days of discharge (survivors)", "binary", BOTH,
                         "hosp_readmit_30d", None),
    "los_icu_days": ("ICU length of stay (days)", "continuous", ICU, "los_icu_days", None),
    "los_hosp_days": ("Hospital length of stay (days)", "continuous", BOTH, "los_hosp_days", None),
}

COVARIATE_SETS = {
    "demographics": ["age", "gender", "race_group", "admission_type"],
    "comorbidity": ["charlson", "congestive_heart_failure", "chronic_pulmonary_disease", "renal_disease",
                    "malignant_cancer", "severe_liver_disease", "diabetes_with_cc"],
    "severity": ["sofa", "sapsii"],
    "labs": ["creatinine_max", "lactate_max", "wbc_max", "hemoglobin_min", "platelets_min", "bilirubin_total_max"],
    "vitals": ["heart_rate_mean", "mbp_mean"],
    "support": ["vasopressor_24h", "invasive_vent_24h", "rrt_24h"],
}
DEFAULT_COVARIATES = {
    "icu": [v for s in ("demographics", "comorbidity", "severity", "labs", "vitals", "support")
            for v in COVARIATE_SETS[s]],
    "hospital": COVARIATE_SETS["demographics"] + COVARIATE_SETS["comorbidity"] + ["ed_admission"],
}

# id: (label, ICD-10 prefixes, ICD-9 prefixes, synonyms)
PHENOTYPES = {
    "diabetes": ("Diabetes mellitus", ["E08", "E09", "E10", "E11", "E13"], ["250"], ["DM", "diabetic"]),
    "type2_diabetes": ("Type 2 diabetes", ["E11"], ["250_0", "250_2"], ["T2DM", "type II diabetes"]),
    "type1_diabetes": ("Type 1 diabetes", ["E10"], ["250_1", "250_3"], ["T1DM"]),
    "dka": ("Diabetic ketoacidosis", ["E101", "E111", "E131"], ["2501"], ["DKA"]),
    "hypertension": ("Hypertension", ["I10", "I11", "I12", "I13", "I15", "I16"], ["401", "402", "403", "404", "405"],
                     ["HTN", "high blood pressure"]),
    "heart_failure": ("Heart failure", ["I50", "I110", "I130", "I132", "I0981"],
                      ["428", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "39891"],
                      ["HF", "CHF", "congestive heart failure", "HFrEF", "HFpEF"]),
    "atrial_fibrillation": ("Atrial fibrillation or flutter", ["I48"], ["42731", "42732"], ["AF", "AFib", "atrial flutter"]),
    "coronary_artery_disease": ("Coronary artery disease", ["I20", "I21", "I22", "I23", "I24", "I25"],
                                ["410", "411", "412", "413", "414"], ["CAD", "ischemic heart disease"]),
    "myocardial_infarction": ("Acute myocardial infarction", ["I21", "I22"], ["410"], ["MI", "heart attack", "STEMI",
                                                                                     "NSTEMI"]),
    "cardiac_arrest": ("Cardiac arrest", ["I46"], ["4275"], ["arrest", "CPR"]),
    "ischemic_stroke": ("Ischemic stroke", ["I63"], ["433_1", "434_1", "436"], ["stroke", "cerebral infarction"]),
    "intracerebral_hemorrhage": ("Intracerebral hemorrhage", ["I61"], ["431"], ["ICH", "hemorrhagic stroke"]),
    "subarachnoid_hemorrhage": ("Subarachnoid hemorrhage", ["I60"], ["430"], ["SAH"]),
    "traumatic_brain_injury": ("Traumatic brain injury", ["S06"], ["850", "851", "852", "853", "854"], ["TBI"]),
    "ckd": ("Chronic kidney disease", ["N18"], ["585"], ["CKD", "chronic renal failure"]),
    "esrd": ("End-stage renal disease / dialysis dependence", ["N186", "Z992"], ["5856", "V4511"], ["ESRD", "ESKD",
                                                                                                   "dialysis"]),
    "aki_dx": ("Acute kidney injury (diagnosis code)", ["N17"], ["584"], ["acute renal failure", "ARF"]),
    "copd": ("COPD", ["J43", "J44"], ["491", "492", "496"], ["chronic obstructive pulmonary disease", "emphysema"]),
    "asthma": ("Asthma", ["J45"], ["493"], []),
    "pneumonia": ("Pneumonia", ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
                  ["480", "481", "482", "483", "484", "485", "486"], ["CAP", "HAP", "lung infection"]),
    "sepsis_dx": ("Sepsis (diagnosis code)", ["A40", "A41", "R652", "A021", "A227", "A267", "A327", "A427", "B377"],
                  ["038", "99591", "99592", "78552"], ["sepsis", "septicemia"]),   # (bacteremia: micro:blood)
    "septic_shock_dx": ("Septic shock (diagnosis code)", ["R6521"], ["78552"], ["septic shock"]),
    "ards": ("Acute respiratory distress syndrome", ["J80"], ["51882", "5185"], ["ARDS"]),
    "respiratory_failure": ("Acute respiratory failure", ["J960", "J962", "J969"], ["51881", "51882", "51884"],
                            ["respiratory failure", "hypoxemic respiratory failure"]),
    "cirrhosis": ("Cirrhosis", ["K703", "K746", "K717"], ["5712", "5715", "5716"], ["liver cirrhosis"]),
    "acute_pancreatitis": ("Acute pancreatitis", ["K85"], ["5770"], ["pancreatitis"]),
    "gi_bleeding": ("Gastrointestinal bleeding", ["K920", "K921", "K922", "K250", "K252", "K254", "K256", "K260", "K262",
                                                   "K264", "K266", "K270", "K272", "K274", "K276", "K280", "K282",
                                                   "K284", "K286", "I8501", "I8511", "K625"],
                    ["578", "5310", "5312", "5314", "5316", "5320", "5322", "5324", "5326", "5330", "5332", "5334",
                     "5336", "5340", "5342", "5344", "5346", "4560", "45620", "5693"], ["GI bleed", "GIB", "melena",
                                                                                         "hematemesis"]),
    "pulmonary_embolism": ("Pulmonary embolism", ["I26"], ["4151"], ["PE"]),
    "dvt": ("Deep vein thrombosis", ["I824"], ["4534"], ["DVT"]),
    "vte": ("Venous thromboembolism", ["I26", "I824"], ["4151", "4534"], ["VTE"]),
    "cancer": ("Malignancy", ["C"], ["14", "15", "16", "17", "18", "19", "20"], ["cancer", "malignant neoplasm",
                                                                                "tumor"]),
    "dementia_dx": ("Dementia", ["F00", "F01", "F02", "F03", "G30"], ["290", "2941", "3310"], ["Alzheimer"]),
    "delirium": ("Delirium", ["F05"], ["2930", "2931"], []),
    "obesity": ("Obesity", ["E66"], ["2780"], ["obese"]),
    "hip_fracture": ("Hip fracture", ["S720", "S721", "S722"], ["820"], ["femoral neck fracture"]),
    "covid19": ("COVID-19", ["U071"], [], ["SARS-CoV-2", "coronavirus"]),
    "hyperkalemia": ("Hyperkalemia", ["E875"], ["2767"], []),
    "hyponatremia": ("Hyponatremia", ["E871"], ["2761"], []),
    "anemia": ("Anemia", ["D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59", "D60", "D61", "D62", "D63",
                          "D64"], ["280", "281", "282", "283", "284", "285"], []),
    "hiv": ("HIV infection", ["B20"], ["042"], ["AIDS"]),
    "alcohol_use_disorder": ("Alcohol use disorder", ["F10"], ["303", "3050"], ["alcoholism", "alcohol abuse"]),
    "depression": ("Depression", ["F32", "F33"], ["2962", "2963", "311"], ["major depressive disorder"]),
    "uti": ("Urinary tract infection", ["N390"], ["5990"], ["UTI"]),
    "endocarditis": ("Endocarditis", ["I33"], ["4210"], ["infective endocarditis"]),
}

# id: (label, ingredient substrings, excluded substrings, synonyms)
_TOPICAL = ["cream", "ointment", "oint", "topical", "ophth", "otic", "gel", "drops", "lotion", "rectal", "enema",
            "suppository", "intrathecal", "foam", "pramoxine", "iodoquin", "aloe"]
# Routes of local administration (mes.drugs.route, lower case); classes that exclude _TOPICAL names count only
# other (systemic) routes, e.g. hydrocortisone foam per rectum is not a septic-shock steroid
LOCAL_ROUTES = ["tp", "pr", "vg", "it", "ih", "inhalation", "neb", "nu", "nas", "ns", "both eyes", "left eye",
                "right eye", "ou", "od", "os", "both ears", "left ear", "right ear", "ad", "as", "au", "ir", "irr",
                "id", "intralesional", "ed", "periphnerve", "lumbar plexus", "interspace", "dwell", "lock", "flush"]


def systemic_only(class_id):
    return "topical" in DRUG_CLASSES[class_id][2]


# Microbiology (mimiciv_hosp.microbiologyevents): a culture of these specimens that grew an organism — tests that were
# cancelled or grew nothing are not positive; possible contaminants (coagulase-negative staphylococci) are not removed.
# key: (label, spec_type_desc values — none: every specimen, synonyms)
MICRO = {
    "blood": ("Positive blood culture", ("BLOOD CULTURE",),
              ["bacteremia", "bloodstream infection", "blood stream infection", "positive blood culture",
               "blood culture positive", "blood culture"]),
    "urine": ("Positive urine culture", ("URINE", "URINE,KIDNEY", "URINE,SUPRAPUBIC ASPIRATE"),
              ["positive urine culture", "urine culture positive", "bacteriuria", "urine culture"]),
    "respiratory": ("Positive respiratory culture", ("SPUTUM", "BRONCHOALVEOLAR LAVAGE", "BRONCHIAL WASHINGS",
                                                     "Mini-BAL", "TRACHEAL ASPIRATE", "BRONCHIAL BRUSH",
                                                     "BRONCHIAL BRUSH - PROTECTED", "ASPIRATE"),
                    ["positive sputum culture", "positive respiratory culture", "positive BAL culture",
                     "sputum culture", "respiratory culture"]),
    "csf": ("Positive CSF culture", ("CSF;SPINAL FLUID",),
            ["positive cerebrospinal fluid culture", "CSF culture", "cerebrospinal fluid culture"]),
    "mrsa": ("Positive MRSA screen", ("MRSA SCREEN",), ["MRSA colonization", "MRSA screen", "MRSA carriage"]),
    "any": ("Any positive culture", (), ["positive culture", "culture positive", "positive microbiology culture"]),
}
NOT_ORGANISMS = ("CANCELLED", "NEGATIVE", "NO GROWTH")


def micro_predicate(key, alias="m"):
    """SQL condition on a microbiologyevents row: a culture of the specimen of ``key`` that grew an organism."""
    _label, specimens, _syn = MICRO[key]
    where = [f"{alias}.org_name IS NOT NULL",
             f"upper({alias}.org_name) NOT IN ({', '.join(repr(x) for x in NOT_ORGANISMS)})",
             f"upper({alias}.org_name) NOT LIKE '%CANCELLED%'"]
    if specimens:
        where.append(f"{alias}.spec_type_desc IN ({', '.join(repr(x) for x in specimens)})")
    return " AND ".join(where)


# Durations computed from the mimic-code episode tables of the ICU stay (NULL when the treatment was never given, so a
# distribution describes the patients who received it). key: (label, unit, SQL of the stay aliased b, synonyms)
DERIVED = {
    "vent_hours": ("Hours of invasive mechanical ventilation in the ICU stay (ventilated patients)", "h",
                   "(SELECT sum(date_diff('second', v.starttime, v.endtime)) / 3600.0 FROM mimiciv_derived.ventilation v "
                   "WHERE v.stay_id = b.stay_id AND v.ventilation_status IN ('InvasiveVent', 'Tracheostomy'))",
                   ["duration of mechanical ventilation", "hours of mechanical ventilation", "ventilation duration",
                    "time on the ventilator", "days of mechanical ventilation", "length of mechanical ventilation"]),
    "vasopressor_hours": ("Hours on vasopressors in the ICU stay (patients who received one)", "h",
                          "(SELECT sum(date_diff('second', v.starttime, v.endtime)) / 3600.0 FROM "
                          "mimiciv_derived.vasoactive_agent v WHERE v.stay_id = b.stay_id AND coalesce(v.norepinephrine, "
                          "v.vasopressin, v.epinephrine, v.phenylephrine, v.dopamine) IS NOT NULL)",
                          ["duration of vasopressors", "hours of vasopressors", "vasopressor duration",
                           "time on vasopressors", "duration of vasopressor therapy"]),
}


DRUG_CLASSES = {
    "metformin": ("Metformin", ["metformin"], [], ["biguanide"]),
    "insulin": ("Insulin", ["insulin"], [], []),
    "sglt2i": ("SGLT2 inhibitors", ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"], [],
               ["SGLT2", "gliflozin"]),
    "glp1ra": ("GLP-1 receptor agonists", ["semaglutide", "liraglutide", "dulaglutide", "exenatide"], [], ["GLP-1"]),
    "statin": ("Statins", ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "pitavastatin",
                           "fluvastatin"], [], ["HMG-CoA reductase inhibitor"]),
    "beta_blocker": ("Beta-blockers", ["metoprolol", "esmolol", "carvedilol", "atenolol", "labetalol", "propranolol",
                                       "bisoprolol", "nadolol", "nebivolol", "sotalol"], [], ["beta blocker"]),
    "esmolol": ("Esmolol", ["esmolol"], [], []),
    "acei": ("ACE inhibitors", ["lisinopril", "enalapril", "captopril", "ramipril", "benazepril", "quinapril",
                                "fosinopril", "perindopril", "trandolapril", "moexipril"], [], ["ACE inhibitor"]),
    "arb": ("Angiotensin receptor blockers", ["losartan", "valsartan", "irbesartan", "olmesartan", "candesartan",
                                              "telmisartan", "eprosartan", "azilsartan"], ["sacubitril"], ["ARB"]),
    "arni": ("Sacubitril/valsartan", ["sacubitril"], [], ["Entresto", "ARNI"]),
    "ccb": ("Calcium channel blockers", ["amlodipine", "nifedipine", "diltiazem", "verapamil", "nicardipine",
                                         "clevidipine"], [], []),
    "loop_diuretic": ("Loop diuretics", ["furosemide", "bumetanide", "torsemide", "ethacrynic"], [], ["diuretic"]),
    "heparin_parenteral": ("Heparin / LMWH / parenteral anticoagulants", ["heparin", "enoxaparin", "argatroban",
                                                                         "bivalirudin", "fondaparinux", "dalteparin"],
                           ["flush", "lock", "dwell"], ["anticoagulation", "LMWH"]),
    "doac": ("Direct oral anticoagulants", ["apixaban", "rivaroxaban", "dabigatran", "edoxaban"], [], ["DOAC", "NOAC"]),
    "apixaban": ("Apixaban", ["apixaban"], [], ["Eliquis"]),
    "warfarin": ("Warfarin", ["warfarin"], [], ["Coumadin", "vitamin K antagonist"]),
    "anticoagulant": ("Any anticoagulant", ["heparin", "enoxaparin", "argatroban", "bivalirudin", "fondaparinux",
                                            "dalteparin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
                                            "warfarin"], ["flush", "lock", "dwell"], []),
    "antiplatelet": ("Antiplatelets", ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"], [], []),
    "aspirin": ("Aspirin", ["aspirin"], [], ["ASA"]),
    "corticosteroid": ("Systemic corticosteroids", ["hydrocortisone", "methylprednisolone", "prednisone",
                                                    "dexamethasone", "prednisolone", "fludrocortisone"], _TOPICAL,
                       ["steroids", "glucocorticoid"]),
    "hydrocortisone": ("Hydrocortisone (systemic)", ["hydrocortisone"], _TOPICAL, []),
    "dexamethasone": ("Dexamethasone (systemic)", ["dexamethasone"], _TOPICAL, []),
    "ppi": ("Proton pump inhibitors", ["pantoprazole", "omeprazole", "esomeprazole", "lansoprazole", "rabeprazole"], [],
            ["PPI"]),
    "h2ra": ("H2 receptor antagonists", ["famotidine", "ranitidine", "cimetidine"], [], ["H2 blocker"]),
    "vasopressor": ("Vasopressors", ["norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"],
                    ["lidocaine", "bupivacaine", "articaine"], []),
    "norepinephrine": ("Norepinephrine", ["norepinephrine"], [], ["noradrenaline", "Levophed"]),
    "vasopressin": ("Vasopressin", ["vasopressin"], [], []),
    "epinephrine": ("Epinephrine", ["epinephrine"], ["norepinephrine", "lidocaine", "bupivacaine"], ["adrenaline"]),
    "phenylephrine": ("Phenylephrine", ["phenylephrine"], _TOPICAL, []),
    "dopamine": ("Dopamine", ["dopamine"], [], []),
    "inotrope": ("Inotropes (dobutamine, milrinone)", ["dobutamine", "milrinone"], [], []),
    "propofol": ("Propofol", ["propofol"], [], []),
    "dexmedetomidine": ("Dexmedetomidine", ["dexmedetomidine"], [], ["Precedex"]),
    "benzodiazepine": ("Benzodiazepines", ["midazolam", "lorazepam", "diazepam", "alprazolam", "clonazepam"], [], []),
    "opioid": ("Opioids", ["fentanyl", "morphine", "hydromorphone", "oxycodone", "methadone", "remifentanil"], [], []),
    "antipsychotic": ("Antipsychotics", ["haloperidol", "quetiapine", "olanzapine", "risperidone"], [], []),
    "antibiotic": ("Antibiotics", ["vancomycin", "piperacillin", "cefepime", "ceftriaxone", "meropenem", "cefazolin",
                                   "ampicillin", "levofloxacin", "ciprofloxacin", "metronidazole", "azithromycin",
                                   "linezolid", "daptomycin", "clindamycin", "gentamicin", "tobramycin", "amikacin",
                                   "ertapenem", "imipenem", "aztreonam", "ceftazidime", "cefuroxime", "cefoxitin",
                                   "doxycycline", "sulfamethoxazole", "nafcillin", "oxacillin", "penicillin",
                                   "amoxicillin", "colistin", "polymyxin", "tigecycline", "rifampin", "erythromycin",
                                   "moxifloxacin", "ceftaroline", "cefotaxime", "cephalexin", "minocycline"], _TOPICAL,
                   ["antimicrobial", "antibacterial"]),
    "vancomycin": ("Vancomycin (IV)", ["vancomycin"], ["oral", " po", "enema"] + _TOPICAL, []),
    "piperacillin_tazobactam": ("Piperacillin-tazobactam", ["piperacillin"], [], ["Zosyn", "pip-tazo"]),
    "cefepime": ("Cefepime", ["cefepime"], [], []),
    "meropenem": ("Meropenem", ["meropenem"], [], ["carbapenem"]),
    "thiamine": ("Thiamine", ["thiamine"], [], ["vitamin B1"]),
    "vitamin_c": ("Vitamin C (ascorbic acid)", ["ascorbic acid", "vitamin c"], [], []),
    "albumin": ("Albumin (IV)", ["albumin"], [], []),
    "balanced_crystalloid": ("Balanced crystalloids", ["lactated ringer", "plasmalyte", "plasma-lyte"], [],
                             ["Ringer's lactate", "LR"]),
    "tranexamic_acid": ("Tranexamic acid", ["tranexamic"], [], ["TXA"]),
    "amiodarone": ("Amiodarone", ["amiodarone"], [], []),
    "digoxin": ("Digoxin", ["digoxin"], ["immune fab"], []),
    "acetaminophen": ("Acetaminophen", ["acetaminophen"], [], ["paracetamol"]),
    "nsaid": ("NSAIDs", ["ibuprofen", "ketorolac", "naproxen", "celecoxib", "indomethacin"], [], []),
    "sodium_bicarbonate": ("Sodium bicarbonate", ["sodium bicarbonate"], [], ["bicarbonate"]),
    "magnesium": ("Magnesium sulfate", ["magnesium sulfate"], [], []),
    # blood products are ICU infusions (inputevents), rarely coded as ICD procedures
    "rbc_transfusion": ("Red blood cell transfusion", ["packed red blood cells", "packed rbc"], [],
                        ["RBC transfusion", "blood transfusion", "PRBC", "packed red cells"]),
    "platelet_transfusion": ("Platelet transfusion", ["platelets", "platelet intake"], ["inhibitor"],
                             ["platelets transfusion"]),
    "plasma_transfusion": ("Plasma transfusion (FFP)", ["fresh frozen plasma"], [], ["FFP", "plasma"]),
}

ANALYSES = {
    "compare": "Compare an exposure (treatment) with a comparator for an outcome — target trial emulation with "
               "inverse probability of treatment weighting",
    "association": "Adjusted association of a risk factor / exposure (condition, lab value, score) with an outcome",
    "rate": "Rate / incidence / prevalence / proportion of an outcome or condition in a population, optionally by subgroup",
    "distribution": "Distribution of a variable (median, IQR, histogram) in a population, optionally by subgroup",
    "utilization": "How often a treatment is used in a population and when it is started",
    "survival": "Survival curves (Kaplan-Meier) of groups for death",
    "trend": "Change of an outcome rate or variable across periods (anchor year groups)",
    "prognostic": "Which characteristics predict an outcome (multivariable model with discrimination)",
    "explore": "Other descriptive questions answered with a custom aggregate SQL query",
    "similar": "Patients like a described patient: the most similar stays, their outcomes, the treatments they "
               "received and the outcomes by treatment, and the risk a model predicts for the patient",
    "characterize": "What patients are like: the characteristics (Table 1) of a cohort, or of two groups — who died "
                    "and who survived, with and without a treatment or condition",
    "trajectory": "How a lab test or vital sign evolves after time zero (median and IQR per interval), by group",
    "pathways": "In which order treatments are started (first, second, third line) and the time to the first dose",
    "catalog": "What the database records about something (data availability) and the questions it can answer",
    "top": "The commonest medications, diagnoses or procedures of a population (the share of patients with each), "
           "for everyone or by group",
}

# What "patients like this" reports the similar stays received (first 24 h), beside the population
SIMILAR_TREATMENTS = ("vasopressor_24h", "norepinephrine_24h", "vasopressin_24h", "invasive_vent_24h", "rrt_24h")


def variables_for(base):
    return {k: v for k, v in VARIABLES.items() if base in v[3]}


def outcomes_for(base):
    return {k: v for k, v in OUTCOMES.items() if base in v[2]}
