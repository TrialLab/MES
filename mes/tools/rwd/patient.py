"""
A described patient → characteristics of the cohort, for "patients like this" questions: "a 68-year-old man with
septic shock, lactate 6, creatinine 2.5, on norepinephrine" → age 68, sex M, septic shock (required), highest lactate
6, highest creatinine 2.5, norepinephrine in the first 24 h. Read from the question with patterns (numbers are not left
to the language model), the model's own reading of the patient merged in by the study designer.

    cue(question)                 -> the question describes one patient (not a population)
    parse(question, base)         -> [{concept, value, weight, must}]
"""

import re

from mes.tools.rwd import library as L

# One patient, not a population: "a 68-year-old", "my patient", "patients like this one", "similar patients"
CUE = re.compile(
    r"\b(?:an?|my|our|this|the)\s+\d{1,3}[- ]?(?:year|yr)s?[- ]?old\b(?!\s+(?:patients|adults|people|men|women))"
    r"|\b(?:my|our|this)\s+patient\b|\bpatients?\s+(?:like|similar\s+to)\s+(?:mine|this(?:\s+one|\s+patient)?|him|her|"
    r"my\s+patient|ours)\b|\bsimilar\s+patients\b|\bpatient\s+similarity\b|\b(?:patients|people)\s+like\s+me\b"
    r"|\b\d{1,3}\s?(?:yo|y/o|y\.o\.)\b|相似(?:的)?(?:患者|病人)|类似(?:的)?(?:患者|病人)|这(?:个|位)(?:患者|病人)"
    r"|我的(?:患者|病人)|\d{1,3}\s*岁(?:的)?(?:男|女)", re.I)

AGE = re.compile(r"\b(\d{1,3})[- ]?(?:year|yr)s?[- ]?old\b|\baged?\s+(\d{1,3})\b|\b(\d{1,3})\s?(?:yo|y/o|y\.o\.)\b"
                 r"|(\d{1,3})\s*岁", re.I)
MALE = re.compile(r"\b(?:man|male|gentleman|boy|he|his|him)\b|男", re.I)
FEMALE = re.compile(r"\b(?:woman|female|lady|girl|she|her)\b|女", re.I)

NUM = r"(?:\s*(?:of|is|was|at|=|:|~|around|about|level|count|score|value|\(|≈))*\s*(-?\d+(?:\.\d+)?)"
# variable: the words that name it before its number (the first variable whose words match wins)
NUMBERS = {
    "sofa": r"sofa(?:\s+score)?|序贯器官衰竭评分",
    "sapsii": r"saps(?:[- ]?ii|[- ]?2)?",
    "oasis": r"oasis",
    "gcs_min": r"gcs|glasgow(?:\s+coma\s+scale)?",
    "charlson": r"charlson(?:\s+(?:index|score))?",
    "lactate_max": r"lactate|lactic\s+acid|乳酸",
    "creatinine_max": r"creatinine|scr|cr|肌酐",
    "bun_max": r"bun|blood\s+urea(?:\s+nitrogen)?|urea|尿素氮",
    "wbc_max": r"wbc|white\s+(?:blood\s+)?cells?|white\s+count|leukocytes?|白细胞",
    "hemoglobin_min": r"ha?emoglobin|hgb|hb|血红蛋白",
    "platelets_min": r"platelets?|plt|血小板",
    "bilirubin_total_max": r"(?:total\s+)?bilirubin|t\.?\s?bili|胆红素",
    "potassium_max": r"potassium|k\+",
    "glucose_max": r"glucose|blood\s+sugar|血糖",
    "albumin_min": r"albumin|白蛋白",
    "inr_max": r"inr",
    "bicarbonate_min": r"bicarbonate|bicarb|hco3",
    "ph_min": r"ph",
    "pao2fio2ratio_min": r"p/f(?:\s+ratio)?|pf\s+ratio|pao2\s*/\s*fio2",
    "heart_rate_mean": r"heart\s+rate|hr|pulse|心率",
    "mbp_min": r"map|mean\s+arterial\s+(?:blood\s+)?pressure|平均动脉压",
    "resp_rate_mean": r"respiratory\s+rate|resp(?:iratory)?\s+rate|rr|呼吸频率",
    "spo2_min": r"spo2|sp02|o2\s+sat(?:uration)?|oxygen\s+saturation",
    "temperature_max": r"temperature|temp|体温",
    "urine_output_24h": r"urine\s+output|uop?",
}
# "sodium 128" is low sodium, "sodium 150" high: the variable follows the value
SODIUM = re.compile(r"(?<![A-Za-z])(?:sodium|na|血钠|钠)(?![A-Za-z])" + NUM, re.I)
WEIGHT = re.compile(r"\b(\d{2,3}(?:\.\d+)?)\s?kg\b|\bweigh(?:s|ing|t)?(?:\s+of)?\s+(\d{2,3}(?:\.\d+)?)", re.I)

# variable: words that say the patient has it ("without …", "no …" before them say the opposite)
FLAGS = {
    "septic_shock": r"septic\s+shock|感染性休克|脓毒性休克|脓毒症休克",
    "sepsis3": r"\bsepsis\b|\bseptic\b(?!\s+shock)|脓毒症",
    "invasive_vent_24h": r"intubated|mechanically\s+ventilated|on\s+(?:a\s+)?(?:the\s+)?ventilator|mechanical\s+ventilation|"
                         r"invasive\s+ventilation|机械通气|插管",
    "vasopressor_24h": r"on\s+(?:vaso)?pressors?|vasopressors?|pressor\s+support|血管活性药",
    "norepinephrine_24h": r"norepinephrine|noradrenaline|levophed|去甲肾上腺素",
    "vasopressin_24h": r"\bvasopressin\b|血管加压素",
    "rrt_24h": r"dialysis|crrt|cvvhd?f?|renal\s+replacement|hemofiltration|透析",
    "congestive_heart_failure": r"heart\s+failure|\bchf\b|hfref|hfpef|心衰|心力衰竭",
    "chronic_pulmonary_disease": r"\bcopd\b|chronic\s+(?:obstructive\s+)?(?:pulmonary|lung)\s+disease|emphysema|慢阻肺",
    "renal_disease": r"\bckd\b|\besrd\b|chronic\s+kidney\s+disease|chronic\s+renal|慢性肾",
    "malignant_cancer": r"cancer|malignan\w*|carcinoma|lymphoma|leuka?emia|tumou?r|癌|肿瘤",
    "metastatic_solid_tumor": r"metasta\w+|转移",
    "severe_liver_disease": r"cirrhosis|liver\s+failure|end[- ]stage\s+liver|肝硬化",
    "myocardial_infarct": r"myocardial\s+infarction|\bheart\s+attack\b|\bnstemi\b|\bstemi\b|心梗|心肌梗死",
    "cerebrovascular_disease": r"\bstroke\b|cerebrovascular|脑卒中|中风",
    "dementia": r"dementia|痴呆",
    "aids": r"\baids\b|\bhiv\b",
}
NEGATION = re.compile(r"(?:\bno\b|\bnot\b|\bwithout\b|\bnon[- ]?|\bdenies\b|\bnever\b|无|没有|未)\s*(?:\w+\s+){0,2}$", re.I)
# conditions that define who is like the patient: matched exactly (the others are weighed)
REQUIRED = {"septic_shock", "sepsis3"}
WEIGHTS = {"gender": 0.5, "age": 1.0, "careunit": 0.5}


def cue(question):
    """The question describes one patient (a "patients like this" question)."""
    return bool(CUE.search(str(question or "")))


def _negated(text, start):
    return bool(NEGATION.search(text[max(0, start - 30):start]))


def parse(question, base="icu", leave_out=()):
    """The characteristics of the patient the question describes (cohort variables of ``base``), each once:
    [{concept, value, weight, must}]. ``leave_out``: variables not to read (e.g. the flag of the treatment the question
    asks about: similar patients are found with and without it)."""
    text = str(question or "")
    variables = L.variables_for(base)
    found = {}

    def put(name, value, must=False):
        if name in variables and name not in leave_out and name not in found:
            found[name] = {"concept": f"var:{name}", "value": value, "weight": WEIGHTS.get(name, 1.0), "must": must}

    m = AGE.search(text)
    if m:
        age = int(next(g for g in m.groups() if g))
        if 18 <= age <= 110:
            put("age", float(age))
    if MALE.search(text) and not FEMALE.search(text):
        put("gender", "M")
    elif FEMALE.search(text) and not MALE.search(text):
        put("gender", "F")
    for name, words in NUMBERS.items():
        # not inside a longer word ("cr" in "crrt"); Chinese names stand right before their number ("乳酸6")
        for m in re.finditer(rf"(?<![A-Za-z/])(?:{words})(?![A-Za-z])" + NUM, text, re.I):
            value = float(m.group(1))
            if name == "ph_min" and not 6.5 <= value <= 8.0:
                continue                                  # "ph" also starts other words and numbers
            put(name, value)
            break
    m = SODIUM.search(text)
    if m:
        value = float(m.group(1))
        put("sodium_min" if value < 140 else "sodium_max", value)
    m = WEIGHT.search(text)
    if m:
        put("weight", float(next(g for g in m.groups() if g)))
    shock = False
    for name, words in FLAGS.items():
        m = re.search(words, text, re.I)
        if not m:
            continue
        value = not _negated(text, m.start())
        if name == "septic_shock" and value:
            shock = True
        if name == "sepsis3" and shock:
            continue                                      # septic shock is Sepsis-3 already
        put(name, 1 if value else 0, must=name in REQUIRED and value)
    return list(found.values())


# short names of characteristics for titles: "68-year-old man · SOFA 10 · lactate 6.2 · ventilated"
SHORT = {"sofa": "SOFA", "sapsii": "SAPS II", "oasis": "OASIS", "gcs_min": "GCS", "charlson": "Charlson",
         "lactate_max": "lactate", "creatinine_max": "creatinine", "bun_max": "BUN", "wbc_max": "WBC",
         "hemoglobin_min": "hemoglobin", "platelets_min": "platelets", "bilirubin_total_max": "bilirubin",
         "sodium_min": "sodium", "sodium_max": "sodium", "potassium_max": "potassium", "glucose_max": "glucose",
         "albumin_min": "albumin", "inr_max": "INR", "bicarbonate_min": "bicarbonate", "ph_min": "pH",
         "pao2fio2ratio_min": "P/F", "heart_rate_mean": "heart rate", "mbp_min": "MAP", "mbp_mean": "MAP",
         "resp_rate_mean": "resp. rate", "spo2_min": "SpO2", "temperature_max": "temperature", "weight": "weight",
         "urine_output_24h": "urine output", "septic_shock": "septic shock", "sepsis3": "sepsis",
         "invasive_vent_24h": "ventilated", "vasopressor_24h": "on vasopressors", "norepinephrine_24h": "on norepinephrine",
         "vasopressin_24h": "on vasopressin", "rrt_24h": "on RRT", "congestive_heart_failure": "heart failure",
         "chronic_pulmonary_disease": "COPD", "renal_disease": "CKD", "malignant_cancer": "cancer",
         "metastatic_solid_tumor": "metastatic cancer", "severe_liver_disease": "cirrhosis",
         "myocardial_infarct": "prior MI", "cerebrovascular_disease": "prior stroke", "dementia": "dementia",
         "aids": "AIDS", "careunit": "ICU"}


def short(profile):
    """The patient in a few words: "68-year-old man · SOFA 10 · lactate 6.2 · ventilated"."""
    feats = {str(f.get("concept") or "")[4:]: f.get("value") for f in profile or [] if isinstance(f, dict)}
    bits = []
    age, sex = feats.pop("age", None), feats.pop("gender", None)
    person = {"M": "man", "F": "woman"}.get(str(sex), "")
    if isinstance(age, (int, float)) and not isinstance(age, bool):
        bits.append(f"{age:g}-year-old" + (f" {person}" if person else ""))
    elif person:
        bits.append(person)
    for name, value in feats.items():
        label = SHORT.get(name) or L.VARIABLES.get(name, (name,))[0].split(",")[0]
        kind = L.VARIABLES.get(name, ("", "numeric"))[1]
        if kind == "binary":
            bits.append(label if value in (1, True, "1", "yes") else f"not {label}" if label.startswith("on ")
                        else f"no {label}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            bits.append(f"{label} {value:g}")
        else:
            bits.append(f"{label} {value}")
    return " · ".join(bits)


def merge(parsed, resolved):
    """The characteristics read from the question, completed with those the model resolved (the question's numbers
    win: they were read, not inferred)."""
    out = {f["concept"]: f for f in parsed}
    for f in resolved:
        if isinstance(f, dict) and f.get("concept") and f["concept"] not in out:
            out[f["concept"]] = f
    return list(out.values())
