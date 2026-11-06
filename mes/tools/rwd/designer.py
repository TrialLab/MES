"""
RWD study designer: question → StudySpec, with the LLM doing only what it is good at.

1. Intent (one LLM call, small JSON): analysis type, population, the population's defining conditions, exposure and
   comparator *as terms* (with thresholds), outcome from a fixed list, strata. No concept ids, no code.
2. Resolution (deterministic): each term is mapped to the best concept for its role with the concept index
   (treatments → drug classes / medications, conditions → curated flags / phenotypes / ICD codes, measurements →
   cohort variables / lab tests); the mapping is kept so the user sees and can change it.
3. Sanitizing (deterministic): contradictions and role mix-ups are fixed (a treatment repeated as an inclusion
   criterion, an exposure equal to its comparator, a characteristic "compared" like a drug, …).
4. Validation; if it fails, the model gets the errors once to correct its intent.
"""

import json
import logging
import re

from mes.tools.rwd import intent_rules as IR
from mes.tools.rwd import library as L
from mes.tools.rwd import patient as PAT
from mes.tools.rwd.spec import OPS, StudySpec, criterion_text, validate
from mes.utils import parse_json_response

logger = logging.getLogger(__name__)

PROMPT = """You plan an observational study on a hospital database to answer a clinical question. Describe the study
with plain clinical terms; do not write code or codes.

DATABASE: {coverage}
POPULATIONS (base): icu = adult ICU patients (first ICU stay; time zero = ICU admission); hospital = adult hospital
admissions (first admission; time zero = hospital admission).
ANALYSIS TYPES: compare = treatment A vs treatment B (or vs no treatment); association = a risk factor / condition /
lab value and an outcome; rate = how often an outcome or condition occurs; distribution = values of a measurement;
utilization = how often a treatment is given and when; survival = survival curves; trend = change across periods;
prognostic = which characteristics predict an outcome; similar = what happened to patients like one described patient
(their outcomes and treatments); characterize = what the patients are like (their characteristics), overall or two
groups; trajectory = how a lab value or vital sign changes over the first hours or days; pathways = in which order
treatments are started; catalog = what the database records about something (is X recorded? which data on Y?);
top = the commonest medications, diagnoses or procedures of the patients (ranked: drugs, diagnoses or procedures);
explore = any other descriptive question.
OUTCOMES: {outcomes}
VARIABLES for stratify / subgroups: {variables}
FOUND IN THE DATABASE for this question: {found}

QUESTION: {question}
{pico}{previous}
Fill in:
- feasibility: "not_feasible" if the database cannot answer (outpatient or long-term outcomes other than death,
  children, imaging, notes, drugs or tests not recorded); otherwise "feasible". reason: one sentence.
- population: terms of the conditions that define who is studied (e.g. "septic shock", "heart failure", "age" with
  op ">=" value "65"). Not the treatments being compared.
- exposure: the treatment (compare / utilization) or risk factor (association), with op/value for a threshold
  (e.g. "lactate", ">", "4"); within_hours = hours after time zero for treatments (default 24; 0 = during the
  whole stay).
- comparator: the other treatment, or "" to compare with patients who did not get the exposure.
- outcome: one key from OUTCOMES ("" for distribution / utilization). variable: the measurement for distribution.
  Never substitute a different outcome, treatment or measurement: if what the question asks about is not in the
  lists (e.g. a specific procedure, complication or score), use analysis explore.
- patient (similar only): the described patient's characteristics, e.g. "age" = "68", "sex" = "male", "lactate" = "6".
- treatments (pathways only): the treatments whose order is described, e.g. norepinephrine, vasopressin.
- stratify: up to 2 variables to break results down by.
- approximation: "" — or, when this design cannot express the measure, grouping, exposure or outcome the question
  asks for (it is not in the lists) and uses something close instead, one sentence saying so. Standard choices
  (first ICU stay, time zero, windows, adjustment set) are not approximations.
Examples:
"28-day mortality in septic shock" → analysis rate, population [septic shock], outcome death_28d
"norepinephrine vs vasopressin in septic shock, 28-day mortality" → analysis compare, population [septic shock],
  exposure norepinephrine, comparator vasopressin, outcome death_28d
"is lactate > 4 associated with in-hospital death in sepsis" → analysis association, population [sepsis],
  exposure lactate > 4, outcome death_hosp
"how often is hydrocortisone given in septic shock" → analysis utilization, population [septic shock],
  exposure hydrocortisone within_hours 72
"which characteristics predict in-hospital death in sepsis" → analysis prognostic, population [sepsis],
  outcome death_hosp, no exposure
"a 70-year-old man in septic shock, lactate 6, on norepinephrine: how did similar patients do with vasopressin?" →
  analysis similar, population [septic shock], patient [age = 70, sex = male, lactate = 6, norepinephrine],
  exposure vasopressin, outcome death_28d
"how do septic shock patients who die differ from survivors" → analysis characterize, population [septic shock],
  outcome death_28d
"lactate over the first 72 h of septic shock, survivors vs non-survivors" → analysis trajectory, population
  [septic shock], variable lactate, outcome death_28d
"which vasopressor is started first in septic shock" → analysis pathways, population [septic shock], treatments
  [norepinephrine, vasopressin, epinephrine, phenylephrine]
"the commonest diagnoses of ICU patients who die within 28 days" → analysis top, ranked diagnoses, outcome death_28d
Return only the JSON object."""

REPAIR = """The study plan above has these problems:
{errors}
Return the corrected complete JSON object."""

TERM = {"type": "object", "properties": {"term": {"type": "string"}, "op": {"type": "string", "enum": [""] + list(OPS)},
                                         "value": {"type": "string"}},
        "required": ["term", "op", "value"]}


def schema():
    variables, outcomes = list(L.VARIABLES), list(L.OUTCOMES)
    exposure = {"type": "object", "properties": {**TERM["properties"], "within_hours": {"type": "number"}},
                "required": ["term", "op", "value", "within_hours"]}
    return {"type": "object", "properties": {
        "feasibility": {"type": "string", "enum": ["feasible", "not_feasible"]},
        "reason": {"type": "string"},
        "analysis": {"type": "string", "enum": list(L.ANALYSES)},
        "base": {"type": "string", "enum": list(L.BASES)},
        "title": {"type": "string"},
        "population": {"type": "array", "items": TERM},
        "exclude": {"type": "array", "items": TERM},
        "exposure": exposure, "comparator": TERM,
        "outcome": {"type": "string", "enum": [""] + outcomes},
        "variable": {"type": "string"},
        "stratify": {"type": "array", "items": {"type": "string", "enum": variables}},
        "subgroups": {"type": "array", "items": {"type": "string", "enum": variables}},
        "sql_question": {"type": "string"}, "approximation": {"type": "string"},
        "patient": {"type": "array", "items": TERM}, "treatments": {"type": "array", "items": TERM},
        "ranked": {"type": "string", "enum": ["", "drugs", "diagnoses", "procedures"]}},
        "required": ["feasibility", "reason", "analysis", "base", "title", "population", "exclude", "exposure",
                     "comparator", "outcome", "variable", "stratify", "subgroups", "sql_question"]}


# Concept kinds acceptable for each role, best first
ROLE_KINDS = {
    "treatment": ["drugclass", "drug", "px10", "px9"],
    "condition": ["var", "phe", "micro", "dx10", "dx9", "drugclass", "drug", "px10", "px9", "lab", "item"],
    "measurement": ["var", "lab", "item"],
}
KIND_BONUS = {"var": 0.25, "phe": 0.2, "drugclass": 0.2, "micro": 0.2}
# Weakest search score (before the kind bonuses) a concept needs to stand for a term: below it the term is reported as
# not found instead of being replaced by a look-alike ("unobtainium" scores 0.62 for vecuronium, real terms ≥ 0.84)
MIN_MATCH = 0.8
# A population term that only restates the base population ("adult ICU patients", "critically ill adults")
GENERIC = re.compile(r"^(?:(?:all|any|adults?|patients?|critical|critically|illness|ill|icu|intensive|care|units?|hospitali[sz]ed|inpatients?|"
                     r"admitted|admissions?|people|subjects|in|the|of|to|with|first|stays?|mimic|iv|mimic-iv|database|"
                     r"cohort|boston|bidmc|beth|israel|deaconess|medical|center|centre|"
                     # the design's own words: "time zero", "index date", "baseline"
                     r"time|zero|index|baseline|date|t0|"
                     # the data source spelled out: "Medical Information Mart for Intensive Care IV database v2.2"
                     r"for|information|mart|dataset|data|records?|electronic|health|ehr|v?\d+(?:\.\d+)*)"
                     r"[\s,()\-]*)+$", re.I)


def generic(term):
    """The term only restates the base population, also with a qualifier in parentheses copied from a dictionary
    label ("Critically ill (MD Progress Note)")."""
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", str(term or "")).replace("_", " ").strip()   # "any_icu"
    return not text or bool(GENERIC.match(text))


# An "exposure" that names a set of characteristics asks which of them predict the outcome
FACTORS = re.compile(r"\b(characteristics|factors|predictors|variables|features)\b", re.I)
# A "variable" that is a statistic, not a measurement ("median", "number of patients")
STATISTIC = re.compile(r"^(?:the\s+)?(?:median|mean|average|distribution|number|count|values?|levels?|proportion|"
                       r"percentage|rate|frequency)(?:\s+of\s+(?:patients|stays|admissions))?$", re.I)
# Diagnosis-code phenotypes with a clinical definition computed for every ICU stay (mimic-code); the definition is
# used unless the question asks for codes
CLINICAL = {"phe:sepsis_dx": "var:sepsis3", "phe:septic_shock_dx": "var:septic_shock"}
CODES_ASKED = re.compile(r"\b(icd|codes?|coded|coding|diagnosis codes?)\b", re.I)
# Words that tie a question to the ICU, or to hospitalized patients in general (choice of the population base)
ICU_WORDS = re.compile(r"\b(icu|intensive care|critical(ly)? ill|critical care|critical illness|ventilat\w*|"
                       r"vasopressor\w*|septic shock|sofa|saps|oasis|apache|intubat\w*|mechanical)\b", re.I)
HOSPITAL_WORDS = re.compile(r"\b(hospitali[sz]\w*|admissions?|admitted|inpatients?|readmi\w*|discharg\w*)\b", re.I)
# A question about the commonest items of a kind ("which drugs are most often given", "the commonest diagnoses")
RANKING = re.compile(r"\b(?:most (?:common(?:ly)?|frequent(?:ly)?|often|used|prescribed|given|administered)|commonest|"
                     r"top \d+|leading)\b|最常见|最常用|最多见", re.I)
RANKED_WORDS = (("drugs", r"\b(?:drugs?|medications?|medicines?|antibiotics?|antimicrobials?|vasopressors?|sedatives?|"
                          r"opioids?|analgesics?|prescriptions?)\b|药"),
                ("diagnoses", r"\b(?:diagnos[ie]s|conditions|comorbidit(?:y|ies)|diseases?|icd codes?)\b|诊断|疾病|合并症"),
                ("procedures", r"\b(?:procedures?|surger(?:y|ies)|operations?)\b|手术|操作"))


def ranked_kind(text):
    """The kind of item a ranking question asks about ("drugs", "diagnoses", "procedures"), or ""."""
    if not RANKING.search(str(text or "")):
        return ""
    return next((k for k, pattern in RANKED_WORDS if re.search(pattern, str(text), re.I)), "")


# A question that counts ("how many received ECMO?"): with one concept, a template counts it exactly
COUNTING = re.compile(r"\b(?:how many|how often|number of|proportion|percentage|percent|what share|frequency|"
                      r"incidence|prevalence)\b|多少|比例", re.I)
# A question about the time spent on a treatment ("how many hours of mechanical ventilation"): no study variable
TREATMENT_TIME = re.compile(r"\b(?:how (?:many|much) (?:hours|days)|how long(?! (?:do|does|did|are|is|were)(?: \w+){1,3} "
                            r"(?:stay|remain))|duration|length(?! of (?:icu |hospital )?stay))\b(?:(?!\bwithin\b)[^.?])"
                            r"{0,40}?"
                            r"\b(?:of |on |with |receiv\w* |need\w* )?(?:invasive |mechanical |continuous )*"
                            r"(ventilat\w*|vasopressor\w*|norepinephrine|vasopressin|dialysis|rrt|crrt|renal replacement|"
                            r"antibiotic\w*|sedat\w*|propofol|dexmedetomidine)", re.I)
# A question about the survival of groups (Kaplan-Meier curves), not only the risk of death
SURVIVAL_WORDS = re.compile(r"\b(surviv\w*|kaplan|time[- ]to[- ](?:death|event)|hazards?)\b|生存", re.I)
ANALYSIS_TITLES = {"compare": "Comparison", "association": "Association", "rate": "Rate", "distribution": "Distribution",
                   "utilization": "Treatment use", "survival": "Survival", "trend": "Trend", "prognostic": "Prognosis",
                   "explore": "Exploratory query", "similar": "Patients like this", "characterize": "Characteristics",
                   "trajectory": "Over time", "pathways": "Treatment pathways", "top": "Commonest"}
SEX = {"female": "F", "females": "F", "women": "F", "woman": "F", "male": "M", "males": "M", "men": "M", "man": "M"}


def _value(text):
    if text is None or text == "":
        return None
    if isinstance(text, (int, float, list)):
        return text
    text = re.sub(r"^\s*(?:>=|<=|[<>=≥≤])\s*(?=-?\d)", "", str(text))           # ">=18": the operator is in "op"
    text = re.sub(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", r"\1,\2", text)     # "2-4" range
    parts = [p.strip() for p in text.replace("–", ",").replace(" to ", ",").split(",")]
    try:
        nums = [float(p) for p in parts if p]
    except ValueError:
        return str(text).strip()
    return nums if len(nums) == 2 else (nums[0] if nums else None)


def _level(index, base, variable, value):
    """The value of a category variable in the data that ``value`` names (None when there is no unique match)."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if variable == "gender":
        text = SEX.get(text.lower(), text)
    levels = index.levels(base, variable)
    exact = [v for v in levels if v.lower() == text.lower()]
    if exact:
        return exact[0]
    partial = [v for v in levels if text.lower() in v.lower() or (len(v) >= 3 and v.lower() in text.lower())]
    return partial[0] if len(partial) == 1 else None


def _needs_level(c):
    """A category variable used as a criterion without a value (e.g. "ICU type"): a breakdown, not a filter."""
    return bool(c) and c["concept"].startswith("var:") and L.VARIABLES[c["concept"][4:]][1] == "category" \
        and c.get("value") is None


class Resolver:
    def __init__(self, index, base):
        self.index, self.base = index, base
        self.mapping = []          # (role, term, concept, label)

    def resolve(self, term, role, op="", numeric=False, min_score=None):
        """Best concept id for a term in a role (None if nothing fits, or nothing scores ``min_score``)."""
        kinds = ROLE_KINDS[role]
        if numeric:
            kinds = ["var", "lab", "item"]
        candidates = self.index.search(term, k=12, kinds=kinds)
        variables = L.variables_for(self.base)
        best, best_score = None, -1e9
        for c in candidates:
            if c.kind == "var":
                name = c.concept_id[4:]
                if name not in variables:
                    continue
                vkind = variables[name][1]
                if numeric and vkind != "numeric":
                    continue
                if role == "condition" and not numeric and vkind == "numeric" and not op:
                    continue
            if c.kind in ("lab", "item") and not numeric and role != "measurement":
                continue
            score = c.score + KIND_BONUS.get(c.kind, 0) - 0.02 * kinds.index(c.kind)
            if c.kind == "var" and L.VARIABLES[c.concept_id[4:]][4] == "syndrome":
                score += 0.2       # clinical definitions (Sepsis-3, KDIGO) over diagnosis codes in the ICU
            if score > best_score:
                best, best_score = c, score
        if best is None or best.score < MIN_MATCH or (min_score is not None and best_score < min_score):
            if not any(m["term"] == term and m["role"] == role for m in self.mapping) and min_score is None:
                self.mapping.append({"role": role, "term": term, "concept": None, "label": "not found in MIMIC-IV",
                                     "patients": None})
            return None
        self.mapping = [m for m in self.mapping if not (m["term"] == term and m["concept"] is None)]
        if not any(m["term"] == term and m["role"] == role for m in self.mapping):
            self.mapping.append({"role": role, "term": term, "concept": best.concept_id, "label": best.label,
                                 "patients": best.n_patients})
        return best.concept_id


def _criterion(resolver, t, role, window=None):
    if not isinstance(t, dict) or not str(t.get("term", "")).strip():
        return None
    term = str(t["term"]).strip()
    if role == "condition" and generic(term):
        return None                        # the base population itself
    if IR.DATA_WORDS.match(IR._norm(term) or "x"):
        return None                        # "drugs", "lab tests": no concept
    op = t.get("op") or ""
    value = _value(t.get("value"))
    number = isinstance(value, (int, float)) or (isinstance(value, list) and len(value) == 2)
    named = IR.exact_variable(term)
    if role == "condition" and named and L.VARIABLES[named][1] == "numeric" and not (number and op not in ("", "=", "!=")):
        return None                        # "age" without a threshold is no criterion (never a look-alike code)
    if named and L.VARIABLES[named][1] == "binary" and named in L.variables_for(resolver.base) \
            and op in ("", "=", ">=") and value in (None, "", 1, 1.0, "1", "yes", "true"):
        # a yes/no variable named by itself ("invasive mechanical ventilation in the first 24 h"): that variable in
        # every role — a treatment search once found an ICD procedure code instead
        resolver.mapping = [m for m in resolver.mapping if not (m["term"] == term and m["concept"] is None)]
        if not any(m["term"] == term for m in resolver.mapping):
            resolver.mapping.append({"role": role, "term": term, "concept": f"var:{named}",
                                     "label": L.VARIABLES[named][0], "patients": None})
        return {"concept": f"var:{named}", "op": "", "value": None, "window_h": window}
    if op and op not in ("=", "!=") and not number:
        op = ""                            # a threshold without a number ("SOFA >= high") is no threshold
    numeric = bool(op) and op not in ("=", "!=")
    concept = None
    if numeric and not named and op in (">=", ">") and value in (1, 1.0, 0, 0.0) and (op, value) != (">=", 0):
        # "delirium >= 1" is the model's way to say "with delirium": a condition before a chart item's value
        concept = resolver.resolve(term, role)
        if concept is not None and concept.split(":")[0] in ("phe", "micro", "dx9", "dx10", "drugclass", "drug", "px9",
                                                             "px10") \
                or (concept or "").startswith("var:") and L.VARIABLES[concept[4:]][1] == "binary":
            op, value, numeric = "", None, False
        else:
            concept = None
    concept = concept or resolver.resolve(term, role, op=op, numeric=numeric)
    if concept is None and numeric:
        # a threshold on something that is not a measurement ("sepsis >= 2"): the condition itself
        concept = resolver.resolve(term, role)
        op, value = "", None
    if concept is None:
        return None
    c = {"concept": concept, "op": "", "value": None, "window_h": window}
    kind = concept.split(":")[0]
    if kind == "var":
        vkind = L.VARIABLES[concept[4:]][1]
        if vkind == "numeric" and op and value is not None:
            c.update(op=op, value=value)
        elif vkind == "category":
            level = _level(resolver.index, resolver.base, concept[4:], value)
            if level is not None:
                c.update(op=op if op in ("=", "!=") else "=", value=level)
    elif kind in ("lab", "item") and op and value is not None:
        c.update(op=op, value=value)
    return c


# Words that show a question asks about an outcome (an outcome it never mentions was added by the model)
OUTCOME_CUES = {"death": ("death", "mortality", "die", "died", "dying", "surviv", "fatal"),
                "aki": ("kidney", "aki", "renal", "creatinine", "kdigo", "nephro"),
                "rrt": ("dialysis", "renal replacement", "rrt", "crrt", "hemofiltration"),
                "vent": ("ventilat", "intubat"), "readmit": ("readmi", "rehospital"),
                "los": ("length of stay", "los", "how long", "duration", "days in")}
STAY = re.compile(r"\b(during (the |their |an? )?(first )?(icu |hospital )?(stay|admission|hospitali[sz]ation)|"
                  r"at any time|ever)\b", re.I)


def _outcome_family(key):
    for family, prefixes in (("death", ("death",)), ("aki", ("aki",)), ("rrt", ("rrt",)), ("vent", ("invasive_vent",)),
                             ("readmit", ("icu_readmit", "hosp_readmit")), ("los", ("los_",))):
        if key.startswith(prefixes):
            return family
    return None


def asked_about(outcome, text):
    """False when ``text`` (question + planner terms, English) never mentions the outcome's subject."""
    family = _outcome_family(outcome or "")
    if not family or not text.strip() or re.search(r"[\u4e00-\u9fff]", text):
        return True                      # unknown outcome family or a non-English question: no judgement
    return any(cue in text.lower() for cue in OUTCOME_CUES[family])


def to_spec(intent, index, question="", terms="", original=""):
    """StudySpec from the model's intent: terms resolved to concepts, then sanitized. ``terms``: the planner's
    concept names, used with the question to recognise an outcome the question does not ask about. ``original``: the
    user's own words when the planner rewrote the question (a described patient, "does MIMIC-IV record …" are read
    from them)."""
    base = intent.get("base") if intent.get("base") in L.BASES else "icu"
    asked = f"{original} {question}".strip() if original and original != question else str(question or "")
    # the model's usual slips, repaired before its terms are resolved (tools/rwd/intent_rules.py)
    intent, base, notes = IR.normalize(intent, asked, base, asked_about)
    resolver = Resolver(index, base)
    analysis = intent.get("analysis") if intent.get("analysis") in L.ANALYSES else "rate"
    if CATALOG.search(asked):                        # "does MIMIC-IV record …": the data catalog answers
        if analysis != "catalog":
            notes.append("the question asks what the data records → data catalog")
        analysis = "catalog"
    # one described patient → patients like this (the question's own numbers are read, not left to the model)
    described = PAT.cue(asked)
    leave = _treatment_flags(intent)
    # the patient's numbers from the user's own words (a rewritten question may round or drop them)
    parsed = PAT.parse(original or question, base, leave_out=leave) if described else []
    if analysis == "similar" and not described:
        analysis = "compare" if (intent.get("exposure") or {}).get("term") and \
            (intent.get("comparator") or {}).get("term") else "rate"
        notes.append(f"the question describes no single patient → {analysis}")
    elif described and analysis != "similar" and (len(parsed) >= 2 or intent.get("patient")):
        notes.append("the question describes one patient → patients like this")
        analysis = "similar"
    ranked = intent.get("ranked") if intent.get("ranked") in ("drugs", "diagnoses", "procedures") else ""
    if analysis not in ("similar", "catalog") and ranked_kind(asked):
        if analysis != "top":
            notes.append(f"the commonest {ranked_kind(asked)} → a ranking")
        analysis, ranked = "top", ranked_kind(asked)
    if analysis == "top" and not ranked:
        ranked = "drugs"
    treat_role = "treatment" if analysis in ("compare", "utilization", "similar", "pathways") else "condition"
    exp = intent.get("exposure") or {}
    if analysis in ("association", "compare") and FACTORS.search(str(exp.get("term") or "")) and intent.get("outcome"):
        analysis = "prognostic"
        notes.append(f"“{exp['term']}” names several characteristics → prognostic model")
    if analysis == "prognostic" and (exp.get("term") or (intent.get("comparator") or {}).get("term")):
        exp, intent = {}, {**intent, "comparator": None}     # a prognostic model uses the covariate set
        notes.append("a prognostic model uses the baseline characteristics as predictors (no single exposure)")
    hours = float(exp.get("within_hours") or (72 if analysis == "utilization" else 24))
    whole_stay = analysis != "compare" and (exp.get("within_hours") == 0 or bool(STAY.search(question)))
    window = [0, None] if whole_stay else [0, hours]
    exposure = _criterion(resolver, exp, treat_role, window=window)
    comparator = _criterion(resolver, intent.get("comparator"), treat_role, window=window)
    if analysis == "compare" and exposure is None and str(exp.get("term") or "").strip():
        # "compare … between patients with and without cirrhosis": the groups are a condition, not a treatment
        exposure = _criterion(resolver, exp, "condition", window=window)
        if exposure is not None:
            comparator = _criterion(resolver, intent.get("comparator"), "condition", window=window)
    outcome = intent.get("outcome") or ""
    if analysis in ("rate", "trend", "distribution", "utilization") and outcome \
            and not asked_about(outcome, f"{question} {terms}"):
        notes.append(f"the question does not ask about {L.OUTCOMES[outcome][0].lower()}: outcome dropped"
                     if outcome in L.OUTCOMES else "outcome dropped")
        outcome = ""
        intent = {**intent, "outcome": ""}
    population = [c for c in (_criterion(resolver, t, "condition") for t in intent.get("population") or []) if c]
    exclude = [c for c in (_criterion(resolver, t, "condition") for t in intent.get("exclude") or []) if c]
    if analysis == "explore" and exposure and comparator is None and not intent.get("stratify") \
            and not str(intent.get("variable") or "").strip() and COUNTING.search(asked):
        # "how many patients received ECMO?": the template counts the concept itself (once a free query counted
        # vasopressors instead)
        analysis = "utilization" if exposure["concept"].split(":")[0] in ("drugclass", "drug", "px9", "px10") else "rate"
        notes.append(f"a count of one concept → {analysis}")

    # ---- sanitize ----------------------------------------------------------------
    exp_kind = exposure["concept"].split(":")[0] if exposure else ""
    if analysis in ("rate", "trend", "distribution", "prognostic") and exposure and intent.get("outcome") \
            and exp_kind not in ("drugclass", "drug", "px9", "px10"):
        # "outcome rate in <condition>": the condition defines the population
        if exposure["concept"] not in {c["concept"] for c in population}:
            population.append({**exposure, "window_h": None})
        exposure, comparator = None, None
        notes.append("the condition was used to define the population")
    roles = {c["concept"] for c in (exposure, comparator) if c}
    before = len(population)
    set_aside = [c for c in population if c["concept"] in roles]      # given back if the exposure is dropped later
    population = [c for c in population if c["concept"] not in roles]
    exclude = [c for c in exclude if c["concept"] not in roles and c["concept"] not in {p["concept"] for p in population}]
    if len(population) < before:
        notes.append("the exposure / comparator was removed from the population criteria")
    seen, unique = set(), []
    for c in population:
        if c["concept"] not in seen:
            seen.add(c["concept"])
            unique.append(c)
    population = unique
    if comparator and exposure and comparator["concept"] == exposure["concept"]:
        comparator = None            # e.g. "lactate > 4 vs ≤ 4": the complement is the reference group
    exp_kind = exposure["concept"].split(":")[0] if exposure else ""
    comp_kind = comparator["concept"].split(":")[0] if comparator else ""
    if analysis == "association" and exp_kind in ("drugclass", "drug") and comp_kind in ("drugclass", "drug"):
        analysis = "compare"
        notes.append("two treatments → treatment comparison")
    elif analysis == "association" and exp_kind in ("drugclass", "drug") and comparator is None and outcome \
            and L.OUTCOMES.get(outcome, ("", ""))[1] == "binary":
        # "does the use of statins relate to lower mortality": a treatment vs none, emulated as a trial (new users,
        # grace period) rather than an association of a drug ever given
        analysis = "compare"
        notes.append("a treatment and an outcome → treatment comparison (vs no treatment)")
    if analysis == "compare" and exp_kind not in ("drugclass", "drug", "px9", "px10"):
        comparator = None
        if exposure and SURVIVAL_WORDS.search(asked) and (L.OUTCOMES.get(outcome) or ("",) * 5)[4]:
            analysis = "survival"          # survival of two groups of patients: their curves and a hazard ratio
            notes.append("the survival of patients with and without a characteristic → survival curves")
        else:
            analysis = "association"
            notes.append("the exposure is a patient characteristic → adjusted association")
    variable = ""
    term = str(intent.get("variable") or "").strip()
    named = IR.exact_variable(term) if term else None
    if named and named in L.variables_for(base) and L.VARIABLES[named][1] == "numeric":
        variable = named                       # the variable named by itself (not a look-alike of another base)
    elif term and not STATISTIC.match(term):
        concept = resolver.resolve(term, "measurement", numeric=True, min_score=0.9)
        if concept:
            variable = concept[4:] if concept.startswith("var:") else concept
    outcome = intent.get("outcome") or ""
    if analysis == "distribution" and not variable and outcome in L.OUTCOMES and L.OUTCOMES[outcome][1] == "continuous":
        variable = L.OUTCOMES[outcome][3]          # "median ICU length of stay" → the length-of-stay variable
    if analysis == "distribution" and not variable and exposure and exp_kind in ("var", "lab", "item"):
        variable = exposure["concept"][4:] if exp_kind == "var" else exposure["concept"]
        exposure = None
    if analysis == "distribution" and exposure:
        if exposure["concept"] in (f"var:{variable}", variable):
            exposure = None
        elif exp_kind in ("phe", "dx9", "dx10") or (exp_kind == "var" and L.VARIABLES[exposure["concept"][4:]][1] == "binary"):
            population.append({**exposure, "window_h": None})
            exposure = None
            notes.append("the condition was used to define the population")
        else:
            exposure = None
            notes.append("a distribution describes one variable; the exposure was not used")
        comparator = None
    stratify = [s for s in intent.get("stratify") or [] if s in L.variables_for(base)]
    for c in [c for c in population if _needs_level(c)] + ([exposure] if _needs_level(exposure) else []):
        name = c["concept"][4:]
        if name not in stratify:
            stratify.insert(0, name)
        notes.append(f"{L.VARIABLES[name][0]} names no single group: results are broken down by it instead")
    population = [c for c in population if not _needs_level(c)]
    exclude = [c for c in exclude if not _needs_level(c)]
    if _needs_level(exposure):
        exposure, comparator = None, None
        if analysis in ("association", "compare", "survival"):
            analysis = "rate"
    timed = TREATMENT_TIME.search(asked)
    duration = None if not timed else "vent_hours" if timed.group(1).lower().startswith("ventilat") else \
        "vasopressor_hours" if re.match(r"vasopressor|norepinephrine|vasopressin", timed.group(1), re.I) else None
    if timed and analysis in ("distribution", "rate", "trend", "utilization", "explore", "characterize") \
            and not str(variable).startswith("derived:"):
        if duration and base == "icu":
            # the hours on ventilation / vasopressors: the episodes of mimic-code (once a model answered "hours of
            # ventilation" with the ICU length of stay)
            analysis, variable, exposure, comparator = "distribution", f"derived:{duration}", None, None
            notes.append(f"the time on {timed.group(1).lower()}: {L.DERIVED[duration][0].lower()}")
        else:
            analysis = "explore"
            notes.append(f"the time on {timed.group(1).lower()} is not a study variable: an exploratory query of the "
                         "mimic-code episodes")
    if analysis in ("rate", "trend") and L.OUTCOMES.get(outcome, ("", ""))[1] == "continuous":
        # a length of stay has no rate: its distribution (by period for a trend)
        if analysis == "trend" and "anchor_year_group" not in stratify:
            stratify.insert(0, "anchor_year_group")
        analysis, variable = "distribution", variable or L.OUTCOMES[outcome][3]
        kind = (exposure or {}).get("concept", ":").split(":")[0]
        if kind in ("phe", "dx9", "dx10") or kind == "var" and L.VARIABLES[exposure["concept"][4:]][1] == "binary":
            population.append({**exposure, "window_h": None})        # a condition: the patients described
        exposure, comparator = None, None
        notes.append(f"{L.OUTCOMES[outcome][0]} is measured, not counted: its distribution")
    if analysis == "rate" and not outcome and exposure and exposure["concept"].split(":")[0] in (
            "drugclass", "drug", "px9", "px10"):
        analysis = "utilization"           # how often a treatment is given
    if analysis in ("rate", "trend") and not intent.get("outcome") and not exposure and not variable \
            and len(population) == 1 and COUNTING.search(asked + " rate"):
        # "the incidence of delirium in the ICU": the condition named is what is counted, not the population
        exposure, population = {**population[0], "window_h": None}, []
        notes.append("the condition asked about is counted in the whole population")
    if analysis in ("rate", "trend") and not outcome and exposure and population:
        # "how many ICU patients have diabetes" with "diabetes with complications" as a criterion: the share of a
        # condition among patients chosen for it is 100 % — such a criterion restates what is counted
        counted = _label_words(index, exposure["concept"])
        same = [c for c in population if counted & _label_words(index, c["concept"])]
        if same:
            population = [c for c in population if c not in same]
            notes.append("criteria naming the condition counted were dropped: " + ", ".join(c["concept"] for c in same))
    if analysis in ("rate", "trend") and not intent.get("outcome") and not exposure and not variable:
        notes.append("no outcome given")
    # "only in women": a criterion, not a breakdown (a small model once broke the results down by sex instead)
    only = re.search(r"\b(?:only|just|exclusively)\s+(?:in\s+|among\s+)?(women|men|females?|males?)\b|"
                     r"\b(?:in|among)\s+(women|men|females?|males?)\s+only\b|只看(女性|男性)|仅(女性|男性)", question, re.I)
    if only:
        word = next(g for g in only.groups() if g)
        level = "F" if word.lower().startswith(("wom", "fem")) or word in ("女性",) else "M"
        population = [c for c in population if c["concept"] != "var:gender"] + \
            [{"concept": "var:gender", "op": "=", "value": level, "window_h": None}]
        stratify = [v for v in stratify if v != "gender"]
        notes.append(f"only {'women' if level == 'F' else 'men'}: a criterion of the population")
    # the base population restates "adults": an age ≥ 18 criterion adds nothing
    population = [c for c in population if not (c["concept"] == "var:age" and c.get("op") in (">=", ">")
                                                 and isinstance(c.get("value"), (int, float)) and c["value"] <= 18)]
    # "adults hospitalized with …" without anything ICU-specific: all hospital admissions, not only ICU stays
    if base == "icu" and HOSPITAL_WORDS.search(question) and not ICU_WORDS.search(question):
        hospital_vars = L.variables_for("hospital")
        used = [c for c in population + exclude + [exposure, comparator] if c]
        if (not intent.get("outcome") or intent.get("outcome") in L.outcomes_for("hospital")) \
                and all(c["concept"][4:] in hospital_vars for c in used if c["concept"].startswith("var:")) \
                and all(v in hospital_vars for v in stratify) and (not variable or variable in hospital_vars):
            base = "hospital"
            notes.append("the question is about hospitalized patients, not only ICU stays: all adult admissions")
    if base == "icu" and not CODES_ASKED.search(question):
        for c in population + exclude + [x for x in (exposure, comparator) if x]:
            if c["concept"] in CLINICAL:      # "sepsis" in the ICU: the Sepsis-3 definition, not discharge codes
                old, c["concept"] = c["concept"], CLINICAL[c["concept"]]
                for m in resolver.mapping:
                    if m["concept"] == old:
                        m.update(concept=c["concept"], label=L.VARIABLES[c["concept"][4:]][0], patients=None)
                notes.append(f"{L.PHENOTYPES[old[4:]][0]} → {L.VARIABLES[c['concept'][4:]][0]} (clinical definition "
                             "of the ICU stay)")
    treatments, group_by = [], ""
    horizon = _horizon(asked)
    if analysis == "top":                  # the first hours or days of the stay, else the whole stay (0)
        early = FIRST_HOURS.search(asked)
        horizon = _horizon(early.group(0)) if early else 0.0
    if analysis == "top" and ranked == "drugs":
        # "the commonest antibiotics": ranked within the class (named, or the model's exposure), not by it
        if exposure and exposure["concept"].startswith("drugclass:"):
            treatments, exposure, comparator = [{"concept": exposure["concept"]}], None, None
        else:
            treatments = [{"concept": c} for c in _named_classes(asked, index)][:1]
    if analysis == "trajectory":
        from mes.tools.rwd.patterns import series_of
        variable, from_exposure = _series_variable(resolver, index, intent, variable, exposure)
        named = str((intent.get("exposure") or {}).get("term") or "").strip().lower() if isinstance(
            intent.get("exposure"), dict) else ""
        own = {c for c, _conversion in series_of(variable)}          # (heart rate > 0 is the heart rate itself)
        if exposure and (from_exposure or exposure["concept"] in (variable, f"var:{variable}") or exposure["concept"] in own
                         or (named and named == str(intent.get("variable") or "").strip().lower())):
            # the measurement named as the exposure is the variable followed; the population it took is given back
            population += [c for c in set_aside if c["concept"] == exposure["concept"]
                           and c["concept"] not in {p["concept"] for p in population}]
            exposure = None
        if not series_of(variable) and variable in L.VARIABLES:
            analysis = "distribution"          # a score of the first 24 h has no time series: its distribution
            notes.append(f"{L.VARIABLES[variable][0]} is recorded once (first 24 h): its distribution instead")
    if analysis == "pathways":
        treatments = _pathway_treatments(resolver, index, intent, exposure, comparator, asked)
        exposure, comparator = None, None
        if len(treatments) < 2:                # one treatment has no order: how often and when it is started
            analysis, exposure, treatments = "utilization", (treatments[0] if treatments else None), []
            notes.append("a pathway needs two treatments: the use of the one named instead")
    if analysis in ("characterize", "trajectory", "top"):
        if comparator:                         # their groups are the outcome, the exposure or a characteristic
            notes.append(f"a description has no comparator: {comparator['concept']} left out")
            comparator = None
        if not (intent.get("outcome") in L.OUTCOMES and L.OUTCOMES[intent["outcome"]][1] == "binary") and \
                re.search(r"\b(die|died|dying|death|deaths|non-?survivors?|survivors?|survived|mortality)\b|死亡|存活",
                          str(question or ""), re.I):
            intent = {**intent, "outcome": "death_28d" if base == "icu" else "death_hosp"}
            notes.append(f"who died and who survived: {L.OUTCOMES[intent['outcome']][0].lower()}")
        binary_outcome = intent.get("outcome") in L.OUTCOMES and L.OUTCOMES[intent["outcome"]][1] == "binary"
        groupable = [v for v in stratify if L.VARIABLES[v][1] in ("binary", "category")]
        group_by = "outcome" if binary_outcome else "exposure" if exposure else groupable[0] if groupable else ""
        if group_by and group_by not in ("outcome", "exposure"):
            stratify = [v for v in stratify if v != group_by]
    profile = []
    if analysis == "similar":
        profile, population = _profile(intent, resolver, index, base, parsed, population)
        described = {f["concept"] for f in profile}
        # never exclude what describes the patient (a small model once excluded the patient's own sex)
        dropped = [c for c in exclude if c["concept"] in described]
        if dropped:
            notes.append("exclusions on the patient's own characteristics were dropped: "
                         + ", ".join(c["concept"] for c in dropped))
        exclude = [c for c in exclude if c["concept"] not in {p["concept"] for p in population} | described]
        # the options compared among similar patients are treatments: a characteristic is part of the patient
        options = [c for c in (exposure, comparator) if c and _treatment(c)]
        for c in (exposure, comparator):
            if c and not _treatment(c):
                notes.append(f"{c['concept']} is a characteristic, not a treatment: it describes the patient")
        exposure, comparator = (options + [None, None])[:2]
        if not intent.get("outcome"):
            intent = {**intent, "outcome": "death_28d" if base == "icu" else "death_hosp"}
            notes.append(f"no outcome asked: {L.OUTCOMES[intent['outcome']][0].lower()}")
    spec = StudySpec(
        analysis=analysis, base=base, title=str(intent.get("title") or "").strip()[:160], profile=profile,
        treatments=treatments, group_by=group_by, horizon_h=horizon, bin_h=_bin(horizon) if horizon else 6.0,
        ranked=ranked,
        include=population, exclude=exclude, exposure=exposure, comparator=comparator, grace_h=hours,
        outcome=intent.get("outcome") or "", variable=variable,
        stratify=stratify[:2],
        subgroups=[s for s in intent.get("subgroups") or [] if s in L.variables_for(base)][:3],
        feasibility="not_feasible" if intent.get("feasibility") == "not_feasible" else "feasible",
        reason=str(intent.get("reason") or "")[:400],
        sql_question=str(intent.get("sql_question") or (asked[:400] if analysis == "explore" else "")))
    spec = StudySpec.from_dict(spec.to_dict())
    if not spec.title:
        spec.title = _default_title(spec, index)
    return spec, resolver.mapping, notes


# flags of what the patient received that name a treatment the question asks about (similar patients are found with
# and without it, so it is not a characteristic to match)
TREATMENT_FLAGS = (("vasopressin_24h", ("vasopressin",)), ("norepinephrine_24h", ("norepinephrine", "noradrenaline")),
                   ("invasive_vent_24h", ("ventilat", "intubat")), ("rrt_24h", ("dialysis", "renal replacement", "rrt")),
                   ("vasopressor_24h", ("vasopressor", "pressor")))


# A question about the data itself: "does MIMIC-IV record ECMO?", "which lab tests are available for …"
CATALOG = re.compile(
    r"\b(?:does|do|is|are)\s+(?:mimic(?:-?iv)?|the\s+(?:database|data(?:set)?))\s+(?:have|has|record|contain|include|"
    r"capture|hold)|\b(?:is|are)\b.{1,60}\b(?:recorded|available|captured|documented|coded)\s+in\s+(?:mimic|the\s+"
    r"(?:database|data))|\bwhat\s+(?:data|variables|information|labs?|lab\s+tests|medications|codes)\b.{0,50}"
    r"\b(?:mimic|database|available|recorded)|\bwhich\s+(?:variables|labs?|lab\s+tests|data|codes|medications)\b"
    r".{0,50}\b(?:available|recorded|mimic|database)|(?:数据库?里?|MIMIC(?:-IV)?).{0,8}(?:有没有|有哪些|记录了|包含|有无)|"
    r"\bhow many (?:patients|admissions|(?:icu )?stays|people|subjects)\s+(?:are|is|were)\s+(?:there\s+)?in\s+"
    r"(?:mimic|the\s+(?:database|data(?:set)?))|\b(?:size|overview|contents?)\s+of\s+(?:mimic|the\s+(?:database|"
    r"dataset))|(?:MIMIC(?:-IV)?|数据库).{0,6}(?:有多少|多少)(?:病人|患者|人)",
    re.I)
HORIZON = re.compile(r"\b(?:first|within|over|during|in the|for)?\s*(\d{1,3})\s*(h|hrs?|hours?|d|days?)\b|"
                     r"(\d{1,3})\s*(小时|天)", re.I)


# the first hours of a stay ("in the first 24 hours", "within 48 h of ICU admission") — not "death within 28 days"
FIRST_HOURS = re.compile(r"\b(?:first|initial)\s+\d{1,3}\s*(?:h|hrs?|hours?|d|days?)\b|\bwithin\s+\d{1,3}\s*"
                         r"(?:h|hrs?|hours?|d|days?)\s+(?:of|after)\s+(?:the\s+)?(?:icu\s+|hospital\s+)?admission", re.I)


def _horizon(question):
    """The hours after time zero a trajectory or pathway follows: "the first 48 h", "3 days" (default 72)."""
    m = HORIZON.search(str(question or ""))
    if not m:
        return 72.0
    n, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    hours = float(n) * (24 if unit.lower().startswith("d") or unit == "天" else 1)
    return min(720.0, max(6.0, hours))


def _bin(horizon):
    """Intervals of a trajectory: about twelve of them, of a usual length (1, 2, 4, 6, 8, 12 or 24 h)."""
    return next((b for b in (1, 2, 4, 6, 8, 12, 24) if horizon / b <= 14), 48.0) * 1.0


def _series_variable(resolver, index, intent, variable, exposure):
    """(variable, named as the exposure) of a trajectory: the design's variable when it has a time series in MIMIC-IV,
    else the lab test or chart item the term names (a measurement the model gave as the exposure counts too)."""
    from mes.tools.rwd.patterns import series_of
    if variable and series_of(variable):
        return variable, False
    terms = [(str(intent.get("variable") or "").strip(), False)]
    if isinstance(intent.get("exposure"), dict):
        terms.append((str(intent["exposure"].get("term") or "").strip(), True))
    for term, is_exposure in [t for t in terms if t[0]]:
        concept = resolver.resolve(term, "measurement", numeric=True, min_score=0.9)
        if concept and series_of(concept[4:] if concept.startswith("var:") else concept):
            return (concept[4:] if concept.startswith("var:") else concept), is_exposure
        hits = [c for c in index.search(term, k=6, kinds=["lab", "item"]) if c.score >= MIN_MATCH]
        if hits:
            return hits[0].concept_id, is_exposure
    return variable, False


_PLAIN_WORDS = {"history", "patients", "patient", "first", "with", "without", "complications", "complication", "chronic",
                "acute", "disease", "disorder", "unspecified", "other", "diagnosis", "condition", "hours", "stay"}


def _label_words(index, concept):
    """The words (5+ letters) of a concept's label that say what it is ("diabetes" of "History: diabetes …")."""
    try:
        label = index.label(concept)
    except Exception:
        label = concept
    return {w for w in re.findall(r"[a-z]{5,}", str(label).lower()) if w not in _PLAIN_WORDS}


def _named_classes(question, index):
    """Drug classes the question names (their label, key or a synonym as words), in the order of the question."""
    text, found = str(question or "").lower(), []
    for key, (label, _ingredients, _excluded, synonyms) in L.DRUG_CLASSES.items():
        names = {label.lower(), key.replace("_", " "), key.replace("_", "-")} | {str(x).lower() for x in synonyms}
        at = [m.start() for n in names if len(n) >= 4
              for m in [re.search(rf"(?<![a-z]){re.escape(n)}(?:e?s)?(?![a-z])", text)] if m]
        if at and (index.count(f"drugclass:{key}") or 0):
            found.append((min(at), f"drugclass:{key}"))
    return [c for _at, c in sorted(found)]


def _pathway_treatments(resolver, index, intent, exposure, comparator, question=""):
    """The treatments of a pathway: the model's list, the exposure and comparator, the drug classes the question
    names (a model once left out the third of three); a class named alone ("which vasopressor first") stands for its
    members (the six most used)."""
    out = []
    for t in list(intent.get("treatments") or []):
        c = _criterion(resolver, t, "treatment")
        if c:
            out.append(c)
    out += [c for c in (exposure, comparator) if c]
    out += [{"concept": c} for c in _named_classes(question, index)]
    seen, unique = set(), []
    for c in out:
        if c["concept"] not in seen:
            seen.add(c["concept"])
            unique.append({**c, "op": "", "value": None, "window_h": None, "negate": False})
    if len(unique) == 1 and unique[0]["concept"].startswith("drugclass:"):
        key = unique[0]["concept"].split(":", 1)[1]
        members = []
        for ingredient in (L.DRUG_CLASSES.get(key, ("", []))[1] or [])[:40]:
            concept = f"drugclass:{ingredient}" if ingredient in L.DRUG_CLASSES else f"drug:{ingredient}"
            n = index.count(concept) or 0
            if n:
                members.append((n, concept))
        if len(members) >= 2:
            unique = [{"concept": c, "op": "", "value": None, "window_h": None, "negate": False}
                      for _n, c in sorted(members, reverse=True)[:6]]
    return unique[:6]


def _treatment_flags(intent):
    words = " ".join(str((intent.get(k) or {}).get("term") or "") for k in ("exposure", "comparator")
                     if isinstance(intent.get(k), dict)).lower()
    return {flag for flag, names in TREATMENT_FLAGS if any(n in words for n in names)}


def _profile(intent, resolver, index, base, parsed, population):
    """(profile, population) of a "patients like this" design: the characteristics read from the question,
    completed with the model's reading of the patient; conditions the patient must have (septic shock, a diagnosis)
    become inclusion criteria — who is like the patient — and the rest is weighed."""
    leave = _treatment_flags(intent)
    resolved = []
    for t in intent.get("patient") or []:
        c = _criterion(resolver, t, "condition")
        if not c:
            continue
        kind = c["concept"].split(":")[0]
        if kind in ("phe", "dx9", "dx10"):
            if c["concept"] not in {p["concept"] for p in population}:
                population.append({**c, "window_h": None})
            continue
        if kind != "var" or c["concept"][4:] in leave:
            continue
        name = c["concept"][4:]
        vkind = L.VARIABLES[name][1]
        if vkind == "binary":
            value = 0 if str(t.get("value") or "").strip().lower() in ("no", "0", "false", "absent", "none") else 1
        elif vkind == "numeric":
            value = c.get("value") if c.get("value") is not None else _value(t.get("value"))
            value = sum(value) / 2 if isinstance(value, list) and len(value) == 2 else value
            if not isinstance(value, (int, float)):
                continue
        else:
            value = c.get("value") if c.get("value") is not None else _level(index, base, name, t.get("value"))
            if value is None:
                continue
        resolved.append({"concept": c["concept"], "value": value, "weight": PAT.WEIGHTS.get(name, 1.0),
                         "must": name in PAT.REQUIRED and value == 1})
    profile = PAT.merge(parsed, resolved)
    for f in [f for f in profile if f.get("must")]:
        if f["concept"] not in {p["concept"] for p in population}:
            population.append({"concept": f["concept"], "op": "", "value": None, "window_h": None})
    soft = {f["concept"][4:] for f in profile if not f.get("must")}
    # a chronic condition of the patient is weighed with the others, not required: "with heart failure and CKD"
    # finds patients like this one, most of whom have both — not only the few with every condition
    population = [p for p in population if not (p["concept"].startswith("phe:") and SOFT.get(p["concept"][4:]) in soft)
                  and not (p["concept"].startswith("var:") and p["concept"][4:] in soft)]
    included = {p["concept"] for p in population}
    return [{**f, "must": False} for f in profile if not f.get("must") and f["concept"] not in included], population


# curated conditions a cohort variable of the patient also stands for (phenotype → variable)
SOFT = {"heart_failure": "congestive_heart_failure", "ckd": "renal_disease", "esrd": "renal_disease",
        "copd": "chronic_pulmonary_disease", "asthma": "chronic_pulmonary_disease", "cirrhosis": "severe_liver_disease",
        "cancer": "malignant_cancer", "myocardial_infarction": "myocardial_infarct", "ischemic_stroke": "cerebrovascular_disease",
        "dementia_dx": "dementia", "hiv": "aids", "hyponatremia": "sodium_min"}
SUPPORT = {k for k, v in L.VARIABLES.items() if v[4] == "support"}


def _treatment(c):
    """A criterion that is something given (a drug, a procedure, organ support), not a characteristic."""
    kind, _, key = str(c.get("concept") or "").partition(":")
    return kind in ("drugclass", "drug", "px9", "px10") or (kind == "var" and key in SUPPORT)


def _default_title(spec, index):
    if spec.analysis == "similar":
        who = PAT.short(spec.profile)
        who = who if len(who) <= 90 else who[:89].rstrip() + "…"
        return f"Patients like this: {who}" if who else "Patients like this"
    what = L.OUTCOMES[spec.outcome][0] if spec.outcome in L.OUTCOMES else \
        L.VARIABLES.get(spec.variable, (spec.variable,))[0] if spec.variable else ""
    if spec.exposure is not None:
        vs = f" vs {index.label(spec.comparator.concept)}" if spec.comparator is not None else ""
        what = f"{index.label(spec.exposure.concept)}{vs}" + (f" — {what}" if what else "")
    who = ", ".join(index.label(c.concept) for c in spec.include[:2])
    return f"{ANALYSIS_TITLES.get(spec.analysis, 'Analysis')}: {what or 'cohort'}" + (f" in {who}" if who else "")


# Asked of Claude only: the small local model designs worse with every extra rule (and leaves approximation empty)
KIND_RULE = """
- approximation_kind: "" without an approximation; else "templates" (MIMIC-IV records it, or it can be computed
  from what it records, but these lists cannot express it), "data" (MIMIC-IV does not record it at that detail, so
  no analysis could do better: anchor year groups instead of calendar years, the exact time of shock onset) or
  "choice" (a design choice or wording: time zero, windows, how the comparator is defined)."""


def _capable(llm):
    return "claude" in str(getattr(llm, "label", "")).lower()


def _prompt(question, plan, index, llm=None):
    found = []
    for entity in (plan.entities or [])[:6]:
        for c in index.search(entity.get("name", ""), k=2):
            found.append(f"{c.label} ({c.n_patients:,} patients)" if c.n_patients else c.label)
    pico = "; ".join(f"{k}: {v}" for k, v in (plan.pico or {}).items() if v)
    text = PROMPT.format(
        coverage="MIMIC-IV, Beth Israel Deaconess Medical Center (Boston), adult hospital and ICU stays 2008-2019; "
                 "inpatient orders and ICU infusions, labs, vitals, ICD codes, death in hospital and up to 1 year "
                 "after discharge.",
        outcomes="; ".join(f"{k} = {v[0]}" for k, v in L.OUTCOMES.items()),
        variables=", ".join(L.VARIABLES), found="; ".join(dict.fromkeys(found)) or "(nothing specific)",
        question=question, pico=f"PICO: {pico}\n" if pico else "",
        previous=(f"PREVIOUS STUDY OF THIS CONVERSATION: {plan.rwd_previous}\nThe question may refine it (another "
                  "outcome, subgroup, comparator, threshold or population): keep what it does not change.\n")
        if getattr(plan, "rwd_previous", "") else "")
    anchor = "are not approximations."
    return text.replace(anchor, anchor + KIND_RULE, 1) if _capable(llm) else text


def design(question, plan, index, llm, catalog=None, report=None, check=None, original=""):
    """
    (spec, info): info has the intent, the term → concept mapping, notes, errors and warnings.
    ``check(spec, mapping)`` runs a valid design on the data and returns a problem (e.g. too few patients, with
    the attrition) or None; a problem gets one more repair round, then the best valid design is kept.
    """
    report = report or (lambda msg: None)
    prompt = _prompt(question, plan, index, llm)
    terms = " ".join([e.get("name", "") for e in plan.entities or []] + [str(v) for v in (plan.pico or {}).values()])
    info = {"attempts": [], "mapping": [], "notes": [], "checks": []}
    spec, errors, warnings = None, ["no design"], []
    valid = None                                   # last design that passed validation: (spec, warnings, mapping, notes)
    attempts = 3 if check else 2
    for attempt in range(attempts):
        try:
            raw = llm.invoke(prompt, json_schema=schema(), reasoning=False, max_tokens=900)
        except Exception as e:
            errors = [f"LLM call failed: {type(e).__name__}: {e}"]
            info["attempts"].append({"error": errors[0]})
            break
        intent = parse_json_response(raw)
        if not isinstance(intent, dict):
            errors, intent = ["the answer was not a JSON object"], {}
        else:
            spec, mapping, notes = to_spec(intent, index, question, terms, original)
            errors, warnings = validate(spec, index)
            specific = [t for t in intent.get("population") or [] if isinstance(t, dict)
                        and str(t.get("term") or "").strip() and not generic(t["term"])
                        and str(t["term"]).strip().lower() != "age"]    # "adults" / "age ≥ 18" restate the base
            if spec.include == [] and spec.analysis != "explore" and specific:
                warnings.append("no population criterion could be matched to the database")
            info.update(mapping=mapping, notes=notes)
        if not errors and valid is not None and spec.feasibility == "not_feasible" \
                and valid[0].feasibility != "not_feasible":
            # a repair round that gives up a design that runs is not an improvement: keep the earlier design
            info["attempts"].append({"intent": intent, "errors": ["repair answered not_feasible; kept the design"]})
            errors = ["kept"]
            break
        if not errors:
            valid = (spec, warnings, info["mapping"], info["notes"])
            info["approximation"] = _approximation(intent)
            info["approximation_kind"] = str(intent.get("approximation_kind") or "").strip().lower()
            problem = check(spec, info["mapping"]) if check else None
            if problem:
                info["checks"].append(problem)
                if attempt < attempts - 1:
                    errors = [problem]
                else:                      # no repair left: the study says why (e.g. too few patients)
                    warnings = list(warnings) + [problem]
        info["attempts"].append({"intent": intent, "errors": errors})
        if not errors:
            break
        if valid is not None and attempt == attempts - 1:
            break
        report(f"RWD: correcting the study plan ({'; '.join(errors)[:140]})")
        prompt += "\n\n" + json.dumps(intent, default=str) + "\n\n" + REPAIR.format(
            errors="\n".join(f"- {e}" for e in errors))
    if errors and valid is not None:        # the repair did not help: keep the valid design (its result says why)
        spec, warnings, info["mapping"], info["notes"] = valid
        warnings = list(warnings) + info["checks"][-1:]
        errors = []
    info.update(errors=errors, warnings=warnings)
    return (spec if not errors else None), info


# Clinical synonyms for the coverage check (word → canonical word)
SYNONYMS = {"mortality": "death", "deaths": "death", "died": "death", "dying": "death", "survival": "death",
            "fatality": "death", "aki": "kidney", "renal": "kidney", "nephrotoxicity": "kidney", "readmissions": "readmission",
            "readmitted": "readmission", "rehospitalization": "readmission", "ventilated": "ventilation",
            "intubation": "ventilation", "dialysis": "replacement", "rrt": "replacement", "los": "stay"}


def _stem(word):
    """Crude suffix stripping, enough to match "transfused" with "transfusion" or "ventilated" with "ventilation"."""
    for suffix in ("ions", "ion", "ings", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _canonical(words):
    return {_stem(SYNONYMS.get(w, w)) for w in words}


# Words that describe the analysis or the setting rather than a concept the design has to represent ("proportion of
# patients transfused", "calendar year", "first-day characteristics", "type of ICU", "time to initiation")
DESCRIPTORS = {"proportion", "percentage", "percent", "rate", "rates", "incidence", "prevalence", "frequency", "number",
               "count", "median", "mean", "average", "distribution", "time", "timing", "year", "years", "calendar",
               "period", "periods", "era", "trend", "trends", "change", "changes", "characteristics", "factors",
               "predictors", "variables", "features", "type", "types", "category", "categories", "group", "groups",
               "level", "levels", "value", "values", "score", "scores", "all", "cause", "overall", "any", "first",
               "day", "days", "hours", "within", "after", "during", "admission", "icu", "intensive", "care", "unit",
               "units", "hospital", "patients", "patient", "adult", "adults", "use", "used", "therapy", "treatment",
               "risk", "early", "received", "receiving", "given", "initiation", "start", "started", "onset", "baseline",
               "mimic", "iv", "database", "cohort", "outcome", "outcomes", "subgroup", "subgroups", "stay", "stays",
               "clinical", "laboratory", "lab", "labs", "vital", "vitals", "sign", "signs", "parameters",
               "measurements", "demographic", "demographics", "admission", "routine", "available", "information",
               "mart", "dataset", "data", "records", "medical"}


def _acronyms(words):
    """Initials of a multi-word name, with and without its trailing descriptor words ("Sequential Organ Failure
    Assessment score" → sofas, sofa)."""
    words = [w for w in words if w.isalpha() and w not in ("of", "the", "and", "in", "for", "to", "with")]
    out = set()
    while len(words) >= 3:
        out.add("".join(w[0] for w in words))
        if words[-1] not in DESCRIPTORS:
            break
        words = words[:-1]
    return out


def _drug_patterns(concept):
    kind, _, key = concept.partition(":")
    if kind == "drugclass" and key in L.DRUG_CLASSES:
        return {p.lower() for p in L.DRUG_CLASSES[key][1]}
    return {key.lower()} if kind == "drug" else set()


def _same_drugs(name, spec, index):
    """A treatment term naming a drug class that contains the design's treatment, or the reverse ("corticosteroids"
    for a hydrocortisone design)."""
    treated = set().union(*(_drug_patterns(c.concept) for c in (spec.exposure, spec.comparator) if c))
    hits = index.search(name, k=1, kinds=["drugclass", "drug"]) if treated else []
    return bool(hits) and bool(_drug_patterns(hits[0].concept_id) & treated)


def _represented(spec, index):
    """Labels of what the design measures, compares and conditions on (with synonyms of curated phenotypes)."""
    def labels(concept):
        kind, _, key = concept.partition(":")
        out = [index.label(concept)]
        if kind == "phe" and key in L.PHENOTYPES:
            out += list(L.PHENOTYPES[key][3])
        if kind == "drugclass" and key in L.DRUG_CLASSES:
            out += list(L.DRUG_CLASSES[key][1][:6]) + list(L.DRUG_CLASSES[key][3])
        return out
    def variable(v):
        if v in L.VARIABLES:
            return [L.VARIABLES[v][0]] + L.VARIABLE_SYNONYMS.get(v, [])
        return [index.label(v), v] if ":" in str(v) else [v]      # a lab test or ICU item by its name
    measured = [L.OUTCOMES[o][0] for o in [spec.outcome] + list(spec.secondary_outcomes) if o in L.OUTCOMES]
    measured += variable(spec.variable) if spec.variable else []
    treatments = [x for c in (spec.exposure, spec.comparator, *spec.treatments) if c for x in labels(c.concept)]
    covariates = spec.covariates or (L.DEFAULT_COVARIATES.get(spec.base, []) if spec.analysis in (
        "association", "prognostic", "compare", "survival") else [])
    context = [x for c in spec.include + spec.exclude for x in labels(c.concept)]
    context += [x for f in spec.profile or [] if isinstance(f, dict) for x in variable(str(f.get("concept") or "")[4:])]
    context += [x for v in list(spec.stratify) + list(spec.subgroups) + list(covariates) if v in L.VARIABLES
                for x in variable(v)]
    return measured, treatments, context


def coverage(plan, spec, index, threshold=0.7):
    """Treatments, comparators and outcomes of the question that the design does not represent (the templates
    could not express them; e.g. "red blood cell transfusion" designed as renal replacement therapy). Outcomes must
    match what is measured; treatments of a comparison the exposure or comparator; other terms anything the design
    uses (a condition the planner called an exposure may be a population criterion or a stratum). Population terms
    and terms that only describe the analysis ("calendar year", "type of ICU") never count."""
    from mes.tools.rwd.concepts import _tokens
    if spec is None or spec.analysis in ("explore", "catalog", "top") or spec.feasibility == "not_feasible":
        return []
    entities = [e for e in plan.entities or [] if e.get("name") and not generic(e["name"])
                and e.get("role") in ("intervention", "exposure", "comparator", "outcome")
                and _canonical(_tokens(e["name"])) - _canonical(DESCRIPTORS)]
    if not entities:
        return []
    measured, treatments, context = _represented(spec, index)
    descriptive = spec.analysis in ("rate", "utilization", "distribution", "trend", "characterize", "trajectory",
                                    "pathways", "similar")
    missing = []
    for e in entities:
        if descriptive:
            targets = measured + treatments + context
        elif e["role"] == "outcome":
            targets = measured
        elif spec.analysis == "compare":
            targets = treatments
        else:                                  # association / survival / prognostic: exposures may be conditions
            targets = treatments + context
        if spec.analysis == "prognostic" and e["role"] != "outcome" and FACTORS.search(e["name"]):
            continue                           # "which characteristics …": the model's predictors
        if not _matches(e["name"], targets, spec, index, drugs=e["role"] != "outcome", threshold=threshold):
            missing.append(e["name"])
    return missing


def _matches(name, targets, spec, index, drugs=True, threshold=0.7):
    """``name`` is one of ``targets``: shared words (stemmed, synonyms), an acronym, a drug class that contains the
    design's treatment, or a close embedding."""
    from mes.tools.ranking_tools import encode
    from mes.tools.rwd.concepts import _tokens
    if not targets:
        return False
    words = _canonical(_tokens(name)) - _canonical(DESCRIPTORS)
    target_words = set().union(*(_canonical(_tokens(t)) for t in targets))
    if words & target_words or _acronyms(str(name).lower().replace("-", " ").split()) & target_words:
        return True
    if drugs and _same_drugs(name, spec, index):
        return True
    return float((encode([name], is_query=True) @ encode(targets).T).max()) >= threshold


# Values a model writes into a term field instead of leaving it empty
JUNK_TERMS = {"any", "none", "all", "other", "others", "unknown", "n/a", "na", "nan", "yes", "no", "true", "false",
              "null", "nil", "not applicable", "unspecified", "various"}


def unmatched(info, spec, index):
    """Terms of the design that MIMIC-IV has no concept for and that the design does not represent otherwise (a
    term the model repeated with another wording, a placeholder like "any" and descriptive words do not count)."""
    from mes.tools.rwd.concepts import _tokens
    terms = list(dict.fromkeys(m["term"] for m in (info or {}).get("mapping") or [] if m.get("concept") is None))
    if not terms:
        return []
    labels = []
    if spec is not None:
        measured, treatments, context = _represented(spec, index)
        labels = measured + treatments + context
    out = []
    for term in terms:
        text = str(term).strip()
        if len(text) < 3 or text.lower() in JUNK_TERMS or generic(text) or \
                not (_canonical(_tokens(text)) - _canonical(DESCRIPTORS)):
            continue
        if text in L.OUTCOMES:                 # an outcome key a small model also listed as a criterion
            continue
        if labels and _matches(text, labels, spec, index):
            continue
        out.append(text)
    return out


def _approximation(intent):
    text = " ".join(str(intent.get("approximation") or "").split())[:300]
    return "" if len(text) < 12 or text.lower().strip(".") in ("none", "n/a", "no approximation") else text


def describe(spec):
    """Short design summary (for logs and tests)."""
    return (f"{spec.feasibility} | {spec.analysis} | {spec.base} | include {[criterion_text(c) for c in spec.include]} | "
            f"exposure {criterion_text(spec.exposure)} | comparator {criterion_text(spec.comparator)} | "
            f"outcome {spec.outcome} | variable {spec.variable} | stratify {spec.stratify}")
