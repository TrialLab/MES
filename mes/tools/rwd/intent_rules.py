"""
Deterministic repairs of the designer's intent before its terms are resolved — the slips a small local model made in
the field, each turned into a rule:

- a term that names a cohort variable by itself ("age", "SOFA", "hours_admit_to_icu") is that variable: without a
  threshold it is a breakdown or the variable described, never a criterion (once "age" became the ICD code
  "age-related physical debility" and 9 patients were left); a variable of the other population moves the base;
- a term that names an outcome ("icu_readmit_30d", "invasive_vent_any") is the outcome, not the exposure: two groups
  of an outcome compared ("readmitted vs not") are a characterization by the outcome;
- the same characteristic with two values among the criteria ("sex = female", "sex = male") is the comparison, not a
  criterion (no patient has both);
- generic words ("drugs", "lab tests", "drug name/type") stand for no concept;
- ICU types named in words ("medical ICU") are levels of the ICU type; two levels compared are a breakdown by it.

    normalize(intent, question, base) -> (intent, base, notes)
"""

import copy
import re

from mes.tools.rwd import library as L

# words around a variable's name that do not change which variable it is ("age group", "SOFA score", "the lactate")
_FILLER = re.compile(r"\b(?:the|a|an|patients?'?|group|groups|band|bands|categor(?:y|ies)|levels?|scores?|values?|"
                     r"distribution|at admission|on admission|first 24 ?h|in the first 24 hours)\b", re.I)
DATA_WORDS = re.compile(r"^(?:all |any |which |the |most common |commonest )*(?:drugs?|medications?|medicines?|"
                        r"drug (?:names?|types?|class(?:es)?)(?:\s*/\s*types?)?|treatments?|therap(?:y|ies)|"
                        r"lab(?:oratory)?(?: tests?)?|tests?|procedures?|diagnos(?:is|es)|conditions?|"
                        r"comorbidit(?:y|ies)|variables?|data|records?|items?|orders?)$", re.I)
# an ICU type in words: the kind of unit right before "ICU" / "intensive care (unit)" / "care unit", or its abbreviation
UNIT_PHRASE = re.compile(r"\b(medical[\s/-]+surgical|medical|surgical|cardiac[\s-]+surg\w*|cardio[\s-]*(?:vascular|"
                         r"thoracic)\w*|coronary(?:\s+care)?|cardiac|trauma|neuro\w*)\s+(?:intensive\s+care(?:\s+unit)?|"
                         r"icu|care\s+unit|unit)\b|\b(micu\s*/\s*sicu|micu|sicu|cvicu|ccu|tsicu|neuro\s*icu)\b", re.I)
ICU_TERM = re.compile(r"\b(?:icu|intensive care|care unit)\b", re.I)    # "ICU admission type" = "Medical"
UNIT_WORDS = (("MICU/SICU", r"medical[\s/-]+surgical|micu\s*/\s*sicu"), ("CVICU", r"cardiac[\s-]+surg|cardio|cvicu"),
              ("CCU", r"coronary|cardiac|\bccu"), ("TSICU", r"trauma|tsicu"), ("Neuro", r"neuro"),
              ("MICU", r"medical|\bmicu"), ("SICU", r"surgical|\bsicu"))
NUMBER = re.compile(r"^\s*[<>=≥≤]*\s*-?\d+(?:\.\d+)?\s*$")
# outcomes of the stay that also have a first-24-h flag (a treatment that can be an exposure)
FIRST_DAY_TWIN = {"invasive_vent_any": "invasive_vent_24h", "rrt_any": "rrt_24h"}
# the words of a sex as a population term ("women", "male")
SEX_WORDS = {"male": "M", "males": "M", "men": "M", "man": "M", "男性": "M", "男": "M", "female": "F", "females": "F",
             "women": "F", "woman": "F", "女性": "F", "女": "F"}


def _norm(text):
    text = re.sub(r"[_\s]+", " ", str(text or "").lower())
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip(" .,:;-")


def _names(name):
    label = L.VARIABLES[name][0]
    out = {_norm(name), _norm(label), _norm(label.split(",")[0])}
    out |= {_norm(s) for s in L.VARIABLE_SYNONYMS.get(name, [])}
    if name == "gender":
        out |= {"sex", "gender"}
    return {n for n in out if n}


_ALL_NAMES = None


def exact_variable(term):
    """The cohort variable a term names by itself (its name, label or a synonym), or None."""
    global _ALL_NAMES
    if _ALL_NAMES is None:
        _ALL_NAMES = {n: name for name in L.VARIABLES for n in _names(name)}
    return _ALL_NAMES.get(_norm(term))


def exact_outcome(term, question=""):
    """The outcome a term names by itself — its key or label ("icu_readmit_30d", "ICU readmission within 30 days"),
    or a readmission / ventilation / dialysis phrase — or None."""
    raw = str(term or "").strip()
    if raw in L.OUTCOMES:
        return raw
    t = _norm(raw)
    for key, spec in L.OUTCOMES.items():
        if t in (_norm(key), _norm(spec[0]), _norm(spec[0].split("(")[0])):
            return key
    if re.fullmatch(r"(?:icu |hospital )?readmi\w*(?: to the (?:icu|hospital))?(?: within 30 days)?", t):
        icu = re.search(r"\b(?:icu|intensive care)\b[^.?]{0,25}readmi|readmi\w*\s+(?:to|in)\s+(?:the\s+)?(?:icu|"
                        r"intensive care)", f"{raw} {question}", re.I)
        return "icu_readmit_30d" if icu or t.startswith("icu") else "hosp_readmit_30d"
    if re.fullmatch(r"(?:invasive )?(?:mechanical )?ventilation|intubation|(?:receiv\w+ )?(?:invasive )?mechanical "
                    r"ventilation", t):
        return "invasive_vent_any"
    if re.fullmatch(r"(?:renal replacement therapy|dialysis|rrt|crrt)", t):
        return "rrt_any"
    return None


def careunit_level(text, bare=False):
    """The ICU type ``text`` names ("medical ICU" → MICU), or None; ``bare``: the word alone ("medical") will do."""
    m = UNIT_PHRASE.search(str(text or ""))
    words = m.group(0) if m else (str(text or "") if bare else "")
    return next((level for level, pattern in UNIT_WORDS if words and re.search(pattern, words, re.I)), None)


def _term(t):
    return str((t or {}).get("term") or "").strip() if isinstance(t, dict) else ""


def _threshold(t):
    """A numeric threshold ("> 4", "between 2 and 4"): a criterion of a numeric variable."""
    op, value = str(t.get("op") or ""), t.get("value")
    if op in ("", "=", "!="):
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 2 and all(NUMBER.match(str(v)) for v in value)
    return bool(NUMBER.match(str(value or ""))) or bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*(?:-|–|and|to)\s*-?\d+"
                                                                       r"(?:\.\d+)?\s*", str(value or "")))


def _two_values(population):
    """Whether the criteria name one characteristic with two values ("sex = female", "sex = male")."""
    seen = {}
    for t in population or []:
        v = exact_variable(_term(t)) if isinstance(t, dict) else None
        if v and str(t.get("op") or "=") in ("=", "") and str(t.get("value") or "").strip():
            seen.setdefault(v, set()).add(str(t.get("value")).strip().lower())
    return any(len(values) > 1 for values in seen.values())


def normalize(intent, question="", base="icu", asked=None):
    """(intent, base, notes): the intent with the rules above applied (the model's own words are kept otherwise).
    ``asked(outcome, text)``: whether the question asks about an outcome (designer.asked_about)."""
    intent = copy.deepcopy(intent) if isinstance(intent, dict) else {}
    notes, analysis = [], intent.get("analysis")
    stratify = [s for s in intent.get("stratify") or [] if isinstance(s, str)]

    # ICU types named in words: a level of the ICU type
    for slot in ("exposure", "comparator"):
        t = intent.get(slot)
        if not isinstance(t, dict):
            continue
        level = careunit_level(_term(t)) or (careunit_level(t.get("value"), bare=True)
                                              if exact_variable(_term(t)) == "careunit" or ICU_TERM.search(_term(t))
                                              else None)
        if level:
            intent[slot] = {**t, "term": "ICU type", "op": "=", "value": level}
    exp, comp = intent.get("exposure") or {}, intent.get("comparator") or {}
    # two levels of one category compared (MICU vs SICU, women vs men): results by that characteristic
    e_var, c_var = exact_variable(_term(exp)), exact_variable(_term(comp))
    if e_var and e_var == c_var and L.VARIABLES[e_var][1] == "category" and analysis not in ("characterize",) and \
            str(exp.get("value") or "") != str(comp.get("value") or ""):
        intent["exposure"], intent["comparator"] = None, None
        stratify = [e_var] + [s for s in stratify if s != e_var]
        outcome = intent.get("outcome") or ""
        kind = L.OUTCOMES.get(outcome, ("", ""))[1]
        if analysis in ("compare", "association", "rate", "distribution", "explore", None, ""):
            intent["analysis"] = analysis = "distribution" if kind == "continuous" else "rate"
            if kind == "continuous":
                intent["variable"] = L.OUTCOMES[outcome][3]
        notes.append(f"{L.VARIABLES[e_var][0]} {exp.get('value')} vs {comp.get('value')}: results by "
                     f"{L.VARIABLES[e_var][0].lower()}")
        exp, comp = {}, {}

    # an outcome named as the exposure (and as its comparator): the outcome, the groups being its two values
    e_out = exact_outcome(_term(exp), question)
    c_out = exact_outcome(_term(comp), question) if e_out else None
    old = intent.get("outcome") or ""
    free = not old or (asked is not None and not asked(old, question))
    if e_out and not free and e_out in FIRST_DAY_TWIN:
        # "survival of ventilated vs non-ventilated patients": the outcome asked stays, the treatment is the flag
        twin = L.VARIABLES[FIRST_DAY_TWIN[e_out]][0]
        intent["exposure"] = {**exp, "term": twin, "op": "", "value": ""}
        if c_out == e_out:
            intent["comparator"] = None
        notes.append(f"“{_term(exp)}” as the exposure: {twin.lower()}")
        exp, comp = intent["exposure"], intent.get("comparator") or {}
    elif e_out and free and e_out in L.outcomes_for(base if base in L.BASES else "icu"):
        intent["exposure"] = None
        if c_out == e_out:
            intent["comparator"] = None
        intent["outcome"] = e_out
        differ = re.search(r"\b(?:differ\w*|characteristics?|compar\w+|vs\.?|versus|unlike|like)\b", question, re.I)
        if (differ or c_out == e_out and not _two_values(intent.get("population"))) and \
                analysis not in ("characterize", "trajectory"):
            # the two groups of the outcome compared ("readmitted vs not") — unless the criteria hold the groups
            # ("women vs men": the exposure, the outcome being ventilation)
            intent["analysis"] = analysis = "characterize"
        notes.append(f"“{_term(exp)}” is an outcome: {L.OUTCOMES[e_out][0].lower()}"
                     + (f" (instead of {old})" if old and old != e_out else ""))
        exp, comp = {}, intent.get("comparator") or {}

    # the same characteristic twice among the criteria with two values: the two groups compared
    population = [t for t in intent.get("population") or [] if isinstance(t, dict)]
    by_var = {}
    for t in population:
        v = exact_variable(_term(t))
        if v and str(t.get("op") or "=") in ("=", "") and str(t.get("value") or "").strip():
            by_var.setdefault(v, []).append(t)
    for v, ts in by_var.items():
        if len({str(t.get("value")).strip().lower() for t in ts}) > 1:
            population = [t for t in population if t not in ts]
            intent["exclude"] = [t for t in intent.get("exclude") or [] if exact_variable(_term(t)) != v]
            if not _term(intent.get("exposure")):
                intent["exposure"] = {"term": ts[0]["term"], "op": "=", "value": ts[0]["value"], "within_hours": 0}
                intent["comparator"] = None
            notes.append(f"{L.VARIABLES[v][0]} {' and '.join(str(t['value']) for t in ts)}: the groups compared, not "
                         "criteria")

    # the exposure names a measurement without a threshold: the variable described, or a breakdown
    exp = intent.get("exposure") or {}
    e_var = exact_variable(_term(exp))
    if e_var and L.VARIABLES[e_var][1] == "numeric" and not _threshold(exp):
        if analysis == "distribution":
            intent["variable"] = intent.get("variable") or e_var
            intent["exposure"] = intent["comparator"] = None
        elif analysis in ("rate", "trend", "utilization"):
            stratify = stratify + [e_var] if e_var not in stratify else stratify
            intent["exposure"] = intent["comparator"] = None
            notes.append(f"“{_term(exp)}” without a threshold: results by {L.VARIABLES[e_var][0].lower()}")

    # generic words for no concept, and variables named without a threshold: breakdowns, not criteria
    kept = []
    for t in population:
        sex = SEX_WORDS.get(_term(t).lower())
        if sex and str(t.get("value") or "").strip().lower() in ("", "yes", "true", "1"):
            kept.append({"term": "sex", "op": "=", "value": sex})     # "women": the criterion, not a breakdown
            continue
        v = exact_variable(_term(t))
        if v == "any_icu" and base == "icu":             # every ICU stay is one: the criterion restates the base
            continue
        if DATA_WORDS.match(_norm(_term(t)) or "x"):
            notes.append(f"“{_term(t)}” names no concept: not a criterion")
            continue
        if v and L.VARIABLES[v][1] in ("numeric", "category") and not _threshold(t) and \
                not (L.VARIABLES[v][1] == "category" and str(t.get("value") or "").strip()):
            if L.VARIABLES[v][1] == "numeric" and analysis == "distribution" and not intent.get("variable"):
                intent["variable"] = v
            elif v not in stratify and analysis in ("rate", "trend", "survival", "characterize", "distribution",
                                                    "utilization"):
                stratify.append(v)
            notes.append(f"“{_term(t)}” is a characteristic without a threshold: not a criterion"
                         + (" (results by it)" if v in stratify else ""))
            continue
        kept.append(t)
    intent["population"] = kept
    intent["exclude"] = [t for t in intent.get("exclude") or [] if isinstance(t, dict)
                         and not DATA_WORDS.match(_norm(_term(t)) or "x")]
    for slot in ("exposure", "comparator"):
        t = intent.get(slot)
        if isinstance(t, dict) and _term(t) and DATA_WORDS.match(_norm(_term(t)) or "x"):
            intent[slot] = None
            notes.append(f"“{_term(t)}” names no concept: no {slot}")
    intent["stratify"] = stratify

    # a variable named by itself that only the other population has moves the base
    named = [exact_variable(intent.get("variable"))] + [exact_variable(_term(intent.get(s))) for s in ("exposure",)]
    for v in [v for v in named if v]:
        bases = L.VARIABLES[v][3]
        if base not in bases and len(bases) == 1:
            notes.append(f"{L.VARIABLES[v][0]} is recorded for {L.BASES[bases[0]]['unit']}s: that population")
            base = bases[0]
            intent["base"] = base
            break
    return intent, base, notes
