"""
RWD study specification (what the study designer produces) and its validation.

A ``StudySpec`` only refers to library entries and concept ids, so it is checked completely before any SQL is
generated: unknown concepts, columns or operators, impossible windows, an exposure equal to its comparator or an
outcome that does not exist for the population are reported as errors; questionable choices become warnings.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

from mes.tools.rwd import library as L

OPS = (">", ">=", "<", "<=", "=", "!=", "between")
# Hours relative to time zero in which a medication / procedure / measurement counts, per role
DEFAULT_WINDOWS = {"include": [-72, 24], "exclude": [-72, 24], "exposure": [0, 24], "comparator": [0, 24]}
# covariates that are a part of an outcome (adjusting for them explains the outcome by itself)
OUTCOME_PARTS = {"invasive_vent_any": "invasive_vent_24h", "rrt_any": "rrt_24h"}


@dataclass
class Criterion:
    concept: str                       # concept id (see concepts.py)
    op: str = ""                       # numeric / categorical comparison for var: and lab:/item: concepts
    value: object = None               # number, [low, high] for "between", or category text
    window_h: list | None = None       # [start, end] hours from time zero (medications, procedures, measurements);
                                       # end None = until the end of the stay (ICU or hospital discharge)
    negate: bool = False               # "without ..."
    label: str = ""

    @staticmethod
    def from_dict(d):
        if isinstance(d, Criterion):
            return d
        if isinstance(d, str):
            return Criterion(concept=d)
        return Criterion(**{k: v for k, v in d.items() if k in Criterion.__dataclass_fields__})


@dataclass
class StudySpec:
    analysis: str                                   # library.ANALYSES
    base: str = "icu"                               # library.BASES
    title: str = ""
    include: list = field(default_factory=list)     # [Criterion]
    exclude: list = field(default_factory=list)
    exposure: Criterion | None = None               # treatment / risk factor / group of interest
    comparator: Criterion | None = None             # None: patients without the exposure
    grace_h: float = 24.0                           # compare: exposure must start within this many hours
    outcome: str = ""                               # library.OUTCOMES
    secondary_outcomes: list = field(default_factory=list)
    covariates: list = field(default_factory=list)  # library.VARIABLES (empty: default set of the base)
    stratify: list = field(default_factory=list)    # variables for rate / distribution / trend / survival
    variable: str = ""                              # distribution: variable (or lab:/item: concept)
    subgroups: list = field(default_factory=list)   # effect modification (binary / category variables)
    new_user: bool = True                           # compare: exclude prior users of either drug
    feasibility: str = "feasible"                   # feasible | partial | not_feasible
    reason: str = ""                                # why (not) feasible / design rationale
    sql_question: str = ""                          # explore: the descriptive question for the SQL explorer
    profile: list = field(default_factory=list)     # similar: the patient [{concept: var:…, value, weight, must,
                                                    # tolerance}] (characteristics of the first 24 h)
    k: int = 200                                    # similar: how many of the most similar stays
    group_by: str = ""                              # characterize / trajectory: "outcome", "exposure", a yes/no or
                                                    # category variable, or "" (one cohort)
    horizon_h: float = 72.0                         # trajectory / pathways: hours after time zero followed
    bin_h: float = 6.0                              # trajectory: the length of each interval (hours)
    treatments: list = field(default_factory=list)  # pathways: the treatments whose order is described [Criterion];
                                                    # top: the drug class the medications are ranked within
    ranked: str = ""                                # top: "drugs" | "diagnoses" | "procedures"

    @staticmethod
    def from_dict(d):
        d = dict(d)
        for key in ("include", "exclude"):
            d[key] = [Criterion.from_dict(c) for c in d.get(key) or []]
        for key in ("exposure", "comparator"):
            d[key] = Criterion.from_dict(d[key]) if d.get(key) else None
        d["treatments"] = [Criterion.from_dict(c) for c in d.get("treatments") or [] if c]
        return StudySpec(**{k: v for k, v in d.items() if k in StudySpec.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)

    def key(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _concept_kind(concept):
    return concept.split(":", 1)[0]


def criterion_text(c):
    """Readable criterion: label plus its comparison, e.g. "Highest lactate, first 24 h > 4"."""
    if c is None:
        return ""
    text = c.label or c.concept
    if c.op == "between" and isinstance(c.value, (list, tuple)):
        text += f" {c.value[0]:g}–{c.value[1]:g}"
    elif c.op:
        value = f"{c.value:g}" if isinstance(c.value, (int, float)) else c.value
        text += f" {c.op.replace('>=', '≥').replace('<=', '≤')} {value}"
    return f"without {text}" if c.negate else text


def _check_group(spec, variables, errors):
    """The groups of a description: the outcome (yes / no), the exposure, a yes/no or category variable, or none."""
    g = str(spec.group_by or "")
    if g == "outcome":
        if spec.outcome not in L.OUTCOMES or L.OUTCOMES[spec.outcome][1] != "binary":
            errors.append(f"{spec.analysis}: grouping by the outcome needs a yes/no outcome")
    elif g == "exposure":
        if spec.exposure is None:
            errors.append(f"{spec.analysis}: grouping by the exposure needs an exposure")
    elif g and (g not in variables or variables[g][1] not in ("binary", "category")):
        errors.append(f"{spec.analysis}: '{g}' cannot form groups (a yes/no or category variable of the population is "
                      "needed)")


def _check_profile(spec, variables, errors, warnings):
    """The described patient of a "patients like this" design: cohort variables of the population with a value of
    their kind; at least two; between 30 and 5000 similar stays."""
    from mes.tools.rwd.similar import MAX_K, MIN_K, features
    usable = features(spec.profile, spec.base)
    names = {f["name"] for f in usable}
    for f in spec.profile or []:
        concept = str((f or {}).get("concept") or "") if isinstance(f, dict) else ""
        if concept[4:] not in names:
            warnings.append(f"patient: {concept or f!r} is not a characteristic of {L.BASES[spec.base]['label']} with "
                            "a value: left out")
    if not usable:
        errors.append("similar: describe the patient with at least one characteristic of the data (age, sex, scores, "
                      "lab values, treatments …)")
    spec.profile = [{"concept": f"var:{f['name']}", "value": f["value"], "weight": f["weight"], "must": f["must"],
                     "tolerance": f["tolerance"]} for f in usable]
    try:
        k = int(spec.k)
    except (TypeError, ValueError):
        k = 200
    if not MIN_K <= k <= MAX_K:
        warnings.append(f"similar: {k} similar stays is outside {MIN_K}–{MAX_K}; {min(MAX_K, max(MIN_K, k))} used")
    spec.k = min(MAX_K, max(MIN_K, k))


def validate(spec, index):
    """Return (errors, warnings); fills defaults (covariates, windows, labels) in place."""
    errors, warnings = [], []
    if spec.analysis not in L.ANALYSES:
        errors.append(f"unknown analysis type '{spec.analysis}'")
    if spec.base not in L.BASES:
        errors.append(f"unknown population '{spec.base}'")
        return errors, warnings
    if spec.feasibility == "not_feasible":
        return errors, warnings
    variables = L.variables_for(spec.base)
    outcomes = L.outcomes_for(spec.base)

    def check(c, role):
        if c is None:
            return
        kind = _concept_kind(c.concept)
        if kind == "var":
            name = c.concept.split(":", 1)[1]
            if name not in variables:
                errors.append(f"{role}: '{name}' is not available for {L.BASES[spec.base]['label']}")
                return
            vkind = variables[name][1]
            continuous_ok = role == "exposure" and spec.analysis in ("association", "survival", "prognostic")
            if vkind == "numeric" and not c.op and continuous_ok:
                pass                     # continuous exposure: per SD and quartiles
            elif vkind == "numeric":
                if c.op not in OPS:
                    errors.append(f"{role}: numeric variable '{name}' needs a comparison (>, >=, <, <=, between)")
                elif c.op == "between":
                    if not (isinstance(c.value, (list, tuple)) and len(c.value) == 2):
                        errors.append(f"{role}: 'between' needs [low, high]")
                else:
                    try:
                        c.value = float(c.value)
                    except (TypeError, ValueError):
                        errors.append(f"{role}: '{name}' {c.op} needs a number")
            elif vkind == "category":
                if c.value in (None, ""):
                    errors.append(f"{role}: categorical variable '{name}' needs a value")
                c.op = c.op if c.op in ("=", "!=") else "="
        elif kind in ("lab", "item"):
            if c.op and c.op not in OPS:
                errors.append(f"{role}: unknown operator '{c.op}'")
            if not index.is_known(c.concept):
                errors.append(f"{role}: unknown concept {c.concept}")
        elif kind in ("phe", "drugclass", "drug", "dx9", "dx10", "px9", "px10", "micro"):
            if not index.is_known(c.concept):
                errors.append(f"{role}: unknown concept {c.concept}")
        else:
            errors.append(f"{role}: unknown concept {c.concept}")
            return
        if kind in ("drugclass", "drug", "px9", "px10", "lab", "item", "micro"):
            if not c.window_h:
                c.window_h = list(DEFAULT_WINDOWS.get(role.split(" ")[0], [-72, 24]))
            if len(c.window_h) != 2 or c.window_h[0] is None or (
                    c.window_h[1] is not None and float(c.window_h[0]) >= float(c.window_h[1])):
                errors.append(f"{role}: window {c.window_h} is not [start, end] hours (end: null = end of the stay)")
        c.label = c.label or index.label(c.concept)

    for c in spec.include:
        check(c, "include")
    for c in spec.exclude:
        check(c, "exclude")
    check(spec.exposure, "exposure")
    check(spec.comparator, "comparator")

    if spec.analysis == "similar":
        _check_profile(spec, variables, errors, warnings)
    if spec.analysis in ("characterize", "trajectory"):
        _check_group(spec, variables, errors)
    if spec.analysis in ("trajectory", "pathways"):
        try:
            spec.horizon_h, spec.bin_h = float(spec.horizon_h or 72), float(spec.bin_h or 6)
        except (TypeError, ValueError):
            spec.horizon_h, spec.bin_h = 72.0, 6.0
        if not 1 <= spec.horizon_h <= 720:
            errors.append(f"{spec.analysis}: follow 1 to 720 hours after time zero")
    if spec.analysis == "trajectory":
        from mes.tools.rwd.patterns import series_of
        if not series_of(spec.variable):
            errors.append("trajectory: choose a lab test or vital sign measured over time (a lab:/item: concept, or "
                          "lactate, creatinine, heart rate, MAP …)")
        elif _concept_kind(spec.variable) in ("lab", "item") and not index.is_known(spec.variable):
            errors.append(f"trajectory: unknown concept {spec.variable}")
        if not 1 <= spec.bin_h <= spec.horizon_h or spec.horizon_h / spec.bin_h > 60:
            errors.append("trajectory: intervals of 1 h or more, at most 60 of them")
    if spec.analysis == "pathways":
        spec.treatments = [c for c in spec.treatments if c is not None][:6]
        for c in spec.treatments:
            if _concept_kind(c.concept) not in ("drugclass", "drug", "px9", "px10"):
                errors.append(f"pathways: {c.concept} is not a treatment (a drug or a procedure)")
                continue
            c.window_h = list(c.window_h) if c.window_h else [0, float(spec.horizon_h)]
            check(c, "exposure")
        if len(spec.treatments) < 2:
            errors.append("pathways: name at least two treatments whose order is described")
    if spec.analysis == "top":
        if spec.ranked not in ("drugs", "diagnoses", "procedures"):
            errors.append("top: rank drugs, diagnoses or procedures")
        if not 0 <= float(spec.horizon_h or 0) <= 720:
            errors.append("top: the window is 0 (the whole stay) to 720 h")
        for c in spec.treatments:
            if _concept_kind(c.concept) != "drugclass" or not index.is_known(c.concept):
                errors.append(f"top: {c.concept} is not a drug class")
            c.label = c.label or index.label(c.concept)
    needs_outcome = spec.analysis in ("compare", "association", "survival", "prognostic", "similar")
    if spec.analysis in ("rate", "trend") and not spec.outcome and spec.exposure is None and not spec.variable:
        errors.append(f"{spec.analysis}: choose the outcome (or condition) whose rate is measured")
    if needs_outcome and not spec.outcome:
        errors.append(f"{spec.analysis}: an outcome is required")
    if spec.analysis in ("rate", "trend") and spec.outcome in L.OUTCOMES and L.OUTCOMES[spec.outcome][1] == "continuous":
        errors.append(f"{spec.analysis}: {L.OUTCOMES[spec.outcome][0].lower()} is a measurement, not a rate (a "
                      "distribution describes it)")
    for o in [spec.outcome] + list(spec.secondary_outcomes):
        if o and o not in outcomes:
            errors.append(f"outcome '{o}' is not available for {L.BASES[spec.base]['label']}")
    if spec.analysis in ("compare", "association", "utilization") and spec.exposure is None:
        errors.append(f"{spec.analysis}: an exposure is required")
    if spec.exposure and spec.comparator and spec.exposure.concept == spec.comparator.concept \
            and spec.exposure.op == spec.comparator.op and spec.exposure.value == spec.comparator.value:
        errors.append("exposure and comparator are the same")
    if spec.analysis == "compare" and spec.exposure and _concept_kind(spec.exposure.concept) == "var":
        warnings.append("the exposure is a patient characteristic, not a treatment: analysed as an adjusted association")
        spec.analysis = "association"
    if spec.analysis == "distribution" and not spec.variable:
        errors.append("distribution: choose the variable")
    if _concept_kind(spec.variable or "") == "derived":
        key = spec.variable.split(":", 1)[1]
        if key not in L.DERIVED:
            errors.append(f"variable '{spec.variable}' is not available")
        elif spec.base != "icu":
            errors.append(f"variable '{L.DERIVED[key][0]}' is recorded for ICU stays only")
    elif spec.variable and spec.variable not in variables and _concept_kind(spec.variable) not in ("lab", "item"):
        errors.append(f"variable '{spec.variable}' is not available")
    if spec.analysis == "explore" and not spec.sql_question:
        errors.append("explore: describe the question for the SQL explorer")
    for name in list(spec.stratify) + list(spec.subgroups):
        if name not in variables:
            errors.append(f"'{name}' is not a variable of this population")
    adjusts = spec.analysis in ("compare", "association", "prognostic", "similar", "characterize")  # (Table 1)
    if not adjusts:
        spec.covariates = []              # descriptive analyses do not adjust
    elif not spec.covariates:
        spec.covariates = list(L.DEFAULT_COVARIATES[spec.base])
    unknown = [c for c in spec.covariates if c not in variables]
    if unknown:
        warnings.append(f"covariates not available and dropped: {', '.join(unknown)}")
        spec.covariates = [c for c in spec.covariates if c in variables]
    # the exposure's own variable is no confounder of itself (women vs men adjusted for sex cannot be estimated)
    own = {c.concept[4:] for c in (spec.exposure, spec.comparator) if c is not None and c.concept.startswith("var:")}
    # nor is the first day of the outcome itself (ventilation in the first 24 h for ventilation during the stay)
    own |= {OUTCOME_PARTS[o] for o in [spec.outcome] if o in OUTCOME_PARTS}
    spec.covariates = [c for c in spec.covariates if c not in own]
    if not 0 <= float(spec.grace_h) <= 168:
        errors.append("grace period must be between 0 and 168 hours")
    return errors, warnings
