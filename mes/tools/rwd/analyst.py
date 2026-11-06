"""
MIMIC-IV data analyst: Claude explores the data and writes its own analysis code, for questions the validated study
templates cannot answer (or when the user asks for a custom analysis).

The loop runs within the answer mode's budget (``ModeConfig.analyst_*``: Fast, Balanced, Deep).
Each model turn has a short plan and one action:

    <concepts>term</concepts>   dictionary search (curated phenotypes, drug classes, variables, ICD codes, labs …)
    <sql>SELECT …</sql>         quick aggregate query (checked like the exploratory SQL tier)
    <python>…</python>          code in a sandboxed Python session (sandbox.py) with MES's tested building blocks
    <finish>…</finish>          end, once ``submit(result)`` was accepted (may follow the submitting code block)

Then, at the same time: an independent review of design and code (a second model call with an epidemiological
checklist) and a reproduction (the cells the result came from re-run in a fresh session must give it again). Major
problems go back to the analyst for a revision when the mode allows it (Balanced 1, Deep 2); unresolved ones are
reported with the result. Every number of the answer comes from ``submit()``, i.e. from code that ran.

Privacy: the model sees the schema, dictionary entries and what its code prints (aggregates; identifiers and dates
masked); patient-level rows stay in the sandbox process.
"""

import hashlib
import json
import logging
import re
import threading
import time

from mes import config
from mes.tools.rwd import DATASET, code_audit, explorer
from mes.tools.rwd import library as L
from mes.tools.rwd import stats
from mes.tools.rwd.result_check import __doc__ as RESULT_FORMAT
from mes.tools.rwd.sandbox import Sandbox, mask_output
from mes.utils import parse_json_response

logger = logging.getLogger(__name__)

ACTION = re.compile(r"<(python|sql|concepts|finish)>(.*?)</\1>", re.S)
OPEN_ACTION = re.compile(r"<(python|sql|concepts)>(.*)$", re.S)        # a reply cut off before its closing tag
PLAN = re.compile(r"<plan>(.*?)</plan>", re.S)
FENCE = re.compile(r"^\s*```(?:python|sql)?\s*\n?(.*?)\n?```\s*$", re.S)
STEP_TOKENS = {"low": 2500, "medium": 3500, "high": 5000}
# Time kept at the end of the budget for the review and the reproduction (they run at the same time)
CHECK_RESERVE = {"osler": 15, "sackett": 35, "snow": 45}

# The statistics API as the model needs it: argument types and what comes back (field runs showed whole steps lost
# to guessing return shapes)
STATS_API = """\
    arrays: numpy arrays or pandas Series of equal length; t = 0/1 treatment indicator; rates are fractions
    stats.wilson(k, n) -> (p, lo, hi)
    stats.summarize(values) -> {n, missing, mean, sd, median, q1, q3, p5, p95, min, max}
    stats.design_matrix(df, columns, kinds=None) -> (X, missing_share)
        kinds: {column: "numeric" | "binary" | "category"} (None: inferred); numeric → median-imputed plus a
        <column>__missing indicator, category → one-hot (8 most frequent levels); X: float DataFrame on df.index
    stats.propensity(X, t) -> (ps, auc)          logistic model with spline terms of continuous covariates
    stats.iptw(ps, t, trim=(0.01, 0.99)) -> (w, info)   stabilized ATE weights; info: truncated_share, ess_treated,
        ess_control, overlap
    stats.overlap_weights(ps, t) -> (w, info)    overlap weights (balance the propensity model's covariates exactly)
    stats.smd(X, t, w=None) -> DataFrame [covariate, mean_treated, mean_control, smd]
    stats.weighted_risks(y, t, w, n_boot=1000) -> {risk_treated, risk_control, rd, rd_ci, rr, rr_ci}
    stats.weighted_or(y, t, w, X=None) -> {value, lo, hi, p}           doubly robust with X
    stats.mean_difference(y, t, w=None, X=None) -> {value, lo, hi, p}
    stats.cox(time, event, x, w=None, X=None) -> {value, lo, hi, p}    hazard ratio of x
    stats.kaplan_meier(time, event, groups, w=None, horizon=None) -> (curves, logrank_p); curves = {group: {t, s,
        at_horizon, n}}
    stats.e_value(value, lo, hi, measure="RR", rare=True) -> (e_point, e_ci)
    stats.match_1to1(ps, t, caliper_sd=0.2) -> [(treated_position, control_position)]
    stats.prognostic_model(X, y, folds=5) -> {auc, calibration_slope, odds_ratios_per_sd: [{term, value}]}
    Typical comparison: X, _ = stats.design_matrix(df, covariates); t = df.treated.values;
        ps, auc = stats.propensity(X, t); w, info = stats.overlap_weights(ps, t); balance = stats.smd(X, t, w);
        risks = stats.weighted_risks(df[outcome].values, t, w)"""

# Facts about the data the analyst and the reviewer must not get wrong
DATA_FACTS = """\
- Deaths: in-hospital (deathtime) and, from state death records, after discharge up to one year after the patient's
  last hospital discharge. death_28d / death_90d / death_1y (from ICU admission in mes.icu_stays, from hospital
  admission in mes.admissions) are therefore fully ascertained: no censoring is needed, a patient without a death
  date is alive at one year. Survival beyond one year cannot be measured.
- mimiciv_derived tables are the official MIT-LCP mimic-code concepts (sepsis3: suspected infection with SOFA ≥ 2,
  the Sepsis-3 definition; kdigo_stages; first_day_sofa …); mes.* variables are built from them. Using them is
  standard practice (Sepsis-3 by this definition also includes post-operative patients, e.g. cardiac surgery).
- mimiciv_derived.norepinephrine_equivalent_dose adds vasopressin (and phenylephrine, dopamine, epinephrine) to
  norepinephrine: when vasopressin is the treatment, take the baseline catecholamine dose from the tables of the
  single drugs (norepinephrine, epinephrine, …), not from this sum.
- mimiciv_derived.sofa has one row per ICU hour (hr 0 = the clock hour of ICU admission): respiration … renal score
  that hour, *_24hours / sofa_24hours the worst of the 24 hours up to it. first_day_sofa and the first-day / *_24h
  columns of mes.icu_stays summarise the whole first day."""


def status(llm=None):
    """(available, reason) of the analyst for the configured backend."""
    from mes.llm_zoo import get_provider
    from mes.tools.rwd.db import get_store
    setting = config.RWD_ANALYST
    if setting == "off":
        return False, "the LLM analyst is turned off (MES_RWD_ANALYST=off)"
    if not get_store().available:
        return False, "the MIMIC-IV store is not built"
    if setting == "on":
        return True, ""
    try:
        provider = get_provider()
    except Exception:
        provider = ""
    if provider == "bedrock":
        return True, ""
    return False, "the LLM analyst writes its own analysis code; set MES_RWD_ANALYST=on to use it with your model"


# ============================================================================ prompts
def _stats_doc():
    missing = [name for name in re.findall(r"stats\.(\w+)\(", STATS_API) if not hasattr(stats, name)]
    assert not missing, f"STATS_API documents functions that do not exist: {missing}"
    return STATS_API


def _variables_doc():
    groups = {}
    for name, (label, kind, unit, bases, group) in L.VARIABLES.items():
        groups.setdefault(group, []).append(f"{name} ({label}{', ' + unit if unit else ''}"
                                            f"{'' if 'icu' in bases and 'hospital' in bases else ', ' + '/'.join(bases)})")
    out = [f"  {g}: " + "; ".join(v) for g, v in groups.items()]
    out.append("  outcomes: " + "; ".join(f"{k} ({v[0]})" for k, v in L.OUTCOMES.items()))
    return "\n".join(out)


SYSTEM = """You are the MES data analyst: an expert clinical epidemiologist and biostatistician. You answer a clinical
question with a new analysis of MIMIC-IV v2.2 (Beth Israel Deaconess Medical Center, Boston; adult hospital and ICU
stays 2008-2019; dates shifted per patient). You work in a persistent Python session on the full database. Only what
your code prints is shown to you, and only the result you submit() is reported — an answer is written from it.

# How to work
Every reply: a short <plan>…</plan> (one or two sentences: what this step does) and exactly ONE action:
<concepts>term</concepts>  search the MIMIC-IV dictionary (curated phenotypes, drug classes, cohort variables, ICD-9/10
                           diagnoses and procedures, lab tests, ICU items, medications) with patient counts
<sql>SELECT …</sql>        quick aggregate DuckDB query to look at the data (≤ 200 rows, no identifier columns,
                           counts < 10 shown as "<10"); compute results in <python> with sql(), never by typing
                           numbers you saw (submit() refuses typed-in counts)
<python>
…code…
</python>                  Python in the session (variables persist between steps)
<finish>summary</finish>   end, after submit() was accepted: two or three sentences on what you found
You may end a reply with <finish> right after the <python> block that calls submit(), to save a step.
Budget: {steps} turns and about {seconds} seconds in total. Work efficiently: explore only what you need, do the
work of several steps in one cell, aggregate in SQL before loading into pandas, submit as soon as the main result is
ready (you may submit again to improve it). When little time is left, submit what you have.
{depth}

# Data (DuckDB SQL)
{schema}
mimiciv_derived tables (official mimic-code concepts): {derived}
mes.icu_stays / mes.admissions variables:
{variables}
Data facts:
{facts}

# Python session
Pre-loaded: np, pd and
  sql(query) -> DataFrame — read-only DuckDB on every table above (patient rows stay in the session)
  cohort(spec) -> object with .df (one row per patient), .attrition [(step, n)], .sql — MES's validated cohort
      builder. spec = {{"analysis": "rate" | "compare" | "association" | "survival" | ..., "base": "icu" | "hospital",
      "include": [criterion], "exclude": [criterion], "exposure": criterion, "comparator": criterion or None,
      "outcome": outcome key, "grace_h": 24, "covariates": [variable names]}};
      criterion = {{"concept": id from <concepts> (var:, phe:, drugclass:, drug:, dx10:, lab: …), "op": ">",
      "value": 4, "window_h": [0, 24]}}. "compare" applies the landmark at grace_h, new users only and drops
      patients whose outcome was present before the landmark; .df then has `treated`. Columns: unit_id, subject_id,
      hadm_id, t0, dischtime, death_time, days_to_death, death_hosp, inc_*/exc_* flags, exposed, comparator,
      exposure_hours, the outcome and the covariates. Use it when it fits and extend its .df with sql() joins.
  stats — tested statistics (rates as fractions; ratios as dicts with value, lo, hi, p):
{stats}
  L — MES library: L.VARIABLES, L.OUTCOMES, L.PHENOTYPES {{id: (label, ICD-10 prefixes, ICD-9 prefixes, synonyms)}},
      L.DRUG_CLASSES; code_predicate(icd10_prefixes, icd9_prefixes) -> SQL condition on mes.diagnoses
  drug_names("drugclass:statin") -> medication names in mes.drugs.drug
  show(df, n=30) prints a table; submit(result) reports the result (see below)
Allowed imports: numpy, pandas, scipy, statsmodels, lifelines, sklearn, math, statistics, itertools, collections,
re, json, datetime. No files, network, os, sys, open, eval, getattr. Random seeds are fixed and sql() rows
without ORDER BY come back sorted by subject_id/hadm_id/stay_id, so a re-run gives the same numbers.

# Method (state each choice in "methods")
- Unit of analysis: one row per patient — the first ICU stay (mes.icu_stays WHERE first_icu_stay AND age >= 18) or
  the first admission (mes.admissions WHERE first_admission AND age >= 18) unless the question needs otherwise.
- Time zero: ICU (or hospital) admission. Eligibility and covariates from information available at time zero
  (first-24-hour values are the usual compromise in ICU data — say so). Exposures in a stated window after time
  zero; outcomes after it. Never select patients on information from the future (immortal time).
- Treatment comparisons emulate a target trial: new users; exposure within a grace period; follow-up from a landmark
  at its end among patients alive and in hospital; exclude outcomes already present at the landmark; adjust for
  confounding (propensity weighting with balance check, SMD < 0.1 — stats.overlap_weights when inverse probability
  weights leave imbalance — or multivariable regression); report crude and adjusted estimates.
- Confounders must be measured before the treatment can start. When the exposure window is shorter than 24 h
  (e.g. within 6 h of ICU admission) do NOT adjust for first-day summaries (first_day_* tables, sofa / sapsii /
  oasis / *_max / *_min / *_24h columns of mes.icu_stays): they are partly consequences of the treatment. Use the
  values of the window before it — mimiciv_derived.sofa is hourly (hr 0 = the hour of ICU admission), vitalsign /
  bg / chemistry have charttime, the vasopressor tables give the dose at baseline — and include the
  treatment-intensity markers that drive the decision (e.g. the catecholamine dose, not a 0/1 flag; see the data
  facts for norepinephrine_equivalent_dose).
- Exclude prevalent users (already treated before time zero, e.g. in the ED or OR: link drugs by hadm_id) or
  analyse them separately. The comparator is "not treated within the window" (later starters stay in it, or are
  censored with weights) — never "never treated during the stay", which uses future information.
- Do not drop patients for an event after time zero other than death (e.g. early ICU discharge); say what the
  weights estimate (overlap weights: the overlap population).
- Missing values are not negatives: exclude or impute explicitly and say how many.
- Joins can duplicate rows: count distinct patients and check that denominators add up.
- Rates as percentages with Wilson 95% CI, n and events; medians with IQR; ratios with 95% CI and p.
- What the data cannot measure (e.g. survival beyond one year, outpatient outcomes): submit {{"feasible": False,
  "reason": …}}; or, when a nearby question still helps (one-year instead of five-year survival), answer that one,
  name it in the title and add a warning that the question as asked cannot be answered.
- Privacy: print only aggregates; never print rows of individual patients, identifiers or dates.

# Result
{result_format}
Mark sensitivity analyses with "sensitivity": True and the main estimate with "primary": True."""

DEPTH = {
    "osler": "Depth Fast: answer the question directly with the main analysis; no extras.",
    "sackett": "Depth Balanced: the main analysis with its essential diagnostics (balance, missing data, "
               "event counts).",
    "snow": "Depth Deep: the main analysis, its diagnostics and at least one sensitivity analysis (another "
            "definition, adjustment set or method), marked as sensitivity.",
}

REVIEW = """You review an observational analysis of MIMIC-IV (adult ICU and hospital stays, one US hospital) that
another analyst wrote and ran to answer a clinical question. Judge the design and the code, not the writing.

QUESTION: {question}

DATA FACTS (true of this database; do not report them as problems):
{facts}

ANALYSIS CODE (cells in the order they ran; a cell stopped by an error still ran up to the error):
{code}

KEY OUTPUTS:
{outputs}

SUBMITTED RESULT:
{result}

CODE CHECK (each ICD code, medication pattern or name list, lab / ICU item and table of the code, looked up in
MIMIC-IV by a program; a PROBLEM is certain, a note needs your judgment):
{audit}

Check: (1) population and unit of analysis (one row per patient, adults, first stay) match the question;
(2) time zero, and eligibility / covariates measured at baseline; (3) exposure definition and window — no immortal
time or selection on future information; codes and medication patterns select what the question means (see the
CODE CHECK: a pattern that also matches other medications, or codes that miss part of the condition, is a problem); (4) outcome definition and window, after time zero, prevalent outcomes;
(5) confounding handled for causal questions (descriptive questions need none), no adjustment for post-exposure
variables; (6) missing data not treated as negative; (7) denominators and counts consistent, no duplicated rows from
joins; (8) statistics and units correct (percent vs fraction, CI, measure); (9) the result answers the question;
(10) the methods text matches the code.
Return JSON: {{"verdict": "sound" | "minor_issues" | "major_issues", "issues": [{{"severity": "major" | "minor",
"problem": "...", "fix": "..."}}], "summary": "one sentence"}}. Major = the main estimate is likely wrong or answers a
different question. Limits every MIMIC-IV study shares (one center, observational, residual confounding) and
stated, standard choices are not problems of this analysis; mention them as minor at most."""

REVIEW_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["sound", "minor_issues", "major_issues"]},
    "issues": {"type": "array", "items": {"type": "object", "properties": {
        "severity": {"type": "string", "enum": ["major", "minor"]}, "problem": {"type": "string"},
        "fix": {"type": "string"}}, "required": ["severity", "problem", "fix"]}},
    "summary": {"type": "string"}}, "required": ["verdict", "issues", "summary"]}

_system_cache = {}


def _system(mode_cfg, store):
    key = (mode_cfg.key, str(store.root))
    if key not in _system_cache:
        _system_cache[key] = SYSTEM.format(
            steps=mode_cfg.analyst_steps, seconds=mode_cfg.analyst_seconds, depth=DEPTH.get(mode_cfg.key, ""),
            schema=explorer.SCHEMA_GUIDE, derived=", ".join(store.tables("mimiciv_derived")),
            variables=_variables_doc(), stats=_stats_doc(), facts=DATA_FACTS,
            result_format=RESULT_FORMAT.split("Format (all keys optional except title and estimates)::", 1)[-1].strip())
    return _system_cache[key]


def _found_concepts(plan, index):
    lines = []
    for entity in (plan.entities or [])[:6]:
        hits = index.search(entity.get("name", ""), k=3)
        if hits:
            lines.append(f"- {entity.get('name')}: " + "; ".join(h.describe() for h in hits))
    return "\n".join(lines)


def _task(question, plan, index, mode_cfg):
    pico = "; ".join(f"{k}: {v}" for k, v in (plan.pico or {}).items() if v)
    found = _found_concepts(plan, index)
    return (f"QUESTION: {question}\n" + (f"PICO: {pico}\n" if pico else "")
            + (f"Dictionary matches for the question's terms (check with <concepts> if unsure):\n{found}\n" if found else "")
            + f"Budget: {mode_cfg.analyst_steps} turns, about {mode_cfg.analyst_seconds} s. Start with your plan.")


# ============================================================================ actions
def concept_text(index, term, k=10):
    """Dictionary search result as text for the model."""
    lines = []
    for c in index.search(term, k=k):
        line = c.describe()
        kind, _, key = c.concept_id.partition(":")
        if kind == "drugclass":
            names = index.drug_names_for(c.concept_id)
            line += f" · {len(names)} medication names, e.g. {', '.join(names[:6])}"
        elif kind == "phe":
            _label, i10, i9, _syn = L.PHENOTYPES[key]
            line += f" · ICD-10 {', '.join(i10[:8])}; ICD-9 {', '.join(i9[:8])}"
        elif kind == "var":
            line += f" · column {key} of mes.icu_stays / mes.admissions"
        elif kind == "lab":
            line += f" · mimiciv_hosp.labevents itemid {key}"
        elif kind == "item":
            line += f" · mimiciv_icu itemid {key}"
        elif kind in ("dx9", "dx10", "px9", "px10"):
            table = "mes.diagnoses" if kind.startswith("dx") else "mes.procedures (source 'icd')"
            line += f" · {table}: icd_version {kind[2:]}, code LIKE '{key}%'"
        elif kind == "drug":
            line += f" · mes.drugs.drug names containing '{key}'"
        lines.append(line)
    return "\n".join(lines) or "no matching concepts"


def sql_text(store, query):
    query = query.strip().rstrip(";")
    problem = explorer.check_sql(store, query)
    if problem:
        return f"SQL not run: {problem}"
    try:
        df = store.query(query, timeout=60)
    except Exception as e:
        return f"SQL error: {type(e).__name__}: {str(e)[:600]}"
    problem = explorer.check_result(df)
    if problem:
        return f"Result not shown: {problem}"
    return mask_output(explorer.suppress(df).head(80).to_string(), 5000)


def parse_reply(text):
    """(plan, action, body, finish_text) of a model reply; action None if the reply has no valid action."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    plan = (PLAN.search(text).group(1).strip() if PLAN.search(text) else "")
    actions = ACTION.findall(text)
    finish = next((body.strip() for kind, body in actions if kind == "finish"), None)
    work = [(kind, body) for kind, body in actions if kind != "finish"]
    if work:
        kind, body = work[0]
    elif finish is not None:
        kind, body = "finish", finish
    else:
        cut = OPEN_ACTION.search(text)
        if cut:                                   # the reply ended inside an action (output limit)
            kind, body = cut.group(1), cut.group(2)
        else:
            block = re.search(r"```python\s*\n(.*?)```", text, re.S)       # lenient: a bare code block
            if not block:
                return plan, None, "", None
            kind, body = "python", block.group(1)
    fenced = FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    if not plan:
        plan = re.sub(r"<[^>]+>.*", "", text, flags=re.S).strip()[:200]
    return plan, kind, body.strip("\n"), finish


def _brief(result):
    """Short summary of a submitted result for the model."""
    if result.get("feasible") is False:
        return f"not answerable: {result.get('reason', '')}"
    from mes.tools.rwd.report import estimate_text
    return "; ".join(estimate_text(e) for e in result["estimates"][:6])


def session_cells(cells, upto=None):
    """[(position, cell)] of the cells whose effects the Python session held when cell ``upto`` ran (default: the
    last): every cell that ran since the session last restarted, including cells stopped by an error (the lines
    before the error ran and may have defined variables that later cells use)."""
    upto = len(cells) - 1 if upto is None else upto
    out = []
    for i, c in enumerate(cells[:upto + 1]):
        if c.get("killed"):              # time or memory limit: the session restarted without its variables
            out = []
        elif c.get("ran", True):         # not refused by the code policy
            out.append((i, c))
    return out


def script(cells):
    """The cells as one script; cells that stopped with an error are kept (in try / except) as in the session."""
    import textwrap
    parts = []
    for i, c in cells:
        if c.get("ok"):
            note = " (its submit() was refused and replaced by a later one)" if c.get("submitted") and \
                not c.get("accepted", True) else ""
            parts.append(f"# --- cell {i + 1}{note}\n{c['code']}")
        else:
            parts.append(f"# --- cell {i + 1} (stopped with an error in the session; later cells use what it defined)\n"
                         f"try:\n    pass\n{textwrap.indent(c['code'], '    ')}\nexcept Exception as error:\n"
                         f"    print('cell {i + 1} stopped:', error)")
    return "\n\n".join(parts)


def _literals(codes):
    """Numbers written as literals in the code cells (the last one submitted the result). Literals of assert
    statements are checks, not values; those of an earlier cell's submit() call left nothing in the session."""
    import ast
    found = set()
    for i, code in enumerate(codes):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        skip = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) or (i < len(codes) - 1 and isinstance(node, ast.Call)
                                                and isinstance(node.func, ast.Name) and node.func.id == "submit"):
                skip.update(id(n) for n in ast.walk(node))
        for node in ast.walk(tree):
            if id(node) not in skip and isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                found.add(float(node.value))
    return found


def typed_numbers(result, codes):
    """Counts / estimates of ``result`` that appear as literals in the code (typed in, not computed). ``codes``: the
    code of the session's cells, the submitting cell last (``session_cells``)."""
    if not result or result.get("feasible") is False:
        return []
    # only numbers that look like copied results: counts ≥ 200 that are not round (thresholds and constants such
    # as 28, 100 or 1000 are legitimate literals) and decimals with at least three digits
    literals = _literals(codes)
    counts = {x for x in literals if x.is_integer() and x >= 200 and x % 50}
    decimals = {x for x in literals if not x.is_integer() and len(f"{x:.10g}".split(".")[-1]) >= 3}
    typed = []
    for e in result["estimates"]:
        for key in ("n", "events"):
            v = e.get(key)
            if isinstance(v, int) and float(v) in counts:
                typed.append(f"{e['label']} {key} = {v}")
        v = e.get("value")
        if isinstance(v, float) and any(abs(v - x) < 1e-9 or abs(v - 100 * x) < 1e-7 for x in decimals):
            typed.append(f"{e['label']} = {v:g}")
    return typed[:6]


def _close(x, y, rel=1e-3):
    if x is None or y is None:
        return x is None and y is None
    return abs(float(x) - float(y)) <= max(1e-6, rel * max(abs(float(x)), abs(float(y))))


def _same(a, b):
    """Two submitted results give the same numbers: the same estimates, counts exactly, values to 0.1 % (iterative
    solvers and rows that tie in the sort order can move the last digits)."""
    if a is None or b is None:
        return False
    if a.get("feasible") is False or b.get("feasible") is False:
        return a.get("feasible") == b.get("feasible")
    ea, eb = a.get("estimates") or [], b.get("estimates") or []
    return len(ea) == len(eb) and all(
        (x.get("label"), x.get("measure"), x.get("n"), x.get("events")) ==
        (y.get("label"), y.get("measure"), y.get("n"), y.get("events"))
        and all(_close(x.get(k), y.get(k)) for k in ("value", "lo", "hi")) for x, y in zip(ea, eb))


# ============================================================================ the analyst
class Analyst:
    def __init__(self, question, plan, llm, mode_cfg, store, index, report=None, cancel=None):
        self.question, self.plan, self.llm, self.mode = question, plan, llm, mode_cfg
        self.cancel = cancel
        self.store, self.index = store, index
        self.report = report or (lambda msg: None)
        self.steps, self.cells, self.reviews = [], [], []
        self.submitted, self.finish_text, self.finished = None, "", False
        self.submitted_cell = None                 # position in self.cells of the accepted submission
        self.reproduction = None
        self.audit = []                            # codes / names / items of the code, looked up in MIMIC-IV
        self.t0 = time.time()
        # the model works until the deadline; the review and the reproduction use the time kept after it
        reserve = CHECK_RESERVE.get(mode_cfg.key, 30)
        self.deadline = self.t0 + max(0.6 * mode_cfg.analyst_seconds, mode_cfg.analyst_seconds - reserve)
        self.box = Sandbox(store.root, cell_timeout=max(30, min(120, mode_cfg.analyst_seconds // 2)))
        self.messages = []
        self.max_steps = mode_cfg.analyst_steps

    # ---------------------------------------------------------------- model turns
    def _remaining(self):
        return self.deadline - time.time()

    def _context(self):
        """Messages for the next call: older outputs shortened to keep the context small."""
        out, n = [], len(self.messages)
        for i, m in enumerate(self.messages):
            content = m["content"]
            if m["role"] == "user" and i not in (0,) and i < n - 6 and len(content) > 700:
                content = content[:600] + "\n… [older output shortened]"
            out.append({"role": m["role"], "content": content})
        return out

    def _call(self):
        effort = self.mode.analyst_effort
        return self.llm.invoke(None, system=_system(self.mode, self.store), messages=self._context(),
                               max_tokens=STEP_TOKENS.get(effort, 3000), reasoning=effort, cache=True)

    def _execute(self, kind, body, finish):
        if kind == "concepts":
            return f"<result>\n{concept_text(self.index, body.strip()[:120])}\n</result>"
        if kind == "sql":
            return f"<result>\n{sql_text(self.store, body)}\n</result>"
        if kind == "python":
            budget = max(10, min(self.box.cell_timeout, self._remaining() - 5))
            r = self.box.run(body, timeout=budget)
            self.cells.append({"code": body, "ok": r["ok"], "ran": not r.get("policy"), "killed": bool(r.get("killed")),
                               "stdout": r.get("stdout", ""), "error": r.get("error", ""), "seconds": r.get("seconds"),
                               "submitted": bool(r.get("submitted")), "accepted": False})
            parts = [f"<output>\n{r.get('stdout', '').rstrip() or '(nothing printed)'}\n</output>"]
            if r.get("error"):
                parts.append(f"<error>\n{r['error']}\n</error>")
            if r.get("submitted"):
                typed = typed_numbers(r["submitted"], [c["code"] for _, c in session_cells(self.cells)])
                if typed:
                    parts.append("<error>submit() refused: these numbers are typed into the code instead of computed "
                                 f"from the data: {'; '.join(typed)}. Compute them with sql() / cohort() in the code "
                                 "so the analysis can be reproduced, then submit again.</error>")
                    return "\n".join(parts)
                self.submitted, self.submitted_cell = r["submitted"], len(self.cells) - 1
                self.cells[-1]["accepted"] = True
                parts.append(f"<submitted>accepted — {_brief(self.submitted)}</submitted>")
                if finish is not None:
                    self.finish_text, self.finished = finish, True
            return "\n".join(parts)
        if kind == "finish":
            if self.submitted is None:
                return "<error>Nothing was submitted yet: call submit(result) in a <python> block first.</error>"
            self.finish_text, self.finished = body, True
            return ""
        return ""

    def _loop(self, steps):
        """Model turns until finish, the step budget or the time budget; returns True if a result is final."""
        for _ in range(steps):
            remaining = self._remaining()
            done = [x["seconds"] for x in self.steps if x["action"] != "error"]
            typical = sum(done[-3:]) / len(done[-3:]) if done else 0
            if remaining < max(12, 0.8 * typical):      # a new turn would end after the budget
                break
            if self.cancelled:
                break
            t = time.time()
            try:
                reply = self._call()
            except Exception as e:
                logger.warning("Analyst model call failed: %s", e)
                self.steps.append({"plan": "", "action": "error", "detail": f"{type(e).__name__}: {e}"[:300],
                                   "seconds": round(time.time() - t, 1)})
                break
            plan, kind, body, finish = parse_reply(reply)
            n = len(self.steps) + 1
            if kind is None:
                observation = ("<error>Format: reply with <plan>…</plan> and exactly one of <concepts>, <sql>, "
                               "<python> or <finish>.</error>")
            else:
                self.report(f"RWD analyst · step {n}: {plan[:140] or kind}")
                self.finished = False
                observation = self._execute(kind, body, finish)
            self.steps.append({"plan": plan, "action": kind or "invalid", "seconds": round(time.time() - t, 1),
                               "code": body if kind in ("python", "sql", "concepts") else ""})
            self.messages.append({"role": "assistant", "content": reply.strip() or "(empty reply)"})
            if self.finished and self.submitted is not None:
                return True
            left = self._remaining()
            note = f"[step {n} of {self.max_steps} · about {max(0, left):.0f} s left"
            note += " — submit your result now]" if left < 25 and self.submitted is None else "]"
            self.messages.append({"role": "user", "content": f"{observation}\n{note}".strip()})
        return self.submitted is not None

    @property
    def cancelled(self):
        return self.cancel is not None and self.cancel.is_set()

    # ---------------------------------------------------------------- checks
    def _session(self):
        """The cells the accepted result came from (``session_cells`` up to the submitting cell)."""
        return session_cells(self.cells, self.submitted_cell)

    def _review(self):
        cells = self._session()
        try:
            self.audit = code_audit.audit(script(cells), self.index, self.store)
        except Exception as e:               # the check must never stop the review
            logger.warning("Code check failed: %s", e)
            self.audit = []
        code = "\n\n".join(f"# --- cell {i + 1}" + ("" if c["ok"] else f" (stopped: {c['error'][-160:].strip()})")
                           + f"\n{c['code']}" for i, c in cells)[-14000:]
        outputs = "\n".join(f"# cell {i + 1}: {c['stdout'][-700:]}" for i, c in cells if c["stdout"].strip())[-5000:]
        prompt = REVIEW.format(question=self.question, facts=DATA_FACTS, code=code, outputs=outputs,
                               result=json.dumps(self.submitted, default=str)[:6000],
                               audit=code_audit.review_text(self.audit)[:4000])
        try:
            raw = self.llm.invoke(prompt, json_schema=REVIEW_SCHEMA, max_tokens=1500,
                                  reasoning=self.mode.analyst_effort)
            review = parse_json_response(raw)
        except Exception as e:
            return {"verdict": "not_reviewed", "issues": [], "summary": f"review failed: {type(e).__name__}"}
        if not isinstance(review, dict) or review.get("verdict") not in ("sound", "minor_issues", "major_issues"):
            return {"verdict": "not_reviewed", "issues": [], "summary": "the review could not be read"}
        review["issues"] = [i for i in review.get("issues") or [] if isinstance(i, dict)][:8]
        return review

    def _reproduce(self):
        """Re-run the cells the accepted result came from in a fresh session (cells stopped by an error included:
        they ran up to the error); the submitting cell must give the same result."""
        cells = self._session()
        expected = self.submitted
        t = time.time()
        got, problem = None, ""
        with Sandbox(self.store.root, cell_timeout=self.box.cell_timeout) as box:
            for i, c in cells:
                r = box.run(c["code"], timeout=max(15, min(self.box.cell_timeout, self.deadline + 120 - time.time())))
                if i == self.submitted_cell:
                    got = r.get("submitted")
                if not r["ok"] and c["ok"]:
                    lines = [x.strip() for x in (r.get("error") or "").splitlines() if x.strip()]
                    problem = f"cell {i + 1} failed when re-run ({(lines[-1] if lines else 'error')[:200]})"
                    break
        same = not problem and _same(got, expected)
        return {"reproduced": same, "seconds": round(time.time() - t, 1),
                "detail": problem or ("same result in a fresh session" if same else
                                      "a fresh session gave different numbers (the result depends on state or "
                                      "randomness)")}

    # the review may end this long after the mode's budget (at least REVIEW_MIN s); a slower reply is given up (it
    # must not hold the answer)
    REVIEW_GRACE, REVIEW_MIN = 30, 20

    def _left(self):
        """Seconds left of the mode's whole budget (the model's turns stop earlier, at the deadline)."""
        return self.t0 + self.mode.analyst_seconds - time.time()

    def _checks(self):
        """The review and the reproduction at the same time; the review is given up when it would end too late."""
        out = {}
        reviewer = threading.Thread(target=lambda: out.update(review=self._review()), daemon=True)
        reviewer.start()
        repro = self._reproduce()
        reviewer.join(max(self.REVIEW_MIN, self._left() + self.REVIEW_GRACE))
        review = out.get("review") or {"verdict": "not_reviewed", "issues": [],
                                       "summary": "the review did not finish within the time budget"}
        return review, repro

    # ---------------------------------------------------------------- run
    def run(self):
        self.report(f"RWD analyst ({self.llm.label}): exploring the data")
        threading.Thread(target=self.box.start, daemon=True).start()       # starts while the model plans
        self.messages = [{"role": "user", "content": _task(self.question, self.plan, self.index, self.mode)}]
        review, repro = None, None
        try:
            done = self._loop(self.max_steps)
            revisions = self.mode.analyst_revisions
            while done and self.submitted.get("feasible") is not False and not self.cancelled:
                self.report("RWD analyst: independent review and reproduction of the analysis")
                review, repro = self._checks()
                self.reviews.append(review)
                # a revision needs time for the analyst's turns and a second review
                if review.get("verdict") != "major_issues" or revisions <= 0 or self._left() < 70:
                    break
                revisions -= 1
                issues = "\n".join(f"- [{i.get('severity')}] {i.get('problem')} → {i.get('fix')}"
                                   for i in review.get("issues", []))
                self.report(f"RWD analyst: revising after review ({len(review.get('issues', []))} issues)")
                self.messages.append({"role": "user", "content":
                                      f"An independent reviewer found problems:\n{issues}\nFix them: re-run what is "
                                      "needed, call submit() with the corrected result, then <finish>."})
                self.finished = False
                done = self._loop(max(4, self.max_steps // 2))
        finally:
            self.box.close()
        return self._study(review, repro)

    def _study(self, review, repro):
        result = self.submitted
        code = script(self._session())
        key = hashlib.sha256((self.question + code).encode()).hexdigest()[:16]
        timings = {"total": round(time.time() - self.t0, 2), "steps": len(self.steps),
                   "model_s": round(sum(s["seconds"] for s in self.steps), 1),
                   "code_s": round(sum(c["seconds"] or 0 for c in self.cells), 1)}
        study = {"key": key, "kind": "analyst", "dataset": DATASET, "llm": self.llm.label, "mode": self.mode.key,
                 "data_built_at": self.store.catalog().get("built_at"), "code": code, "timings": timings,
                 "cells": [{**c, "stdout": c["stdout"][-3000:]} for c in self.cells], "submitted_cell": self.submitted_cell,
                 "steps": self.steps, "finish": self.finish_text, "reviews": self.reviews, "review": review,
                 "reproduction": repro, "code_audit": self.audit}
        if result is None:
            study.update(status="failed", spec=None, result=None,
                         design_info={"errors": ["the analyst did not submit a result within the "
                                                 f"{self.mode.label} budget ({len(self.steps)} steps, "
                                                 f"{timings['total']:.0f} s)"]})
            return study
        if result.get("feasible") is False:
            study.update(status="not_feasible", result=None,
                         spec={"analysis": "custom", "base": "icu", "title": result.get("title", ""),
                               "reason": result.get("reason", "")})
            return study
        warnings = list(result.get("warnings", []))
        if review and review.get("verdict") == "major_issues":
            warnings.append("An independent review found unresolved problems: "
                            + "; ".join(i.get("problem", "") for i in review.get("issues", [])
                                        if i.get("severity") == "major")[:600])
        elif review and review.get("verdict") == "not_reviewed" and len(self.reviews) > 1 \
                and self.reviews[-2].get("verdict") == "major_issues":
            warnings.append("The analysis was revised after an independent review found problems, but the revision "
                            "was not reviewed again within the time budget. The problems found: "
                            + "; ".join(i.get("problem", "") for i in self.reviews[-2].get("issues", [])
                                        if i.get("severity") == "major")[:500])
        if repro and not repro.get("reproduced"):
            warnings.append(f"Not reproduced in a fresh session: {repro.get('detail')}")
        if code_audit.problems(self.audit):
            warnings.append("Code check: " + "; ".join(code_audit.problems(self.audit))[:600])
        study.update(status="done", result=result_record(result, warnings), sql=None,
                     spec={"analysis": "custom", "base": "icu", "title": result["title"]},
                     design_info={"source": f"written by the MES analyst ({self.llm.label})"})
        return study


def result_record(result, warnings):
    """The study result shown and cited (``study["result"]``) from a submitted result."""
    return {"analysis": "custom", "title": result["title"], "population": result.get("population", ""),
            "n": result.get("n"), "attrition": result.get("attrition", []), "estimates": result["estimates"],
            "tables": result.get("tables", {}), "diagnostics": result.get("diagnostics", {}), "warnings": warnings,
            "methods": result.get("methods", ""),
            "analyst": {k: result.get(k) for k in ("analysis_type", "time_zero", "exposure", "comparator", "outcome",
                                                   "limitations")}}


def analyze(question, plan, llm, mode_cfg, store, index, report=None, cancel=None):
    """Run the analyst for ``question``; returns the study record (see ``Analyst._study``). ``cancel`` (an Event)
    stops it before its next step."""
    return Analyst(question, plan, llm, mode_cfg, store, index, report, cancel).run()


EDITED_REVIEW = "not reviewed: the code was edited by the user after the analyst's review"
EDITED_WARNING = ("The analysis code was edited by the user; the edited code was not reviewed independently (the code "
                  "check ran again).")


def run_edited(code, study, store, index, mode_cfg, report=None):
    """
    The study record of analysis ``code`` edited by the user (``study``: the analyst study it was edited from), run
    once as a script in a fresh sandboxed session: the same code policy, result checks and refusal of typed-in
    numbers as the analyst's code, and the code check again. The independent review is not repeated: the record
    says so (``review`` "not_reviewed", a warning) instead of keeping the review of the code before the edit.
    """
    report = report or (lambda msg: None)
    t0 = time.time()
    code = str(code or "").strip("\n")
    key = hashlib.sha256(f"{study.get('key', '')}\n{code}".encode()).hexdigest()[:16]
    timeout = max(90, min(300, 2 * mode_cfg.analyst_seconds))
    record = {"key": key, "kind": "analyst", "dataset": DATASET, "llm": study.get("llm", "the analyst"),
              "mode": mode_cfg.key, "data_built_at": store.catalog().get("built_at"), "code": code, "finish": "",
              "reviews": [], "review": {"verdict": "not_reviewed", "issues": [], "summary": EDITED_REVIEW},
              "reproduction": None, "edited": {"kind": "code", "from_key": study.get("key")}}
    report("RWD: running the edited analysis code in a fresh session")
    with Sandbox(store.root, cell_timeout=timeout) as box:
        r = box.run(code, timeout=timeout)
    submitted = r.get("submitted")
    cell = {"code": code, "ok": r["ok"], "ran": not r.get("policy"), "killed": bool(r.get("killed")),
            "stdout": (r.get("stdout") or "")[-3000:], "error": r.get("error", ""), "seconds": r.get("seconds"),
            "submitted": submitted is not None, "accepted": False}
    errors = []
    if submitted is not None:
        # the script's cells as the analyst ran them ("# --- cell n" lines of ``script``): the numbers of an earlier
        # cell's refused submit() are not typed into the final result
        cells = [c for c in re.split(r"(?m)^# --- cell \d+.*$", code) if c.strip()] or [code]
        typed = typed_numbers(submitted, cells)
        if typed:
            errors.append("submit() refused: these numbers are typed into the code instead of computed from the "
                          f"data: {'; '.join(typed)}")
            submitted = None
    elif r.get("error"):
        lines = [x.strip() for x in r["error"].splitlines() if x.strip()]
        errors.append(f"the code stopped with an error: {(lines[-1] if lines else 'error')[:300]}")
    else:
        errors.append("the code ran but did not call submit(result)")
    try:
        audit = code_audit.audit(code, index, store)
    except Exception as e:                   # the check must never stop the run
        logger.warning("Code check failed: %s", e)
        audit = []
    seconds = round(time.time() - t0, 2)
    record.update(cells=[cell], submitted_cell=0 if submitted is not None else None, code_audit=audit,
                  steps=[{"plan": "The analysis code as edited by the user, run as one script in a fresh session",
                          "action": "python", "seconds": round(r.get("seconds") or 0, 1), "code": ""}],
                  timings={"total": seconds, "steps": 1, "model_s": 0.0, "code_s": round(r.get("seconds") or 0, 1)})
    if submitted is None:
        record.update(status="failed", spec=None, result=None, design_info={"errors": errors})
        return record
    cell["accepted"] = True
    if submitted.get("feasible") is False:
        record.update(status="not_feasible", result=None,
                      spec={"analysis": "custom", "base": "icu", "title": submitted.get("title", ""),
                            "reason": submitted.get("reason", "")})
        return record
    warnings = [EDITED_WARNING] + list(submitted.get("warnings", []))
    if code_audit.problems(audit):
        warnings.append("Code check: " + "; ".join(code_audit.problems(audit))[:600])
    record.update(status="done", result=result_record(submitted, warnings), sql=None,
                  spec={"analysis": "custom", "base": "icu", "title": submitted["title"]},
                  design_info={"source": f"written by the MES analyst ({record['llm']}) and edited by the user"})
    return record
