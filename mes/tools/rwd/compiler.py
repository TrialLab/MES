"""
StudySpec → DuckDB SQL → analysis cohort (deterministic; the SQL is shown to the user).

One query returns the base population with one flag column per criterion, the exposure / comparator
assignment, outcomes and covariates. The attrition is then applied step by step in pandas, so every exclusion
is counted. Only library-whitelisted column names reach the SQL; values are quoted or cast to numbers.

Comparative designs follow a target trial emulation with a landmark at the end of the grace period:
time zero = ICU (or hospital) admission; exposure = first dose within [0, grace) hours; patients who died or
left the hospital before the landmark are excluded, so both groups start follow-up alive at the same time
(no immortal time); prior users before time zero are excluded (new-user design).
"""

from dataclasses import dataclass, field

import pandas as pd

from mes.tools.rwd import library as L
from mes.tools.rwd.concepts import code_predicate
from mes.tools.rwd.db import get_store
from mes.tools.rwd.spec import DEFAULT_WINDOWS, criterion_text


# First occurrence (hours from ICU admission) of outcomes that can already be present at time zero: a comparison
# counts only new events after the landmark (prevalent AKI is not caused by a drug started later)
_ONSET_SQL = "SELECT min(date_diff('minute', b.intime, {time})) / 60.0 FROM mimiciv_derived.{table} WHERE {where}"
ONSET = {          # outcome: (what is excluded, SQL of the onset)
    "aki_7d": ("acute kidney injury", _ONSET_SQL.format(
        time="k.charttime", table="kdigo_stages k", where="k.stay_id = b.stay_id AND k.charttime >= b.intime "
                                                          "AND k.aki_stage >= 1")),
    "aki_stage2_7d": ("AKI stage ≥ 2", _ONSET_SQL.format(
        time="k.charttime", table="kdigo_stages k", where="k.stay_id = b.stay_id AND k.charttime >= b.intime "
                                                          "AND k.aki_stage >= 2")),
    "rrt_any": ("renal replacement therapy", _ONSET_SQL.format(
        time="r.charttime", table="rrt r", where="r.stay_id = b.stay_id AND r.dialysis_active = 1")),
    "invasive_vent_any": ("invasive ventilation", _ONSET_SQL.format(
        time="v.starttime", table="ventilation v", where="v.stay_id = b.stay_id AND v.ventilation_status IN "
                                                         "('InvasiveVent', 'Tracheostomy')")),
}


def _q(text):
    return "'" + str(text).replace("'", "''") + "'"


def _num(value):
    return repr(float(value))


@dataclass
class Cohort:
    df: pd.DataFrame
    sql: str
    attrition: list = field(default_factory=list)       # [(step, n)]
    columns: dict = field(default_factory=dict)          # role → column name
    notes: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)           # column → readable name (a lab / ICU item measurement)
    step_ids: list = field(default_factory=list)         # one id per attrition step: base, inc_0 …, exc_0 …, groups
    store: object = None                                 # the store it was built on (later queries of the analysis)


class Compiler:
    def __init__(self, index, store=None):
        self.index = index
        self.store = store or get_store()

    # ------------------------------------------------------------------ criteria
    def _window(self, c, role, t0):
        lo, hi = c.window_h or DEFAULT_WINDOWS.get(role, [-72, 24])
        alias = t0.split(".")[0]
        end = f"{alias}.outtime" if t0.endswith("intime") else f"{alias}.dischtime"     # end of the stay
        return f"{t0} + INTERVAL ({_num(lo)}) HOUR", end if hi is None else f"{t0} + INTERVAL ({_num(hi)}) HOUR"

    def criterion(self, c, base, role="include", alias="b", first_time=False):
        """SQL boolean (or, with ``first_time``, the hours from time zero to the first event) for a criterion."""
        kind, _, key = c.concept.partition(":")
        t0 = f"{alias}.{L.BASES[base]['t0']}"
        if kind == "var":
            vkind = L.VARIABLES[key][1]
            col = f"{alias}.{key}"
            if vkind == "binary":
                sql = f"coalesce(CAST({col} AS INTEGER), 0) = 1"
            elif vkind == "category":
                sql = f"{col} {'!=' if c.op == '!=' else '='} {_q(c.value)}"
            elif c.op == "between":
                sql = f"{col} BETWEEN {_num(c.value[0])} AND {_num(c.value[1])}"
            else:
                sql = f"{col} {c.op} {_num(c.value)}"
        elif kind == "phe":
            _label, i10, i9, _syn = L.PHENOTYPES[key]
            sql = f"{alias}.hadm_id IN (SELECT hadm_id FROM mes.diagnoses WHERE {code_predicate(i10, i9)})"
        elif kind in ("dx9", "dx10"):
            version = 9 if kind == "dx9" else 10
            sql = (f"{alias}.hadm_id IN (SELECT hadm_id FROM mes.diagnoses WHERE icd_version = {version} "
                   f"AND icd_code LIKE {_q(key + '%')})")
        elif kind in ("px9", "px10"):
            version = 9 if kind == "px9" else 10
            lo, hi = self._window(c, role, t0)
            sub = (f"FROM mes.procedures p WHERE p.hadm_id = {alias}.hadm_id AND p.source = 'icd' "
                   f"AND p.icd_version = {version} AND p.code LIKE {_q(key + '%')} "
                   f"AND p.time >= date_trunc('day', {lo}) AND p.time < {hi}")
            if first_time:
                return f"(SELECT min(date_diff('minute', {t0}, p.time)) / 60.0 {sub})"
            sql = f"EXISTS (SELECT 1 {sub})"
        elif kind in ("drugclass", "drug"):
            names = self.index.drug_names_for(c.concept)
            lst = ", ".join(_q(n) for n in names) or "NULL"
            lo, hi = self._window(c, role, t0)
            route = ""
            if kind == "drugclass" and L.systemic_only(key):
                route = (f" AND lower(coalesce(d.route, '')) NOT IN ({', '.join(_q(r) for r in L.LOCAL_ROUTES)})")
            sub = (f"FROM mes.drugs d WHERE d.hadm_id = {alias}.hadm_id AND d.drug IN ({lst}){route} "
                   f"AND d.starttime >= {lo} AND d.starttime < {hi}")
            if first_time:
                return f"(SELECT min(date_diff('minute', {t0}, d.starttime)) / 60.0 {sub})"
            sql = f"EXISTS (SELECT 1 {sub})"
        elif kind == "micro":
            # the culture's time (its date at midnight when no time was recorded); cultures of the ED or the ward
            # before the ICU have no ICU stay: linked by the patient and the window
            lo, hi = self._window(c, role, t0)
            when = "coalesce(m.charttime, CAST(m.chartdate AS TIMESTAMP))"
            sub = (f"FROM mimiciv_hosp.microbiologyevents m WHERE m.subject_id = {alias}.subject_id "
                   f"AND {L.micro_predicate(key)} AND {when} >= {lo} AND {when} < {hi}")
            if first_time:
                return f"(SELECT min(date_diff('minute', {t0}, {when})) / 60.0 {sub})"
            sql = f"EXISTS (SELECT 1 {sub})"
        elif kind in ("lab", "item"):
            lo, hi = self._window(c, role, t0)
            if kind == "lab":
                sub = (f"FROM mimiciv_hosp.labevents m WHERE m.hadm_id = {alias}.hadm_id AND m.itemid = {int(key)} "
                       f"AND m.charttime >= {lo} AND m.charttime < {hi}")
            else:
                link = f"m.stay_id = {alias}.stay_id" if base == "icu" else f"m.hadm_id = {alias}.hadm_id"
                sub = (f"FROM mimiciv_icu.chartevents m WHERE {link} AND m.itemid = {int(key)} "
                       f"AND m.charttime >= {lo} AND m.charttime < {hi}")
            if first_time:
                return f"(SELECT min(date_diff('minute', {t0}, m.charttime)) / 60.0 {sub})"
            value = ""
            if c.op == "between":
                value = f" AND m.valuenum BETWEEN {_num(c.value[0])} AND {_num(c.value[1])}"
            elif c.op:
                value = f" AND m.valuenum {c.op} {_num(c.value)}"
            sql = f"EXISTS (SELECT 1 {sub}{value})"
        else:
            raise ValueError(f"unsupported concept {c.concept}")
        if first_time:
            return "NULL"
        return f"NOT ({sql})" if c.negate else f"({sql})"

    def measurement(self, concept, base, window=(0, 24), agg="max", alias="b"):
        """Scalar value of a lab:/item: concept within a window (for distributions and associations)."""
        kind, _, key = concept.partition(":")
        if kind == "derived":
            return L.DERIVED[key][2].replace("b.", f"{alias}.")
        t0 = f"{alias}.{L.BASES[base]['t0']}"
        lo, hi = f"{t0} + INTERVAL ({_num(window[0])}) HOUR", f"{t0} + INTERVAL ({_num(window[1])}) HOUR"
        if kind == "lab":
            src, link = "mimiciv_hosp.labevents", f"m.hadm_id = {alias}.hadm_id"
        else:
            src = "mimiciv_icu.chartevents"
            link = f"m.stay_id = {alias}.stay_id" if base == "icu" else f"m.hadm_id = {alias}.hadm_id"
        return (f"(SELECT {agg}(m.valuenum) FROM {src} m WHERE {link} AND m.itemid = {int(key)} "
                f"AND m.charttime >= {lo} AND m.charttime < {hi})")

    # ------------------------------------------------------------------ cohort
    def build(self, spec, timeout=90, counts_only=False):
        """The cohort of ``spec`` with its attrition. ``counts_only``: only the columns the attrition needs (the live
        counts of the cohort builder), not the outcomes, covariates and measurements of the analysis."""
        if spec.analysis == "compare":   # treatment = first dose within the grace period
            for c in (spec.exposure, spec.comparator):
                if c is not None and c.concept.split(":")[0] in ("drugclass", "drug", "px9", "px10"):
                    c.window_h = [0, float(spec.grace_h)]
        base = L.BASES[spec.base]
        table, id_col, t0 = base["table"], base["id"], base["t0"]
        cols = [f"b.{id_col} AS unit_id", "b.subject_id", "b.hadm_id", f"b.{t0} AS t0", "b.dischtime",
                "b.death_time", "b.days_to_death", "b.death_hosp"]
        roles = {}
        for i, c in enumerate(spec.include):
            cols.append(f"{self.criterion(c, spec.base, 'include')} AS inc_{i}")
        for i, c in enumerate(spec.exclude):
            cols.append(f"{self.criterion(c, spec.base, 'exclude')} AS exc_{i}")
        exposure_kind = spec.exposure.concept.split(":")[0] if spec.exposure is not None else ""
        continuous = (exposure_kind == "var" and not spec.exposure.op
                      and L.VARIABLES[spec.exposure.concept[4:]][1] == "numeric")
        if spec.exposure is not None and not continuous:
            cols.append(f"{self.criterion(spec.exposure, spec.base, 'exposure')} AS exposed")
            roles["exposure"] = "exposed"
            if spec.exposure.concept.split(":")[0] in ("drugclass", "drug", "px9", "px10"):
                if not counts_only:
                    cols.append(f"{self.criterion(spec.exposure, spec.base, 'exposure', first_time=True)} AS exposure_hours")
                if spec.analysis == "compare" and spec.new_user:
                    prior = type(spec.exposure)(concept=spec.exposure.concept, window_h=[-24 * 30, 0])
                    cols.append(f"{self.criterion(prior, spec.base, 'exposure')} AS prior_exposed")
        if spec.exposure is not None and not continuous and spec.exposure.op and exposure_kind in ("var", "lab", "item"):
            # a patient without a value cannot be classified: a missing lactate is not "lactate ≤ 4"
            if exposure_kind == "var":
                cols.append(f"b.{spec.exposure.concept[4:]} IS NULL AS exposure_missing")
            else:
                measured = type(spec.exposure)(concept=spec.exposure.concept, window_h=spec.exposure.window_h)
                cols.append(f"NOT {self.criterion(measured, spec.base, 'exposure')} AS exposure_missing")
        if continuous and not counts_only:
            cols.append(f"b.{spec.exposure.concept[4:]} AS exposure_value")
        if spec.comparator is not None:
            cols.append(f"{self.criterion(spec.comparator, spec.base, 'comparator')} AS comparator")
            roles["comparator"] = "comparator"
            if spec.analysis == "compare" and spec.new_user and spec.comparator.concept.split(":")[0] in (
                    "drugclass", "drug"):
                prior = type(spec.comparator)(concept=spec.comparator.concept, window_h=[-24 * 30, 0])
                cols.append(f"{self.criterion(prior, spec.base, 'comparator')} AS prior_comparator")
        outcome_cols = []
        for o in [spec.outcome] + list(spec.secondary_outcomes):
            if o and o in L.OUTCOMES:
                col = L.OUTCOMES[o][3]
                if col not in outcome_cols:
                    outcome_cols.append(col)
        variables = list(dict.fromkeys(list(spec.covariates) + list(spec.stratify) + list(spec.subgroups)))
        if spec.variable and spec.variable in L.VARIABLES:
            variables.append(spec.variable)
        if spec.group_by in L.VARIABLES:    # characterize / trajectory: the variable that forms the groups
            variables.append(spec.group_by)
        if spec.analysis == "pathways" and not counts_only:     # the hour of the first dose of each treatment
            for i, c in enumerate(spec.treatments):
                c.window_h = list(c.window_h) if c.window_h else [0, float(spec.horizon_h)]
                cols.append(f"{self.criterion(c, spec.base, 'exposure', first_time=True)} AS start_{i}")
        if spec.analysis == "similar":      # the patient's characteristics and what the similar stays received
            present = self.store.columns(table)
            variables += [str(f.get("concept") or "")[4:] for f in spec.profile or [] if isinstance(f, dict)
                          and str(f.get("concept") or "").startswith("var:")]
            variables += [v for v in L.SIMILAR_TREATMENTS if not present or v in present]
            variables = [v for v in dict.fromkeys(variables) if v in L.variables_for(spec.base)]
        for col in dict.fromkeys([] if counts_only else outcome_cols + variables):
            if col in L.VARIABLES or col in {v[3] for v in L.OUTCOMES.values()}:
                cols.append(f"b.{col}")
        if spec.variable and spec.variable.split(":")[0] in ("lab", "item", "derived") and not counts_only:
            cols.append(f"{self.measurement(spec.variable, spec.base)} AS variable_value")
        if spec.analysis == "compare" and spec.base == "icu" and spec.outcome in ONSET:
            cols.append(f"({ONSET[spec.outcome][1]}) AS outcome_onset_h")
        if spec.analysis == "trend" and "anchor_year_group" not in variables and not counts_only:
            cols.append("b.anchor_year_group")
        sql = "SELECT\n    " + ",\n    ".join(dict.fromkeys(cols)) + f"\nFROM {table} b\nWHERE {base['where']}"
        df = self.store.query(sql, timeout=timeout)
        cohort = Cohort(df=df, sql=sql, columns=roles, store=self.store)
        if spec.variable and spec.variable.split(":")[0] in ("lab", "item"):
            cohort.labels["variable_value"] = f"{self.index.label(spec.variable)}, highest in the first 24 h"
        elif spec.variable and spec.variable.split(":")[0] == "derived":
            cohort.labels["variable_value"] = L.DERIVED[spec.variable.split(":", 1)[1]][0]
        self._attrition(spec, cohort)
        return cohort

    def _attrition(self, spec, cohort):
        df = cohort.df
        steps, ids = [(L.BASES[spec.base]["label"], len(df))], ["base"]
        for i, c in enumerate(spec.include):
            df = df[df[f"inc_{i}"].fillna(False).astype(bool)]
            steps.append((f"with {criterion_text(c)}", len(df)))
            ids.append(f"inc_{i}")
        for i, c in enumerate(spec.exclude):
            df = df[~df[f"exc_{i}"].fillna(False).astype(bool)]
            steps.append((f"excluding {criterion_text(c)}", len(df)))
            ids.append(f"exc_{i}")
        if "exposure_missing" in df:
            df = df[~df["exposure_missing"].fillna(True).astype(bool)]
            name = L.VARIABLES.get(spec.exposure.concept[4:], (spec.exposure.label or spec.exposure.concept,))[0]
            steps.append((f"with a measured value of {name} (needed to classify the exposure)", len(df)))
            ids.append("measured")
        if {spec.outcome, *spec.secondary_outcomes} & {"hosp_readmit_30d", "icu_readmit_30d"}:
            df = df[df["death_hosp"].fillna(0).astype(int) == 0]
            steps.append(("discharged alive (readmission is defined for survivors)", len(df)))
            ids.append("survivors")
        if spec.analysis == "compare":
            grace = pd.to_timedelta(float(spec.grace_h), unit="h")
            if spec.grace_h > 0:
                landmark = df["t0"] + grace
                alive = ~(df["death_time"].notna() & (df["death_time"] <= landmark)) & (df["dischtime"] > landmark)
                df = df[alive]
                steps.append((f"alive and in hospital {spec.grace_h:g} h after {L.BASES[spec.base]['t0_label']} "
                              f"(landmark)", len(df)))
                ids.append("landmark")
            if "outcome_onset_h" in df:
                df = df[~(df["outcome_onset_h"] < float(spec.grace_h))]
                steps.append((f"excluding patients with {ONSET[spec.outcome][0]} before the landmark (outcome already "
                              "present)", len(df)))
                ids.append("onset")
            if spec.new_user:
                prior = df.get("prior_exposed", False) | df.get("prior_comparator", False)
                if isinstance(prior, pd.Series):
                    df = df[~prior.fillna(False).astype(bool)]
                    steps.append(("excluding patients already treated before time zero (new users only)", len(df)))
                    ids.append("new_users")
            exposed = df["exposed"].fillna(False).astype(bool)
            if spec.comparator is not None:
                comparator = df["comparator"].fillna(False).astype(bool)
                df = df[exposed ^ comparator]
                steps.append(("received exactly one of the two treatments within the grace period", len(df)))
                ids.append("one_treatment")
                df = df.assign(treated=df["exposed"].fillna(False).astype(int))
            else:
                df = df.assign(treated=exposed.astype(int))
            steps.append((f"treated: {int(df['treated'].sum()):,} · comparison: {int((1 - df['treated']).sum()):,}",
                          len(df)))
            ids.append("groups")
        cohort.df = df.reset_index(drop=True)
        cohort.attrition = steps
        cohort.step_ids = ids
