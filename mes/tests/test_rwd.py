"""
Offline tests of the RWD analysis path on a synthetic MIMIC-like store with planted truths:
a confounded treatment with a true hazard ratio of 0.7, early deaths before the landmark (immortal-time trap),
exact counts for rates, unmeasured values of a threshold exposure, plus the SQL explorer guards, the designer's sanitizing rules and number grounding.

    python -m pytest mes/tests/test_rwd.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tools.rwd import analyses, explorer, report  # noqa: E402
from mes.tools.rwd import library as L  # noqa: E402
from mes.tools.rwd.compiler import Compiler  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402
from mes.tools.rwd.spec import StudySpec, validate  # noqa: E402

TRUE_HR = 0.7
# The weighted Cox model estimates the population-average (marginal) hazard ratio, which is closer to 1 than the
# conditional one (non-collapsibility): ≈ 0.72 for this mechanism, from a simulated randomized trial of 200,000
MARGINAL_HR = 0.72
N = 6000


def _synthetic_store(root):
    rng = np.random.default_rng(42)
    t0 = pd.Timestamp("2150-01-01")
    sofa = rng.poisson(4, N).astype(float)
    age = rng.normal(65, 12, N).clip(18, 95)
    p_treat = 1 / (1 + np.exp(-(-1.0 + 0.45 * (sofa - 4))))           # sicker patients are treated more often
    treated = rng.random(N) < p_treat
    hazard = 0.004 * np.exp(0.25 * (sofa - 4)) * np.where(treated, TRUE_HR, 1.0)
    days = rng.exponential(1 / hazard)
    early = rng.random(N) < 0.03                                        # 3% die within 12 h (before the landmark)
    days = np.where(early, rng.uniform(0.05, 0.5, N), days + 1.0)       # others survive the first day
    died = days <= 365
    intime = t0 + pd.to_timedelta(np.arange(N), unit="D")
    death_time = pd.Series(intime + pd.to_timedelta(np.where(died, days, np.nan), unit="D"))
    dischtime = intime + pd.to_timedelta(np.minimum(np.where(died, days, 30), 10) + 0.01, unit="D")
    icu = pd.DataFrame({
        "subject_id": np.arange(N) + 1, "hadm_id": np.arange(N) + 100001, "stay_id": np.arange(N) + 300001,
        "intime": intime, "outtime": intime + pd.to_timedelta(2, unit="D"), "admittime": intime - pd.Timedelta(hours=6),
        "dischtime": dischtime, "first_icu_stay": True, "age": age, "gender": rng.choice(["M", "F"], N),
        "careunit": rng.choice(["MICU", "SICU"], N), "anchor_year_group": rng.choice(["2008 - 2010", "2011 - 2013"], N),
        "sofa": sofa, "sapsii": sofa * 6 + rng.normal(0, 5, N), "lactate_max": rng.gamma(2, 1.2, N),
        "sepsis3": rng.random(N) < 0.4, "death_time": death_time, "days_to_death": np.where(died, days, np.nan),
        "death_hosp": (died & (days <= 10)).astype(int), "death_28d": (died & (days <= 28)).astype(int),
        "death_1y": died.astype(int), "los_icu_days": rng.gamma(2, 1.5, N), "los_hosp_days": rng.gamma(3, 3, N),
    })
    icu.loc[np.random.default_rng(7).random(N) < 0.2, "lactate_max"] = np.nan      # lactate not measured
    drugs = pd.DataFrame({
        "subject_id": icu.subject_id[treated], "hadm_id": icu.hadm_id[treated], "stay_id": icu.stay_id[treated],
        "source": "order", "drug": "testdrug 10mg", "route": "iv",
        "starttime": icu.intime[treated] + pd.to_timedelta(rng.uniform(0.5, 20, treated.sum()), unit="h"),
        "endtime": icu.intime[treated] + pd.to_timedelta(30, unit="h")})
    (root / "mes").mkdir(parents=True)
    for folder in ("hosp", "icu", "derived"):
        (root / folder).mkdir()
    icu.to_parquet(root / "mes" / "icu_stays.parquet")
    drugs.to_parquet(root / "mes" / "drugs.parquet")
    pd.DataFrame({"drug": ["testdrug 10mg", "otherdrug"], "n_patients": [int(treated.sum()), 5],
                  "sources": "order"}).to_parquet(root / "mes" / "drug_names.parquet")
    pd.DataFrame({"subject_id": [1], "hadm_id": [100001], "seq_num": [1], "icd_code": ["A419"],
                  "icd_version": [10]}).to_parquet(root / "mes" / "diagnoses.parquet")
    dictionary = pd.DataFrame({"kind": ["drug"], "concept_id": ["drug:testdrug 10mg"], "label": ["testdrug 10mg"],
                               "code": ["testdrug 10mg"], "vocabulary": ["order"], "n_patients": [int(treated.sum())]})
    dictionary.to_parquet(root / "dictionary.parquet")
    np.save(root / "dictionary_vectors.npy", np.zeros((1, 384), dtype=np.float16))
    (root / "catalog.json").write_text(json.dumps({"coverage": "synthetic", "built_at": "test"}))
    return icu, treated, early


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, treated, early = _synthetic_store(root)
    store = Store(root)
    index = ConceptIndex(store)
    return store, index, icu, treated, early


def test_iptw_recovers_the_planted_hazard_ratio_and_the_landmark_removes_early_deaths(synth):
    store, index, icu, treated, early = synth
    spec = StudySpec.from_dict({"analysis": "compare", "base": "icu", "exposure": {"concept": "drug:testdrug"},
                                "outcome": "death_1y", "grace_h": 24, "covariates": ["sofa"]})
    errors, _ = validate(spec, index)
    assert not errors
    cohort = Compiler(index, store).build(spec)
    steps = dict(cohort.attrition)
    landmark = next(n for s, n in steps.items() if "landmark" in s)
    assert landmark == N - int(early.sum())             # every early death is excluded, nobody else
    res = analyses.run(spec, cohort, "osler")
    hr = next(e for e in res["estimates"] if e["measure"] == "HR")
    assert hr["lo"] <= MARGINAL_HR <= hr["hi"]
    from mes.tools.rwd import stats as S
    from mes.tools.rwd.analyses import _time_to_event
    time, event = _time_to_event(cohort.df, 365, offset_days=1)
    crude = S.cox(time, event, cohort.df["treated"].values)
    assert crude["value"] > hr["value"] + 0.1           # confounding by severity biases the crude estimate
    assert res["diagnostics"]["max_smd_after"] < 0.1 < res["diagnostics"]["max_smd_before"]


def test_rate_counts_are_exact(synth):
    store, index, icu, treated, early = synth
    spec = StudySpec.from_dict({"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                                "outcome": "death_28d", "stratify": ["careunit"]})
    assert not validate(spec, index)[0]
    res = analyses.run(spec, Compiler(index, store).build(spec), "osler")
    sub = icu[icu.sepsis3]
    rate = res["estimates"][0]
    assert rate["n"] == len(sub) and rate["events"] == int(sub.death_28d.sum())
    assert abs(rate["value"] - round(100 * sub.death_28d.mean(), 1)) < 1e-9
    micu = next(r for r in res["tables"]["strata"] if r["level"] == "MICU")
    assert micu["events"] == int(sub[sub.careunit == "MICU"].death_28d.sum())


def test_validation_rejects_unknown_concepts_and_wrong_outcomes(synth):
    store, index, *_ = synth
    bad = StudySpec.from_dict({"analysis": "compare", "base": "hospital", "exposure": {"concept": "drug:nosuchdrug"},
                               "outcome": "death_28d"})
    errors, _ = validate(bad, index)
    assert any("unknown concept" in e for e in errors) and any("death_28d" in e for e in errors)


def test_explorer_guards(synth):
    store, *_ = synth
    assert explorer.check_sql(store, "SELECT careunit, count(*) AS n FROM mes.icu_stays GROUP BY 1") is None
    for sql in ("COPY (SELECT 1) TO 'x.csv'", "SELECT 1; DELETE FROM mes.icu_stays",
                "SELECT * FROM read_csv('/etc/hosts')"):
        assert explorer.check_sql(store, sql) is not None
    assert explorer.check_result(pd.DataFrame({"subject_id": [1]})) is not None
    a = pd.DataFrame({"unit": ["MICU", "SICU"], "n": [100, 7]})
    assert explorer.agree(a, a.iloc[::-1].reset_index(drop=True))
    assert not explorer.agree(a, a.assign(n=[100, 9]))
    assert explorer.suppress(a).n.tolist() == [100, "<10"]


def test_designer_sanitizes_role_mixups(synth):
    store, index, *_ = synth
    from mes.tools.rwd.designer import to_spec
    # a condition named as the exposure of a rate question defines the population
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "outcome": "death_28d",
                               "exposure": {"term": "Sepsis-3", "op": "", "value": ""}}, index)
    assert spec.exposure is None and spec.include and spec.include[0].concept == "var:sepsis3"
    # a characteristic "compared" like a treatment becomes an adjusted association
    spec, _m, notes = to_spec({"analysis": "compare", "base": "icu", "outcome": "death_28d",
                               "exposure": {"term": "lactate", "op": ">", "value": "4"},
                               "comparator": {"term": "lactate", "op": "<=", "value": "4"}}, index)
    assert spec.analysis == "association" and spec.comparator is None and spec.exposure.op == ">"


def test_rwd_evidence_numbers_are_grounded(synth):
    from mes.tools import synthesis_tools
    store, index, icu, *_ = synth
    spec = StudySpec.from_dict({"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                                "outcome": "death_28d"})
    validate(spec, index)
    res = analyses.run(spec, Compiler(index, store).build(spec), "osler")
    item = report.to_items(spec, res, "k")[0]
    item.id = "R1"
    e = res["estimates"][0]
    answer = (f"## Answer\nIn MIMIC-IV, 28-day mortality was {e['value']:.1f}% (95% CI {e['lo']:.1f}–{e['hi']:.1f}) "
              f"among {e['n']:,} patients [R1].")
    g = synthesis_tools.ground_sentences(answer, {"R1": item})
    assert g["sentences"][0]["status"] == "grounded"
    assert L.OUTCOMES["death_28d"][0] in item.passages[0].text


def test_unmeasured_threshold_exposure_is_excluded_not_counted_as_unexposed(synth):
    store, index, icu, *_ = synth
    spec = StudySpec.from_dict({"analysis": "association", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                                "exposure": {"concept": "var:lactate_max", "op": ">", "value": 4},
                                "outcome": "death_28d", "covariates": ["age"]})
    assert not validate(spec, index)[0]
    cohort = Compiler(index, store).build(spec)
    measured = icu[icu.sepsis3 & icu.lactate_max.notna()]
    assert len(cohort.df) == len(measured) and "measured value" in cohort.attrition[-1][0]
    res = analyses.run(spec, cohort, "osler")
    exposed, unexposed = [e for e in res["estimates"] if e["measure"] == "risk"]
    assert exposed["n"] == int((measured.lactate_max > 4).sum())
    assert unexposed["n"] == int((measured.lactate_max <= 4).sum())


def test_designer_turns_vague_criteria_into_breakdowns(synth):
    store, index, *_ = synth
    from mes.tools.rwd.designer import to_spec
    # "adult ICU patients" restates the base population; "ICU type" names no group: break down by it
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "outcome": "death_28d",
                               "population": [{"term": "adult ICU patients", "op": "", "value": ""},
                                              {"term": "ICU type", "op": "", "value": ""}]}, index)
    assert spec.include == [] and spec.stratify == ["careunit"]
    assert not validate(spec, index)[0]
    # a named level is matched to the data's value
    spec, _m, _n = to_spec({"analysis": "rate", "base": "icu", "outcome": "death_28d",
                            "population": [{"term": "ICU type", "op": "=", "value": "micu"}]}, index)
    assert spec.include[0].value == "MICU"
    # a threshold without a number is not a threshold; a prognostic model has no single exposure
    spec, _m, _n = to_spec({"analysis": "prognostic", "base": "icu", "outcome": "death_28d",
                            "exposure": {"term": "SOFA", "op": ">=", "value": "high"}}, index)
    assert spec.exposure is None and not validate(spec, index)[0]
    # "median" is a statistic, not a variable: the distribution describes the continuous outcome
    spec, _m, _n = to_spec({"analysis": "distribution", "base": "icu", "outcome": "los_icu_days", "variable": "median",
                            "stratify": ["careunit"]}, index)
    assert spec.variable == "los_icu_days"
    # "sepsis" in the ICU means the Sepsis-3 definition unless the question asks for diagnosis codes
    intent = {"analysis": "rate", "base": "icu", "outcome": "death_28d",
              "population": [{"term": "Sepsis (diagnosis code)", "op": "", "value": ""}]}
    assert to_spec(intent, index)[0].include[0].concept == "var:sepsis3"
    assert to_spec(intent, index, "sepsis by ICD codes")[0].include[0].concept == "phe:sepsis_dx"


def test_overlap_weights_balance_exactly_when_iptw_cannot():
    from mes.tools.rwd import stats as S
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"age": rng.normal(65, 12, 4000), "sofa": rng.poisson(4, 4000).astype(float)})
    t = (rng.random(4000) < 1 / (1 + np.exp(-(-6 + 0.08 * X.age + 0.3 * X.sofa)))).astype(int)   # strong confounding
    ps, _auc = S.propensity(X, t)
    w, _info = S.overlap_weights(ps, t)
    assert S.smd(X, t, w).smd.abs().max() < 0.01 < S.smd(X, t).smd.abs().max()


def test_statistics_accept_the_shapes_analysts_pass():
    """Whole (values, info) pairs, kinds as a list or inferred, a pandas treatment column: the same numbers."""
    from mes.tools.rwd import stats as S
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"age": rng.normal(65, 12, 3000), "sex": rng.choice(["M", "F"], 3000),
                       "vent": rng.integers(0, 2, 3000), "lactate": np.where(rng.random(3000) < 0.2, np.nan,
                                                                              rng.gamma(2, 1.2, 3000))},
                      index=np.arange(3000) + 7000)
    df["treated"] = (rng.random(3000) < 1 / (1 + np.exp(-(-3 + 0.04 * df.age)))).astype(int)
    cols = ["age", "sex", "vent", "lactate"]
    X, info = S.design_matrix(df, cols, {"age": "numeric", "sex": "category", "vent": "binary", "lactate": "numeric"})
    assert S.design_matrix(df, cols, ["numeric", "category", "binary", "numeric"])[0].equals(X)
    assert S.design_matrix(df, cols)[0].equals(X)                                   # kinds inferred
    ps, _auc = S.propensity(X, df.treated.values)
    pair = S.propensity(S.design_matrix(df, cols), df.treated)                      # pair and Series passed on
    assert np.allclose(pair[0], ps)
    w = S.overlap_weights(ps, df.treated.values)[0]
    assert np.allclose(S.overlap_weights(pair, df.treated)[0], w)
    assert np.allclose(S.iptw(pair, df.treated)[0], S.iptw(ps, df.treated.values)[0])
    assert np.allclose(S.smd(X, df.treated, S.overlap_weights(ps, df.treated)).smd, S.smd(X, df.treated.values, w).smd)
    assert S.weighted_risks(df.vent, df.treated, S.overlap_weights(ps, df.treated))["rd"] == \
        S.weighted_risks(df.vent.values, df.treated.values, w)["rd"]


def test_designer_base_title_and_coverage(synth):
    store, index, *_ = synth
    from mes.tools.result_schema import QueryPlan
    from mes.tools.rwd.designer import coverage, to_spec
    # "hospitalized" without anything ICU-specific: all admissions; a redundant "age ≥ 18" criterion is dropped
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "outcome": "hosp_readmit_30d",
                               "population": [{"term": "age", "op": ">=", "value": "18"}]}, index,
                              "What proportion of adults hospitalized with pneumonia are readmitted within 30 days?")
    assert spec.base == "hospital" and spec.include == [] and spec.title
    # an outcome the templates cannot express is reported instead of silently replaced
    plan = QueryPlan(question="q", entities=[{"name": "red blood cell transfusion", "role": "outcome"}])
    assert coverage(plan, StudySpec.from_dict({"analysis": "rate", "base": "icu", "outcome": "rrt_any"}), index) \
        == ["red blood cell transfusion"]
    plan = QueryPlan(question="q", entities=[{"name": "28-day mortality", "role": "outcome"}])
    assert coverage(plan, StudySpec.from_dict({"analysis": "rate", "base": "icu", "outcome": "death_28d"}), index) == []
    # terms that describe the analysis, inflected forms, strata and population criteria count as represented
    sepsis = {"concept": "var:sepsis3"}
    for entities, spec in (
            ([("calendar year", "intervention"), ("in-hospital mortality", "outcome")],
             {"analysis": "trend", "outcome": "death_hosp", "include": [sepsis]}),
            ([("first-day characteristics", "intervention"), ("in-hospital death", "outcome")],
             {"analysis": "prognostic", "outcome": "death_hosp", "include": [sepsis]}),
            ([("type of ICU", "intervention"), ("acute kidney injury", "outcome")],
             {"analysis": "rate", "outcome": "aki_7d", "stratify": ["careunit"]}),
            ([("Sepsis-3 criteria", "intervention"), ("28-day mortality", "outcome")],
             {"analysis": "rate", "outcome": "death_28d", "include": [sepsis]}),
            ([("Sequential Organ Failure Assessment score", "intervention"), ("in-hospital mortality", "outcome")],
             {"analysis": "rate", "outcome": "death_hosp", "stratify": ["sofa"]}),
            ([("proportion of patients given testdrug", "outcome")],
             {"analysis": "utilization", "exposure": {"concept": "drug:testdrug", "window_h": [0, None]}})):
        plan = QueryPlan(question="q", entities=[{"name": n, "role": r} for n, r in entities])
        assert coverage(plan, StudySpec.from_dict({"base": "icu", **spec}), index) == [], entities
    # … but a treatment of a comparison must be the exposure or comparator
    plan = QueryPlan(question="q", entities=[{"name": "vasopressin", "role": "intervention"}])
    assert coverage(plan, StudySpec.from_dict({"analysis": "compare", "base": "icu", "outcome": "death_28d",
                                               "exposure": {"concept": "drug:testdrug"}, "include": [sepsis]}),
                    index) == ["vasopressin"]


def test_treatment_use_during_the_stay_and_no_invented_outcome(synth):
    store, index, icu, treated, _early = synth
    from mes.tools.rwd.designer import to_spec
    # "how often … during their ICU stay": the model added an outcome the question never mentions
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "outcome": "death_hosp",
                               "exposure": {"term": "testdrug", "op": "", "value": "", "within_hours": 24}}, index,
                              "How often do ICU patients receive testdrug during their ICU stay?", "testdrug")
    assert spec.analysis == "utilization" and spec.outcome == "" and spec.exposure.window_h == [0, None]
    assert not validate(spec, index)[0]
    res = analyses.run(spec, Compiler(index, store).build(spec), "osler")
    e = res["estimates"][0]
    assert e["events"] == int(treated.sum()) and "during the ICU stay" in e["label"]
    # an outcome rate with an exposure is computed among the exposed, never with the exposure ignored
    spec = StudySpec.from_dict({"analysis": "rate", "base": "icu", "outcome": "death_28d",
                                "exposure": {"concept": "drug:testdrug", "window_h": [0, 24]}})
    assert not validate(spec, index)[0]
    res = analyses.run(spec, Compiler(index, store).build(spec), "osler")
    assert res["estimates"][0]["n"] == int(treated.sum()) and "among patients with" in res["estimates"][0]["label"]


def test_a_threshold_on_a_condition_keeps_the_condition(synth):
    """"sepsis >= 2" (a threshold the model added to a condition) still means Sepsis-3, not "not found"."""
    store, index, *_ = synth
    from mes.tools.rwd.designer import to_spec, unmatched
    spec, mapping, _notes = to_spec({"analysis": "association", "base": "icu", "outcome": "death_hosp",
                                     "population": [{"term": "sepsis", "op": ">=", "value": "2"}],
                                     "exposure": {"term": "lactate", "op": ">", "value": "4", "within_hours": 24}},
                                    index, "Is lactate above 4 associated with death in sepsis?")
    assert [c.concept for c in spec.include] == ["var:sepsis3"] and not unmatched({"mapping": mapping}, spec, index)


def test_tracebacks_in_study_records_are_reduced_to_the_error_line():
    from mes.tools.rwd.report import no_traceback
    text = ("Not reproduced in a fresh session: cell 3 failed when re-run (Traceback (most recent call last):\n"
            "  cell line 2: submit(res)\n  in result_check.py, _num()\nValueError: value must be a number)")
    assert no_traceback(text) == ("Not reproduced in a fresh session: cell 3 failed when re-run "
                                  "(ValueError: value must be a number)")
    assert no_traceback("plain") == "plain" and no_traceback(None) == ""
    from mes.ui import components as C
    from mes.tools.result_schema import EvidenceItem, Passage
    item = EvidenceItem(id="rwd:x", source_type="rwd", title="t", passages=[Passage(text="Results: " + text,
                                                                                     section="RWD results")])
    assert "Traceback" not in C._card_passage(item, limit=1000)


def test_an_edited_design_runs_without_the_language_model(synth, monkeypatch):
    # the "Edit design and re-run" path of the study panel (no designer call, no analyst)
    store, index, icu, *_ = synth
    from mes import config
    from mes.tools import rwd_tools
    from mes.tools.result_schema import QueryPlan
    from mes.tools.rwd import analyst, concepts
    monkeypatch.setattr(rwd_tools, "get_store", lambda: store)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (False, "test"))
    spec = {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}], "outcome": "death_28d"}
    items, stats = rwd_tools.search_rwd(QueryPlan(question="28-day mortality in sepsis"), config.get_mode("osler"),
                                        "28-day mortality in sepsis", spec_override=spec)
    study = stats["studies"][0]
    assert study["status"] == "done" and study["design_info"]["source"] == "edited by the user"
    assert study["result"]["estimates"][0]["n"] == int(icu.sepsis3.sum()) and items[0].source_type == "rwd"


def test_the_data_source_named_in_the_question_is_not_a_concept():
    # Claude's planner spelled out "MIMIC-IV" as a concept; the template then looked unable to express it
    from mes.tools.rwd.designer import generic
    assert generic("Medical Information Mart for Intensive Care IV database") and generic("MIMIC-IV (v2.2)")
    assert not generic("sepsis") and not generic("mental health") and not generic("data on 28-day mortality")


def test_code_check_of_an_analysis_finds_codes_and_names_mimic_does_not_have(synth):
    store, index, *_ = synth
    from mes.tools.rwd import code_audit
    code = ("sql(\"SELECT * FROM mes.drugs d WHERE d.drug ILIKE '%testdrug%' OR lower(drug) LIKE '%nosuchdrug%'\")\n"
            "sql(\"SELECT * FROM mes.diagnoses WHERE icd_code LIKE 'Q999%'\")\n"
            "sql(\"SELECT * FROM mes.nosuchtable\")")
    rows = code_audit.audit(code, index, store)
    by = {(r["kind"], r["written"]): r for r in rows}
    assert by[("medication pattern", "%testdrug%")]["patients"] > 0 and not by[("medication pattern", "%testdrug%")]["problem"]
    assert by[("medication pattern", "%nosuchdrug%")]["problem"]
    assert by[("ICD code", "Q999%")]["problem"] and by[("table", "mes.nosuchtable")]["problem"]
    assert len(code_audit.problems(rows)) == 3 and "PROBLEM" in code_audit.review_text(rows)
    assert code_audit.audit("", index, store) == []


def test_a_redundant_covariate_does_not_break_the_doubly_robust_estimate():
    # a duplicated missing-value indicator once left the robust standard errors undefined (CI "nan")
    import numpy as np
    import pandas as pd
    from mes.tools.rwd import stats as S
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x + 0.3 * t - 1)))).astype(int)
    X = pd.DataFrame({"x": x, "miss_a": np.zeros(n), "dup": 2 * x, "c": rng.integers(0, 2, n)})
    X["c2"] = X["c"]
    est = S.weighted_or(y, t, np.ones(n), X)
    assert all(np.isfinite([est["value"], est["lo"], est["hi"]])) and est["lo"] < est["value"] < est["hi"]
    assert S._independent(pd.DataFrame({"a": x, "treated": t, "b": x})) == ["treated", "a"]


def test_estimates_a_model_could_not_compute_are_not_shown(monkeypatch):
    from mes.tools.rwd import analyses
    fake = {"estimates": [{"label": "ok", "measure": "OR", "value": 1.1, "lo": 0.9, "hi": 1.3},
                          {"label": "nan CI", "measure": "OR", "value": 1.0, "lo": float("nan"), "hi": float("nan")},
                          {"label": "zero", "measure": "OR", "value": 0.0, "lo": 0.0, "hi": 0.0},
                          {"label": "count", "measure": "count", "value": "<10"}], "warnings": []}
    monkeypatch.setitem(analyses.RUNNERS, "rate", lambda spec, cohort, mode: fake)
    res = analyses.run(type("S", (), {"analysis": "rate"})(), None)
    assert [e["label"] for e in res["estimates"]] == ["ok", "count"] and "nan CI; zero" in res["warnings"][0]


def test_a_lab_test_distribution_is_named_by_the_test():
    """A distribution of a lab test (lab:…) is labelled with the test's name, not its concept id."""
    from mes.tools.rwd.compiler import Cohort
    spec = StudySpec.from_dict({"analysis": "distribution", "base": "icu", "variable": "lab:50813"})
    cohort = Cohort(df=pd.DataFrame({"variable_value": [1.0, 2.0, 3.0, 4.0, np.nan]}), sql="", attrition=[("all", 5)],
                    labels={"variable_value": "Lactate (Blood, Blood Gas), highest in the first 24 h"})
    res = analyses.run_distribution(spec, cohort, "osler")
    assert res["estimates"][0]["label"].startswith("Lactate (Blood, Blood Gas), highest in the first 24 h: median")
    assert res["estimates"][0]["n"] == 4 and "lab:50813" not in res["methods"]
