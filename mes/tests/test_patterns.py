"""
Offline tests of the descriptive RWD questions and of answering every RWD question: characterizations (one cohort, two
groups), trajectories (a lab test per interval, by outcome), treatment pathways (lines, sequences, the Sankey links),
the data catalog ("does MIMIC-IV record …"), the designer's routing of each, the follow-up that refines the previous
study, the ladder that always answers (template → exploratory query → catalog), and their cards and reports — on a
synthetic MIMIC-IV store with a lab test and a second drug.

    python -m pytest mes/tests/test_patterns.py -q
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools import rwd_tools  # noqa: E402
from mes.tools.rwd import analyses, analyst, catalog, concepts, counts, db, patterns, report  # noqa: E402
from mes.tools.rwd.compiler import Compiler  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402
from mes.tools.rwd.spec import StudySpec, validate  # noqa: E402


def _extend(root, icu, treated):
    """A lactate (labevents 50813) every 4 h for 60 h, higher in those who die; a second drug given to some."""
    rng = np.random.default_rng(3)
    rows = []
    for _i, r in icu.sample(1500, random_state=1).iterrows():
        base = 4.0 if r.death_28d else 2.0
        for hour in range(0, 60, 4):
            rows.append({"subject_id": r.subject_id, "hadm_id": r.hadm_id, "itemid": 50813,
                         "charttime": r.intime + pd.Timedelta(hours=hour + 0.5), "valuenum": base * np.exp(-hour / 80)
                         + rng.normal(0, 0.2)})
    pd.DataFrame(rows).to_parquet(root / "hosp" / "labevents.parquet")
    drugs = pd.read_parquet(root / "mes" / "drugs.parquet")
    other = icu[treated].sample(400, random_state=2)
    more = pd.DataFrame({"subject_id": other.subject_id, "hadm_id": other.hadm_id, "stay_id": other.stay_id,
                         "source": "order", "drug": "otherdrug", "route": "iv",
                         "starttime": other.intime + pd.to_timedelta(rng.uniform(21, 40, len(other)), unit="h"),
                         "endtime": other.intime + pd.to_timedelta(50, unit="h")})
    pd.concat([drugs, more], ignore_index=True).to_parquet(root / "mes" / "drugs.parquet")
    names = pd.read_parquet(root / "mes" / "drug_names.parquet")
    names.loc[names.drug == "otherdrug", "n_patients"] = len(other)
    names.to_parquet(root / "mes" / "drug_names.parquet")
    d = pd.read_parquet(root / "dictionary.parquet")
    d = pd.concat([d, pd.DataFrame({"kind": ["lab"], "concept_id": ["lab:50813"], "label": ["Lactate (Blood, Blood Gas)"],
                                    "code": ["50813"], "vocabulary": ["labevents"], "n_patients": [1500]})])
    d.to_parquet(root / "dictionary.parquet")
    np.save(root / "dictionary_vectors.npy", np.zeros((len(d), 384), dtype=np.float16))
    # cultures: positive blood (E. coli) 5 h after ICU admission, cancelled, no growth, positive long before, urine
    picked = icu.sample(630, random_state=4)
    groups = [picked.iloc[a:b] for a, b in ((0, 300), (300, 400), (400, 500), (500, 550), (550, 630))]
    rows = []
    for g, (spec, org, hours) in zip(groups, (("BLOOD CULTURE", "ESCHERICHIA COLI", 5), ("BLOOD CULTURE", "CANCELLED", 5),
                                              ("BLOOD CULTURE", None, 5), ("BLOOD CULTURE", "KLEBSIELLA", -100),
                                              ("URINE", "PROTEUS MIRABILIS", 10))):
        for _i, r in g.iterrows():
            rows.append({"subject_id": r.subject_id, "hadm_id": r.hadm_id, "spec_type_desc": spec, "org_name": org,
                         "charttime": r.intime + pd.Timedelta(hours=hours), "chartdate": (r.intime + pd.Timedelta(
                             hours=hours)).normalize()})
    pd.DataFrame(rows).to_parquet(root / "hosp" / "microbiologyevents.parquet")
    other.attrs["blood"] = set(groups[0].stay_id)
    return other


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_patterns")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, treated, _early = _synthetic_store(root)
    other = _extend(root, icu, treated)
    store = Store(root)
    return SimpleNamespace(store=store, index=ConceptIndex(store), icu=icu, treated=treated, other=other)


@pytest.fixture
def patched(world, monkeypatch):
    monkeypatch.setattr(rwd_tools, "get_store", lambda: world.store)
    monkeypatch.setattr(db, "get_store", lambda: world.store)
    monkeypatch.setattr(concepts, "get_index", lambda: world.index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (False, "test"))
    counts.clear()
    return world


def _run(world, design):
    spec = StudySpec.from_dict(design)
    errors, _ = validate(spec, world.index)
    assert not errors, errors
    cohort = Compiler(world.index, world.store).build(spec)
    return spec, cohort, analyses.run(spec, cohort, "osler")


# ============================================================================ characterize
def test_a_characterization_of_one_cohort_and_of_two_groups(patched):
    w = patched
    _s, cohort, res = _run(w, {"analysis": "characterize", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                               "covariates": ["age", "sofa", "gender"]})
    t1 = res["table1"]
    assert t1["single"] and t1["n"] == [int(w.icu.sepsis3.sum())] and "control_raw" not in t1["rows"][0]
    spec, cohort, res = _run(w, {"analysis": "characterize", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                                 "outcome": "death_28d", "group_by": "outcome", "covariates": ["age", "sofa", "gender"]})
    sub = w.icu[w.icu.sepsis3]
    assert res["groups"]["n_treated"] == int(sub.death_28d.sum()) and res["groups"]["treated"] == "died"
    diffs = res["tables"]["differences"]
    assert diffs[0]["variable"] == "sofa" and abs(diffs[0]["smd"]) == max(abs(d["smd"]) for d in diffs)
    text = report.pattern_text(spec, res)
    assert text.startswith("Characteristics, died vs survived") and "SOFA" in text
    spec, _c, res = _run(w, {"analysis": "characterize", "base": "icu", "exposure": {"concept": "drug:testdrug",
                                                                                     "window_h": [0, 24]},
                             "group_by": "exposure", "covariates": ["age", "sofa"]})
    assert res["groups"]["n_treated"] == int(w.treated.sum())
    bad = StudySpec.from_dict({"analysis": "characterize", "base": "icu", "group_by": "age"})
    assert any("cannot form groups" in e for e in validate(bad, w.index)[0])


# ============================================================================ trajectory
def test_a_trajectory_is_the_median_of_each_stays_mean_per_interval(patched):
    w = patched
    spec, cohort, res = _run(w, {"analysis": "trajectory", "base": "icu", "variable": "lactate_max", "outcome": "death_28d",
                                 "group_by": "outcome", "horizon_h": 48, "bin_h": 8})
    chart = res["charts"]["trajectory"]
    assert [g["label"] for g in chart["groups"]] == ["died", "survived"] and chart["unit"] == "mmol/L"
    died = chart["groups"][0]["points"]
    assert [p["t0"] for p in died] == [0, 8, 16, 24, 32, 40] and died[-1]["t1"] == 48
    # the 8–16 h interval of those who died, computed again by hand
    lab = pd.read_parquet(w.store.root / "hosp" / "labevents.parquet")
    lab = lab.merge(w.icu[["hadm_id", "intime", "death_28d"]], on="hadm_id")
    hours = (lab.charttime - lab.intime).dt.total_seconds() / 3600
    part = lab[(hours >= 8) & (hours < 16) & (lab.death_28d == 1)]
    per_stay = part.groupby("hadm_id").valuenum.mean()
    lo, hi = pd.concat([lab.valuenum]).quantile([0.005, 0.995])           # (the trimming of the analysis)
    assert died[1]["n"] == int(((per_stay >= lo - 1) & (per_stay <= hi + 1)).sum()) or died[1]["n"] <= len(per_stay)
    assert abs(died[1]["median"] - float(per_stay.median())) < 0.05
    assert died[0]["median"] > chart["groups"][1]["points"][0]["median"]          # higher in those who die
    assert all(p["n"] >= 10 for g in chart["groups"] for p in g["points"])
    assert "Median (mmol/L) and IQR of Lactate" in report.pattern_text(spec, res)
    spec = StudySpec.from_dict({"analysis": "trajectory", "base": "icu", "variable": "sofa"})
    assert any("measured over time" in e for e in validate(spec, w.index)[0])


def test_a_measurement_just_before_the_horizon_is_in_the_last_interval(patched):
    w = patched
    lab = pd.read_parquet(w.store.root / "hosp" / "labevents.parquet").sort_values("charttime")
    first = lab.groupby("hadm_id").head(1)
    c = first[first.charttime.dt.second < 30].iloc[0]
    # 47 h 59 m 30 s + under 30 s: inside 48 h, but 48 minute-boundaries of hours (the bin past the end, before)
    t0 = c.charttime.floor("min") - pd.Timedelta(hours=48) + pd.Timedelta(seconds=30)
    cohort = SimpleNamespace(df=pd.DataFrame({"unit_id": [1], "hadm_id": [c.hadm_id], "t0": [t0]}))
    m = patterns.measurements(w.store, cohort, [("lab:50813", None)], 48, 8, "icu")
    assert m.bin.tolist() == [5]


# ============================================================================ pathways
def test_treatment_pathways_count_lines_and_sequences(patched):
    w = patched
    spec, cohort, res = _run(w, {"analysis": "pathways", "base": "icu", "horizon_h": 48,
                                 "treatments": [{"concept": "drug:testdrug"}, {"concept": "drug:otherdrug"}]})
    rows = {r["treatment"]: r for r in res["tables"]["treatments"]}
    test, other = rows["Medication containing “testdrug”"], rows["Medication containing “otherdrug”"]
    assert test["events"] == int(w.treated.sum()) and other["events"] == len(w.other)
    # testdrug always comes first (0.5–20 h), otherdrug after it (21–40 h)
    seqs = {q["sequence"]: q["n"] for q in res["tables"]["sequences"]}
    both = "Medication containing “testdrug” → Medication containing “otherdrug”"
    assert seqs[both] == len(w.other) and seqs["Medication containing “testdrug”"] == int(w.treated.sum()) - len(w.other)
    assert seqs["none of them"] == len(w.icu) - int(w.treated.sum())
    links = res["charts"]["sankey"]["links"]
    assert all(link["n"] >= 10 for link in links)
    assert {"from": [1, "Medication containing “testdrug”"], "to": [2, "Medication containing “otherdrug”"],
            "n": len(w.other)} in links
    assert test["first"] == pytest.approx(100 * w.treated.sum() / len(w.icu), abs=0.1)
    assert "Commonest sequences" in report.pattern_text(spec, res)
    assert patterns.sequences(np.array([[2.0, np.nan], [5.0, 1.0], [np.nan, np.nan]]), ["a", "b"]) == [("a",), ("b", "a"), ()]
    bad = StudySpec.from_dict({"analysis": "pathways", "base": "icu", "treatments": [{"concept": "var:sofa"}]})
    errors = validate(bad, w.index)[0]
    assert any("not a treatment" in e for e in errors) and any("at least two" in e for e in errors)


# ============================================================================ the catalog and the ladder
def test_the_catalog_says_what_the_data_records(patched):
    w = patched
    res = catalog.run("Does MIMIC-IV record testdrug and unobtainium?", ["testdrug", "unobtainium"], w.index, w.store)
    rows = res["tables"]["concepts"]
    found = [r for r in rows if r["concept"]]
    assert found and found[0]["term"] == "testdrug" and found[0]["patients"] == int(w.treated.sum())
    assert any(r["term"] == "unobtainium" and r["concept"] is None for r in rows)
    assert "Not found in MIMIC-IV: unobtainium" in res["warnings"]
    assert res["tables"]["suggestions"] and "How often is" in res["tables"]["suggestions"][0]["question"]
    text = catalog.text(res)
    assert "“unobtainium”: not found" in text and "patients" in text


class FakeLLM:
    label = "fake"

    def __init__(self, sql=None):
        self.sql, self.n_calls = sql, 0

    def invoke(self, prompt, **_kw):
        self.n_calls += 1
        if self.sql is None:
            return "not json"
        return json.dumps({"sql": self.sql, "explanation": "test"})


def _plan(question, names):
    from mes.tools.result_schema import QueryPlan
    return QueryPlan(question=question, standalone_question=question, entities=[{"name": n} for n in names])


def test_every_question_gets_an_answer(patched, monkeypatch):
    from mes import config
    from mes.tools.rwd import designer
    w = patched
    mode = config.get_mode("osler")
    monkeypatch.setattr(designer, "design", lambda *a, **k: (None, {"errors": ["no template fits"], "mapping": []}))
    monkeypatch.setattr(designer, "coverage", lambda *a, **k: [])
    # 1. no template: an exploratory query answers
    sql = "SELECT careunit, count(*) AS n FROM mes.icu_stays GROUP BY 1 ORDER BY 1"
    items, stats = rwd_tools.search_rwd(_plan("How many stays per ICU type?", ["ICU type"]), mode,
                                        "How many stays per ICU type?", llm=FakeLLM(sql), engine="templates")
    study = stats["studies"][0]
    assert study["status"] == "done" and study["spec"]["analysis"] == "explore" and study["result"]["tables"]["result"]
    assert "exploratory query instead" in study["design_info"]["routed"]
    # 2. neither a template nor a query: what the data records, and what to ask
    items, stats = rwd_tools.search_rwd(_plan("Something odd about testdrug", ["testdrug"]), mode,
                                        "Something odd about testdrug", llm=FakeLLM(None), engine="templates")
    kinds = [(s["status"], (s.get("spec") or {}).get("analysis")) for s in stats["studies"]]
    assert kinds == [("failed", "explore"), ("done", "catalog")]
    assert items and items[-1].meta["analysis"] == "catalog" and "testdrug" in items[-1].passages[0].text


def test_explorer_queries_in_another_dialect_run_in_duckdb(patched):
    from mes.tools.rwd import explorer
    w = patched
    grouped = ("SELECT careunit, CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY los_icu_days) OVER (PARTITION BY "
               "careunit) AS REAL) AS median_los FROM mes.icu_stays GROUP BY careunit")
    fixed = explorer.dialect(grouped)
    assert "OVER" not in fixed and "WITHIN GROUP (ORDER BY los_icu_days)" in fixed
    window = explorer.dialect("SELECT DISTINCT careunit, percentile_cont(0.25) WITHIN GROUP (ORDER BY sofa DESC) OVER "
                              "(PARTITION BY careunit) AS q FROM mes.icu_stays")
    assert "quantile_cont(sofa, 0.75) OVER (PARTITION BY careunit)" in window
    assert "quantile_cont(los_icu_days, 0.5)" in explorer.dialect(
        "SELECT APPROX_QUANTILES(los_icu_days, 4)[OFFSET(2)] AS m FROM mes.icu_stays")
    hours = explorer.dialect("SELECT avg(DATEDIFF(hour, intime, outtime)) AS a, avg(TIMESTAMP_DIFF(outtime, intime, "
                             "HOUR)) AS b FROM mes.icu_stays")
    assert hours.count("date_diff('hour', intime, outtime)") == 2
    assert explorer.dialect("SELECT json_object_agg(careunit, n) AS j, NVL(x, 0) FROM t") == \
        "SELECT json_group_object(careunit, n) AS j, coalesce(x, 0) FROM t"
    duck = "SELECT careunit, quantile_cont(los_icu_days, 0.5) AS m, date_diff('hour', intime, outtime) AS h FROM mes.icu_stays"
    assert explorer.dialect(duck) == duck                      # DuckDB's own SQL is left as written
    alias = ("SELECT count(DISTINCT subject_id) AS n FROM mes.icu_stays WHERE EXISTS (SELECT 1 FROM mes.drugs d WHERE "
             "d.stay_id = t.stay_id)")
    error = 'BinderException: Binder Error: Referenced table "t" not found!\nCandidate tables: "icu_stays"\n'
    assert "FROM mes.icu_stays AS t WHERE" in explorer.repair(alias, error)
    assert explorer.repair(alias, error.replace('"icu_stays"', '"icu_stays", "drugs"')) == alias
    # the independent query agrees when it only rounded more (a median of 1.8 days cast to 2), not otherwise
    a = pd.DataFrame({"unit": ["A", "B"], "n": [6585, 3621], "median": [1.8, 2.5]})
    assert explorer.agree(a, pd.DataFrame({"unit": ["A", "B"], "n": [6585, 3621], "median": [2, 3]}))
    assert not explorer.agree(a, pd.DataFrame({"unit": ["A", "B"], "n": [6585, 3621], "median": [1.9, 2.5]}))
    assert not explorer.agree(a, pd.DataFrame({"unit": ["A", "B"], "n": [6585, 3621], "median": [2, 4]}))
    assert not explorer.agree(a.assign(median=[3.0, 4.0]), a.assign(median=[4.0, 5.0]))
    # the explorer runs the rewritten query, and adds the alias without asking the model again
    for sql, calls in ((grouped, 2), (alias, 2)):
        llm = FakeLLM(sql)
        res = explorer.explore("q", llm, w.store)
        assert res["tables"]["result"], res["warnings"]
        assert llm.n_calls == calls                            # the query and its independent check, once each


def test_the_designer_routes_each_kind_of_question(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu"}, w.index, "Does MIMIC-IV record testdrug doses?")
    assert spec.analysis == "catalog" and any("data catalog" in n for n in notes)
    # the user's own words, when the planner rewrote the question
    spec, _m, _n = to_spec({"analysis": "explore", "base": "icu"}, w.index, "Count of testdrug records in the database",
                           original="Does MIMIC-IV record testdrug doses?")
    assert spec.analysis == "catalog"
    spec, _m, notes = to_spec({"analysis": "characterize", "base": "icu",
                               "population": [{"term": "sepsis", "op": "", "value": ""}],
                               "comparator": {"term": "ICU type", "op": "", "value": ""}}, w.index,
                              "How do patients with sepsis who die differ from those who survive?")
    assert (spec.group_by, spec.outcome, spec.comparator) == ("outcome", "death_28d", None)
    spec, _m, _n = to_spec({"analysis": "trajectory", "base": "icu", "variable": "lactate",
                            "exposure": {"term": "lactate", "op": ">=", "value": "", "within_hours": 72},
                            "outcome": "death_28d"}, w.index, "lactate over the first 48 hours, survivors vs non-survivors")
    assert (spec.variable, spec.horizon_h, spec.bin_h, spec.exposure) == ("lactate_max", 48.0, 4.0, None)
    assert not validate(spec, w.index)[0]
    spec, _m, _n = to_spec({"analysis": "pathways", "base": "icu",
                            "treatments": [{"term": "testdrug", "op": "", "value": ""},
                                           {"term": "otherdrug", "op": "", "value": ""}]}, w.index,
                           "In what order are testdrug and otherdrug started?")
    assert [c.concept for c in spec.treatments] == ["drug:testdrug", "drug:otherdrug"]
    # a comparison of patients with and without a condition (no treatment): survival curves, or an association
    intent = {"analysis": "compare", "base": "icu", "outcome": "death_28d",
              "population": [{"term": "sepsis", "op": "=", "value": "yes"}],
              "exposure": {"term": "sepsis", "op": "=", "value": "yes", "within_hours": 0},
              "comparator": {"term": "sepsis", "op": "=", "value": "no"}}
    spec, mapping, notes = to_spec(intent, w.index, "Compare 28-day survival between ICU patients with and without sepsis")
    assert (spec.analysis, spec.exposure.concept, spec.comparator, spec.include) == ("survival", "var:sepsis3", None, [])
    assert not validate(spec, w.index)[0] and all(m["concept"] for m in mapping)
    spec, _m, _n = to_spec(intent, w.index, "Is sepsis linked to 28-day mortality in the ICU?")
    assert (spec.analysis, spec.exposure.concept) == ("association", "var:sepsis3")
    spec, _m, notes = to_spec({"analysis": "compare", "base": "icu", "outcome": "death_28d",
                               "exposure": {"term": "testdrug", "op": "", "value": "", "within_hours": 24},
                               "stratify": ["gender"]}, w.index, "And only in women?")
    assert any(c.concept == "var:gender" and c.value == "F" for c in spec.include) and "gender" not in spec.stratify


def test_a_follow_up_carries_the_previous_design(patched):
    from mes.tools.pipeline_tools import conversation_turn
    from mes.tools.result_schema import MESResult
    from mes.tools.rwd.designer import _prompt
    w = patched
    spec, cohort, res = _run(w, {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}],
                                 "outcome": "death_28d"})
    study = {"key": "r1", "kind": "template", "status": "done", "spec": spec.to_dict(), "result": res}
    turn = conversation_turn(MESResult(question="q", mode="osler", sources=["rwd"], plan={"question": "q"}, evidence=[],
                                       sections={"Answer": "x"}, rwd_studies=[study]))
    assert "rate" in turn["rwd_design"] and "Sepsis-3" in turn["rwd_design"]
    plan = _plan("And 1-year mortality?", [])
    plan.rwd_previous = turn["rwd_design"]
    prompt = _prompt("And 1-year mortality?", plan, w.index)
    assert "PREVIOUS STUDY OF THIS CONVERSATION" in prompt and "Sepsis-3" in prompt


# ============================================================================ the cards
def test_the_cards_of_the_new_kinds_of_study(patched):
    from mes.tests.test_viz import _result
    from mes.tools.rwd_tools import _plain
    from mes.ui import study_card, study_report
    from mes.ui.viz import clean
    w = patched
    designs = {"characterize": {"analysis": "characterize", "base": "icu", "outcome": "death_28d", "group_by": "outcome",
                                "covariates": ["age", "sofa"]},
               "trajectory": {"analysis": "trajectory", "base": "icu", "variable": "lactate_max", "outcome": "death_28d",
                              "group_by": "outcome", "horizon_h": 48, "bin_h": 8},
               "pathways": {"analysis": "pathways", "base": "icu", "treatments": [{"concept": "drug:testdrug"},
                                                                                  {"concept": "drug:otherdrug"}]}}
    expect = {"characterize": {"forest"}, "trajectory": {"trajectory"}, "pathways": {"sankey", "received", "ranges"}}
    for name, design in designs.items():
        spec, cohort, res = _run(w, design)
        study = {"key": name, "kind": "template", "status": "done", "evidence_id": "R1", "spec": spec.to_dict(),
                 "result": _plain(res), "sql": cohort.sql, "timings": {"total": 1.0}}
        card = study_card.card(_result([study]), study)
        charts = {b["chart"]["type"] for t in card["tabs"] for b in t["blocks"] if b.get("type") == "chart"}
        assert expect[name] <= charts, (name, charts)
        assert card["kpis"] and card["design"] != "Study"
        json.dumps(clean(card))
        assert study_report.report_html(_result([study]), study)
    res = catalog.run("q", ["testdrug"], w.index, w.store)
    study = {"key": "cat", "kind": "template", "status": "done", "evidence_id": "R1", "result": res,
             "spec": {"analysis": "catalog", "base": "icu", "title": res["title"]}}
    card = study_card.card(_result([study]), study)
    assert card["design"] == "Data catalog" and [t["label"] for t in card["tabs"]] == ["What MIMIC-IV has"]
    assert "no study design" in rwd_tools.editable(study, "design")          # (the panel shows no editor for it)
    assert any(b["type"] == "chips" for b in card["tabs"][0]["blocks"])


def test_a_cohort_under_ten_patients_shows_nothing_but_its_flow(patched):
    w = patched
    top = sorted(w.icu.sofa)[-5]                                          # the 5 sickest stays at most
    spec, cohort, res = _run(w, {"analysis": "distribution", "base": "icu", "variable": "age",
                                 "include": [{"concept": "var:sofa", "op": ">=", "value": float(top)}]})
    assert 0 < len(cohort.df) < 10
    assert res["estimates"] == [] and res["n"] is None and res["attrition"][-1][1] is None
    assert any("Fewer than 10 patients" in x for x in res["warnings"])
    items = report.to_items(spec, res, "k1")
    text = " ".join(p.text for it in items for p in it.passages)
    assert "<10" in text and "Fewer than 10 patients" in text and str(len(cohort.df)) + " " not in text.split("→")[-1]


def test_a_count_of_one_concept_is_counted_by_a_template(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    spec, _m, notes = to_spec({"analysis": "explore", "base": "icu", "sql_question": "x",
                               "exposure": {"term": "testdrug", "op": "", "value": "", "within_hours": 0}}, w.index,
                              "How many patients received testdrug?")
    assert spec.analysis == "utilization" and spec.exposure.concept == "drug:testdrug" and spec.exposure.window_h == [0, None]
    assert any("a count of one concept" in n for n in notes)
    # a measurement named without a threshold is never a criterion (once "age" matched an ICD code)
    spec, _m, _n = to_spec({"analysis": "distribution", "base": "icu", "population": [{"term": "sepsis"}],
                            "exposure": {"term": "age", "op": "", "value": ""}, "variable": "age"}, w.index,
                           "What is the median age of ICU patients with sepsis?")
    assert spec.variable == "age" and [c.concept for c in spec.include] == ["var:sepsis3"]


def test_a_term_without_a_concept_gets_an_exploratory_query_first(patched, monkeypatch):
    from mes import config
    from mes.tools.rwd import designer
    w = patched
    spec = StudySpec.from_dict({"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}]})
    mapping = [{"role": "condition", "term": "positive blood culture", "concept": None, "label": "not found in MIMIC-IV"}]
    monkeypatch.setattr(designer, "design", lambda *a, **k: (spec, {"errors": [], "mapping": mapping}))
    monkeypatch.setattr(designer, "coverage", lambda *a, **k: [])
    sql = "SELECT careunit, count(*) AS n FROM mes.icu_stays GROUP BY 1 ORDER BY 1"
    q = "How many ICU patients had a positive blood culture?"
    items, stats = rwd_tools.search_rwd(_plan(q, ["positive blood culture"]), config.get_mode("osler"), q,
                                        llm=FakeLLM(sql), engine="templates")
    study = stats["studies"][0]
    assert study["spec"]["analysis"] == "explore" and study["status"] == "done"
    assert "no entry for “positive blood culture”" in study["design_info"]["routed"]
    warnings = study["result"]["warnings"]
    assert any("no entry for “positive blood culture”" in x for x in warnings)
    assert any("not independently confirmed" in x for x in warnings)          # the fake writes the same query twice


def test_the_catalog_says_how_big_the_data_is(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    spec, _m, _n = to_spec({"analysis": "explore", "base": "icu"}, w.index, "How many patients are in MIMIC-IV?")
    assert spec.analysis == "catalog"
    res = catalog.run("How many patients are in MIMIC-IV?", ["MIMIC-IV database patients"], w.index, w.store)
    assert res["tables"]["concepts"] == [] and not res["warnings"]
    icu = next(r for r in res["tables"]["overview"] if "icu population" in r["what"])
    assert icu["n"] == len(w.icu)
    assert f"{len(w.icu):,}" in catalog.text(res)


def test_the_incidence_of_a_condition_counts_it_in_everyone(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    # "X >= 1" is the model's way to say "with X"; the one condition of a rate without an outcome is what is counted
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "outcome": "aki_7d",
                               "population": [{"term": "adult ICU patients (first ICU stay)"},
                                              {"term": "sepsis", "op": ">=", "value": "1"}]}, w.index,
                              "What is the incidence of sepsis in the ICU?")
    assert (spec.analysis, spec.include, spec.exposure.concept, spec.outcome) == ("rate", [], "var:sepsis3", "")
    spec2, cohort, res = _run(w, spec.to_dict())
    assert res["estimates"][0]["value"] == pytest.approx(100 * w.icu.sepsis3.mean(), abs=0.05)


class ThreeLLM(FakeLLM):
    """An explorer whose first query is wrong, its check and a third right."""

    def __init__(self, wrong, right, plain):
        super().__init__()
        self.queries = [wrong, right, plain]

    def invoke(self, prompt, **_kw):
        self.n_calls += 1
        return json.dumps({"sql": self.queries[min(self.n_calls, 3) - 1], "explanation": "test"})


def test_a_third_query_decides_when_two_disagree(patched):
    from mes.tools.rwd import explorer
    w = patched
    wrong = "SELECT count(*) AS n FROM mes.icu_stays WHERE sofa > 100"
    right = "WITH s AS (SELECT stay_id FROM mes.icu_stays WHERE sepsis3) SELECT count(*) AS n FROM s"
    plain = "SELECT count(stay_id) AS n FROM mes.icu_stays WHERE sepsis3"
    res = explorer.explore("How many ICU stays have sepsis?", ThreeLLM(wrong, right, plain), w.store,
                           terms=["sepsis"])
    assert res["diagnostics"]["independent_query_agrees"] is True and "2 of 3" in res["diagnostics"]["queries_agreeing"]
    assert res["tables"]["result"][0]["n"] == int(w.icu.sepsis3.sum()) and res["sql"] == right


def test_a_length_of_stay_is_never_a_rate(patched):
    from mes.tools.rwd import stats as S
    from mes.tools.rwd.designer import to_spec
    w = patched
    # ICU type without a level the data has: a breakdown, and the rate of a length of stay its distribution
    spec, _m, notes = to_spec({"analysis": "compare", "base": "icu", "outcome": "los_icu_days",
                               "exposure": {"term": "admission type", "op": "=", "value": "unknown"}}, w.index,
                              "ICU length of stay by admission type")
    assert (spec.analysis, spec.variable) == ("distribution", "los_icu_days") and not validate(spec, w.index)[0]
    bad = StudySpec.from_dict({"analysis": "rate", "base": "icu", "outcome": "los_icu_days"})
    assert any("not a rate" in e for e in validate(bad, w.index)[0])
    assert S.wilson(30, 10)[0] == 1.0 and S.wilson(0, 0) == (None, None, None)


def test_a_positive_culture_is_a_concept(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    blood = w.other.attrs["blood"]
    assert w.index.is_known("micro:blood") and w.index.label("micro:blood") == "Positive blood culture"
    assert w.index.count("micro:blood") == 300 + 50                    # anywhere: cancelled and no growth are not
    spec, _m, _n = to_spec({"analysis": "rate", "base": "icu", "exposure": {"term": "positive blood culture",
                                                                            "within_hours": 0}}, w.index,
                           "How many ICU patients had a positive blood culture?")
    assert spec.exposure.concept == "micro:blood" and spec.exposure.window_h == [0, None]
    spec, cohort, res = _run(w, spec.to_dict())
    assert res["estimates"][0]["events"] == len(blood)                 # during the stay: not the one 100 h before
    _s, cohort, _r = _run(w, {"analysis": "rate", "base": "icu", "outcome": "death_28d",
                              "include": [{"concept": "micro:urine"}]})
    assert len(cohort.df) == 80
    from mes.tools.rwd.definitions import describe
    d = describe({"analysis": "rate", "base": "icu", "exposure": {"concept": "micro:blood", "window_h": [0, None]}},
                 w.index)[0]
    assert d["kind"] == "microbiology" and d["codes"] == {"Specimen": ["BLOOD CULTURE"]} and "contaminants" in d["detail"]


# ============================================================================ the commonest items, durations
def test_the_commonest_drugs_by_ingredient_and_window(patched):
    from mes.tools.rwd import ranking
    from mes.tools.rwd.designer import to_spec
    w = patched
    spec, _m, notes = to_spec({"analysis": "utilization", "base": "icu", "exposure": {"term": "drugs"}}, w.index,
                              "Which drugs are most commonly given in the first 24 hours of ICU admission?")
    assert (spec.analysis, spec.ranked, spec.horizon_h) == ("top", "drugs", 24.0) and not validate(spec, w.index)[0]
    spec, cohort, res = _run(w, spec.to_dict())
    rows = {r["item"]: r for r in res["tables"]["ranking"]}
    assert rows["Testdrug"]["patients"] == int(w.treated.sum())                       # given 0.5–20 h after
    assert rows["Otherdrug"]["patients"] < len(w.other) / 3                            # started 21–40 h after
    spec, cohort, res = _run(w, {"analysis": "top", "base": "icu", "ranked": "drugs", "horizon_h": 0,
                                 "outcome": "death_28d", "group_by": "outcome"})
    rows = {r["item"]: r for r in res["tables"]["ranking"]}
    assert list(rows) == ["Testdrug", "Otherdrug"] and rows["Otherdrug"]["patients"] == len(w.other)  # the whole stay
    died = w.icu.death_28d == 1
    a = 100 * (w.treated & died).sum() / died.sum()
    assert rows["Testdrug"]["a"] == pytest.approx(round(a, 1), abs=0.11) and res["groups"]["names"] == ["died", "survived"]
    text = ranking.text(spec, res)
    assert text.startswith("The commonest medications of the") and "died" in text
    assert ranking.ingredient("iso-osmotic dextrose") == "dextrose" and ranking.ingredient("kcl (bolus)") == \
        "potassium chloride" and ranking.ingredient("syringe") is None
    from mes.tests.test_viz import _result
    from mes.tools.rwd_tools import _plain
    from mes.ui import study_card
    study = {"key": "t1", "kind": "template", "status": "done", "evidence_id": "R1", "spec": spec.to_dict(),
             "result": _plain(res), "sql": cohort.sql, "timings": {"total": 1.0}}
    card = study_card.card(_result([study]), study)
    charts = [b["chart"] for t in card["tabs"] for b in t["blocks"] if b.get("type") == "chart"]
    assert card["design"] == "Commonest items" and any(c["type"] == "received" for c in charts)
    assert card["kpis"][0]["value"] == "Testdrug"


def test_the_time_on_ventilation_is_its_own_variable(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    spec, _m, notes = to_spec({"analysis": "distribution", "base": "icu", "variable": "ventilation"}, w.index,
                              "How many hours of mechanical ventilation do ICU patients receive?")
    assert (spec.analysis, spec.variable) == ("distribution", "derived:vent_hours") and not validate(spec, w.index)[0]
    spec, _m, _n = to_spec({"analysis": "distribution", "base": "icu", "variable": "dialysis"}, w.index,
                           "How long are patients on dialysis?")
    assert spec.analysis == "explore" and spec.sql_question
    bad = StudySpec.from_dict({"analysis": "distribution", "base": "hospital", "variable": "derived:vent_hours"})
    assert any("ICU stays only" in e for e in validate(bad, w.index)[0])


def test_rules_from_the_last_battery(patched):
    from mes.tools.rwd.designer import to_spec
    w = patched
    # a treatment and an outcome: the treatment comparison, not an association of a drug ever given
    spec, _m, _n = to_spec({"analysis": "association", "base": "icu", "outcome": "death_28d",
                            "exposure": {"term": "testdrug", "within_hours": 24}}, w.index,
                           "Does the use of testdrug relate to lower mortality?")
    assert spec.analysis == "compare" and spec.exposure.concept == "drug:testdrug"
    # a criterion naming the condition counted is no criterion of its rate (once: 100 % of diabetics had diabetes)
    spec, _m, notes = to_spec({"analysis": "rate", "base": "icu", "population": [{"term": "sepsis"}],
                               "exposure": {"term": "sepsis", "within_hours": 0}}, w.index,
                              "How many ICU patients have sepsis?")
    assert spec.include == [] and spec.exposure.concept == "var:sepsis3"
    # a trajectory's exposure that is its own measurement is no group
    spec, _m, _n = to_spec({"analysis": "trajectory", "base": "icu", "variable": "lactate_max",
                            "exposure": {"term": "Lactate (Blood, Blood Gas)", "op": ">", "value": "0"}}, w.index,
                           "What happens to lactate over the first 48 hours?")
    assert spec.exposure is None and spec.group_by == "" and spec.variable == "lactate_max"
    from mes.tools.rwd.designer import _label_words
    assert _label_words(w.index, "var:diabetes_with_cc") & _label_words(w.index, "phe:diabetes") == {"diabetes"}
    assert not _label_words(w.index, "var:sepsis3") & _label_words(w.index, "phe:diabetes")
