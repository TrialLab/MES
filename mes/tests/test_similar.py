"""
Offline tests of "patients like this": the distance and the ranking (weights, missing values, must-match), the study on
a synthetic MIMIC-IV store (the similar patients, their outcome beside everyone's and by treatment, the patient's
predicted risk, aggregates only), the patient read from a question (English and Chinese, negations), the designer's
routing, the card and its report, the live preview, and the patient explorer on the page (AppTest).

    python -m pytest mes/tests/test_similar.py -q
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tests.test_cohort_builder import _page, store, synth  # noqa: E402,F401  (fixtures)
from mes.tools.rwd import analyses, report  # noqa: E402
from mes.tools.rwd import patient as PAT  # noqa: E402
from mes.tools.rwd import similar as SIM  # noqa: E402
from mes.tools.rwd.compiler import Compiler  # noqa: E402
from mes.tools.rwd.spec import StudySpec, validate  # noqa: E402

PROFILE = [{"concept": "var:age", "value": 70}, {"concept": "var:gender", "value": "M", "weight": 0.5},
           {"concept": "var:sofa", "value": 8}, {"concept": "var:lactate_max", "value": 4.5}]
DESIGN = {"analysis": "similar", "base": "icu", "include": [{"concept": "var:sepsis3"}], "outcome": "death_28d", "k": 200,
          "profile": PROFILE, "exposure": {"concept": "drug:testdrug", "window_h": [0, 24]},
          "covariates": ["age", "sofa", "sapsii"]}                  # (the synthetic store has no default covariate set)


# ============================================================================ the distance
def _frame():
    return pd.DataFrame({"age": [70, 72, 40, np.nan, 70], "gender": ["M", "F", "M", "M", None],
                         "sepsis3": [True, False, True, None, True], "sofa": [8, 8, 2, 8, 8]})


def test_the_distance_weighs_numbers_flags_categories_and_missing_values():
    feats = SIM.features([{"concept": "var:age", "value": 70}, {"concept": "var:gender", "value": "M"},
                          {"concept": "var:sepsis3", "value": 1}, {"concept": "var:sofa", "value": 8, "weight": 2}], "icu")
    df = _frame()
    sc = {"age": 10.0, "sofa": 2.0}
    d = SIM.distance(df, feats, sc)
    assert d[0] == 0                                                 # the patient's own values
    assert d[4] == pytest.approx(np.sqrt(0.5 ** 2 / 5))              # missing sex counts half (weights 1, 1, 1, 2)
    assert d[1] == pytest.approx(np.sqrt((0.2 / 3) ** 2 / 5 + 1 / 5 + 1 / 5))
    assert d[2] > d[1] > d[0]
    assert d[3] == pytest.approx(np.sqrt(0.5 ** 2 * 2 / 5))          # missing age and Sepsis-3
    assert list(SIM.nearest(d, 3)) == [0, 4, 3]
    assert list(SIM.nearest(np.zeros(4), 2)) == [0, 1]               # ties in the order of the stays
    must = SIM.features([{"concept": "var:gender", "value": "M", "must": True},
                         {"concept": "var:age", "value": 70, "must": True, "tolerance": 1}], "icu")
    assert SIM.must_mask(df, must, sc).tolist() == [True, False, False, False, False]


def test_a_profile_keeps_only_characteristics_of_the_data():
    feats = SIM.features([{"concept": "var:age", "value": "70"}, {"concept": "var:careunit", "value": "MICU"},
                          {"concept": "var:nothing", "value": 1}, {"concept": "drug:testdrug", "value": 1},
                          {"concept": "var:sofa", "value": "high"}, "garbage", None, {"concept": "var:age", "value": 71}],
                         "icu")
    assert [(f["name"], f["value"]) for f in feats] == [("age", 71.0), ("careunit", "MICU")]
    assert SIM.features(PROFILE, "hospital", columns={"age", "gender"})[0]["name"] == "age"
    assert SIM.numbers([2.4, 2.0, 2.6]) == ["2.4", "2.0", "2.6"] and SIM.numbers([10, 9, 12]) == ["10", "9", "12"]


# ============================================================================ the study
def _study(synth, design=DESIGN):
    s, index, icu = synth
    spec = StudySpec.from_dict(design)
    errors, _ = validate(spec, index)
    assert not errors, errors
    cohort = Compiler(index, s).build(spec)
    return spec, cohort, analyses.run(spec, cohort, "osler")


def test_patients_like_this_on_the_synthetic_store(store, synth):
    spec, cohort, res = _study(synth)
    icu = synth[2]
    pool = icu[icu.sepsis3].reset_index(drop=True)
    assert res["n"] == 200 and res["attrition"][-1] == ("the 200 stays most similar to the patient", 200)
    near_rate, all_rate = [e for e in res["estimates"] if e["measure"] == "rate"][:2]
    assert near_rate["primary"] and near_rate["n"] == 200 and all_rate["n"] == len(pool)
    assert all_rate["events"] == int(pool.death_28d.sum())
    # the similar patients are near the patient on every characteristic
    profile = {r["variable"]: r for r in res["tables"]["profile"]}
    assert abs(profile["age"]["median"] - 70) < 6 and abs(profile["sofa"]["median"] - 8) <= 1
    assert profile["gender"]["share_similar"] > profile["gender"]["share_cohort"]
    # by treatment: the two groups are the similar patients
    groups = res["tables"]["by_treatment"]
    assert sum(g["n"] for g in groups) == 200 and {g["role"] for g in groups} == {"treated", "control"}
    assert any(e["measure"] == "RD" for e in res["estimates"]) and any("crude" in w for w in res["warnings"])
    risk = next(e for e in res["estimates"] if e["measure"] == "risk")
    assert 0 < risk["lo"] <= risk["value"] <= risk["hi"] < 100 and 0.5 <= res["diagnostics"]["risk_auc"] <= 1
    assert res["table1"]["groups"] == ["similar patients", "other patients"]
    m = res["charts"]["map"]
    assert m["hexes"] and all(h["n"] >= 10 and (h["k"] is None or h["k"] >= 10) for h in m["hexes"])
    assert m["ellipse"]["rx"] > 0 and len(m["axes"]) == 2
    json.dumps(res, default=str)
    # what the language model reads: aggregates only — no stay, admission or patient id
    text = " ".join(p.text for p in report.to_items(spec, res, "k")[0].passages)
    assert "most similar" in text and "The described patient: Age 70" in text
    ids = set(map(str, pool.stay_id.head(400))) | set(map(str, pool.hadm_id.head(400)))
    assert not any(f" {i}" in text for i in ids)


def test_small_groups_and_too_few_patients(store, synth):
    _spec, _cohort, res = _study(synth, {**DESIGN, "k": 30, "include": [{"concept": "var:sepsis3"},
                                                                         {"concept": "var:sofa", "op": ">=", "value": 12}]})
    for e in res["estimates"]:
        if e.get("n") is not None and e["n"] < 10:
            assert e["value"] is None and e.get("small_cell")
        if e.get("events") is not None:
            assert e["events"] >= 10
    _spec, _cohort, res = _study(synth, {**DESIGN, "include": [{"concept": "var:sofa", "op": ">=", "value": 40}]})
    assert res["n"] < 30 and not res["estimates"] and "at least 30" in res["warnings"][0]


def test_a_must_match_characteristic_filters(store, synth):
    must = {**DESIGN, "profile": PROFILE + [{"concept": "var:careunit", "value": "MICU", "must": True}]}
    _spec, cohort, res = _study(synth, must)
    assert "matching the patient on ICU type" in res["attrition"][-2][0]
    assert res["attrition"][-2][1] == int((cohort.df.careunit == "MICU").sum())


# ============================================================================ the patient in a question
def test_the_patient_is_read_from_the_question():
    q = ("A 68-year-old man with septic shock, lactate 6.2 mmol/L, creatinine 2.5, SOFA 10, intubated, no CKD, on "
         "norepinephrine: how did similar patients do with vasopressin?")
    assert PAT.cue(q)
    got = {f["concept"][4:]: f["value"] for f in PAT.parse(q, "icu", leave_out={"vasopressin_24h"})}
    assert got == {"age": 68.0, "gender": "M", "sofa": 10.0, "lactate_max": 6.2, "creatinine_max": 2.5,
                   "septic_shock": 1, "invasive_vent_24h": 1, "renal_disease": 0, "norepinephrine_24h": 1}
    zh = {f["concept"][4:]: f["value"] for f in PAT.parse("一位68岁男性感染性休克患者，乳酸6，肌酐2.5，血钠128，相似患者怎样？", "icu")}
    assert zh == {"age": 68.0, "gender": "M", "lactate_max": 6.0, "creatinine_max": 2.5, "sodium_min": 128.0,
                  "septic_shock": 1}
    assert PAT.cue("相似患者的28天死亡率") and PAT.cue("my patient has sepsis") and PAT.cue("patients like mine")
    for q in ("In 70-year-old patients with sepsis, is hydrocortisone better?", "Does vasopressin reduce mortality?",
              "28-day mortality of septic shock"):
        assert not PAT.cue(q), q
    assert PAT.short(PAT.parse("a 45 yo woman, MAP 58, heart rate 120, no dialysis", "icu")) == \
        "45-year-old woman · heart rate 120 · MAP 58 · not on RRT"


def test_the_designer_routes_a_described_patient(store, synth):
    from mes.tools.rwd.designer import to_spec
    index = synth[1]
    q = "A 70-year-old man with sepsis, SOFA 8, lactate 4.5: how did similar patients do with testdrug?"
    intent = {"analysis": "compare", "base": "icu", "outcome": "death_28d",
              "population": [{"term": "sepsis", "op": "", "value": ""}],
              "exposure": {"term": "testdrug", "op": "", "value": "", "within_hours": 24},
              "comparator": {"term": "", "op": "", "value": ""}}
    spec, _mapping, notes = to_spec(intent, index, q)
    assert spec.analysis == "similar" and "one patient" in " ".join(notes)
    assert {f["concept"] for f in spec.profile} == {"var:age", "var:gender", "var:sofa", "var:lactate_max"}
    assert [c.concept for c in spec.include] == ["var:sepsis3"]          # the condition: who is like the patient
    assert spec.exposure is not None and spec.exposure.concept.startswith("drug")
    assert spec.title.startswith("Patients like this: 70-year-old man")
    assert not validate(spec, index)[0]
    # a small model's slips: the patient's own sex excluded, a comorbidity required, a characteristic as the treatment
    q2 = "My patient is an 80-year-old woman with heart failure, sodium 128: what happened to similar patients?"
    slips = {"analysis": "prognostic", "base": "icu", "outcome": "death_hosp",
             "population": [{"term": "heart failure", "op": "", "value": ""}],
             "exclude": [{"term": "gender", "op": "=", "value": "female"}],
             "exposure": {"term": "sodium", "op": "<=", "value": "128", "within_hours": 0},
             "comparator": {"term": "age", "op": ">=", "value": "80"}}
    spec, _m, notes = to_spec(slips, index, q2)
    assert spec.analysis == "similar" and spec.exclude == [] and spec.include == []
    assert spec.exposure is None and spec.comparator is None
    assert {f["concept"] for f in spec.profile} >= {"var:age", "var:gender", "var:congestive_heart_failure",
                                                    "var:sodium_min"}
    assert any("own characteristics" in n for n in notes) and not validate(spec, index)[0]
    # "similar" without a described patient is not a patients-like-this study
    spec, _m, notes = to_spec({**intent, "analysis": "similar"}, index, "Does testdrug lower 28-day mortality?")
    assert spec.analysis != "similar"


# ============================================================================ the card, the report, the preview
def _record(synth, design=DESIGN):
    from mes.tools.rwd_tools import _plain
    spec, cohort, res = _study(synth, design)
    return {"key": "sim1", "kind": "template", "status": "done", "evidence_id": "R1", "spec": spec.to_dict(),
            "result": _plain(res), "sql": cohort.sql, "timings": {"total": 1.0}, "dataset": "MIMIC-IV v2.2"}


def test_the_card_and_the_report_of_patients_like_this(store, synth):
    from mes.tests.test_viz import _result
    from mes.ui import study_card, study_report
    from mes.ui.viz import clean
    study = _record(synth)
    card = study_card.card(_result([study]), study)
    assert [t["id"] for t in card["tabs"] if t["blocks"]][:3] == ["results", "profile", "map"]
    assert card["title"].startswith("Patients like this: 70-year-old man") and card["design"] == "Patients like this"
    assert [k["label"] for k in card["kpis"]][:2] == ["Patients like this", "28-day death · patients like this"]
    charts = {b["chart"]["type"] for t in card["tabs"] for b in t["blocks"] if b.get("type") == "chart"}
    assert {"dots", "received", "profile", "simmap", "flow"} <= charts
    json.dumps(clean(card))
    html_ = study_report.report_html(_result([study]), study)
    assert html_.count("<svg") >= 3 and "Patients like this" in html_ and "This patient beside" in html_
    damaged = json.loads(json.dumps(study))
    damaged["result"]["tables"] = {"profile": "x", "treatments": [None, 3]}
    damaged["result"]["charts"] = {"map": {"hexes": "bad"}}
    assert study_card.card(_result([damaged]), damaged) is not None


def test_the_live_preview_ranks_again_in_milliseconds(store, synth):
    res, errors = SIM.preview(DESIGN)
    assert not errors and res["n"] == 200 and "map" not in (res.get("charts") or {})
    t = time.time()
    older, _ = SIM.preview({**DESIGN, "profile": [{**PROFILE[0], "value": 85}] + PROFILE[1:]})
    assert time.time() - t < 1.0                                        # the cohort is built once
    age = lambda r: next(x for x in r["tables"]["profile"] if x["variable"] == "age")["median"]
    assert age(older) > age(res)
    _res, errors = SIM.preview({**DESIGN, "profile": []})
    assert errors and "at least one characteristic" in errors[0]


# ============================================================================ the patient explorer (AppTest)
def test_the_patient_explorer_on_the_page(tmp_path, monkeypatch, store, synth):
    from mes.tests.test_protocol import _base
    study = _record(synth)
    at, turn = _page(tmp_path, monkeypatch, _base([study]))
    key = f"mes_px_{turn['id']}_sim1"
    at.button(key=f"{key}_toggle").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.number_input(key=f"{key}_g0_p0_value").set_value(85.0)          # an older patient
    at.run()
    at.selectbox(key=f"{key}_new").set_value("sapsii")
    at.run()
    at.button(key=f"{key}_addgo").click()
    at.run()
    assert not at.exception and len(at.session_state[f"{key}_state"]["rows"]) == 5
    assert at.button(key=f"{key}_run").disabled                         # locked
    at.toggle(key=f"{key}_lock").set_value(True)
    at.run()
    at.button(key=f"{key}_run").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    thread = at.session_state["mes_thread"]
    assert len(thread) == 2 and thread[1]["version"] == 2 and thread[1]["status"] == "done"
    profile = {f["concept"]: f["value"] for f in thread[1]["edit"]["spec"]["profile"]}
    assert profile["var:age"] == 85.0 and "var:sapsii" in profile
    # the cohort builder is offered beside it, and other studies get no patient explorer
    assert any(b.key == f"mes_cb_{turn['id']}_sim1_toggle" for b in at.button)
    assert not any(b.key and b.key.startswith("mes_px_") and "_sim1" not in b.key for b in at.button)
