"""
Offline tests of trial vs MIMIC-IV (tools/rwd/trial_population.py, ui/trial_compare.py): a trial's baseline is read
from its total group, its characteristics are matched to MIMIC-IV's with units converted, the standardized
differences are those computed by hand, small groups are not shown, implausible units are refused, the study's
cohort is built again on a synthetic MIMIC-IV store, and the comparison's card. ClinicalTrials.gov is not called.

    python -m pytest mes/tests/test_trial_population.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools.rwd import concepts, db  # noqa: E402
from mes.tools.rwd import trial_population as TP  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402

TOTAL = "BG002"


def _measure(title, param, values, disp=None, unit="", classes=None):
    """A baseline measure: ``values`` [(category, value, spread, lo, hi)] of the total group (and a different arm)."""
    cats = [{"title": c, "measurements": [{"groupId": "BG000", "value": "1"},
                                          {"groupId": TOTAL, "value": v, "spread": s, "lowerLimit": lo, "upperLimit": hi}]}
            for c, v, s, lo, hi in values]
    return {"title": title, "paramType": param, "dispersionType": disp, "unitOfMeasure": unit,
            "classes": classes or [{"categories": [{k: x for k, x in c.items() if not (k == "title" and x is None)}
                                                   for c in cats]}]}


def trial_record(nct="NCT00000001"):
    """A ClinicalTrials.gov record (API v2 shape) with a posted baseline of 1,000 participants."""
    n = lambda v: [(None, v, None, None, None)]          # noqa: E731
    measures = [
        _measure("Age, Continuous", "MEAN", [(None, "60.0", "15.0", None, None)], "STANDARD_DEVIATION", "years"),
        _measure("Sex: Female, Male", "COUNT_OF_PARTICIPANTS", [("Female", "400", None, None, None),
                                                               ("Male", "600", None, None, None)], unit="Participants"),
        _measure("Race (NIH/OMB)", "COUNT_OF_PARTICIPANTS", [("White", "700", None, None, None),
                                                            ("Black or African American", "150", None, None, None),
                                                            ("Unknown or Not Reported", "150", None, None, None)]),
        _measure("Sequential Organ Failure Assessment (SOFA) score", "MEDIAN", [(None, "3", None, "2", "5")],
                 "INTER_QUARTILE_RANGE", "units on a scale"),
        _measure("Lactate", "MEAN", [(None, "27.0", "9.0", None, None)], "STANDARD_DEVIATION", "mg/dL"),
        _measure("Serum creatinine", "MEAN", [(None, "150", "80", None, None)], "STANDARD_DEVIATION", "µmol/L"),
        _measure("Temperature", "MEAN", [(None, "100.4", "1.8", None, None)], "STANDARD_DEVIATION", "°F"),
        _measure("APACHE II score", "MEAN", [(None, "20", "7", None, None)], "STANDARD_DEVIATION", "units on a scale"),
        _measure("Weight", "MEAN", [(None, "12", "2", None, None)], "STANDARD_DEVIATION", "stone"),
        _measure("Coexisting Conditions", "COUNT_OF_PARTICIPANTS", [], classes=[
            {"title": "Diabetes", "categories": [{"title": "Yes", "measurements": [{"groupId": TOTAL, "value": "300"}]},
                                                 {"title": "No", "measurements": [{"groupId": TOTAL, "value": "700"}]}]}]),
        _measure("Age, Categorical", "COUNT_OF_PARTICIPANTS", [("<=18 years", "0", None, None, None),
                                                              ("Between 18 and 65 years", "550", None, None, None),
                                                              (">=65 years", "450", None, None, None)]),
        _measure("Region of Enrollment", "NUMBER", [], classes=[
            {"title": "United States", "categories": [{"measurements": [{"groupId": TOTAL, "value": "1000"}]}]}]),
        _measure("Septic shock", "COUNT_OF_PARTICIPANTS", n("4"), unit="Participants"),
    ]
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": "A test trial"}},
            "resultsSection": {"baselineCharacteristicsModule": {
                "groups": [{"id": "BG000", "title": "Drug"}, {"id": "BG001", "title": "Placebo"},
                           {"id": TOTAL, "title": "Total"}],
                "denoms": [{"units": "Participants", "counts": [{"groupId": "BG000", "value": "500"},
                                                               {"groupId": "BG001", "value": "500"},
                                                               {"groupId": TOTAL, "value": "1000"}]}],
                "measures": measures}}}


# ============================================================================ reading and matching
def test_a_trial_baseline_is_read_from_its_total_group():
    trial = TP.parse(trial_record())
    assert trial["nct"] == "NCT00000001" and trial["n"] == 1000
    age = next(e for e in trial["entries"] if e["title"] == "Age, Continuous")
    assert (age["value"], age["spread"], age["n"]) == (60.0, 15.0, 1000)       # the total, not an arm
    assert TP.parse({"protocolSection": {}, "resultsSection": {}})["entries"] == []


def test_characteristics_are_matched_with_units_converted():
    rows, other = TP.match(TP.parse(trial_record())["entries"], "icu")
    by = {r["key"]: r for r in rows}
    assert {"age", "female", "race_White", "race_Black", "sofa", "lactate_max", "creatinine_max", "temperature_max",
            "diabetes_without_cc+diabetes_with_cc", "age65", "septic_shock"} <= set(by)
    assert by["female"]["trial"]["p"] == pytest.approx(0.4) and by["age65"]["trial"]["p"] == pytest.approx(0.45)
    assert by["lactate_max"]["trial"]["mean"] == pytest.approx(27 / 9.008)                 # mg/dL → mmol/L
    assert by["creatinine_max"]["trial"]["mean"] == pytest.approx(150 / 88.4)              # µmol/L → mg/dL
    assert by["temperature_max"]["trial"] == pytest.approx({"mean": 38.0, "sd": 1.0})      # °F → °C
    assert by["sofa"]["trial"] == {"median": 3.0, "q1": 2.0, "q3": 5.0}
    why = {o["characteristic"]: o["why"] for o in other}
    assert "analysis variables" in why["APACHE II score"] and "unit" in why["Weight"]
    assert "analysis variables" in why["Region of Enrollment"] and "Race (NIH/OMB)" not in why
    rows, other = TP.match(TP.parse(trial_record())["entries"], "hospital")
    assert "sofa" not in {r["key"] for r in rows}
    assert any(o["characteristic"].startswith("Sequential") and "admissions" in o["why"] for o in other)


# ============================================================================ the comparison
def _cohort(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"age": rng.normal(66, 14, n), "gender": rng.choice(["F", "M"], n, p=[0.45, 0.55]),
                         "race_group": rng.choice(["White", "Black", "Unknown"], n, p=[0.6, 0.2, 0.2]),
                         "sofa": rng.poisson(5, n).astype(float), "lactate_max": rng.gamma(2, 1.4, n),
                         "temperature_max": rng.normal(37.8, 0.8, n),
                         "diabetes_without_cc": rng.random(n) < 0.2, "diabetes_with_cc": rng.random(n) < 0.1,
                         "septic_shock": np.r_[np.ones(5), np.zeros(n - 5)]})


def test_standardized_differences_are_those_computed_by_hand():
    df = _cohort()
    rows, other = TP.compare(TP.parse(trial_record()), df, "icu")
    by = {r["key"]: r for r in rows}
    m, s = df.age.mean(), df.age.std()
    assert by["age"]["smd"] == pytest.approx((m - 60) / np.sqrt((s ** 2 + 15 ** 2) / 2))
    p = (df.gender == "F").mean()
    assert by["female"]["smd"] == pytest.approx((p - 0.4) / np.sqrt((p * (1 - p) + 0.4 * 0.6) / 2))
    assert by["female"]["mimic"] == f"{100 * p:.1f}%" and by["female"]["trial"] == "40.0%"
    q1, med, q3 = df.sofa.quantile([0.25, 0.5, 0.75])
    assert by["sofa"]["approx"] and by["sofa"]["smd"] == pytest.approx(
        (med - 3) / np.sqrt((((q3 - q1) / 1.349) ** 2 + (3 / 1.349) ** 2) / 2))
    either = (df.diabetes_without_cc | df.diabetes_with_cc).mean()
    assert by["diabetes_without_cc+diabetes_with_cc"]["mimic"] == f"{100 * either:.1f}%"
    # five patients: not shown, and no difference that would tell their number
    assert by["septic_shock"]["mimic"] == "<10 patients" and by["septic_shock"]["smd"] is None
    assert any(o["characteristic"].startswith("Serum creatinine") and "not in this cohort" in o["why"] for o in other)
    res = {"nct": "NCT00000001", "n_trial": 1000, "n_cohort": len(df), "rows": rows}
    text = TP.summary(res)
    assert text.startswith("Compared with the 1,000 participants of NCT00000001, the 400 patients")
    assert "Age higher" in text and "years)" in text


def test_a_value_in_other_units_is_not_compared():
    record = trial_record()
    lactate = next(m for m in record["resultsSection"]["baselineCharacteristicsModule"]["measures"]
                   if m["title"] == "Lactate")
    lactate["unitOfMeasure"] = "mmol/L"                     # 27 "mmol/L": mg/dL in truth
    rows, other = TP.compare(TP.parse(record), _cohort(), "icu")
    assert "lactate_max" not in {r["key"] for r in rows}
    assert any("outside MIMIC-IV's range" in o["why"] for o in other)


# ============================================================================ on a synthetic store
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_trial")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, _treated, _early = _synthetic_store(root)
    store = Store(root)
    return store, ConceptIndex(store), icu


@pytest.fixture
def patched(synth, monkeypatch):
    store, index, icu = synth
    monkeypatch.setattr(db, "get_store", lambda: store)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    calls = []

    def get_json(url, params=None, **_kw):
        calls.append(url)
        nct = url.rsplit("/", 1)[-1]
        if nct == "NCT00000009":
            raise ConnectionError("offline")
        return trial_record(nct) if nct != "NCT00000002" else {"protocolSection": {"identificationModule": {
            "nctId": nct, "briefTitle": "No results"}}}
    monkeypatch.setattr(TP, "get_json", get_json)
    TP.clear()
    yield icu, calls
    TP.clear()


def test_the_study_cohort_is_built_again_beside_the_trial(patched):
    icu, calls = patched
    spec = {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}], "outcome": "death_28d"}
    res = TP.run(spec, "nct00000001")
    sub = icu[icu.sepsis3]
    assert not res.get("error") and res["nct"] == "NCT00000001" and res["n_cohort"] == len(sub)
    age = next(r for r in res["rows"] if r["key"] == "age")
    m, s = sub.age.mean(), sub.age.std()
    assert age["smd"] == pytest.approx((m - 60) / np.sqrt((s ** 2 + 15 ** 2) / 2))
    assert {r["key"] for r in res["rows"]} >= {"age", "female", "sofa", "lactate_max", "age65"}
    # what the synthetic store does not have is listed, not guessed
    assert any(o["characteristic"].startswith("Race") for o in res["other"])
    TP.run(spec, "NCT00000001")
    assert len(calls) == 1                                   # the trial is read once
    assert "no baseline" in TP.run(spec, "NCT00000002")["error"]
    assert "could not be reached" in TP.run(spec, "NCT00000009")["error"]
    assert "NCT and 8 digits" in TP.run(spec, "NCT123")["error"]
    bad = TP.run({"analysis": "rate", "base": "icu", "include": [{"concept": "drug:nosuch"}]}, "NCT00000001")
    assert "could not be built" in bad["error"]


def test_the_card_of_a_comparison(patched):
    from mes.ui import trial_compare
    from mes.ui.viz import clean
    res = TP.run({"analysis": "rate", "base": "icu", "outcome": "death_28d"}, "NCT00000001")
    card = trial_compare.card(res)
    json.dumps(clean(card))
    assert [t["id"] for t in card["tabs"]] == ["results", "other"]
    blocks = card["tabs"][0]["blocks"]
    forest = next(b for b in blocks if b.get("type") == "chart")["chart"]["rows"]
    assert [abs(r["value"]) for r in forest] == sorted((abs(r["value"]) for r in forest), reverse=True)
    table = next(b for b in blocks if b.get("id") == "trial-table")
    assert table["smd"] and len(table["rows"]) == len(res["rows"])
    assert card["kpis"][0]["value"] == "1,000" and card["kpis"][1]["value"] == f"{res['n_cohort']:,}"
