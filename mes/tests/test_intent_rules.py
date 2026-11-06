"""
Offline tests of the designer's intent rules (tools/rwd/intent_rules.py) on the intents a small local model wrote in
the field: a variable or an outcome named by itself, the two groups of a comparison written as criteria, generic
words, ICU types in words — and the explorer's checks of its own queries (the same query twice, a term the query
never mentions).

    python -m pytest mes/tests/test_intent_rules.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tools.rwd import explorer  # noqa: E402
from mes.tools.rwd import intent_rules as IR  # noqa: E402
from mes.tools.rwd.designer import asked_about  # noqa: E402


def _norm(intent, question, base="icu"):
    return IR.normalize(intent, question, base, asked_about)


def test_names_of_variables_outcomes_and_icu_types():
    assert [IR.exact_variable(t) for t in ("age", "Age group", "SOFA score", "hours_admit_to_icu", "sex", "ICU type",
                                           "lactate", "the weight", "Admission type", "sepsis", "drug")] == \
        ["age", "age", "sofa", "hours_admit_to_icu", "gender", "careunit", "lactate_max", "weight", "admission_type",
         None, None]
    assert IR.exact_outcome("icu_readmit_30d") == "icu_readmit_30d"
    assert IR.exact_outcome("Readmission within 30 days", "ICU patients readmitted to the ICU") == "icu_readmit_30d"
    assert IR.exact_outcome("Readmission within 30 days", "patients readmitted to hospital") == "hosp_readmit_30d"
    assert IR.exact_outcome("mechanical ventilation") == "invasive_vent_any" and IR.exact_outcome("death") is None
    assert [IR.careunit_level(t) for t in ("Medical ICU Admission", "surgical intensive care unit", "the CVICU",
                                           "cardiac surgery ICU", "coronary care unit", "trauma ICU", "MICU/SICU",
                                           "Neuromuscular blockers in ICU", "medical")] == \
        ["MICU", "SICU", "CVICU", "CVICU", "CCU", "TSICU", "MICU/SICU", None, None]
    assert IR.careunit_level("medical", bare=True) == "MICU"


def test_two_icu_types_compared_are_results_by_icu_type():
    intent, base, notes = _norm({"analysis": "compare", "base": "icu", "outcome": "los_icu_days",
                                 "exposure": {"term": "Medical ICU Admission", "op": "=", "value": "medical"},
                                 "comparator": {"term": "Surgical ICU Admission", "op": "=", "value": "surgical"}},
                                "Compare ICU length of stay between the medical and the surgical ICU.")
    assert (intent["analysis"], intent["variable"], intent["stratify"], intent["exposure"]) == \
        ("distribution", "los_icu_days", ["careunit"], None)
    intent, _b, _n = _norm({"analysis": "compare", "base": "icu", "outcome": "death_28d",
                            "exposure": {"term": "sex", "op": "=", "value": "female"},
                            "comparator": {"term": "sex", "op": "=", "value": "male"}},
                           "28-day mortality of women vs men in the ICU")
    assert (intent["analysis"], intent["stratify"]) == ("rate", ["gender"])


def test_a_variable_named_without_a_threshold_is_no_criterion():
    intent, _b, notes = _norm({"analysis": "rate", "base": "icu", "outcome": "death_icu", "stratify": [],
                               "exposure": {"term": "age", "op": "between", "value": "Age Group A and Age Group B"},
                               "comparator": {"term": "age", "op": "between", "value": "C and D"}},
                              "What percentage of ICU patients die in the ICU, by age group?")
    assert intent["exposure"] is None and intent["comparator"] is None and intent["stratify"] == ["age"]
    intent, _b, _n = _norm({"analysis": "distribution", "base": "icu", "variable": "",
                            "population": [{"term": "sepsis"}, {"term": "SOFA score", "op": "", "value": ""}],
                            "exposure": {"term": "age", "op": "", "value": ""}}, "median age in sepsis")
    assert intent["variable"] == "age" and [t["term"] for t in intent["population"]] == ["sepsis"]
    kept, _b, _n = _norm({"analysis": "rate", "base": "icu", "population": [{"term": "age", "op": ">=", "value": "80"}]},
                         "1-year mortality over 80")
    assert kept["population"] == [{"term": "age", "op": ">=", "value": "80"}]       # a threshold is a criterion


def test_a_variable_of_the_other_population_moves_the_base():
    intent, base, notes = _norm({"analysis": "distribution", "base": "hospital", "variable": "hours_admit_to_icu",
                                 "population": [{"term": "admission_type", "op": "!=", "value": ""}],
                                 "exposure": {"term": "hours_admit_to_icu", "op": "=", "value": ""}},
                                "Average time from hospital admission to ICU admission by admission type")
    assert base == "icu" and intent["population"] == [] and intent["stratify"] == ["admission_type"]


def test_an_outcome_named_as_the_exposure_is_the_outcome():
    intent, _b, notes = _norm({"analysis": "compare", "base": "icu", "outcome": "icu_readmit_30d",
                               "population": [{"term": "sex", "op": "=", "value": "female"},
                                              {"term": "sex", "op": "=", "value": "male"}],
                               "exclude": [{"term": "sex", "op": "=", "value": "female"}],
                               "exposure": {"term": "invasive_vent_any", "op": ">=", "value": "1"},
                               "comparator": {"term": "sex", "op": "=", "value": "male"}},
                              "Are women less likely than men to receive invasive mechanical ventilation in the ICU?")
    assert intent["outcome"] == "invasive_vent_any" and intent["population"] == [] and intent["exclude"] == []
    assert intent["exposure"]["term"] == "sex" and intent["exposure"]["value"] == "female"
    intent, _b, _n = _norm({"analysis": "compare", "base": "icu", "outcome": "",
                            "population": [{"term": "any_icu", "op": "=", "value": "1"}],
                            "exposure": {"term": "icu_readmit_30d", "op": "=", "value": "1"},
                            "comparator": {"term": "icu_readmit_30d", "op": "=", "value": "0"}},
                           "How do ICU patients readmitted within 30 days differ from those not readmitted?")
    assert (intent["analysis"], intent["outcome"], intent["population"]) == ("characterize", "icu_readmit_30d", [])
    # an outcome the question does ask about stays: readmission is then the exposure the model meant
    intent, _b, _n = _norm({"analysis": "association", "base": "icu", "outcome": "death_1y",
                            "exposure": {"term": "ICU readmission"}}, "Is ICU readmission associated with 1-year mortality?")
    assert intent["outcome"] == "death_1y" and intent["exposure"] == {"term": "ICU readmission"}


def test_generic_words_name_no_concept():
    intent, _b, notes = _norm({"analysis": "utilization", "base": "icu",
                               "population": [{"term": "lab tests"}, {"term": "sepsis"}],
                               "exposure": {"term": "drug name/type", "op": "", "value": ""}},
                              "Which drugs are most commonly given in the first 24 hours?")
    assert intent["exposure"] is None and [t["term"] for t in intent["population"]] == ["sepsis"]


def test_the_explorer_knows_its_check_was_no_check():
    assert explorer._same("SELECT 1 ", "select 1;") and not explorer._same("", "")
    assert explorer.missing_terms(["MIMIC-IV database patients", "Extracorporeal Membrane Oxygenation (ECMO)"],
                                  "SELECT count(*) FROM mes.icu_stays WHERE vasopressor_24h = 1") == \
        ["Extracorporeal Membrane Oxygenation (ECMO)"]
    assert explorer.missing_terms(["Positive blood culture"], "SELECT count(*) FROM mimiciv_hosp.microbiologyevents "
                                                              "WHERE spec_type_desc = 'BLOOD CULTURE'") == []


def test_an_outcome_asked_about_keeps_its_place():
    # "survival of ventilated vs non-ventilated": death is asked, ventilation is the exposure (its first-24-h flag)
    intent, _b, notes = _norm({"analysis": "survival", "base": "icu", "outcome": "death_1y",
                               "exposure": {"term": "invasive_vent_any", "op": "=", "value": "1"},
                               "comparator": {"term": "invasive_vent_any", "op": "=", "value": "0"}},
                              "Survival of ventilated vs non-ventilated patients with sepsis over one year")
    assert intent["outcome"] == "death_1y" and intent["analysis"] == "survival" and intent["comparator"] is None
    assert intent["exposure"]["term"] == "Invasive mechanical ventilation in the first 24 h"


def test_a_sex_named_as_a_population_is_its_criterion():
    intent, _b, _n = _norm({"analysis": "rate", "base": "icu", "outcome": "death_28d",
                            "population": [{"term": "women"}, {"term": "sepsis"}]}, "28-day mortality of women with sepsis")
    assert intent["population"] == [{"term": "sex", "op": "=", "value": "F"}, {"term": "sepsis"}]


def test_aliases_do_not_count_as_referring_to_a_term():
    sql = ("WITH icu_delirium AS (SELECT DISTINCT subject_id FROM mes.icu_stays WHERE sofa > 15) SELECT count(*) AS "
           "count_delirium FROM icu_delirium")
    assert explorer.missing_terms(["Delirium"], sql) == ["Delirium"]
    assert explorer.missing_terms(["Delirium"], "SELECT count(*) AS n FROM mes.diagnoses WHERE lower(title) LIKE "
                                                "'%delirium%'") == []


def test_an_icu_admission_type_valued_medical_is_the_icu_type():
    intent, _b, _n = _norm({"analysis": "compare", "base": "icu", "outcome": "los_icu_days",
                            "exposure": {"term": "ICU admission type", "op": "=", "value": "Medical"},
                            "comparator": {"term": "ICU admission type", "op": "=", "value": "Surgical"}},
                           "Compare ICU length of stay between the medical and the surgical ICU.")
    assert (intent["analysis"], intent["stratify"]) == ("distribution", ["careunit"])


def test_the_time_on_a_treatment_is_explored():
    from mes.tools.rwd.designer import TREATMENT_TIME
    asks = ["How many hours of mechanical ventilation do patients with ARDS receive?",
            "What is the duration of vasopressor therapy in septic shock?", "How long are patients on dialysis?",
            "length of mechanical ventilation in ARDS"]
    not_asks = ["How many patients receive mechanical ventilation?", "mortality within 28 days of ventilation",
                "ICU length of stay of ventilated patients", "How long do ventilated patients stay in the ICU?"]
    assert all(TREATMENT_TIME.search(q) for q in asks) and not any(TREATMENT_TIME.search(q) for q in not_asks)


def test_two_outcome_values_with_the_groups_in_the_criteria_are_an_association():
    intent, _b, _n = _norm({"analysis": "compare", "base": "icu", "outcome": "",
                            "population": [{"term": "sex", "op": "=", "value": "female"},
                                           {"term": "sex", "op": "=", "value": "male"}],
                            "exposure": {"term": "invasive_vent_any", "op": "=", "value": "true"},
                            "comparator": {"term": "invasive_vent_any", "op": "=", "value": "false"}},
                           "Are women less likely than men to receive invasive mechanical ventilation in the ICU?")
    assert intent["analysis"] == "compare" and intent["outcome"] == "invasive_vent_any"
    assert intent["exposure"]["term"] == "sex" and intent["exposure"]["value"] == "female"
