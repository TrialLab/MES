"""
Offline tests of the cohort builder: the live counts are the analysis's own (the compiler's attrition, step by step,
cached), the builder's working design and its flow beside the study's, and the builder on the page (AppTest with the
fakes of scripts/ui_monkey.py on a synthetic MIMIC-IV store): open it, switch a criterion off, change a threshold,
add and remove criteria, unlock and run the edited design as a new version. Also the trial comparison under the same
card (a trial of the answer, or a typed NCT number, beside the study's cohort).

    python -m pytest mes/tests/test_cohort_builder.py -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes import config  # noqa: E402
from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools import protocol_tools as PT  # noqa: E402
from mes.tools import rwd_tools  # noqa: E402
from mes.tools.result_schema import MESResult  # noqa: E402
from mes.tools.rwd import analyst, concepts, counts, db  # noqa: E402
from mes.tools.rwd.compiler import Compiler  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402
from mes.tools.rwd.spec import StudySpec, validate  # noqa: E402
from mes.ui import cohort_state as CS  # noqa: E402

COMPARE = {"analysis": "compare", "base": "icu", "include": [{"concept": "var:sepsis3"}],
           "exclude": [{"concept": "var:lactate_max", "op": ">", "value": 6}],
           "exposure": {"concept": "drug:testdrug"}, "outcome": "death_1y", "grace_h": 24, "covariates": ["sofa"]}
DESIGNS = [COMPARE,
           {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sofa", "op": ">=", "value": 5},
                                                           {"concept": "var:gender", "op": "=", "value": "F"}],
            "outcome": "death_28d"},
           {"analysis": "association", "base": "icu", "exposure": {"concept": "var:lactate_max", "op": ">", "value": 4},
            "outcome": "death_28d", "covariates": ["sofa"]},
           {"analysis": "rate", "base": "icu", "include": [{"concept": "drug:testdrug", "window_h": [0, None]}],
            "exclude": [{"concept": "var:sepsis3", "negate": True}], "outcome": "death_hosp"},
           {"analysis": "compare", "base": "icu", "exposure": {"concept": "drug:testdrug"}, "outcome": "death_28d",
            "grace_h": 6, "new_user": False, "covariates": ["sofa"]}]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_builder")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, treated, early = _synthetic_store(root)
    store = Store(root)
    return store, ConceptIndex(store), icu


@pytest.fixture
def store(synth, monkeypatch):
    s, index, icu = synth
    monkeypatch.setattr(rwd_tools, "get_store", lambda: s)
    monkeypatch.setattr(db, "get_store", lambda: s)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (False, "test"))
    counts.clear()
    return s, index, icu


# ============================================================================ the counts
@pytest.mark.parametrize("design", DESIGNS, ids=lambda d: d["analysis"])
def test_the_live_counts_are_the_analysis_own(store, design):
    s, index, icu = store
    spec = StudySpec.from_dict(design)
    assert not validate(spec, index)[0]
    cohort = Compiler(index, s).build(spec)
    flow = counts.flow(design)
    assert not flow["errors"], flow["errors"]
    assert [(x["label"], x["n"]) for x in flow["steps"]] == [(a, int(n)) for a, n in cohort.attrition]
    assert [x["id"] for x in flow["steps"]] == cohort.step_ids and flow["n"] == len(cohort.df)
    if design["analysis"] == "compare":
        assert flow["groups"] == {"treated": int(cohort.df.treated.sum()), "control": int((1 - cohort.df.treated).sum())}
        assert cohort.step_ids[-1] == "groups"
    again = counts.flow(json.loads(json.dumps(design)))
    assert again.get("cached") and again["steps"] == flow["steps"]


def test_counts_of_a_broken_design_say_why(store):
    for design, words in (({"analysis": "rate", "base": "icu", "include": [{"concept": "drug:nosuchdrug"}],
                            "outcome": "death_28d"}, "unknown concept"),
                          ({"analysis": "rate", "base": "icu", "include": [{"concept": "drug:testdrug",
                                                                           "window_h": [24, 0]}],
                            "outcome": "death_28d"}, "window"),
                          ({"analysis": "rate", "base": "nowhere"}, "unknown population"),
                          ({"analysis": "rate", "include": "nonsense", "bogus": 1}, "")):
        flow = counts.flow(design)
        assert flow["errors"] and not flow["steps"] and words in " ".join(flow["errors"])


def test_small_counts_are_not_shown():
    assert [counts.shown(n) for n in (0, 1, 9, 10, 2500, None)] == [0, None, None, 10, 2500, None]


# ============================================================================ the working design
def test_rows_make_the_study_design_again(store):
    spec = StudySpec.from_dict(COMPARE)
    validate(spec, store[1])
    spec = spec.to_dict()
    rows = CS.rows_of(spec)
    assert [(r["origin"], r["role"]) for r in rows] == [("inc_0", "include"), ("exc_0", "exclude"),
                                                      ("exposure", "exposure")]
    assert not PT.design_changes(spec, CS.draft_of(spec, rows, spec["grace_h"], spec["new_user"]))


def test_the_flow_beside_the_study_marks_what_changed(store):
    spec = StudySpec.from_dict(COMPARE)
    validate(spec, store[1])
    spec = spec.to_dict()
    originals = {r["origin"]: r for r in CS.rows_of(spec)}
    rows = CS.rows_of(spec)
    rows[0]["on"] = False                                          # Sepsis-3 switched off
    rows[1]["value"] = 4.0                                         # lactate > 6 → > 4
    rows.append(CS.new_row("var:sofa", "SOFA score, first 24 h", "include", "n1", typical=4.0))
    draft = CS.draft_of(spec, rows, 24, True)
    study, now = counts.flow(spec), counts.flow(draft)
    steps, groups = CS.beside(study, now, rows, originals)
    by_key = {s["key"]: s for s in steps}
    assert by_key["inc_0"]["state"] == "off" and by_key["inc_0"]["n"] is None and by_key["inc_0"]["was"] is not None
    assert by_key["n1"]["state"] == "new" and by_key["n1"]["was"] is None and by_key["n1"]["n"] > 0
    assert by_key["exc_0"]["state"] == "changed" and by_key["exc_0"]["n"] != by_key["exc_0"]["was"]
    assert by_key["base"]["n"] == by_key["base"]["was"] and [s["key"] for s in steps][:1] == ["base"]
    assert groups["treated"]["was"] == study["groups"]["treated"] and groups["treated"]["n"] == now["groups"]["treated"]
    assert {c["what"] for c in PT.design_changes(spec, draft)} == {"Inclusion", "Exclusion"}
    # a step of the comparison the draft no longer has (no comparator here: none) and new users switched off
    steps, _ = CS.beside(study, counts.flow(CS.draft_of(spec, rows, 24, False)), rows, originals)
    assert next(s for s in steps if s["key"] == "new_users")["state"] == "off"


def test_a_new_criterion_starts_with_a_sensible_comparison(store):
    row = CS.new_row("var:lactate_max", "Highest lactate", "include", "n1", typical=counts.typical("icu", "lactate_max"))
    assert row["op"] == ">=" and row["value"] == pytest.approx(float(np.nanmedian(store[2].lactate_max)))
    drug = CS.new_row("drug:testdrug", "Testdrug", "exclude", "n2")
    assert drug["window_h"] == [-72, 24] and drug["op"] == "" and drug["value"] is None
    assert counts.typical("icu", "gender") is None and counts.typical("icu", "no_such_variable") is None
    assert CS.text({**drug, "negate": True}) == "without Testdrug"


# ============================================================================ the page (AppTest)
def _result(synth):
    s, index, _icu = synth
    spec = StudySpec.from_dict(COMPARE)
    validate(spec, index)
    from mes.tools.rwd import analyses
    res = analyses.run(spec, Compiler(index, s).build(spec), "sackett")
    study = {"key": "cb1", "kind": "template", "status": "done", "evidence_id": "R1", "spec": spec.to_dict(),
             "result": json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))),
             "sql": "SELECT 1", "timings": {"total": 1.0}, "dataset": "MIMIC-IV v2.2"}
    from mes.tests.test_protocol import _base
    return _base([study])


def _page(tmp_path, monkeypatch, result):
    from streamlit.testing.v1 import AppTest
    from mes.tools.rwd import trial_population as TP
    from mes.ui import page, team_builder, warmup
    (tmp_path / "fixture.json").write_text(json.dumps(result.to_dict(), default=str))
    for name, value in (("MES_MONKEY_FIXTURES", tmp_path), ("MES_MONKEY_CACHE", tmp_path / "cache"),
                        ("MES_HISTORY_DIR", tmp_path / "history"), ("MES_MONKEY_BEHAVIOUR", "ok")):
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(config, "PROTOCOL_EDIT", "on")
    for module, name in [(page, n) for n in ("iter_answer", "iter_rerun", "create_llm", "composer", "_llm_health",
                                             "kg_status")] + [(team_builder, "canvas"), (warmup, "start"),
                                                              (warmup, "ready"), (rwd_tools, "edited_study"),
                                                              (TP, "get_json")]:
        monkeypatch.setattr(module, name, getattr(module, name))     # ui_monkey.install() replaces them: undone after
    turn = page._new_turn(result.question, result.mode, result.sources, False, True, [])
    turn.update(status="done", result=result, shown=True)
    at = AppTest.from_file(str(Path(page.__file__).parents[1] / "scripts" / "ui_monkey_app.py"), default_timeout=90)
    at.session_state["mes_no_kick"] = True
    at.session_state["mes_thread"] = [turn]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at, turn


def _counted(at):
    return [c.value for c in at.caption if c.value.startswith("Counted on")]


def test_the_builder_on_the_page(tmp_path, monkeypatch, store, synth):
    at, turn = _page(tmp_path, monkeypatch, _result(synth))
    key = f"mes_cb_{turn['id']}_cb1"
    assert not _counted(at)                                        # closed until asked
    at.button(key=f"{key}_toggle").click()
    at.run()
    assert not at.exception and _counted(at), [e.value for e in at.exception]
    assert at.button(key=f"{key}_run").disabled                    # nothing changed, and locked
    at.toggle(key=f"{key}_g0_inc_0_on").set_value(False)          # Sepsis-3 off
    at.run()
    at.number_input(key=f"{key}_g0_exc_0_value").set_value(3.0)   # lactate > 6 → > 3
    at.run()
    assert not at.exception and _counted(at)
    changes = " ".join(m.value for m in at.markdown if "mes-cb-changes" in m.value)
    assert "Inclusion" in changes and "Exclusion" in changes
    assert at.button(key=f"{key}_run").disabled                    # still locked
    at.toggle(key=f"{key}_lock").set_value(True)
    at.run()
    assert not at.button(key=f"{key}_run").disabled
    at.button(key=f"{key}_run").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    thread = at.session_state["mes_thread"]
    assert len(thread) == 2 and thread[1]["version"] == 2 and thread[1]["status"] == "done"
    edit = thread[1]["edit"]
    assert edit["kind"] == "design" and edit["study"] == "cb1"
    assert edit["spec"]["include"] == [] and edit["spec"]["exclude"][0]["value"] == 3.0
    assert thread[0]["result"].rwd_studies[0]["spec"]["include"][0]["concept"] == "var:sepsis3"   # v1 untouched


def test_the_builder_resets_and_refuses_what_it_cannot_count(tmp_path, monkeypatch, store, synth):
    at, turn = _page(tmp_path, monkeypatch, _result(synth))
    key = f"mes_cb_{turn['id']}_cb1"
    at.button(key=f"{key}_toggle").click()
    at.run()
    at.selectbox(key=f"{key}_g0_exc_0_op").set_value("between")
    at.run()
    at.number_input(key=f"{key}_g0_exc_0_lo").set_value(9.0)
    at.number_input(key=f"{key}_g0_exc_0_hi").set_value(2.0)       # a backwards range is put in order
    at.run()
    assert not at.exception and _counted(at)
    at.button(key=f"{key}_reset").click()
    at.run()
    assert not at.exception
    assert at.button(key=f"{key}_reset").disabled                  # back to the study's design
    assert at.toggle(key=f"{key}_g1_inc_0_on").value is True       # fresh fields after the reset


def test_the_builder_is_locked_where_it_cannot_work(tmp_path, monkeypatch, store):
    from mes.tests.test_protocol import _page_result
    at, turn = _page(tmp_path, monkeypatch, _page_result())
    locked = {b.key: b for b in at.button if b.key and b.key.startswith("mes_cb_") and b.key.endswith("_locked")}
    assert set(locked) == {f"mes_cb_{turn['id']}_ex1_locked", f"mes_cb_{turn['id']}_an1_locked"}
    assert all(b.disabled for b in locked.values())
    assert "SQL" in locked[f"mes_cb_{turn['id']}_ex1_locked"].help and "code" in locked[f"mes_cb_{turn['id']}_an1_locked"].help


def test_a_result_without_the_store_gets_a_locked_builder(tmp_path, monkeypatch, store, synth):
    monkeypatch.setattr(rwd_tools, "available", lambda: False)
    at, turn = _page(tmp_path, monkeypatch, _result(synth))
    b = at.button(key=f"mes_cb_{turn['id']}_cb1_locked")
    assert b.disabled and "not built" in b.help
    assert MESResult.from_dict(_result(synth).to_dict()).rwd_studies[0]["key"] == "cb1"


def test_the_trial_comparison_on_the_page(tmp_path, monkeypatch, store, synth):
    from mes.tests.test_trial_population import trial_record
    from mes.tools.result_schema import EvidenceItem
    from mes.tools.rwd import trial_population as TP
    TP.clear()
    result = _result(synth)
    result.evidence.append(EvidenceItem(id="T2", source_type="trial", title="A test trial", identifiers={
        "nct": "NCT00000001"}, meta={"has_results": True}))
    monkeypatch.setenv("MES_MONKEY_REAL_TRIALS", "1")      # the app keeps this test's ClinicalTrials.gov
    monkeypatch.setattr(TP, "get_json", lambda url, params=None, **kw: trial_record(url.rsplit("/", 1)[-1]))
    at, turn = _page(tmp_path, monkeypatch, result)
    key = f"mes_tp_{turn['id']}_cb1"
    at.button(key=f"{key}_toggle").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key=f"{key}_pick").value == "NCT00000001"   # the answer's trial with results first
    at.button(key=f"{key}_go").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state[f"{key}_res"]["NCT00000001"]
    assert not res.get("error") and res["rows"] and res["n_cohort"] > 0 and res["title"] == "A test trial"
    at.selectbox(key=f"{key}_pick").set_value("__other__")
    at.run()
    at.text_input(key=f"{key}_nct").set_value("not a trial")
    at.run()
    assert at.button(key=f"{key}_go").disabled
    for typed in ("see nct00000003 please", "NCT 0000-0003"):
        at.text_input(key=f"{key}_nct").set_value(typed)
        at.run()
        assert not at.button(key=f"{key}_go").disabled, typed
    TP.clear()


def test_the_trial_comparison_is_locked_without_the_store(tmp_path, monkeypatch, store, synth):
    monkeypatch.setattr(rwd_tools, "available", lambda: False)
    at, turn = _page(tmp_path, monkeypatch, _result(synth))
    b = at.button(key=f"mes_tp_{turn['id']}_cb1_locked")
    assert b.disabled and "not built" in b.help
