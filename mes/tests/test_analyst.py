"""
Offline tests of the MIMIC-IV analyst: sandbox policy and limits, result checks, and the analyst loop driven by a
scripted model (explore → code → submit → review → reproduction → evidence), on the synthetic store of test_rwd.

    python -m pytest mes/tests/test_analyst.py -q
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes import config  # noqa: E402
from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools.result_schema import QueryPlan  # noqa: E402
from mes.tools.rwd import analyst, report  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402
from mes.tools.rwd.result_check import normalize_result  # noqa: E402
from mes.tools.rwd.sandbox import Sandbox, check_code, mask_output  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_analyst")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, treated, early = _synthetic_store(root)
    store = Store(root)
    return store, ConceptIndex(store), icu


class ScriptedLLM:
    """Replies in order to the analyst's turns; the reviewer gets ``reviews`` in order."""
    label = "scripted"

    def __init__(self, replies, reviews):
        self.replies, self.reviews = list(replies), list(reviews)
        self.turns, self.n_calls = [], 0

    def invoke(self, prompt, messages=None, **_):
        self.n_calls += 1
        if prompt and "You review an observational analysis" in prompt:
            return json.dumps(self.reviews.pop(0))
        self.turns.append(messages)
        return self.replies.pop(0) if self.replies else "<plan>nothing left</plan>"


MORTALITY = """<plan>28-day mortality among Sepsis-3 patients.</plan>
<python>
d = sql("SELECT count(*) AS n, sum(death_28d) AS k FROM mes.icu_stays WHERE first_icu_stay AND age >= 18 AND sepsis3")
n, k = int(d.n[0]), int(d.k[0])
p, lo, hi = stats.wilson(k, n)
submit({"title": "28-day mortality in Sepsis-3", "analysis_type": "descriptive", "population": "adult first ICU stays with Sepsis-3",
        "n": n, "estimates": [{"label": "28-day mortality", "measure": "rate", "value": p, "lo": lo, "hi": hi, "n": n, "events": k, "primary": True}],
        "methods": "Proportion with Wilson CI."})
</python>
<finish>Mortality computed.</finish>"""
SOUND = {"verdict": "sound", "issues": [], "summary": "Appropriate descriptive analysis."}


def _mode(**changes):
    return dataclasses.replace(config.get_mode("sackett"), **{"analyst_seconds": 90, "analyst_steps": 4, **changes})


def test_policy_blocks_files_processes_and_dunders():
    assert check_code("import numpy as np\nx = np.arange(3).mean()\nprint(x)") == []
    for code in ("import os", "open('/etc/hosts').read()", "pd.read_csv('/etc/hosts')", "df.to_csv('x.csv')",
                 "().__class__.__bases__", "import subprocess", "eval('1')", "np.save('x', 1)",
                 "from mes.tools.rwd import db"):
        assert check_code(code), code
    text = mask_output("subject 10012345 stay 30000001 at 2150-03-04 12:00:00; n = 22518; itemid 220045")
    assert "10012345" not in text and "30000001" not in text and "2150-03-04" not in text
    assert "22518" in text and "220045" in text


def test_result_check_normalizes_and_refuses_row_level_output():
    import pandas as pd
    r = normalize_result({"title": "t", "estimates": [{"label": "rate", "measure": "rate", "value": 0.25, "lo": 0.2,
                                                       "hi": 0.3, "n": 400, "events": 100}],
                          "tables": {"by unit": pd.DataFrame({"unit": ["A", "B"], "n": [395, 5]})}})
    e = r["estimates"][0]
    assert (e["value"], e["lo"], e["hi"], e["unit"]) == (25.0, 20.0, 30.0, "%")       # fraction → percent
    assert r["tables"]["by unit"][1]["n"] == "<10"                                     # small cell suppressed
    named = normalize_result({"title": "t", "estimates": [
        {"label": m, "measure": m, "value": 1.2, "lo": 1.1, "hi": 1.3} for m in
        ("risk_ratio", "odds_ratio", "Hazard-Ratio", "risk_difference", "c-statistic", "something_else")]})
    assert [e["measure"] for e in named["estimates"]] == ["RR", "OR", "HR", "RD", "AUC", "other"]   # as Claude writes them
    with pytest.raises(ValueError, match="does not contain"):
        normalize_result({"title": "t", "estimates": [{"label": "OR", "measure": "OR", "value": 2, "lo": 2.5, "hi": 3}]})
    with pytest.raises(ValueError, match="identifier"):
        normalize_result({"title": "t", "estimates": [{"label": "x", "value": 1}],
                          "tables": {"rows": [{"subject_id": 1, "x": 2}]}})
    with pytest.raises(ValueError, match="rows"):
        normalize_result({"title": "t", "estimates": [{"label": "x", "value": 1}],
                          "tables": {"rows": [{"x": i} for i in range(500)]}})


def test_sandbox_session_persists_limits_time_and_restarts(synth):
    store, *_ = synth
    with Sandbox(store.root, cell_timeout=20) as box:
        assert box.run("x = 41")["ok"]
        assert box.run("print(x + 1)")["stdout"].strip() == "42"
        r = box.run("while True:\n    pass", timeout=2)
        assert not r["ok"] and "restarted" in r["error"]
        r = box.run("print('x' in dir())")
        assert r["stdout"].strip() == "False"                                         # fresh session
        assert "ZeroDivisionError" in box.run("1/0")["error"]


def test_analyst_explores_codes_submits_and_is_reproduced(synth):
    store, index, icu = synth
    llm = ScriptedLLM(["<plan>Find the Sepsis-3 flag.</plan>\n<concepts>sepsis</concepts>", MORTALITY], [SOUND])
    # a longer budget: the first cell also waits for the sandbox to load its libraries (slow on a cold start)
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, _mode(analyst_seconds=240),
                            store, index)
    assert study["status"] == "done" and len(study["steps"]) == 2, study.get("error") or study.get("steps")
    sub = icu[icu.sepsis3]
    e = study["result"]["estimates"][0]
    assert e["n"] == len(sub) and e["events"] == int(sub.death_28d.sum())
    assert abs(e["value"] - 100 * sub.death_28d.mean()) < 1e-9
    assert study["reproduction"]["reproduced"] and study["review"]["verdict"] == "sound"
    assert "<result>" in llm.turns[1][-1]["content"]                                    # the search result was shown
    item = report.analyst_items(study)[0]
    assert f"{e['value']:.1f}%" in item.passages[1].text and item.design.startswith("Custom analysis")


def test_analyst_revises_after_a_major_review_finding(synth):
    store, index, _ = synth
    first = MORTALITY.replace("death_28d", "death_hosp").replace("28-day mortality in", "Mortality in")
    major = {"verdict": "major_issues", "summary": "wrong outcome",
             "issues": [{"severity": "major", "problem": "uses in-hospital death", "fix": "use death_28d"}]}
    llm = ScriptedLLM([first, MORTALITY], [major, SOUND])
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, _mode(analyst_revisions=1),
                            store, index)
    assert [r["verdict"] for r in study["reviews"]] == ["major_issues", "sound"]
    assert study["result"]["title"] == "28-day mortality in Sepsis-3" and not study["result"]["warnings"]
    assert "independent reviewer found problems" in llm.turns[1][-1]["content"]


def test_analyst_without_a_result_fails_cleanly(synth):
    store, index, _ = synth
    llm = ScriptedLLM(["<plan>look</plan>\n<sql>SELECT count(*) AS n FROM mes.icu_stays</sql>",
                       "no action here", "<plan>stop</plan>\n<finish>done</finish>"], [])
    study = analyst.analyze("q", QueryPlan(question="q"), llm, _mode(analyst_steps=3), store, index)
    assert study["status"] == "failed" and len(study["steps"]) == 3
    assert "Nothing was submitted" in llm.turns[-1][-1]["content"] or study["steps"][-1]["action"] == "finish"
    assert report.analyst_failed_item(study).passages[0].text.startswith("Real-world data: the MES analyst")


def test_typed_in_numbers_are_refused(synth):
    store, index, _ = synth
    typed = MORTALITY.replace('n, k = int(d.n[0]), int(d.k[0])', 'n, k = 2405, 612')
    llm = ScriptedLLM([typed, MORTALITY], [SOUND])
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, _mode(), store, index)
    assert "submit() refused" in llm.turns[1][-1]["content"]
    assert study["status"] == "done" and study["result"]["estimates"][0]["n"] != 2405


def test_cells_stopped_by_an_error_are_part_of_the_reproduction(synth):
    """A cell that defines n, k and then fails still changed the session: the review sees it, the reproduction re-runs
    it, and the downloadable script keeps it."""
    store, index, icu = synth
    partial = ('<plan>Counts, then a typo.</plan>\n<python>\nd = sql("SELECT count(*) AS n, sum(death_28d) AS k FROM '
               'mes.icu_stays WHERE first_icu_stay AND age >= 18 AND sepsis3")\nn, k = int(d.n[0]), int(d.k[0])\n'
               'print(undefined_name)\n</python>')
    use = MORTALITY.replace('d = sql("SELECT count(*) AS n, sum(death_28d) AS k FROM mes.icu_stays WHERE first_icu_stay '
                            'AND age >= 18 AND sepsis3")\nn, k = int(d.n[0]), int(d.k[0])\n', "")
    llm = ScriptedLLM([partial, use], [SOUND])
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, _mode(), store, index)
    assert study["status"] == "done" and [c["ok"] for c in study["cells"]] == [False, True]
    assert study["reproduction"]["reproduced"], study["reproduction"]
    assert study["result"]["estimates"][0]["n"] == int(icu.sepsis3.sum())
    assert "try:" in study["code"] and "undefined_name" in study["code"]
    compile(study["code"], "analysis.py", "exec")


def test_a_refused_submission_does_not_block_the_corrected_one(synth):
    """Numbers typed into a refused submit() call left nothing in the session: the corrected, computed submission is
    accepted although it reports the same count."""
    store, index, icu = synth
    n = int(icu.sepsis3.sum())
    assert n >= 200 and n % 50, "the synthetic count must look like a copied result"
    typed = MORTALITY.replace('"n": n, "estimates"', f'"n": n, "attrition": [["Sepsis-3", {n}]], "estimates"')
    fixed = MORTALITY.replace('"n": n, "estimates"', '"n": n, "attrition": [["Sepsis-3", n]], "estimates"')
    llm = ScriptedLLM([typed, fixed], [SOUND])
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, _mode(), store, index)
    assert "submit() refused" in llm.turns[1][-1]["content"]
    assert study["status"] == "done" and study["submitted_cell"] == 1 and study["reproduction"]["reproduced"]
    assert analyst.typed_numbers(study["result"], [c["code"] for _, c in analyst.session_cells(study["cells"], 1)]) == []
    # assert statements are checks, not typed values
    assert analyst.typed_numbers({"estimates": [{"label": "x", "n": n}]}, [f"assert len(df) == {n}"]) == []


def test_auto_falls_back_to_the_template_when_the_analyst_does_not_finish(synth, monkeypatch):
    """Auto mode routes a design that misses a treatment of the question to the analyst; when the analyst runs out
    of budget, the closest validated design still answers, with the reason."""
    from mes.tools import rwd_tools
    from mes.tools.rwd import concepts
    store, index, _ = synth
    intent = {"feasibility": "feasible", "reason": "", "analysis": "rate", "base": "icu", "title": "Testdrug",
              "population": [], "exclude": [], "exposure": {"term": "testdrug", "op": "", "value": "", "within_hours": 24},
              "comparator": {"term": "", "op": "", "value": ""}, "outcome": "death_28d", "variable": "",
              "stratify": [], "subgroups": [], "sql_question": ""}
    designer_llm = type("D", (), {"label": "scripted", "invoke": lambda self, *a, **k: json.dumps(intent)})()
    failed = {"key": "a", "kind": "analyst", "status": "failed", "result": None, "spec": None,
              "design_info": {"errors": ["the analyst did not submit a result within the Fast budget"]}}
    monkeypatch.setattr(rwd_tools, "get_store", lambda: store)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (True, ""))
    monkeypatch.setattr(analyst, "analyze", lambda *a, **k: dict(failed))
    plan = QueryPlan(question="q", entities=[{"name": "vasopressin", "role": "intervention"},
                                             {"name": "28-day mortality", "role": "outcome"}])
    items, stats = rwd_tools.search_rwd(plan, config.get_mode("osler"), "Is vasopressin better?", llm=designer_llm)
    study = stats["studies"][0]
    assert study["kind"] == "template" and study["status"] == "done" and study["result"]["estimates"]
    assert study["design_info"]["routed"] == "the study templates cannot express vasopressin"
    assert any("did not finish" in w for w in study["result"]["warnings"])
    assert any("could not express vasopressin" in w for w in study["result"]["warnings"])


def test_session_cells_follow_restarts_and_skip_refused_code():
    cells = [{"ok": True, "ran": True}, {"ok": False, "ran": True, "killed": True}, {"ok": False, "ran": True},
             {"ok": False, "ran": False}, {"ok": True, "ran": True}]
    assert [i for i, _ in analyst.session_cells(cells)] == [2, 4]
    assert [i for i, _ in analyst.session_cells(cells, 0)] == [0]


def test_reply_parsing_is_lenient():
    plan, kind, body, finish = analyst.parse_reply("<plan>p</plan>\n<python>\n```python\nx = 1\n```\n</python>")
    assert (plan, kind, body, finish) == ("p", "python", "x = 1", None)
    assert analyst.parse_reply("Here:\n```python\ny = 2\n```")[1:3] == ("python", "y = 2")
    assert analyst.parse_reply("<plan>p</plan><python>z = 3\nprint(z)")[1] == "python"      # cut off at the limit
    assert analyst.parse_reply("just text")[1] is None


def test_a_term_without_a_concept_stops_the_template_study(synth, monkeypatch):
    """"unobtainium" is never replaced by a look-alike concept: no study of a different treatment is run."""
    from mes.tools import rwd_tools
    from mes.tools.rwd import concepts
    from mes.tools.rwd.concepts import _spelled_like
    store, index, _ = synth
    intent = {"feasibility": "feasible", "reason": "", "analysis": "rate", "base": "icu", "title": "",
              "population": [], "exclude": [],
              "exposure": {"term": "unobtainium", "op": "", "value": "", "within_hours": 24},
              "comparator": {"term": "", "op": "", "value": ""}, "outcome": "death_28d", "variable": "",
              "stratify": [], "subgroups": [], "sql_question": ""}
    designer_llm = type("D", (), {"label": "scripted", "invoke": lambda self, *a, **k: json.dumps(intent)})()
    monkeypatch.setattr(rwd_tools, "get_store", lambda: store)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (False, "off"))
    plan = QueryPlan(question="q", entities=[{"name": "unobtainium", "role": "intervention"}])
    items, stats = rwd_tools.search_rwd(plan, config.get_mode("osler"), "Is unobtainium associated with death?",
                                        llm=designer_llm)
    study = stats["studies"][0]
    assert study["status"] == "failed" and "unobtainium" in items[0].passages[0].text
    assert not _spelled_like("vecuronium", {"unobtainium"}) and _spelled_like("metformin", {"metfromin"})


def test_reproduction_tolerates_the_last_digits_of_fitted_models():
    a = {"estimates": [{"label": "RD", "measure": "RD", "value": 4.82345, "lo": 0.61, "hi": 9.3, "n": 6240, "events": None}]}
    b = {"estimates": [{"label": "RD", "measure": "RD", "value": 4.82351, "lo": 0.61, "hi": 9.3, "n": 6240, "events": None}]}
    c = {"estimates": [{"label": "RD", "measure": "RD", "value": 4.9, "lo": 0.61, "hi": 9.3, "n": 6240, "events": None}]}
    d = {"estimates": [{"label": "RD", "measure": "RD", "value": 4.82345, "lo": 0.61, "hi": 9.3, "n": 6241, "events": None}]}
    assert analyst._same(a, b) and not analyst._same(a, c) and not analyst._same(a, d)


def test_sandbox_query_rows_come_back_in_a_fixed_order():
    import pandas as pd
    from mes.tools.rwd.sandbox_worker import _stable
    df = pd.DataFrame({"stay_id": [3, 1, 2, 1], "x": [None, "b", "a", None], "l": [[1], [2], [3], [4]]})
    assert list(_stable(df, "SELECT * FROM t")["stay_id"]) == [1, 1, 2, 3]
    assert list(_stable(df, "SELECT * FROM t ORDER BY x")["stay_id"]) == [3, 1, 2, 1]
    groups = pd.DataFrame({"g": ["b", "a"], "n": [2, 1]})
    assert list(_stable(groups, "SELECT g, count(*) n FROM t GROUP BY 1")["g"]) == ["a", "b"]


def test_the_code_check_goes_to_the_review_and_to_the_warnings(synth):
    store, index, icu = synth
    code = MORTALITY.replace('AND sepsis3")', 'AND sepsis3 AND hadm_id NOT IN (SELECT hadm_id FROM mes.drugs '
                             'WHERE drug ILIKE \'%nosuchdrug%\')")')
    prompts = []

    class Recording(ScriptedLLM):
        def invoke(self, prompt, messages=None, **kw):
            if prompt and "You review" in prompt:
                prompts.append(prompt)
            return super().invoke(prompt, messages, **kw)

    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), Recording([code], [SOUND]),
                            _mode(), store, index)
    assert study["status"] == "done"
    rows = {r["written"]: r for r in study["code_audit"]}
    assert rows["%nosuchdrug%"]["problem"] and rows["sepsis3"]["kind"] == "MES variable"
    assert "CODE CHECK" in prompts[0] and "%nosuchdrug%" in prompts[0]
    assert any(w.startswith("Code check:") for w in study["result"]["warnings"])


def test_a_slow_review_is_given_up_within_the_budget(synth, monkeypatch):
    import time
    store, index, icu = synth
    monkeypatch.setattr(analyst.Analyst, "REVIEW_GRACE", -55)            # i.e. 5 s after the start
    monkeypatch.setattr(analyst.Analyst, "REVIEW_MIN", 2)

    class Slow(ScriptedLLM):
        def invoke(self, prompt, messages=None, **kw):
            if prompt and "You review" in prompt:
                time.sleep(30)                          # e.g. a Bedrock reply that takes minutes
            return super().invoke(prompt, messages, **kw)

    t = time.time()
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), Slow([MORTALITY], [SOUND]),
                            _mode(analyst_seconds=60), store, index)
    assert study["status"] == "done" and study["review"]["verdict"] == "not_reviewed"
    assert "time budget" in study["review"]["summary"] and time.time() - t < 28


def test_only_what_the_templates_cannot_express_goes_to_the_analyst(synth, monkeypatch):
    """Claude's designer names the kind of an approximation: a limit of the data itself (anchor year groups) is
    reported with the template study; something the templates cannot express goes to the analyst."""
    from mes.tools import rwd_tools
    from mes.tools.rwd import concepts
    store, index, _ = synth
    base = {"feasibility": "feasible", "reason": "", "analysis": "rate", "base": "icu", "title": "Sepsis",
            "population": [{"term": "sepsis3", "op": "", "value": ""}], "exclude": [],
            "exposure": {"term": "", "op": "", "value": ""}, "comparator": {"term": "", "op": "", "value": ""},
            "outcome": "death_28d", "variable": "", "stratify": [], "subgroups": [], "sql_question": ""}
    monkeypatch.setattr(rwd_tools, "get_store", lambda: store)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (True, ""))
    ran = []
    done = {"key": "a", "kind": "analyst", "status": "not_feasible", "result": None,
            "spec": {"reason": "x"}, "design_info": {}}
    monkeypatch.setattr(analyst, "analyze", lambda *a, **k: ran.append(1) or dict(done))
    plan = QueryPlan(question="q", entities=[{"name": "sepsis", "role": "population"},
                                             {"name": "28-day mortality", "role": "outcome"}])
    for kind, routed in (("data", False), ("choice", False), ("templates", True)):
        intent = {**base, "approximation": "Calendar years are only available as anchor year groups here.",
                  "approximation_kind": kind}
        llm = type("C", (), {"label": "claude-bedrock · test", "invoke": lambda self, *a, **k: json.dumps(intent)})()
        ran.clear()
        items, stats = rwd_tools.search_rwd(plan, config.get_mode("osler"), "q", llm=llm)
        study = stats["studies"][0]
        assert bool(ran) == routed, kind
        if not routed:
            assert study["kind"] == "template" and any("approximates" in w for w in study["result"]["warnings"])


def test_a_design_that_leaves_out_a_treatment_mimic_has_is_repaired_not_routed(synth, monkeypatch):
    """Claude's designer once dropped the comparator (cefepime): MIMIC-IV has it, so the check asks the designer
    for a repair instead of handing the question to the analyst; a term without a clear concept is not hinted."""
    from mes.tools import rwd_tools
    from mes.tools.rwd.concepts import Candidate
    from mes.tools.rwd.spec import StudySpec
    store, index, _ = synth
    real = index.search
    monkeypatch.setattr(index, "search", lambda term, k=8, **kw: [Candidate("drugclass:cefepime", "Cefepime",
                        "drugclass", 18078, 1.81)] if term == "cefepime" else real(term, k=k, **kw))
    spec = StudySpec.from_dict({"analysis": "compare", "base": "icu", "exposure": {"concept": "drug:testdrug"},
                                "outcome": "death_28d"})
    hint = rwd_tools._expressible(["cefepime"], spec, index)
    assert "drugclass:cefepime" in hint and "comparator" in hint
    assert rwd_tools._expressible(["no corticosteroid of any kind"], spec, index) == ""
    used = StudySpec.from_dict({**spec.to_dict(), "comparator": {"concept": "drugclass:cefepime"}})
    assert rwd_tools._expressible(["cefepime"], used, index) == ""              # already in the design
