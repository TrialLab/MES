"""
Offline tests of edited protocols (tools/protocol_tools.py, ui/protocol.py, ui/rwd_panel.py): search edits rebuilt
like the planner's, only the affected sources searched again, the other evidence kept untouched, edited SQL and code
run under the same guards on a synthetic MIMIC-like store and marked as not independently checked, the version
record, the history, the exports, and the 🔒 switches of the page (AppTest with the fakes of scripts/ui_monkey.py).

    python -m pytest mes/tests/test_protocol.py -q
"""

import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes import config  # noqa: E402
from mes.tests.test_analyst import MORTALITY, SOUND, ScriptedLLM  # noqa: E402
from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools import protocol_tools as PT  # noqa: E402
from mes.tools import rwd_tools  # noqa: E402
from mes.tools.planner_tools import build_pubmed_queries, build_trial_terms  # noqa: E402
from mes.tools.result_schema import EvidenceItem, MESResult, Passage, QueryPlan  # noqa: E402
from mes.tools.rwd import analyst, concepts  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402

CONCEPTS = [{"role": "population", "label": "septic shock", "kind": "disease", "synonyms": ["vasoplegic shock"]},
            {"role": "intervention", "label": "vasopressin", "kind": "drug", "synonyms": ["AVP"]},
            {"role": "comparator", "label": "norepinephrine", "kind": "drug", "synonyms": []},
            {"role": "outcome", "label": "28-day mortality", "kind": "outcome", "synonyms": []}]


def _plan(facets=False):
    precise, broad, tiers = build_pubmed_queries(CONCEPTS)
    plan = QueryPlan(question="Vasopressin in septic shock?", standalone_question="Vasopressin in septic shock?",
                     question_type="therapy", entities=[{"name": c["label"], "type": c["kind"], "role": c["role"],
                                                         "synonyms": c["synonyms"]} for c in CONCEPTS],
                     pico={"population": "septic shock"}, pubmed_query=precise, pubmed_query_broad=broad,
                     pubmed_tiers=tiers, trial_terms=build_trial_terms(CONCEPTS))
    if facets:
        main = build_pubmed_queries(CONCEPTS[:2])[0]
        plan.facets = [{"name": "safety", "question": "harms?", "pubmed_query": f"{main} AND (ischemia)"}]
    return dataclasses.asdict(plan)


def _item(id_, source_type, title, origin="", **ids):
    return EvidenceItem(id=id_, source_type=source_type, title=title, origin=origin, identifiers=ids, score=0.5,
                        passages=[Passage(text=f"{title}: mortality was 20%.", section="Abstract")])


def _base(rwd_studies=None, rwd_items=None, sources=("literature", "trials", "kg", "rwd")):
    evidence = [_item("L1", "literature", "Old trial", "PubMed", pmid="1"),
                _item("L2", "literature", "Old review", "PubMed + Europe PMC full text", pmid="2"),
                _item("L3", "literature", "My PDF", "Local PDF library", file="my.pdf"),
                _item("T1", "trial", "VANISH", "ClinicalTrials.gov", nct="NCT01"),
                _item("K1", "kg", "vasopressin treats septic shock", "iBKH")] + list(rwd_items or [])
    return MESResult(question="Vasopressin in septic shock?", mode="osler", sources=list(sources), plan=_plan(),
                     evidence=evidence, answer_markdown="## Answer\nx [L1].", answer_display="## Answer\nx [1].",
                     sections={"Answer": "x"}, citation_order=["L1"], llm="fake", rwd_studies=list(rwd_studies or []),
                     source_status={"literature": {"ok": True, "queries": {"precise": "old"}},
                                    "pdfs": {"ok": True}, "trials": {"ok": True}, "kg": {"ok": True},
                                    "rwd": {"ok": True}})


class LLM:
    label, n_calls = "fake", 0


@pytest.fixture
def fakes(monkeypatch):
    """Retrieval and synthesis without network or model; records which sources ran."""
    from mes.tools import pipeline_tools as P
    ran = []

    def literature(plan, mode, query, qv, report, fulltext=True):
        ran.append(("literature", plan.pubmed_query))
        return [_item("lit:new", "literature", "New trial", "PubMed", pmid="9"),
                _item("lit:old", "literature", "Old trial", "PubMed", pmid="1")], {"queries": {"precise": plan.pubmed_query}}

    def trials(plan, mode, query, qv, report):
        ran.append(("trials", plan.trial_terms))
        return [_item("t:new", "trial", "A new trial", "ClinicalTrials.gov", nct="NCT02")], {}

    def kg(plan, mode, query, qv, report):
        ran.append(("kg", [e["name"] for e in plan.entities]))
        return [_item("k:new", "kg", "AVP relation", "iBKH")], {"links": {}}

    def stream(plan, evidence, llm, mode_cfg, sources, budget, history):
        ids = [e.id for e in evidence]
        yield "text", "## Key takeaways\n- Mortality was 20% [L1].\n\n## Answer\nSee " + " ".join(f"[{i}]" for i in ids) + ".\n"

    monkeypatch.setattr(P, "_literature", literature)
    monkeypatch.setattr(P.trials_tools, "search_trials", trials)
    monkeypatch.setattr(P, "_kg", kg)
    monkeypatch.setattr(P, "stream_synthesis", stream)
    monkeypatch.setattr(PT, "encode", lambda texts, is_query=False: np.zeros((len(texts), 8)))
    return ran


def _run(base, edit, **kw):
    events = list(PT.iter_rerun(base.question, LLM(), base, edit, **kw))
    results = [e["result"] for e in events if e["type"] == "result"]
    return events, (results[0] if results else None)


# ============================================================================ search protocol
def test_search_edit_rebuilds_queries_like_the_planner_and_reruns_only_what_changed():
    plan = _plan(facets=True)
    same, changes, rerun, errors = PT.search_edit_plan(plan, {"concepts": copy.deepcopy(CONCEPTS)}, ["literature"])
    assert not changes and not rerun and not errors and same.entities == plan["entities"]

    edited = copy.deepcopy(CONCEPTS)
    edited[1]["synonyms"] = "AVP; arginine vasopressin; AVP"            # as typed in the table: ";"-separated
    new, changes, rerun, errors = PT.search_edit_plan(plan, {"concepts": edited}, ["literature", "trials", "kg", "rwd"])
    assert not errors and rerun == ["literature", "trials", "kg"]       # MIMIC-IV has its own protocol
    concepts_now = PT.concepts_of(dataclasses.asdict(new))
    assert concepts_now[1]["synonyms"] == ["AVP", "arginine vasopressin"]
    assert (new.pubmed_query, new.pubmed_query_broad, new.pubmed_tiers) == build_pubmed_queries(concepts_now)
    assert new.trial_terms == build_trial_terms(concepts_now) and "arginine vasopressin" in new.pubmed_query
    assert new.facets[0]["pubmed_query"] == build_pubmed_queries(concepts_now[:2])[0] + " AND (ischemia)"
    assert [c["what"] for c in changes][:2] == ["Concept changed", "PubMed query"]

    only_type, changes, rerun, errors = PT.search_edit_plan(plan, {"question_type": "harm"}, ["literature"])
    assert only_type.question_type == "harm" and rerun == [] and [c["what"] for c in changes] == ["Question type"]
    empty = {**plan, "trial_terms": {"condition": "", "intervention": "", "outcome": ""}}      # empty already: not the edit's
    assert PT.search_edit_plan(empty, {"question_type": "harm"}, ["literature", "trials"])[3] == []
    assert PT.search_edit_plan(plan, {"concepts": [{"role": "population", "label": "", "synonyms": "x"},
                                                   {"role": "bogus", "label": "y"}]}, ["literature"])[3]
    # a stored plan without its question (an old or foreign record) takes the answer's
    headless = {k: v for k, v in plan.items() if k != "question"}
    assert PT.search_edit_plan(headless, {"question_type": "harm"}, ["literature"], "Q?")[0].question == "Q?"
    assert PT.search_edit_plan({**headless, "question": None}, {"question_type": "harm"}, ["literature"])[0].question == ""


def test_queries_written_by_hand_run_as_written_and_are_checked():
    plan = _plan()
    queries = {"pubmed_query": plan["pubmed_query"] + " AND 2015:2025[dp]", "relaxed": [], "facets": {},
               "trial_terms": plan["trial_terms"]}
    new, changes, rerun, errors = PT.search_edit_plan(plan, {"queries": queries}, ["literature", "trials", "kg"])
    assert not errors and rerun == ["literature"] and new.pubmed_tiers == [queries["pubmed_query"]]
    for bad, problem in (("(a OR b", "not closed"), ("a OR b)", "no opening"), ('"septic shock', "quote"),
                         ("x" * 4001, "too long")):
        errors = PT.search_edit_plan(plan, {"queries": {**queries, "pubmed_query": bad}}, ["literature"])[3]
        assert any(problem in e for e in errors), (bad[:20], errors)
    assert PT.search_edit_plan(plan, {"queries": {**queries, "pubmed_query": '"septic (shock" AND x'}},
                               ["literature"])[3] == []                  # parentheses inside quotes are text
    errors = PT.search_edit_plan(plan, {"queries": {**queries, "pubmed_query": " "}}, ["literature"])[3]
    assert any("PubMed needs a query" in e for e in errors)
    # terms given back as they were (an empty field, other spacing) are not a change: ClinicalTrials.gov is kept
    plan["trial_terms"] = {**plan["trial_terms"], "outcome": ""}
    plan["pubmed_query"] = plan["pubmed_query"].replace(" AND ", "  AND ")
    same = {"pubmed_query": " ".join(plan["pubmed_query"].split()) + " AND humans[mh]",
            "relaxed": plan["pubmed_tiers"][1:], "facets": {}, "trial_terms": {**plan["trial_terms"],
                                                                                "condition": plan["trial_terms"]["condition"] + " "}}
    new, changes, rerun, errors = PT.search_edit_plan(plan, {"queries": same}, ["literature", "trials"])
    assert not errors and rerun == ["literature"] and new.trial_terms == plan["trial_terms"]
    assert [c["what"] for c in changes] == ["PubMed query"]


def test_an_edited_search_reuses_the_other_evidence_and_records_the_version(fakes):
    base = _base(sources=("literature", "trials", "kg"))
    before = copy.deepcopy(base.to_dict())
    queries = {"pubmed_query": "vasopressin AND septic shock", "relaxed": [], "facets": {},
               "trial_terms": base.plan["trial_terms"]}
    events, v2 = _run(base, {"part": "search", "queries": queries})
    assert [s for s, _ in fakes] == ["literature"]                       # trials and iBKH kept
    assert base.to_dict() == before                                      # the edited answer is untouched
    titles = {e.title for e in v2.evidence}
    assert {"New trial", "Old trial", "My PDF", "VANISH", "vasopressin treats septic shock"} <= titles
    assert "Old review" not in titles                                    # PubMed results are the new search's
    kept = [e for e in events if e["type"] == "stage" and e["status"] == "kept"]
    assert {e["stage"] for e in kept} == {"pdfs", "trials", "kg"}
    p = v2.protocol
    assert (p["version"], p["from_version"], p["rerun"], p["part"]) == (2, 1, ["literature"], "search")
    assert set(p["kept"]) == {"pdfs", "trials", "kg"} and p["checks"] and "PubMed query" in p["summary"]
    lit = p["comparison"]["sources"]["literature"]
    assert lit["added"] == ["New trial"] and lit["removed"] == ["Old review"]
    assert v2.source_status["trials"]["kept"] and v2.source_status["literature"]["queries"]["precise"] == queries["pubmed_query"]
    assert v2.plan["pubmed_query"] == queries["pubmed_query"]

    events, v3 = _run(v2, {"part": "search", "question_type": "harm"})       # an edit of the edited version
    assert v3.protocol["version"] == 3 and [t["version"] for t in v3.protocol["trail"]] == [2, 3]
    assert v3.protocol["rerun"] == [] and len(fakes) == 1                    # nothing searched again


def test_nothing_changed_or_a_failing_source_ends_without_an_answer(fakes, monkeypatch):
    from mes.tools import pipeline_tools as P
    base = _base(sources=("literature",))
    events, result = _run(base, {"part": "search", "concepts": copy.deepcopy(CONCEPTS)})
    assert result is None and "Nothing was changed" in [e for e in events if e["type"] == "error"][0]["message"]

    def down(*a):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(P, "_literature", down)
    events, result = _run(base, {"part": "search", "queries": {"pubmed_query": "x AND y"}})
    assert result is None and "PubMed failed (HTTP 503)" in [e for e in events if e["type"] == "error"][0]["message"]


# ============================================================================ study protocols
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_protocol")
    with np.errstate(over="ignore", invalid="ignore"):   # pandas' NaN → NaT conversion of the synthetic deaths
        icu, treated, early = _synthetic_store(root)
    store = Store(root)
    return store, ConceptIndex(store), icu


@pytest.fixture
def store(synth, monkeypatch):
    s, index, icu = synth
    monkeypatch.setattr(rwd_tools, "get_store", lambda: s)
    monkeypatch.setattr(concepts, "get_index", lambda: index)
    monkeypatch.setattr(analyst, "status", lambda llm=None: (False, "test"))
    return s, index, icu


def _explore_study(sql="SELECT careunit, count(*) AS n FROM mes.icu_stays GROUP BY 1"):
    return {"key": "ex1", "kind": "template", "status": "done", "evidence_id": "R1", "sql": sql,
            "spec": {"analysis": "explore", "base": "icu", "sql_question": "patients by ICU type"},
            "result": {"analysis": "explore", "title": "patients by ICU type", "estimates": [], "warnings": [],
                       "tables": {}, "diagnostics": {}, "population": "MIMIC-IV"}}


def test_an_edited_query_runs_under_the_explorer_guards(store):
    _s, _index, icu = store
    study = _explore_study()
    mode = config.get_mode("osler")
    sql = "SELECT careunit, count(*) AS n, sum(death_28d) AS deaths FROM mes.icu_stays GROUP BY 1 ORDER BY 1"
    record = rwd_tools.edited_study(study, {"kind": "sql", "sql": sql + ";", "study": "ex1"}, "q", mode)
    rows = record["result"]["tables"]["result"]
    assert record["status"] == "done" and record["edited"] == {"kind": "sql", "from_key": "ex1"}
    assert {r["careunit"]: r["n"] for r in rows} == icu.careunit.value_counts().to_dict()
    assert record["result"]["sql_check"] is None and "not run" in record["result"]["diagnostics"]["independent_query"]
    item = rwd_tools.study_items(record)[0]
    assert "edited by the user" in item.passages[0].text and "Query result" in item.passages[1].text
    for bad, problem in (("SELECT subject_id, age FROM mes.icu_stays", "identifiers"),
                         ("SELECT * FROM read_csv('/etc/hosts')", "file access"),
                         ("SELECT 1; DELETE FROM mes.icu_stays", "statement")):
        record = rwd_tools.edited_study(study, {"kind": "sql", "sql": bad, "study": "ex1"}, "q", mode)
        assert record["status"] == "failed" and problem in record["design_info"]["errors"][0], (bad, record["design_info"])
    assert rwd_tools.editable({**study, "sql": ""}, "sql") == ""        # a query the model could not write: write it
    with pytest.raises(ValueError):                     # the SQL of a template study is compiled from its design
        rwd_tools.edited_study({**study, "spec": {"analysis": "rate", "base": "icu"}}, {"kind": "sql", "sql": sql}, "q",
                               mode)


def _analyst_study(store_, index):
    llm = ScriptedLLM([MORTALITY], [SOUND])
    mode = dataclasses.replace(config.get_mode("sackett"), analyst_seconds=90, analyst_steps=3)
    study = analyst.analyze("28-day mortality in sepsis?", QueryPlan(question="q"), llm, mode, store_, index)
    study["result"] = rwd_tools._plain(study["result"])
    return study


def test_edited_code_runs_in_a_fresh_session_and_is_marked_unreviewed(store):
    s, index, icu = store
    study = _analyst_study(s, index)
    assert study["status"] == "done" and study["review"]["verdict"] == "sound"
    code = study["code"].replace("death_28d", "death_hosp")
    mode = config.get_mode("sackett")
    record = rwd_tools.edited_study(study, {"kind": "code", "code": code, "study": study["key"]}, "q", mode)
    e = record["result"]["estimates"][0]
    sub = icu[icu.sepsis3]
    assert record["status"] == "done" and e["events"] == int(sub.death_hosp.sum()) and e["n"] == len(sub)
    assert record["review"]["verdict"] == "not_reviewed" and record["reproduction"] is None
    assert record["edited"]["from_key"] == study["key"] and record["key"] != study["key"]
    assert record["result"]["warnings"][0] == analyst.EDITED_WARNING
    item = rwd_tools.study_items(record)[0]
    assert "edited by the user" in item.passages[0].text and "not reviewed independently" in item.passages[0].text
    assert item.meta["edited"] and "reproduced in a fresh session: yes" not in item.passages[0].text

    typed = code.replace("n, k = int(d.n[0]), int(d.k[0])", f"n, k = {len(sub)}, int(d.k[0])")
    record = rwd_tools.edited_study(study, {"kind": "code", "code": typed, "study": study["key"]}, "q", mode)
    assert record["status"] == "failed" and "typed into the code" in record["design_info"]["errors"][0]
    for bad, problem in (("import os\nos.system('ls')", "not allowed"), ("x = 1", "did not call submit")):
        record = rwd_tools.edited_study(study, {"kind": "code", "code": bad, "study": study["key"]}, "q", mode)
        assert record["status"] == "failed" and problem in " ".join(record["design_info"]["errors"]), record


def test_a_refused_submit_of_an_earlier_cell_does_not_block_the_script(store):
    """The analyst's script keeps a cell whose submit() was refused (typed-in count) before the computed one: run as
    one script it must still be accepted."""
    s, index, icu = store
    n = int(icu.sepsis3.sum())
    script = (f"# --- cell 1 (its submit() was refused and replaced by a later one)\n"
              f"submit({{'title': 't', 'estimates': [{{'label': 'x', 'measure': 'count', 'value': 1, 'n': {n}}}]}})\n\n"
              "# --- cell 2\nd = sql(\"SELECT count(*) AS n FROM mes.icu_stays WHERE first_icu_stay AND age >= 18 "
              "AND sepsis3\")\nsubmit({'title': 't', 'estimates': [{'label': 'x', 'measure': 'count', "
              "'value': int(d.n[0]), 'n': int(d.n[0])}]})")
    study = {"key": "a1", "kind": "analyst", "code": "print(1)", "status": "failed", "llm": "fake"}
    record = rwd_tools.edited_study(study, {"kind": "code", "code": script, "study": "a1"}, "q", config.get_mode("osler"))
    assert record["status"] == "done", record.get("design_info")
    assert record["result"]["estimates"][0]["n"] == n


def test_an_rwd_edit_runs_only_the_study_and_uses_the_tested_run(store, fakes, monkeypatch):
    s, index, icu = store
    explore, other = _explore_study(), {"key": "t2", "kind": "template", "status": "done", "evidence_id": "R2",
                                        "spec": {"analysis": "rate", "base": "icu"}, "result": {"estimates": []}}
    items = [EvidenceItem(id="R1", source_type="rwd", title="explore", meta={"study_key": "ex1"},
                          passages=[Passage(text="old", section="RWD results")]),
             EvidenceItem(id="R2", source_type="rwd", title="rate", meta={"study_key": "t2"},
                          passages=[Passage(text="kept", section="RWD results")])]
    base = _base([explore, other], items)
    edit = {"part": "rwd", "study": "ex1", "kind": "sql",
            "sql": "SELECT careunit, count(*) AS n FROM mes.icu_stays GROUP BY 1 ORDER BY 1"}
    tested = rwd_tools.edited_study(explore, edit, base.question, config.get_mode("osler"))

    def never(*a, **k):
        raise AssertionError("the tested edit ran again")
    monkeypatch.setattr(rwd_tools, "edited_study", never)
    events, v2 = _run(base, edit, tested=tested)
    assert not fakes                                                     # no source searched again
    assert [st_["key"] for st_ in v2.rwd_studies] == [tested["key"], "t2"]
    rwd_titles = [e.title for e in v2.evidence if e.source_type == "rwd"]
    assert "rate" in rwd_titles and "explore" not in rwd_titles
    p = v2.protocol
    assert (p["rerun"], p["study"], p["study_ref"], p["new_study"]) == (["rwd"], "ex1", "R1", tested["key"])
    assert "+SELECT careunit" in p["diff"] and "independently written query" in p["checks"][0]
    assert set(p["kept"]) == {"literature", "pdfs", "trials", "kg"}

    unreproduced = {"key": "an9", "kind": "analyst", "status": "done", "code": "x = 1", "reproduction": {"reproduced": False}}
    errors, _ = PT.check_edit(_base([unreproduced]), {"part": "rwd", "study": "an9", "kind": "code", "code": "x = 2"})
    assert not errors and "other numbers in a fresh session" in PT.prepare(
        _base([unreproduced]), {"part": "rwd", "study": "an9", "kind": "code", "code": "x = 2"})["checks"][1]
    events, result = _run(base, {**edit, "sql": "SELECT subject_id FROM mes.icu_stays"})
    assert result is None and any(e["type"] == "error" for e in events)     # the failed edit raises (never())
    monkeypatch.undo()


def test_an_edited_design_reruns_the_template_study(store, fakes):
    s, index, icu = store
    spec = {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}], "outcome": "death_28d"}
    items, stats = rwd_tools.search_rwd(QueryPlan(question="q"), config.get_mode("osler"), "q", spec_override=spec)
    study = {**stats["studies"][0], "evidence_id": "R1"}
    base = _base([study], items)
    edited = {**study["spec"], "outcome": "death_hosp"}
    errors, preview = PT.check_edit(base, {"part": "rwd", "study": study["key"], "kind": "design", "spec": edited})
    assert not errors and preview["changes"][0]["what"] == "Outcome" and preview["rerun"] == ["rwd"]
    events, v2 = _run(base, {"part": "rwd", "study": study["key"], "kind": "design", "spec": edited})
    new = v2.rwd_studies[0]
    assert new["edited"] == {"kind": "design", "from_key": study["key"]} and new["spec"]["outcome"] == "death_hosp"
    assert new["result"]["estimates"][0]["events"] == int(icu[icu.sepsis3].death_hosp.sum())
    assert v2.protocol["comparison"]["estimates"][0]["n_changed"] >= 1
    errors, _ = PT.check_edit(base, {"part": "rwd", "study": study["key"], "kind": "design", "spec": study["spec"]})
    assert errors == ["Nothing was changed in the study design"]
    errors, _ = PT.check_edit(base, {"part": "rwd", "study": "missing", "kind": "design", "spec": edited})
    assert "not part of this answer" in errors[0]


# ============================================================================ records, history, exports
def _edited_version():
    v2 = _base()
    v2.protocol = {"version": 2, "from_version": 1, "part": "search", "label": "Search protocol",
                   "summary": "PubMed query", "changes": [{"what": "PubMed query", "before": "a", "after": "b"}],
                   "diff": "", "rerun": ["literature"], "kept": ["trials", "kg", "rwd"], "checks": ["checked"],
                   "edited_at": "2026-11-01T10:00:00", "trail": [{"version": 2, "from_version": 1,
                                                                  "label": "Search protocol", "summary": "PubMed query"}],
                   "comparison": {"sources": {"literature": {"before": 2, "after": 2, "n_added": 1, "n_removed": 1,
                                                             "added": ["New"], "removed": ["Old"]}},
                                  "cited": {"before": 1, "after": 2}, "estimates": []}}
    return v2


def test_history_keeps_edited_versions(tmp_path, monkeypatch):
    from mes.ui import history
    monkeypatch.setenv("MES_HISTORY_DIR", str(tmp_path))
    v1, v2 = _base(), _edited_version()
    thread = [{"id": "a", "question": v1.question, "mode": "osler", "sources": v1.sources, "status": "done",
               "result": v1},
              {"id": "b", "question": v1.question, "mode": "osler", "sources": v1.sources, "status": "done",
               "result": v2, "edit": {"part": "search", "question_type": "harm"}, "parent": "a", "root": "a",
               "version": 2, "base": v1, "tested": {"big": "record"}}]
    history.save("t1", thread)
    saved = json.loads((tmp_path / "t1.json").read_text())
    assert "base" not in saved["turns"][1] and "tested" not in saved["turns"][1]
    turns = history.load("t1")
    assert (turns[1]["parent"], turns[1]["version"], turns[1]["edit"]["part"]) == ("a", 2, "search")
    assert turns[1]["result"].protocol["version"] == 2 and turns[0]["result"].protocol == {}


def test_team_states_and_exports_of_an_edited_version():
    from mes.ui import components as C
    from mes.ui import exports, teams
    v2 = _edited_version()
    states = teams.result_states(v2)
    assert states["planner"][0] == "edited" and states["trials"][0] == states["kg"][0] == "kept"
    assert states["literature"][0] == "done"
    html_ = exports.report_html(v2)
    assert "Protocol version 2, edited by the user from version 1" in html_ and "Versions: v1 designed by MES" in html_
    md_ = C.export_markdown(v2)
    assert "## Protocol" in md_ and md_.index("## Protocol") < md_.index("## References")
    assert "Protocol" not in exports.report_html(_base())
    tracker = teams.TeamTracker(["literature", "trials"], False)
    tracker.update({"type": "stage", "stage": "plan", "status": "edited"})
    tracker.update({"type": "stage", "stage": "trials", "status": "kept"})
    tracker.update({"type": "stage", "stage": "literature", "status": "done", "elapsed": 2.0})
    tracker.update({"type": "stage", "stage": "pdfs", "status": "kept"})
    s = tracker.states()
    assert (s["planner"][0], s["trials"][0], s["literature"][0]) == ("edited", "kept", "done")


# ============================================================================ the page (AppTest)
def _page(tmp_path, monkeypatch, result, setting="on"):
    from streamlit.testing.v1 import AppTest
    from mes.ui import page
    (tmp_path / "fixture.json").write_text(json.dumps(result.to_dict(), default=str))
    monkeypatch.setenv("MES_MONKEY_FIXTURES", str(tmp_path))
    monkeypatch.setenv("MES_MONKEY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MES_HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("MES_MONKEY_BEHAVIOUR", "ok")
    monkeypatch.setattr(config, "PROTOCOL_EDIT", setting)
    monkeypatch.setattr(rwd_tools, "available", lambda: True)
    from mes.ui import team_builder, warmup
    for module, name in [(page, n) for n in ("iter_answer", "iter_rerun", "create_llm", "composer", "_llm_health",
                                             "kg_status")] + [(team_builder, "canvas"),
                                                              (warmup, "start"), (warmup, "ready"),
                                                              (rwd_tools, "edited_study")]:
        monkeypatch.setattr(module, name, getattr(module, name))  # ui_monkey.install() replaces them: undone after
    turn = page._new_turn(result.question, result.mode, result.sources, False, True, [])
    turn.update(status="done", result=result, shown=True)
    at = AppTest.from_file(str(Path(page.__file__).parents[1] / "scripts" / "ui_monkey_app.py"), default_timeout=60)
    at.session_state["mes_no_kick"] = True
    at.session_state["mes_thread"] = [turn]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at, turn


def _page_result():
    analyst_study = {"key": "an1", "kind": "analyst", "status": "done", "evidence_id": "R2", "llm": "fake",
                     "code": "d = sql('SELECT 1 AS n')\nsubmit({'title': 't', 'estimates': []})", "cells": [], "steps": [],
                     "review": {"verdict": "sound"}, "reproduction": {"reproduced": True}, "timings": {"total": 3},
                     "spec": {"analysis": "custom", "base": "icu", "title": "t"},
                     "result": {"analysis": "custom", "title": "custom", "estimates": [], "warnings": [], "tables": {}}}
    return _base([_explore_study(), analyst_study])


def test_every_protocol_editor_is_locked_until_its_switch_is_turned_on(tmp_path, monkeypatch):
    at, turn = _page(tmp_path, monkeypatch, _page_result())
    switches = {t.key: t for t in at.toggle if t.key and t.key.startswith("mes_lock_")}
    tid = turn["id"]
    assert set(switches) == {f"mes_lock_{tid}_search", f"mes_lock_{tid}_sql_ex1", f"mes_lock_{tid}_design_ex1",
                             f"mes_lock_{tid}_code_an1"}
    assert all(t.value is False and t.label == "🔒 Locked" for t in switches.values())
    buttons = [b for b in at.button if b.key and b.key.startswith("mes_lock_")]
    assert len(buttons) == 4 and all(b.disabled for b in buttons)
    at.toggle(key=f"mes_lock_{tid}_sql_ex1").set_value(True)
    at.run()
    assert at.toggle(key=f"mes_lock_{tid}_sql_ex1").label == "🔓 Unlocked"
    assert not at.button(key=f"mes_lock_{tid}_sql_ex1_go").disabled and at.button(key=f"mes_lock_{tid}_code_an1_go").disabled


@pytest.mark.parametrize("setting, locked", [("off", {"search", "sql", "design", "code"}), ("no_code", {"code"})])
def test_a_server_can_keep_editors_locked(tmp_path, monkeypatch, setting, locked):
    at, turn = _page(tmp_path, monkeypatch, _page_result(), setting)
    disabled = {t.key.split("_")[3] for t in at.toggle if t.key and t.key.startswith("mes_lock_") and t.disabled}
    assert disabled == locked, disabled


def test_an_edit_becomes_a_new_version_and_the_edited_answer_stays(tmp_path, monkeypatch):
    at, turn = _page(tmp_path, monkeypatch, _page_result())
    tid = turn["id"]
    at.toggle(key=f"mes_lock_{tid}_sql_ex1").set_value(True)
    at.run()
    at.button(key=f"mes_lock_{tid}_sql_ex1_go").click()
    at.run()
    run = at.button(key=f"mes_test_{tid}_sql_ex1_run")
    assert run.disabled                                                  # not tested yet
    at.text_area(key=f"mes_test_{tid}_sql_ex1_text").set_value("SELECT careunit, count(*) AS n FROM mes.icu_stays "
                                                               "GROUP BY 1 ORDER BY 2")
    at.button(key=f"mes_test_{tid}_sql_ex1_test").click()
    at.run()
    assert not at.button(key=f"mes_test_{tid}_sql_ex1_run").disabled
    at.text_area(key=f"mes_test_{tid}_sql_ex1_text").set_value("SELECT 1 AS failing")       # changed after the test
    at.run()
    assert at.button(key=f"mes_test_{tid}_sql_ex1_run").disabled
    at.text_area(key=f"mes_test_{tid}_sql_ex1_text").set_value("SELECT careunit, count(*) AS n FROM mes.icu_stays "
                                                               "GROUP BY 1 ORDER BY 2")
    at.button(key=f"mes_test_{tid}_sql_ex1_test").click()
    at.run()
    at.button(key=f"mes_test_{tid}_sql_ex1_run").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    thread = at.session_state["mes_thread"]
    assert [t["status"] for t in thread] == ["done", "done"] and thread[0]["result"].protocol == {}
    v2 = thread[1]
    assert (v2["version"], v2["parent"], v2["edit"]["kind"]) == (2, tid, "sql")
    assert v2["result"].protocol["version"] == 2 and v2["result"].rwd_studies[0]["edited"]["kind"] == "sql"
    text = " ".join(m.value for m in at.markdown)
    assert "Protocol v2 · edited by you from v1" in text and "mes_editor" not in at.session_state
    assert any("protocol v2 (edited)" in e.label for e in at.expander) or "protocol v2 (edited)" in text


def _rate_result():
    study = {"key": "rt1", "kind": "template", "status": "done", "evidence_id": "R1", "sql": "SELECT 1",
             "spec": {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3", "label": "Sepsis-3"}],
                      "outcome": "death_28d", "grace_h": 24.0},
             "result": {"analysis": "rate", "title": "28-day mortality", "warnings": [], "tables": {},
                        "estimates": [{"label": "28-day mortality", "measure": "rate", "value": 18.4, "n": 100,
                                       "events": 18, "primary": True}]}}
    return _base([study])


def _open_design(at, tid):
    at.toggle(key=f"mes_lock_{tid}_design_rt1").set_value(True)
    at.run()
    at.button(key=f"mes_lock_{tid}_design_rt1_go").click()
    at.run()
    return next(s for s in at.selectbox if s.label == "Analysis")


def test_the_design_editor_asks_what_each_analysis_needs(tmp_path, monkeypatch, store):
    """Distribution needs a variable (a list of the population's numeric variables, or a lab / ICU item concept),
    an exploratory query needs its question: the editor asks for them instead of refusing the design."""
    at, turn = _page(tmp_path, monkeypatch, _rate_result())
    tid = turn["id"]
    _open_design(at, tid).set_value("distribution")
    at.run()
    variable = at.selectbox(key="mes_design_var_rt1")
    assert "SOFA score, first 24 h" in variable.options and "Sex" not in variable.options    # numeric variables only
    at.text_input(key="mes_design_var_concept_rt1").set_value("lab:99999999")
    at.run()
    at.button(key="mes_design_run_rt1").click()
    at.run()
    assert any("not a lab test or ICU item" in e.value for e in at.error)          # an unknown concept is refused
    at.text_input(key="mes_design_var_concept_rt1").set_value("var:lactate_max")   # as the concept search lists it
    variable.set_value("sofa")
    at.run()
    at.button(key="mes_design_run_rt1").click()
    at.run()
    assert not at.exception and not [e.value for e in at.error if "distribution" in e.value]
    thread = at.session_state["mes_thread"]
    spec = thread[-1]["edit"]["spec"]
    assert (spec["analysis"], spec["variable"]) == ("distribution", "lactate_max") and thread[-1]["version"] == 2

    at, turn = _page(tmp_path, monkeypatch, _rate_result())
    tid = turn["id"]
    _open_design(at, tid).set_value("explore")
    at.run()
    at.button(key="mes_design_run_rt1").click()
    at.run()
    assert any("describe the question" in e.value for e in at.error)              # asked, not silently refused
    at.text_input(key="mes_design_question_rt1").set_value("How many ICU stays per ICU type?")
    at.run()
    at.button(key="mes_design_run_rt1").click()
    at.run()
    edit = at.session_state["mes_thread"][-1]["edit"]
    assert edit["spec"]["sql_question"] == "How many ICU stays per ICU type?"
    assert "Question of the exploratory query" in [c["what"] for c in PT.design_changes(
        {**_rate_result().rwd_studies[0]["spec"], "analysis": "explore"}, edit["spec"])]
