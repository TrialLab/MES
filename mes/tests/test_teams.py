"""
Offline tests of the agent team builder's logic (ui/teams.py and ui/team_graph.js): templates, graph rules, the
compiler, saved teams, the team state shown while a question runs, and that the browser's copy of the rules gives the
same results as Python's (needs Node; skipped without it).

    python -m pytest mes/tests/test_teams.py -q
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # repository root

from mes import config  # noqa: E402
from mes.tools import pipeline_tools, planner_tools, synthesis_tools  # noqa: E402
from mes.tools.rwd import analyst, designer, explorer  # noqa: E402
from mes.ui import team_canvas  # noqa: E402
from mes.ui import teams  # noqa: E402

# the browser runs the shared graph plus the MES compiler, in this order
GRAPH_JS = [Path(team_canvas.__file__).with_name("graph.js"), Path(teams.__file__).with_name("team_graph.js")]


def edges(graph):
    return {(e["from"], e["to"]) for e in graph["edges"]}


def types(graph):
    return [n["type"] for n in graph["nodes"]]


# ============================================================================ templates
EXPECTED = {
    "builtin:standard": (["literature", "trials", "kg", "rwd"], "osler", False),
    "builtin:appraisal": (["literature", "trials", "kg"], "sackett", True),
    "builtin:comprehensive": (["literature", "trials", "kg"], "snow", False),
    "builtin:rwe": (["literature", "trials", "rwd"], "sackett", False),
    "builtin:literature_only": (["literature"], "osler", False),
    "builtin:mechanism": (["literature", "kg"], "osler", False),
}


def test_builtin_templates_compile_to_their_settings():
    templates = teams.builtin_templates()
    assert [t["id"] for t in templates] == list(EXPECTED)
    for t in templates:
        c = teams.compile_team(t["graph"])
        sources, mode, verify = EXPECTED[t["id"]]
        assert c["ok"] and not c["messages"], t["id"]
        assert (c["sources"], c["mode"], c["verify"], c["rwd_engine"], c["use_cache"]) == \
            (sources, mode, verify, "auto", True), t["id"]
        assert t["graph"] == teams.normalize(t["graph"])           # templates are normalized graphs
        assert t["graph"]["name"] == t["name"]


def test_default_team_is_the_pipelines_default():
    c = teams.compile_team(teams.default_graph())
    assert c["sources"] == list(pipeline_tools.DEFAULT_SOURCES)
    assert c["mode"] == config.DEFAULT_MODE and not c["verify"]


def test_templates_use_the_default_layout_without_overlaps():
    for t in teams.builtin_templates():
        boxes = [(n["x"], n["y"]) for n in t["graph"]["nodes"]]
        for i, (x1, y1) in enumerate(boxes):
            for x2, y2 in boxes[i + 1:]:
                assert abs(x1 - x2) >= teams.NODE_W or abs(y1 - y2) >= 70, t["id"]


# ============================================================================ normalize
GARBAGE = [None, [], "team", 5, True, {"nodes": "x", "edges": 3}, {"nodes": [None, 1, "a", [], {"type": 3}]},
           {"depth": "turbo", "use_cache": "yes", "name": 42, "description": ["x"]},
           {"nodes": [{"type": "planner", "x": float("nan"), "y": float("inf")}, {"type": "planner", "x": 1, "y": 2},
                      {"type": "__proto__"}, {"type": "statistician"}, {"type": "rwd", "x": 10 ** 400, "y": -1e9,
                                                                         "tools": {"templates": "no", "analyst": False}}],
            "edges": [["question", "planner"], ["planner", "planner"], ["answer", "question"], {"from": "rwd"},
                      "ab", ["planner", "rwd", "x"], {"from": "planner", "to": "rwd"}, {"from": "planner", "to": "rwd"},
                      ["rwd", "synthesizer"], ["synthesizer", "answer"]]}]


@pytest.mark.parametrize("raw", GARBAGE)
def test_normalize_survives_garbage(raw):
    g = teams.normalize(raw)
    assert set(teams.CORE) <= set(types(g))
    assert len(types(g)) == len(set(types(g))) and set(types(g)) <= set(teams.TYPES)
    assert types(g) == [t for t in teams.TYPES if t in types(g)]               # canonical order
    assert g["depth"] in config.MODES and isinstance(g["use_cache"], bool) and g["name"]
    for n in g["nodes"]:
        assert isinstance(n["x"], int) and isinstance(n["y"], int)
        assert abs(n["x"]) <= teams.COORD_LIMIT and abs(n["y"]) <= teams.COORD_LIMIT
        assert set(n["tools"]) == set(teams.TOGGLES.get(n["type"], {}))
        assert all(isinstance(v, bool) for v in n["tools"].values())
    for a, b in edges(g):
        assert teams.compatible(a, b)
    targets = [e["to"] for e in g["edges"] if e["to"] not in teams.MULTI_INPUT]
    assert len(targets) == len(set(targets))                                     # one edge per single input
    assert teams.normalize(g) == g                                               # idempotent
    teams.compile_team(g)


def test_normalize_keeps_valid_parts_of_a_damaged_graph():
    g = teams.normalize(GARBAGE[-1])
    assert types(g) == ["question", "planner", "rwd", "synthesizer", "grounding", "answer"]
    assert edges(g) == {("question", "planner"), ("planner", "rwd"), ("rwd", "synthesizer")}
    assert next(n for n in g["nodes"] if n["type"] == "planner")["x"] == teams.layout(types(g))["planner"][0]
    rwd = next(n for n in g["nodes"] if n["type"] == "rwd")
    assert rwd["tools"] == {"templates": True, "analyst": False}
    assert (rwd["x"], rwd["y"]) == teams.layout(types(g))["rwd"]          # one bad coordinate: the default spot
    far = teams.normalize({"nodes": [{"type": "kg", "x": 12.5, "y": -1e9}]})
    assert next((n["x"], n["y"]) for n in far["nodes"] if n["type"] == "kg") == (13, -teams.COORD_LIMIT)


def test_text_fields_are_one_short_line():
    g = teams.normalize({"name": "  My\tteam\n\n name  " + "x" * 200, "description": "a\r\nb"})
    assert g["name"].startswith("My team name") and len(g["name"]) == teams.MAX_NAME
    assert g["description"] == "a b"
    assert teams.normalize({"name": "   "})["name"] == "Untitled team"


# ============================================================================ wiring
def test_evidence_agents_are_wired_between_planner_and_synthesizer():
    g = teams.empty_graph()
    g = teams.add_node(g, "rwd")
    assert {("planner", "rwd"), ("rwd", "synthesizer")} <= edges(g)
    assert teams.add_node(g, "rwd") == g                                         # one of each agent
    g = teams.remove_node(g, "rwd")
    assert "rwd" not in types(g) and not any("rwd" in e for e in edges(g))


def test_safechecker_is_inserted_before_the_answer_and_removed_cleanly():
    g = teams.add_node(teams.empty_graph(), "safechecker")
    assert ("grounding", "answer") not in edges(g)
    assert {("grounding", "safechecker"), ("safechecker", "answer")} <= edges(g)
    g = teams.remove_node(g, "safechecker")
    assert ("grounding", "answer") in edges(g)


def test_core_agents_cannot_be_removed():
    g = teams.default_graph()
    for t in teams.CORE:
        assert teams.remove_node(g, t) == g


def test_connect_checks_port_kinds_and_replaces_single_inputs():
    g = teams.default_graph()
    assert teams.connect(g, "planner", "answer")[1] == "incompatible"
    assert teams.connect(g, "planner", "planner")[1] == "incompatible"
    assert teams.connect(teams.remove_node(g, "rwd"), "planner", "rwd")[1] == "missing"   # not in the team
    g2, err = teams.connect(teams.add_node(g, "safechecker"), "grounding", "answer")     # bypass the checker
    assert err is None and ("grounding", "answer") in edges(g2) and ("safechecker", "answer") not in edges(g2)
    c = teams.compile_team(g2)
    assert not c["verify"] and [m["code"] for m in c["messages"]] == ["safechecker_off"]
    g3 = teams.disconnect(g, "kg", "synthesizer")
    assert teams.compile_team(g3)["sources"] == ["literature", "trials", "rwd"]


# ============================================================================ compiler
def test_rwd_tools_choose_the_engine():
    g = teams.add_node(teams.empty_graph(), "rwd")
    for templates, analyst_on, engine in ((True, True, "auto"), (True, False, "templates"), (False, True, "analyst")):
        g2 = teams.set_tool(teams.set_tool(g, "rwd", "templates", templates), "rwd", "analyst", analyst_on)
        c = teams.compile_team(g2)
        assert c["ok"] and c["rwd_engine"] == engine and c["sources"] == ["rwd"]
    none = teams.set_tool(teams.set_tool(g, "rwd", "templates", False), "rwd", "analyst", False)
    c = teams.compile_team(none)
    assert not c["ok"] and [m["code"] for m in c["messages"]] == ["rwd_no_tool"]


def test_availability_turns_into_warnings_not_errors():
    g = teams.add_node(teams.default_graph(), "rwd")
    c = teams.compile_team(g, {"rwd": False, "analyst": False})
    assert c["ok"] and [m["code"] for m in c["messages"]] == ["rwd_unavailable"]
    c = teams.compile_team(g, {"rwd": True, "analyst": False})
    assert c["ok"] and [(m["code"], m["level"]) for m in c["messages"]] == [("analyst_unavailable", "info")]
    templates_only = teams.set_tool(g, "rwd", "analyst", False)
    assert not teams.compile_team(templates_only, {"rwd": True, "analyst": False})["messages"]


def test_broken_chain_is_reported_on_the_right_agents():
    g = teams.add_node(teams.empty_graph(), "kg")
    g = teams.disconnect(teams.disconnect(teams.disconnect(g, "question", "planner"), "kg", "synthesizer"),
                         "grounding", "answer")
    g = teams.disconnect(g, "synthesizer", "grounding")
    c = teams.compile_team(g)
    codes = {(m["code"], m["node"], m["level"]) for m in c["messages"]}
    assert codes == {("no_question", "planner", "error"), ("no_evidence", "synthesizer", "error"),
                     ("not_connected", "kg", "warning"), ("no_grounding", "grounding", "error"),
                     ("no_answer", "answer", "error")}
    assert not c["ok"] and c["sources"] == []
    assert any(m["text"] == "KG Specialist is not connected: it will not run." for m in c["messages"])


def test_matches_compares_the_settings_of_a_question_box():
    cfg = teams.compile_team(teams.template("builtin:rwe")["graph"])
    assert teams.matches(cfg, ["rwd", "trials", "literature"], "sackett", False, True, "auto")
    assert not teams.matches(cfg, ["literature", "trials"], "sackett")
    assert not teams.matches(cfg, ["literature", "trials", "rwd"], "osler")
    assert not teams.matches(cfg, ["literature", "trials", "rwd"], "sackett", verify=True)
    assert not teams.matches(cfg, ["literature", "trials", "rwd"], "sackett", rwd_engine="templates")
    plain = teams.compile_team(teams.build_graph("Plain", "", ("literature", "trials", "kg"), "osler"))
    assert teams.matches(plain, ["literature", "trials", "kg"], "osler", rwd_engine="analyst")   # engine unused
    assert not teams.matches(None, ["literature"], "osler")


# ============================================================================ saved teams
def test_save_list_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("MES_TEAMS_DIR", str(tmp_path))
    g = teams.normalize({**teams.template("builtin:rwe")["graph"], "name": "Sepsis RWE team"})
    team_id = teams.save(g)
    assert team_id == "saved:sepsis-rwe-team"
    saved = teams.list_saved()
    assert [t["name"] for t in saved] == ["Sepsis RWE team"] and saved[0]["graph"] == g
    teams.save({**g, "depth": "snow"})                                # same name: replaced
    assert [t["graph"]["depth"] for t in teams.list_saved()] == ["snow"]
    builtin = teams.save({**g, "name": "MES standard"})              # never shadows a built-in template
    assert teams.template(builtin)["name"] == "MES standard (custom)"
    assert teams.template("builtin:standard")["builtin"]
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert len(teams.list_saved()) == 2                               # a damaged file is skipped
    for bad in ("builtin:standard", "saved:../x", "saved:", None, 5, "saved:A B"):
        assert teams.delete(bad) is False
    assert teams.delete(team_id) and not teams.delete(team_id)
    assert [t["name"] for t in teams.list_saved()] == ["MES standard (custom)"]


# ============================================================================ team state of a run
def test_tracker_follows_the_stage_events():
    tr = teams.TeamTracker(["literature", "rwd"], True)
    assert tr.agents == ["planner", "literature", "rwd", "synthesizer", "grounding", "safechecker"]
    assert set(s for s, _ in tr.states().values()) == {"waiting"}
    assert tr.update({"type": "stage", "stage": "plan", "status": "running"})
    assert tr.update({"type": "stage", "stage": "plan", "status": "done", "elapsed": 4.2})
    tr.update({"type": "stage", "stage": "literature", "status": "running"})
    tr.update({"type": "stage", "stage": "pdfs", "status": "done", "elapsed": 0.5})
    tr.update({"type": "stage", "stage": "rwd", "status": "skipped", "elapsed": 0.1})
    s = tr.states()
    assert s["planner"] == ("done", 4.2) and s["literature"] == ("running", None) and s["rwd"][0] == "skipped"
    tr.update({"type": "stage", "stage": "literature", "status": "done", "elapsed": 1.5})
    tr.update({"type": "stage", "stage": "synthesis", "status": "done", "elapsed": 20})
    s = tr.states()
    assert s["literature"] == ("done", 1.5) and s["grounding"][0] == "done" and s["safechecker"][0] == "waiting"
    assert not tr.update({"type": "log", "stage": "plan", "message": "x"})
    assert not tr.update({"type": "stage", "stage": "something", "status": "done"})
    cached = teams.TeamTracker(["kg"], False)
    cached.update({"type": "stage", "stage": "cache", "status": "done"})
    assert set(cached.states().values()) == {("cached", None)}


def test_result_states_and_status_card():
    result = SimpleNamespace(sources=["literature", "trials", "rwd"], verification={"llm_verified": 3},
                             timings={"plan": 4.0, "literature": 1.2, "pdfs": 2.5, "trials": 0.4, "synthesis": 18,
                                      "verify": 6.1},
                             source_status={"trials": {"ok": False}, "rwd": {"ok": True, "available": False}},
                             from_cache=False)
    s = teams.result_states(result)
    assert s["literature"] == ("done", 2.5) and s["trials"][0] == "error" and s["rwd"][0] == "skipped"
    assert s["safechecker"] == ("done", 6.1)
    html = teams.status_html(s, "sackett", "<script>alert(1)</script>")
    assert "Agent team · Balanced" in html and "&lt;script&gt;" in html and "<script>" not in html
    assert html.count("mes-team-ev") == 3 and "2.5 s" in html and "18 s" in html
    pending = SimpleNamespace(**{**vars(result), "source_status": {"rwd": {"ok": None, "pending": True}}})
    assert teams.result_states(pending)["rwd"][0] == "running"
    old = SimpleNamespace(sources=["literature"], verification={}, timings={}, source_status={}, mode="osler")
    assert teams.status_html(teams.result_states(old), "bogus")                  # answers of older versions
    rail = teams.rail_html(s, "sackett", "<b>team</b>")
    assert rail.count("mes-team-chip ") == 7 and "&lt;b&gt;team" in rail and "<b>" not in rail
    assert "mes-team-error" in rail and "mes-team-skipped" in rail and "2.5 s" in rail and "6.1 s" in rail
    assert teams.rail_html(teams.result_states(old), None)


# ============================================================================ catalogue
def test_catalog_shows_the_real_prompts_and_settings():
    cat = teams.catalog()
    agents = cat["agents"]
    assert list(agents) == list(teams.TYPES)
    prompts = {p["label"]: p for a in agents.values() for p in a["prompts"]}
    assert prompts["Planner prompt"]["text"] == planner_tools.PLAN_PROMPT
    assert prompts["Rules"]["text"] == synthesis_tools.RULES
    assert prompts["Style"]["by_depth"] == synthesis_tools.MODE_STYLE
    assert prompts["Fact-check prompt"]["text"] == synthesis_tools.VERIFY_PROMPT
    assert prompts["Study designer prompt"]["text"] == designer.PROMPT
    assert prompts["LLM analyst prompt"]["text"] == analyst.SYSTEM
    for t, a in agents.items():
        assert a["title"] == teams.TITLES[t] and a["ports"] == list(teams.PORTS[t]) and a["core"] == (t in teams.CORE)
        for tool in a["tools"]:
            assert tool["kind"] in ("toggle", "locked", "depth") and tool["key"]
            if tool["kind"] != "toggle":
                assert set(tool["on"]) == set(config.MODES)
        toggles = {tool["key"] for tool in a["tools"] if tool["kind"] == "toggle"}
        assert toggles == set(teams.TOGGLES.get(t, {}))                           # only real switches are editable
        for b in a["budgets"]:
            assert b["key"] and set(b["values"]) == set(config.MODES)
    lit = {b["key"]: b["values"] for b in agents["literature"]["budgets"]}
    assert lit["literature_max"] == {m: c.literature_max for m, c in config.MODES.items()}
    assert prompts["Design repair prompt"]["text"] == designer.REPAIR
    assert prompts["Exploratory SQL prompt"]["text"] == explorer.PROMPT
    assert prompts["Answer format"]["by_depth"]["sackett"] == synthesis_tools._format_instructions(
        "sackett", SimpleNamespace(facets=[{"name": "{facet}"}]), list(teams.EVIDENCE))
    assert [a["key"] for a in cat["locked_agents"]] == ["statistician"]
    assert cat["locked_agents"][0]["prompts"][0]["text"] == synthesis_tools.EXTRACT_PROMPT
    for t in ("literature", "trials", "kg", "grounding"):                         # strategy / rules instead of a prompt
        assert agents[t]["panel"] and agents[t]["prompts"][0]["text"].strip()
    for t in ("literature", "trials", "kg"):
        assert agents[t]["model"].endswith(config.ENCODER.split("/")[-1]) and agents[t]["model_role"]
    json.dumps(cat)                                                              # sent to the browser


def test_the_builder_never_belittles_an_agent():
    """No "deterministic", "0 LLM calls" or "no language model" anywhere the user can read it."""
    texts = [json.dumps(teams.catalog(), ensure_ascii=False).lower(),
             *[p.read_text(encoding="utf-8").lower() for p in GRAPH_JS],
             (Path(team_canvas.__file__).with_name("canvas.js")).read_text(encoding="utf-8").lower(),
             teams.status_html({a: ("done", 1.0) for a in teams.run_agents(list(teams.EVIDENCE), True)}).lower(),
             teams.rail_html({a: ("done", 1.0) for a in teams.run_agents(list(teams.EVIDENCE), True)}).lower()]
    for text in texts:
        for phrase in ("deterministic", "0 llm", "0 language-model", "no language model", "reproducible"):
            assert phrase not in text, phrase


def test_strategies_match_the_code():
    """The search strategies show the code's real filters and weights; a change in the code must be shown too."""
    import inspect
    from mes.tools import ibkh_tools, literature_tools, ranking_tools, trials_tools
    rules = teams.strategies()
    lit = rules["literature"]
    assert literature_tools.HIGH_EVIDENCE_FILTER in lit and literature_tools.TRIAL_FILTER in lit
    assert f"screened {literature_tools.SCREEN_PMIDS} deep" in lit and f"its {literature_tools.TOP_CITED} most" in lit
    rank = inspect.getsource(literature_tools.rank_documents)
    for code in ("0.52 * best[i]", "0.13 * DESIGN_PRIOR", "0.12 * impact", "0.15 * rank_prior", "0.08 * recency",
                 "best_dense[i] >= 0.25"):
        assert code in rank, code
    for shown in ("0.52 × passage", "0.13 × study design", "0.12 × citations", "0.15 × PubMed", "0.08 × recency",
                  "relevance < 0.25"):
        assert shown in lit, shown
    assert inspect.signature(ranking_tools.hybrid_scores).parameters["dense_weight"].default == 0.65
    assert "0.65 dense + 0.35 TF-IDF" in lit and "0.65 dense + 0.35 TF-IDF" in rules["trials"]
    trials = inspect.getsource(trials_tools.search_trials)
    for code in ("0.7 * scores", '0.15 * r["has_results"]', '0.05 * (r["status"] == "COMPLETED")', "0.1 * any(",
                 '0.05 * (r["allocation"] == "RANDOMIZED")', "0.05 * min(1.0", "dense[i] >= 0.25",
                 "max(10, 2 * mode.trials_max)"):
        assert code in trials, code
    assert inspect.signature(trials_tools.fetch_results).parameters["max_measures"].default == 3
    assert "Up to 3 outcome measures" in rules["trials"]
    kg = inspect.getsource(ibkh_tools.IBKHGraph.retrieve) + inspect.getsource(ibkh_tools.IBKHGraph.link_entities)
    for code in ('("direct", [e], 1.0', '("semantic-link", [e], 0.7', "p[\"edges\"], 0.6", '("neighborhood", [e], 0.0',
                 "0.55 * sim[text] + 0.3 * weight + 0.15 * bonus", "score >= 0.8", "k // 3"):
        assert code in kg, code
    for label, reliability in ibkh_tools.CLASS_RELIABILITY.items():
        assert f"{ibkh_tools.CLASS_LABEL[label].lower()} → {reliability}" in rules["kg"]


# ============================================================================ the browser's rules = Python's
_HARNESS = r"""
const fs = require("fs");
const src = process.argv.slice(2).map((f) => fs.readFileSync(f, "utf8")).join("\n");
const TeamGraph = new Function(src + "\nreturn TeamGraph;")();
const input = JSON.parse(fs.readFileSync(0, "utf8"));
TeamGraph.configure(input.rules);
const cases = input.cases.map((c) => ({ normalized: TeamGraph.normalize(c.graph),
                                        compiled: TeamGraph.compile(c.graph, c.availability),
                                        layout: TeamGraph.layout(c.layout) }));
const sequences = input.sequences.map((s) => {
  let g = s.start;
  const states = [];
  for (const op of s.ops) {
    let err = null;
    if (op[0] === "add") g = TeamGraph.addNode(g, op[1], op[2], op[3]);
    else if (op[0] === "remove") g = TeamGraph.removeNode(g, op[1]);
    else if (op[0] === "connect") [g, err] = TeamGraph.connect(g, op[1], op[2]);
    else if (op[0] === "disconnect") g = TeamGraph.disconnect(g, op[1], op[2]);
    else if (op[0] === "tool") g = TeamGraph.setTool(g, op[1], op[2], op[3]);
    states.push({ graph: g, err, compiled: TeamGraph.compile(g, s.availability) });
  }
  return states;
});
process.stdout.write(JSON.stringify({ cases, sequences }));
"""

NAMES = ["", "  Sepsis   team ", "Ünïcödé 团队 🧪🧪", "x" * 300, "\ttabs\nand\rreturns\f", " nbsp ", 7,
         None, "emoji " + "😀" * 90]
ALL_TYPES = list(teams.TYPES) + ["statistician", "bogus", "__proto__", "constructor"]


def _random_graph(rng):
    nodes = []
    for _ in range(rng.randint(0, 14)):
        node = {"type": rng.choice(ALL_TYPES)}
        for axis in ("x", "y"):
            node[axis] = rng.choice([rng.randint(-5000, 5000), rng.uniform(-4500, 4500), rng.randint(-50, 50) + 0.5,
                                     None, "12", True, 1e300, -1e300])
        if rng.random() < 0.6:
            node["tools"] = {"templates": rng.choice([True, False, "yes", None]),
                             "analyst": rng.choice([True, False, 0, 1]), "extra": True}
        nodes.append(rng.choice([node, node, node, None, 3, "planner"]))
    pairs = [[rng.choice(ALL_TYPES), rng.choice(ALL_TYPES)] for _ in range(rng.randint(0, 25))]
    pairs += [[a, b] for a in teams.TYPES for b in teams.TYPES if rng.random() < 0.08]
    edge_list = [rng.choice([p, {"from": p[0], "to": p[1]}, p + ["x"], {"from": p[0]}]) for p in pairs]
    rng.shuffle(edge_list)
    return {"name": rng.choice(NAMES), "description": rng.choice(NAMES),
            "depth": rng.choice(list(config.MODES) + ["turbo", None, 3]),
            "use_cache": rng.choice([True, False, "true", None]), "nodes": nodes, "edges": edge_list}


def _random_ops(rng, n):
    ops = []
    for _ in range(n):
        kind = rng.choice(["add", "add", "remove", "connect", "connect", "disconnect", "tool"])
        if kind == "add":
            ops.append(["add", rng.choice(ALL_TYPES)] + ([rng.randint(-900, 900), rng.randint(-200, 900)]
                                                         if rng.random() < 0.5 else [None, None]))
        elif kind == "remove":
            ops.append(["remove", rng.choice(ALL_TYPES)])
        elif kind in ("connect", "disconnect"):
            ops.append([kind, rng.choice(teams.TYPES), rng.choice(teams.TYPES)])
        else:
            ops.append(["tool", rng.choice(["rwd", "kg"]), rng.choice(["templates", "analyst", "x"]),
                        rng.random() < 0.5])
    return ops


def _availability(rng):
    return rng.choice([None, {}, {"rwd": False}, {"rwd": True, "analyst": False}, {"rwd": True, "analyst": True}])


def _python_sequence(start, ops, availability):
    g, states = start, []
    for op in ops:
        err = None
        if op[0] == "add":
            g = teams.add_node(g, op[1], op[2], op[3])
        elif op[0] == "remove":
            g = teams.remove_node(g, op[1])
        elif op[0] == "connect":
            g, err = teams.connect(g, op[1], op[2])
        elif op[0] == "disconnect":
            g = teams.disconnect(g, op[1], op[2])
        else:
            g = teams.set_tool(g, op[1], op[2], op[3])
        states.append({"graph": g, "err": err, "compiled": teams.compile_team(g, availability)})
    return states


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_python_and_javascript_agree(tmp_path):
    rng = random.Random(20261027)
    checked = teams.template("builtin:appraisal")["graph"]
    competing = [{**checked, "edges": checked["edges"] + [{"from": "grounding", "to": "answer"}]},     # two inputs
                 {**checked, "edges": [{"from": "grounding", "to": "answer"}] + checked["edges"]},     # of the Answer
                 {**checked, "edges": checked["edges"] * 2}]                                           # duplicates
    graphs = ([t["graph"] for t in teams.builtin_templates()] + GARBAGE + competing
              + [_random_graph(rng) for _ in range(400)])
    cases = [{"graph": g, "availability": _availability(rng),
              "layout": rng.sample(list(teams.TYPES), rng.randint(0, len(teams.TYPES)))} for g in graphs]
    starts = [t["graph"] for t in teams.builtin_templates()] + [teams.empty_graph(), _random_graph(rng)]
    sequences = [{"start": rng.choice(starts), "ops": _random_ops(rng, 30), "availability": _availability(rng)}
                 for _ in range(120)]
    # what JSON can carry (the browser never sees NaN or infinities: JSON.stringify writes null)
    payload = json.loads(json.dumps({"rules": teams.RULES, "cases": cases, "sequences": sequences},
                                    allow_nan=True).replace("NaN", "null").replace("-Infinity", "null")
                         .replace("Infinity", "null"))
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(harness), *map(str, GRAPH_JS)], input=json.dumps(payload),
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    js = json.loads(run.stdout)
    for case, out in zip(payload["cases"], js["cases"]):
        assert out["normalized"] == teams.normalize(case["graph"]), case["graph"]
        assert out["compiled"] == teams.compile_team(case["graph"], case["availability"]), case["graph"]
        assert {k: list(v) for k, v in teams.layout(case["layout"]).items()} == out["layout"], case["layout"]
    for seq, states in zip(payload["sequences"], js["sequences"]):
        py = _python_sequence(seq["start"], seq["ops"], seq["availability"])
        for i, (a, b) in enumerate(zip(py, states)):
            assert a == b, (seq["ops"][:i + 1])
    # the cases cover what they should: valid and broken teams, every agent, warnings and errors
    compiled = [teams.compile_team(c["graph"], c["availability"]) for c in payload["cases"]] + [
        s["compiled"] for seq in js["sequences"] for s in seq]
    assert sum(c["ok"] for c in compiled) > 200 and sum(not c["ok"] for c in compiled) > 200
    assert {m["code"] for c in compiled for m in c["messages"]} == set(teams.MESSAGES)
    assert {s for c in compiled for s in c["sources"]} == set(teams.EVIDENCE)
    assert len(js["cases"]) == len(cases)


def test_the_depth_buttons_show_the_times_of_the_question_box():
    """Above each depth name the builder shows its typical time: the same strings as the question box's menu."""
    from mes.ui import page
    assert teams.MODE_LEVELS == {k: v[1] for k, v in page.MODE_MENU.items()}
    assert teams.RULES["mode_levels"] == teams.MODE_LEVELS
