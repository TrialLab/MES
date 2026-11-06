"""
Agent teams of MES: the agents the team builder offers (``ui/team_builder.py``), the built-in team templates, saved
teams, and the compiler that turns a team graph into the settings of a run.

A team graph is ``{"version", "name", "description", "depth", "use_cache", "nodes", "edges"}``. Every agent appears
at most once and its node id is its type. Edges join an output port to an input port of the same kind:

    Question → Query Planner → evidence agents (Literature, Trials, iBKH, RWD) → Evidence Synthesizer
             → Citation Check → (SafeChecker) → Answer

The pipeline itself is code (``tools/pipeline_tools.py``). A team chooses which agents take part and the switches the
pipeline has:

* the evidence agents wired between the Planner and the Synthesizer → ``sources``;
* SafeChecker between the check and the Answer → ``verify``;
* the RWD Analyst's two tools → ``rwd_engine``;
* the depth → ``mode``;
* reuse of cached answers → ``use_cache``.

Settings the pipeline does not expose yet (prompts, budgets, models, single tools) are part of the catalogue as
locked fields. Each has its key, so it can be unlocked when the backend accepts it.

The graph itself knows nothing of MES: ``team_canvas.Graph`` runs on the rules below, and the browser runs the same
logic on the same rules (``team_canvas/graph.js`` plus the MES compiler in ``team_graph.js``).
``tests/test_teams.py`` checks that both give the same results.
"""

import inspect
import json
import logging
import math
import os
import re
import time
from html import escape
from pathlib import Path

from mes.ui.team_canvas import Graph
from mes import config
from mes.tools.rwd import DATABASE

logger = logging.getLogger(__name__)

GRAPH_VERSION = 1
EVIDENCE = ("literature", "trials", "kg", "rwd")          # in the order of the pipeline's sources
# Canonical node order (also the order of nodes in a normalized graph)
TYPES = ("question", "planner", "literature", "trials", "kg", "rwd", "synthesizer", "grounding", "safechecker",
         "answer")
CORE = ("question", "planner", "synthesizer", "grounding", "answer")     # always in a team, cannot be removed
# Port kinds: (input, output); None = no port on that side
PORTS = {"question": (None, "question"), "planner": ("question", "plan"),
         **{t: ("plan", "evidence") for t in EVIDENCE},
         "synthesizer": ("evidence", "draft"), "grounding": ("draft", "checked"), "safechecker": ("checked", "checked"),
         "answer": ("checked", None)}
MULTI_INPUT = ("synthesizer",)          # every other input port takes one edge (a new edge replaces the old one)
PORT_LABELS = {"question": "Question", "plan": "Query plan", "evidence": "Evidence", "draft": "Draft answer",
               "checked": "Checked answer"}
# Tools the user can switch: type → {tool: default}
TOGGLES = {"rwd": {"templates": True, "analyst": True}}
MAX_NODES, MAX_EDGES = 50, 100
MAX_NAME, MAX_DESCRIPTION = 80, 400
COORD_LIMIT = 4000

# Layout (top to bottom): node width, gap between evidence agents and the y of each row ("parallel" = the evidence row)
NODE_W = 204
GAP_X = 22
ROW_Y = {"question": 0, "planner": 74, "parallel": 190, "synthesizer": 306, "grounding": 422, "safechecker": 538,
         "answer": 654}

MESSAGES = {
    "no_question": "Connect the Question to the Query Planner.",
    "no_evidence": "Connect at least one evidence agent between the Query Planner and the Evidence Synthesizer.",
    "not_connected": "{agent} is not connected: it will not run.",
    "no_grounding": "Connect the Evidence Synthesizer to the Citation Check.",
    "no_answer": "Connect the checks to the Answer.",
    "safechecker_off": "SafeChecker is not connected: cited claims will not be checked by the language model.",
    "rwd_no_tool": "Give the RWD Analyst at least one tool: study templates or the LLM analyst.",
    "rwd_unavailable": f"The {DATABASE} store is not built on this computer: the RWD Analyst will be skipped.",
    "analyst_unavailable": "On this computer the RWD Analyst uses the study templates (the LLM analyst runs where it "
                           "is switched on: MES_RWD_ANALYST).",
}

TITLES = {"question": "Question", "planner": "Query Planner", "literature": "Literature Miner",
          "trials": "Trials Agent", "kg": "KG Specialist", "rwd": "RWD Analyst",
          "synthesizer": "Evidence Synthesizer", "grounding": "Citation Check", "safechecker": "SafeChecker",
          "answer": "Answer"}

# shown above the depth names on the depth buttons and beside them on the canvas: how long an answer typically
# takes (the times of the question box, ui/page.py MODE_MENU — the names themselves say fast / balanced / deep)
MODE_LEVELS = {"osler": "~30 s", "sackett": "~1 min", "snow": "2–3 min"}

RULES = {"version": GRAPH_VERSION, "types": list(TYPES), "core": list(CORE), "parallel": list(EVIDENCE),
         "ports": {t: list(p) for t, p in PORTS.items()}, "multi_input": list(MULTI_INPUT), "toggles": TOGGLES,
         "modes": list(config.MODES), "default_mode": config.DEFAULT_MODE, "mode_levels": MODE_LEVELS,
         "titles": TITLES, "messages": MESSAGES, "input_node": "question",
         "wire": {"parallel": ["planner", "synthesizer"]},                    # evidence agents are wired between these
         "insert": {"safechecker": {"after": "grounding", "before": "answer"}},
         "chain": [["question", "planner"], ["synthesizer", "grounding"], ["grounding", "answer"]],
         "limits": {"nodes": MAX_NODES, "edges": MAX_EDGES, "name": MAX_NAME, "description": MAX_DESCRIPTION,
                    "coord": COORD_LIMIT},
         "layout": {"node_w": NODE_W, "gap_x": GAP_X, "rows": ROW_Y, "collapse": {"answer": "safechecker"}}}


# ============================================================================ graph
_G = Graph(RULES)                          # the graph machinery; the browser runs the same on the same rules

_text = _G.text
compatible = _G.compatible
layout = _G.layout
normalize = _G.normalize
_with_edge = _G.with_edge
connect = _G.connect
disconnect = _G.disconnect
add_node = _G.add_node
remove_node = _G.remove_node
set_tool = _G.set_tool


def compile_team(graph, availability=None):
    """
    The run settings of a team: {"ok", "sources", "mode", "verify", "rwd_engine", "use_cache", "agents", "messages"}.
    ``availability``: {"rwd": real-world data store built, "analyst": Claude analyst usable} (default: both available).
    ``messages``: [{"level": "error" | "warning" | "info", "code", "node", "text"}]; any error makes the team not
    runnable; a note ("info") only says how this computer will run it.
    """
    ctx = _G.context(graph, availability)
    nodes, edge, say = ctx.nodes, ctx.edge, ctx.say
    rwd_ok = ctx.av.get("rwd", True) is not False
    analyst_ok = ctx.av.get("analyst", True) is not False

    if not edge("question", "planner"):
        say("error", "no_question", "planner")
    connected = ctx.connected
    if not connected:
        say("error", "no_evidence", "synthesizer")
    for t in EVIDENCE:
        if t in nodes and t not in connected:
            say("warning", "not_connected", t)
    if not edge("synthesizer", "grounding"):
        say("error", "no_grounding", "grounding")
    verify = "safechecker" in nodes and edge("grounding", "safechecker") and edge("safechecker", "answer")
    if not verify and not edge("grounding", "answer"):
        say("error", "no_answer", "answer")
    if "safechecker" in nodes and not verify:
        say("warning", "safechecker_off", "safechecker")
    tools = nodes["rwd"]["tools"] if "rwd" in nodes else dict(TOGGLES["rwd"])
    templates, analyst = tools.get("templates", True), tools.get("analyst", True)
    engine = "auto" if templates and analyst else "templates" if templates else "analyst" if analyst else None
    if "rwd" in connected:
        if engine is None:
            say("error", "rwd_no_tool", "rwd")
        elif not rwd_ok:
            say("warning", "rwd_unavailable", "rwd")
        elif engine != "templates" and not analyst_ok:
            say("info", "analyst_unavailable", "rwd")
    agents = ["planner"] + connected + ["synthesizer", "grounding"] + (["safechecker"] if verify else [])
    return {"ok": ctx.ok(), "sources": connected, "mode": ctx.graph["depth"], "verify": verify,
            "rwd_engine": engine or "auto", "use_cache": ctx.graph["use_cache"], "agents": agents,
            "messages": ctx.messages}


def matches(team_config, sources, mode, verify=None, use_cache=None, rwd_engine=None):
    """Settings of a question box are still those of the team (``None``: not compared)."""
    if not team_config:
        return False
    sources = list(sources or [])
    if set(sources) != set(team_config.get("sources") or []) or mode != team_config.get("mode"):
        return False
    if verify is not None and bool(verify) != bool(team_config.get("verify")):
        return False
    if use_cache is not None and bool(use_cache) != bool(team_config.get("use_cache", True)):
        return False
    return rwd_engine is None or "rwd" not in sources or rwd_engine == team_config.get("rwd_engine")


# ============================================================================ templates
_BUILTIN = [
    ("standard", "MES standard", "The default team: published literature, the trial registry, the iBKH knowledge "
     "graph and a new study on real-world data, for a quick cited answer.", ("literature", "trials", "kg", "rwd"),
     "osler", False),
    ("appraisal", "Evidence appraisal", "Weighs the evidence: an evidence table, GRADE-style certainty and a language "
     "model check of the cited claims.", ("literature", "trials", "kg"), "sackett", True),
    ("comprehensive", "Comprehensive review", "Every facet of the question searched on its own, with a coverage "
     "check and one section per facet.", ("literature", "trials", "kg"), "snow", False),
    ("rwe", "Real-world evidence", "Published evidence plus a new study on real-world data: study templates, and the "
     "LLM analyst when they do not fit.", ("literature", "trials", "rwd"), "sackett", False),
    ("literature_only", "Literature only", "The fastest team: PubMed, open-access full text and your PDFs.",
     ("literature",), "osler", False),
    ("mechanism", "Mechanism explorer", "Published studies with knowledge-graph relations and 2-hop paths that "
     "explain mechanisms.", ("literature", "kg"), "osler", False),
]
DEFAULT_TEMPLATE = "builtin:standard"


def empty_graph(name="Untitled team", description="", depth=config.DEFAULT_MODE):
    """The core agents, wired, without evidence agents."""
    return _G.empty_graph(name, description, depth)


def build_graph(name, description, agents, depth, verify=False):
    return _G.build_graph(name, description, agents, depth, ["safechecker"] if verify else [])


def builtin_templates():
    return [{"id": f"builtin:{key}", "name": name, "description": desc, "builtin": True,
             "graph": build_graph(name, desc, agents, depth, verify)}
            for key, name, desc, agents, depth, verify in _BUILTIN]


def default_graph():
    return template(DEFAULT_TEMPLATE)["graph"]


# ============================================================================ saved teams
def _dir():
    path = Path(os.environ.get("MES_TEAMS_DIR") or config.CACHE_DIR / "teams")       # tests use their own folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:48]
    return s or "team"


def list_saved():
    """Saved teams, newest first: [{"id", "name", "description", "builtin": False, "graph", "saved_at"}]."""
    out = []
    try:
        files = sorted(_dir().glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    except OSError:
        return out
    for f in files:
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            graph = normalize(record.get("graph"))
        except Exception as e:                 # a damaged file is skipped, not shown
            logger.warning("Could not read the saved team %s: %s", f.name, e)
            continue
        out.append({"id": f"saved:{f.stem}", "name": graph["name"], "description": graph["description"],
                    "builtin": False, "graph": graph, "saved_at": record.get("saved_at") or f.stat().st_mtime})
    return out


def save(graph):
    """Store a team under its name (a team of the same name is replaced); returns its id. A built-in name gets
    " (custom)", so a built-in template is never shadowed."""
    graph = normalize(graph)
    if graph["name"].lower() in {name.lower() for _, name, *_ in _BUILTIN}:
        graph["name"] = _text(graph["name"], MAX_NAME - 9) + " (custom)"
    path = _dir() / f"{slug(graph['name'])}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"saved_at": time.time(), "graph": graph}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return f"saved:{path.stem}"


def delete(team_id):
    """Remove a saved team (built-in templates cannot be removed); True if a file was removed."""
    if not isinstance(team_id, str) or not team_id.startswith("saved:"):
        return False
    name = team_id.split(":", 1)[1]
    if not re.fullmatch(r"[a-z0-9-]{1,48}", name):
        return False
    path = _dir() / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def all_templates():
    return builtin_templates() + list_saved()


def template(team_id):
    return next((t for t in all_templates() if t["id"] == team_id), None)


# ============================================================================ catalogue (shown in the builder)
LOCKED = "Fixed by the current MES pipeline — it will become editable when the backend exposes it."
BY_DEPTH = "Set by the depth of the team (FAST · BALANCED · DEEP)."


def _locked_tool(key, label, desc, on=True):
    return {"key": key, "label": label, "desc": desc, "kind": "locked", "on": {m: on for m in config.MODES}}


def _depth_tool(key, label, desc, modes):
    return {"key": key, "label": label, "desc": desc, "kind": "depth", "on": {m: m in modes for m in config.MODES}}


def _budget(key, label, values, unit=""):
    return {"key": key, "label": label, "unit": unit, "values": values}


def _mode_values(attr, fmt=None):
    return {m: (fmt(getattr(c, attr)) if fmt else getattr(c, attr)) for m, c in config.MODES.items()}


def model_labels():
    """(main model, planner model) of the configured backend, for display."""
    from mes.llm_zoo import bedrock_model_id, get_provider, lmstudio_model, openai_model
    try:
        provider = get_provider()
    except Exception:
        provider = os.environ.get("MES_LLM_PROVIDER", "unknown")
    if provider == "openai":
        main = f"OpenAI · {openai_model()}"
        return main, main
    if provider == "lmstudio":
        main = f"LM Studio · {lmstudio_model()}"
        return main, main
    if provider == "bedrock":
        main = f"Bedrock · {bedrock_model_id()}"
        planner = os.environ.get("MES_BEDROCK_PLANNER_MODEL_ID", "").strip()
        return main, (f"Bedrock · {planner}" if planner and planner != bedrock_model_id() else main)
    return provider, provider


def _design_weights():
    from mes.tools.literature_tools import DESIGN_PRIOR
    names = {1: "systematic review / meta-analysis / guideline", 2: "randomized or other clinical trial",
             3: "cohort / case-control / observational", 4: "narrative review / case report", 5: "editorial / comment"}
    return "; ".join(f"{names[level]} {weight:g}" for level, weight in sorted(DESIGN_PRIOR.items()))


def strategies():
    """The search strategies and rules of the agents that work from code, written from its constants (the weights
    are those of literature_tools.rank_documents, trials_tools.search_trials and ibkh_tools.retrieve; the tests
    check that they still match)."""
    from mes.tools.ibkh_tools import CLASS_LABEL, CLASS_RELIABILITY, CONTRADICTORY
    from mes.tools.literature_tools import HIGH_EVIDENCE_FILTER, SCREEN_PMIDS, TOP_CITED, TRIAL_FILTER
    contradictions = "; ".join(f"{' / '.join(sorted(a))} vs {' / '.join(sorted(b))}" for a, b in CONTRADICTORY)
    reliability = "; ".join(f"{CLASS_LABEL[c].lower()} → {r}" for c, r in CLASS_RELIABILITY.items())
    return {
        "literature": f"""PUBMED QUERIES (from the planner's PICO concepts and synonyms, AND / OR tiers)
1. precise: every concept; screened {SCREEN_PMIDS} deep, its {TOP_CITED} most cited studies always considered
2. high-evidence: (precise query) AND {HIGH_EVIDENCE_FILTER}
3. head-to-head trials: intervention AND comparator AND {TRIAL_FILTER}
4. most cited trials: Europe PMC, population and treatment in title or abstract, sorted by citations
5. DEEP: one query per facet of the question
Articles the question names (PMID, DOI, PubMed link) are fetched directly and put first.

RANKING OF EACH ARTICLE
score = 0.52 × passage relevance (0.65 dense + 0.35 TF-IDF)
      + 0.13 × study design ({_design_weights()})
      + 0.12 × citations per year (landmark studies)
      + 0.15 × PubMed Best Match rank
      + 0.08 × recency (15 years)
Off-topic articles (relevance < 0.25) are left out; diverse articles are chosen (maximal marginal relevance).
Landmark slots: the most cited relevant trials, meta-analyses and guidelines are always considered.

FULL TEXT
Europe PMC JATS, then Unpaywall PDFs; passages with results and effect estimates first.""",
        "trials": """CLINICALTRIALS.GOV QUERIES (condition, intervention and outcome terms from the planner), until twice the
trials kept are found
1. condition AND intervention AND outcome
2. condition AND intervention
3. intervention (with the first condition term)
4. free-text term
Trials the question names (NCT numbers) are fetched directly and always kept.

RANKING OF EACH TRIAL
score = 0.7 × relevance (0.65 dense + 0.35 TF-IDF)
      + 0.15 posted results + 0.05 completed + 0.1 phase 3 / 4 + 0.05 randomized + up to 0.05 enrolment
Off-topic trials (relevance < 0.25) are left out; diverse trials are chosen (maximal marginal relevance).

POSTED RESULTS
Up to 3 outcome measures per trial with arm values and effect estimates (hazard ratio, 95% CI, p).""",
        "kg": f"""ENTITY LINKING (each concept and its synonyms)
exact name → token match → embedding similarity (≥ 0.8); generic words (disease, patients, drug …) are never
linked on their own

CANDIDATE RELATIONS
direct relations between the question's concepts (weight 1.0) · relations to semantically close nodes (0.7) ·
2-hop paths A → X → B (0.6, at most a third of the relations) · 1-hop relations of the treatments and conditions (0)
score = 0.55 × similarity to the question + 0.3 × relation weight + 0.15 × candidate weight

RELIABILITY
{reliability}; a path is as reliable as its weakest relation; contradicting relations ({contradictions})
lower it one level. Relations are mechanism and context, never evidence of a clinical effect.""",
        "grounding": """CITATIONS
1. Citations to IDs that are not in the evidence are removed; the others are renumbered [1..n].
2. A source ID written without brackets ("… in L1") becomes a citation.

NUMBERS
3. Every number of a cited sentence must occur in a cited source: badge "grounded" or "check number". Correct
   roundings (49.7 → 50), percentages of a source's fractions and a lead-in sentence whose numbers the next
   sentence cites count as grounded.
4. Citation repair: a number found in exactly one other source gets that source ("source corrected").
5. Numbers without a citation are flagged ("uncited").

TEXT
6. Sentences written twice are dropped; a second copy of the whole answer is cut off.""",
    }


def _encoder_label():
    return f"Sentence encoder · {config.ENCODER.split('/')[-1]}"


def _format_by_depth():
    """The answer format the synthesis prompt asks for, at each depth (facets shown as a placeholder)."""
    from types import SimpleNamespace
    from mes.tools.synthesis_tools import _format_instructions
    plan = SimpleNamespace(facets=[{"name": "{facet}"}])
    return {m: _format_instructions(m, plan, list(EVIDENCE)) for m in config.MODES}


def catalog():
    """Everything the builder shows about each agent: role, description, tools, locked settings and prompts (for
    the agents that work from code: their search strategy or rules)."""
    from mes.tools.planner_tools import FACET_OFF, FACET_ON, PLAN_PROMPT
    from mes.tools.rwd import analyst, designer, explorer
    from mes.tools.synthesis_tools import EXTRACT_PROMPT, MODE_STYLE, RULES as SYNTHESIS_RULES, SYSTEM_PROMPT, \
        VERIFY_PROMPT, verify_claims
    main_model, planner_model = model_labels()
    claims = inspect.signature(verify_claims).parameters
    max_claims, batch = claims["max_claims"].default, claims["batch_size"].default
    rules = strategies()
    yes_no = lambda v: "yes" if v else "no"
    placeholders = "Placeholders in {braces} are filled in at run time. Editable in a later version."
    later = "Editable in a later version."
    agents = {
        "question": {
            "group": "io", "icon": "❓", "color": "slate", "role": "Input",
            "subtitle": "Your clinical question",
            "desc": "The question you ask; for a follow-up, earlier turns of the conversation give the context.",
            "llm": None, "tools": [], "budgets": [
                _budget("max_question", "Characters used", {m: 4000 for m in config.MODES})],
            "model": None, "prompts": []},
        "planner": {
            "group": "core", "icon": "🧭", "color": "slate", "role": "Core agent",
            "subtitle": "PICO concepts · search queries",
            "desc": "Reads the question: its type, the PICO concepts with synonyms, its language, three related "
                    "questions and, in DEEP, facets. The PubMed and ClinicalTrials.gov queries are then built in "
                    "code (AND / OR tiers).",
            "llm": "1 language-model call (JSON)",
            "tools": [_locked_tool("pico", "PICO concepts and synonyms", "Population, intervention, comparator and "
                                   "outcome, with abbreviations, brand names and development codes."),
                      _locked_tool("named", "Records named in the question", "A PMID, DOI, PubMed link or NCT "
                                   "number in the question is fetched directly and put first."),
                      _depth_tool("facets", "Facet sub-questions", "3–4 angles a review needs (efficacy, "
                                  "harms, subgroups, mechanism), each searched on its own.", ("snow",)),
                      _locked_tool("follow_ups", "Related questions", "Three follow-up questions shown under the "
                                   "answer.")],
            "budgets": [], "model": planner_model,
            "prompts": [{"label": "Planner prompt", "text": PLAN_PROMPT, "hint": placeholders,
                         "by_depth": {m: ("facets: " + (FACET_ON if c.facets else FACET_OFF))
                                      for m, c in config.MODES.items()}}]},
        "literature": {
            "group": "evidence", "icon": "📄", "color": "1", "role": "Evidence agent", "source": "literature",
            "subtitle": "PubMed · full text · PDFs",
            "desc": "Searches PubMed (relevance and high-evidence queries), reads open-access full text (Europe PMC, "
                    "Unpaywall), your PDF library and uploaded PDFs, and ranks them (dense + TF-IDF + design and "
                    "recency, landmark studies first).",
            "llm": None,
            "tools": [_locked_tool("pubmed", "PubMed search", "Relevance and high-evidence queries, abstracts and "
                                   "publication types."),
                      _locked_tool("landmarks", "Landmark studies", "The most cited relevant trials and "
                                   "meta-analyses (Europe PMC) are always considered."),
                      _locked_tool("fulltext", "Open-access full text", "Europe PMC JATS, then Unpaywall PDFs."),
                      _locked_tool("pdfs", "PDF library and uploaded PDFs", f"Searched for every question "
                                   f"(library folder: {config.PDF_DIR.name}/; uploads under Options)."),
                      _depth_tool("coverage", "Coverage check", "A targeted search for facets the first search "
                                  "left uncovered.", ("snow",))],
            "budgets": [_budget("pubmed_max", "PMIDs per query", _mode_values("pubmed_max")),
                        _budget("literature_max", "Articles kept", _mode_values("literature_max")),
                        _budget("fulltext_max", "Full texts read", _mode_values("fulltext_max")),
                        _budget("passages_per_doc", "Passages per article", _mode_values("passages_per_doc"))],
            "model": _encoder_label(), "model_role": "Ranking model", "panel": "Search strategy",
            "prompts": [{"label": "PubMed search and ranking", "text": rules["literature"], "hint": later}]},
        "trials": {
            "group": "evidence", "icon": "🧪", "color": "2", "role": "Evidence agent", "source": "trials",
            "subtitle": "ClinicalTrials.gov",
            "desc": "Searches ClinicalTrials.gov: registry records and posted results (arm values, hazard ratios, "
                    "confidence intervals, p-values).",
            "llm": None,
            "tools": [_locked_tool("registry", "Registry records", "Design, status, enrolment and outcomes of each "
                                   "registered trial."),
                      _locked_tool("results", "Posted results", "Outcome measures with arm values and effect "
                                   "estimates.")],
            "budgets": [_budget("trials_max", "Trials kept", _mode_values("trials_max"))],
            "model": _encoder_label(), "model_role": "Ranking model", "panel": "Search strategy",
            "prompts": [{"label": "Registry search and ranking", "text": rules["trials"], "hint": later}]},
        "kg": {
            "group": "evidence", "icon": "🕸️", "color": "3", "role": "Evidence agent", "source": "kg",
            "subtitle": "iBKH knowledge graph",
            "desc": "Links the question's entities to iBKH (34M relations) and returns direct relations, 2-hop paths "
                    "and semantic neighbours with their reliability (curated, SemMedDB, inferred). Mechanism and "
                    "context only — never evidence of a clinical effect.",
            "llm": None,
            "tools": [_locked_tool("linking", "Entity linking", "Exact, token and embedding matches."),
                      _locked_tool("paths", "Relations and 2-hop paths", "Between the entities of the question."),
                      _locked_tool("semantic", "Semantic neighbours", "Dense re-ranking of nearby relations.")],
            "budgets": [_budget("kg_max", "Relations kept", _mode_values("kg_max"))],
            "model": _encoder_label(), "model_role": "Linking and ranking model", "panel": "Retrieval rules",
            "prompts": [{"label": "Linking, relations and reliability", "text": rules["kg"], "hint": later}]},
        "rwd": {
            "group": "evidence", "icon": "📊", "color": "4", "role": "Evidence agent", "source": "rwd",
            "subtitle": "New studies on real-world data",
            "desc": "Designs an observational study for the question, runs it on the connected real-world database and "
                    "reports the estimates with their diagnostics. Only aggregates reach the language model; patient rows never leave "
                    "this computer.",
            "llm": "1–2 language-model calls (study design); the LLM analyst: up to its turns",
            "tools": [{"key": "templates", "label": "Study templates", "kind": "toggle",
                       "desc": "Validated designs that run in seconds: target trial emulation, adjusted association, "
                               "rates, survival, trends, prognostic models.",
                       "note": {"flag": "rwd", "text": f"The {DATABASE} store is not built on this computer."}},
                      {"key": "analyst", "label": "LLM data analyst", "kind": "toggle",
                       "desc": "Explores the data and writes its own analysis code in a sandbox; the result is "
                               "reviewed and reproduced before it is used.",
                       "info": {"flag": "analyst", "text": "Runs where it is switched on (MES_RWD_ANALYST); on this "
                                                           "computer the study templates are used."}}],
            "budgets": [_budget("analyst_steps", "Analyst turns", _mode_values("analyst_steps")),
                        _budget("analyst_seconds", "Analyst time budget", _mode_values("analyst_seconds"), "s"),
                        _budget("analyst_effort", "Thinking effort", _mode_values("analyst_effort")),
                        _budget("analyst_revisions", "Revisions after review", _mode_values("analyst_revisions")),
                        _budget("analyst_sensitivity", "Sensitivity analysis", _mode_values("analyst_sensitivity",
                                                                                            yes_no))],
            "model": main_model,
            "prompts": [{"label": "Study designer prompt", "text": designer.PROMPT, "hint": placeholders},
                        {"label": "Design repair prompt", "text": designer.REPAIR, "hint": placeholders},
                        {"label": "Design rules (approximations)", "text": designer.KIND_RULE.strip(), "hint": later},
                        {"label": "Exploratory SQL prompt", "text": explorer.PROMPT, "hint": placeholders},
                        {"label": "LLM analyst prompt", "text": analyst.SYSTEM, "hint": placeholders},
                        {"label": "Analysis reviewer prompt", "text": analyst.REVIEW, "hint": placeholders}]},
        "synthesizer": {
            "group": "core", "icon": "✍️", "color": "red", "role": "Core agent",
            "subtitle": "Cited answer, streamed",
            "desc": "Writes the answer only from the numbered evidence, citing every statement; evidence classes kept "
                    "apart, conflicts and gaps stated.",
            "llm": "1 language-model call (streamed); +1 when the real-world data results arrive later",
            "tools": [_locked_tool("cited", "Cited answer and key takeaways", "Inline citations to evidence IDs "
                                   "only."),
                      _depth_tool("table", "Evidence table and certainty", "A table of the most informative "
                                  "studies and a GRADE-style certainty per outcome.", ("sackett", "snow")),
                      _depth_tool("facet_sections", "One section per facet", "A section for each facet of the "
                                  "question.", ("snow",))],
            "budgets": [_budget("context_budget", "Evidence budget",
                                {m: int(config.CONTEXT_BUDGET * c.budget_scale) for m, c in config.MODES.items()},
                                "characters"),
                        _budget("synthesis_tokens", "Answer length (max tokens)", _mode_values("synthesis_tokens")),
                        _budget("reasoning", "Model reasoning", _mode_values("reasoning", yes_no))],
            "model": main_model,
            "prompts": [{"label": "System prompt", "text": SYSTEM_PROMPT, "hint": later},
                        {"label": "Rules", "text": SYNTHESIS_RULES, "hint": later},
                        {"label": "Style", "text": "", "by_depth": dict(MODE_STYLE), "hint": later},
                        {"label": "Answer format", "text": "", "by_depth": _format_by_depth(), "hint": placeholders}]},
        "grounding": {
            "group": "core", "icon": "🛡️", "color": "green", "role": "Core check",
            "subtitle": "Citations · numbers · always on",
            "desc": "Removes citations to unknown sources, checks every number of a cited sentence against the cited "
                    "source and moves a citation to the source that contains the number.",
            "llm": None,
            "tools": [_locked_tool("invalid", "Invalid citations removed", "Unknown IDs dropped, citations "
                                   "renumbered [1..n]."),
                      _locked_tool("numbers", "Number grounding", "Each number must occur in the cited source "
                                   "(badges: grounded / check number)."),
                      _locked_tool("repair", "Citation repair", "A number found in exactly one other source gets "
                                   "that source.")],
            "budgets": [], "model": None, "panel": "Check rules",
            "prompts": [{"label": "Citation and number checks", "text": rules["grounding"], "hint": later}]},
        "safechecker": {
            "group": "quality", "icon": "🔍", "color": "green", "role": "Quality agent",
            "subtitle": "LLM check of cited claims",
            "desc": "A second language-model pass that checks each cited claim against its sources (claims with numbers "
                    "first): supported, partial or unsupported.",
            "llm": f"{math.ceil(max_claims / batch)} language-model calls (batched)",
            "tools": [_locked_tool("claims", "Claim check", "Verdicts shown as badges in the answer.")],
            "budgets": [_budget("max_claims", "Claims checked", {m: max_claims for m in config.MODES}),
                        _budget("batch_size", "Claims per call", {m: batch for m in config.MODES})],
            "model": main_model,
            "prompts": [{"label": "Fact-check prompt", "text": VERIFY_PROMPT, "hint": placeholders}]},
        "answer": {
            "group": "io", "icon": "📝", "color": "slate", "role": "Output",
            "subtitle": "Answer page", "canvas_sub": "Cited answer · {depth}",
            "desc": "Key takeaways, the answer with numbered citations and hover cards, grounding badges, the sources "
                    "rail and exports (HTML report, Markdown, JSON).",
            "llm": None, "tools": [], "budgets": [], "model": None, "prompts": []},
    }
    for t, a in agents.items():
        a["title"] = TITLES[t]
        a["ports"] = list(PORTS[t])
        a["core"] = t in CORE
    locked_agents = [{
        "key": "statistician", "title": "Statistician", "icon": "📐", "color": "slate", "role": "Evidence agent",
        "subtitle": "Separate study extraction",
        "desc": "A separate extraction pass that tabulates design, population, sample size, comparison, outcome and "
                "effect estimate of every study. In this version the Evidence Synthesizer writes the evidence table "
                "itself (BALANCED and DEEP).",
        "prompts": [{"label": "Extraction prompt", "text": EXTRACT_PROMPT, "hint": placeholders}]}]
    groups = [{"name": "Evidence agents", "hint": "Drag onto the canvas — wired automatically", "types": list(EVIDENCE)},
              {"name": "Quality", "hint": "", "types": ["safechecker"]},
              {"name": "Core · always in the team", "hint": "", "types": ["planner", "synthesizer", "grounding"]}]
    cache = {"name": "Reuse recent answers",
             "desc": "An answer to the same question with the same team (last 72 h) is shown at once."}
    return {"agents": agents, "locked_agents": locked_agents, "port_labels": PORT_LABELS, "groups": groups,
            "cache": cache,
            "modes": {m: {"label": c.label, "tagline": c.tagline} for m, c in config.MODES.items()},
            "locked_text": LOCKED, "depth_text": BY_DEPTH}


# ============================================================================ team status (run view, search trace)
# Pipeline stages → the agent that runs them
STAGE_AGENT = {"plan": "planner", "literature": "literature", "pdfs": "literature", "coverage": "literature",
               "trials": "trials", "kg": "kg", "rwd": "rwd", "extract": "synthesizer", "synthesis": "synthesizer",
               "verify": "safechecker"}
STATE_TEXT = {"waiting": "waiting", "running": "running", "done": "done", "error": "error", "skipped": "skipped",
              "cached": "cached", "kept": "kept", "edited": "edited"}


def run_agents(sources, verify):
    """The agents a run uses, in pipeline order."""
    return (["planner"] + [s for s in EVIDENCE if s in (sources or [])] + ["synthesizer", "grounding"]
            + (["safechecker"] if verify else []))


class TeamTracker:
    """States of the agents of a running question, from the pipeline's stage events."""

    def __init__(self, sources, verify):
        self.agents = run_agents(sources, verify)
        self.stages = {}           # stage → (status, elapsed)

    def update(self, event):
        """Record a stage event; True when an agent's state changed."""
        stage = event.get("stage")
        if event.get("type") != "stage" or stage not in STAGE_AGENT and stage not in ("cache", "done"):
            return False
        before = self.states()
        self.stages[stage] = (event.get("status"), event.get("elapsed"))
        return self.states() != before

    def states(self):
        """{agent: (state, seconds or None)}"""
        if self.stages.get("cache", ("",))[0] == "done":
            return {a: ("cached", None) for a in self.agents}
        out = {}
        for agent in self.agents:
            mine = [v for s, v in self.stages.items() if STAGE_AGENT.get(s) == agent]
            if agent == "grounding":       # runs right after the synthesis
                synth = self.stages.get("synthesis")
                mine = [("done", None)] if synth and synth[0] == "done" else []
            statuses = [s for s, _ in mine]
            seconds = [e for _, e in mine if isinstance(e, (int, float))]
            if "running" in statuses:
                state = "running"
            elif "error" in statuses:
                state = "error"
            elif statuses and all(s == "skipped" for s in statuses):
                state = "skipped"
            elif "done" in statuses:
                state = "done"
            elif "edited" in statuses:      # an edited protocol (tools/protocol_tools.py): the user's own plan
                state = "edited"
            elif "kept" in statuses:        # the evidence of the version that was edited
                state = "kept"
            else:
                state = "waiting"
            out[agent] = (state, max(seconds) if seconds and state != "running" else None)
        return out


def result_states(result, verify=False):
    """{agent: (state, seconds)} of a finished answer, from its timings and source status."""
    agents = run_agents(result.sources, verify or bool(result.verification.get("llm_verified")))
    timings, status = result.timings or {}, result.source_status or {}
    if getattr(result, "from_cache", False):
        return {a: ("cached", None) for a in agents}
    protocol = getattr(result, "protocol", None) or {}
    kept = set(protocol.get("kept") or [])
    out = {}
    for agent in agents:
        if agent == "planner" and protocol:
            out[agent] = ("edited" if protocol.get("part") == "search" else "kept", None)
        elif agent in EVIDENCE and agent in kept:
            out[agent] = ("kept", None)
        elif agent == "planner":
            out[agent] = ("done", timings.get("plan"))
        elif agent in EVIDENCE:
            st = status.get(agent) or {}
            state = ("error" if st.get("ok") is False else "running" if st.get("pending")      # first answer shown
                     else "skipped" if st.get("available") is False else "done")
            secs = [timings.get(k) for k in ("literature", "pdfs", "coverage") if timings.get(k) is not None] \
                if agent == "literature" else [timings.get(agent)]
            secs = [s for s in secs if isinstance(s, (int, float))]
            out[agent] = (state, max(secs) if secs else None)
        elif agent == "synthesizer":
            out[agent] = ("done", timings.get("synthesis"))
        elif agent == "grounding":
            out[agent] = ("done", None)
        else:
            out[agent] = ("done", timings.get("verify"))
    return out


def _fmt_seconds(value):
    if not isinstance(value, (int, float)):
        return ""
    return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"


ICONS = {"question": "❓", "planner": "🧭", "literature": "📄", "trials": "🧪", "kg": "🕸️", "rwd": "📊",
         "synthesizer": "✍️", "grounding": "🛡️", "safechecker": "🔍", "answer": "📝"}
SHORT = {"literature": "Literature", "trials": "Trials", "kg": "iBKH", "rwd": "RWD"}


RAIL_NAMES = {"planner": "Planner", "synthesizer": "Synthesizer", "grounding": "Citation Check",
              "safechecker": "SafeChecker", **SHORT}


def rail_html(states, mode=None, team=None):
    """The agent team above the sources of an answer: one chip per agent with its time."""
    chips = []
    for agent, (state, secs) in states.items():
        right = _fmt_seconds(secs) if state == "done" and secs is not None else (
            "" if state == "done" else STATE_TEXT.get(state, state))
        chips.append(f'<span class="mes-team-chip mes-team-{state}" title="{escape(TITLES[agent])}">'
                     f'<span class="mes-team-dot"></span>{ICONS[agent]} {escape(RAIL_NAMES[agent])}'
                     + (f'<span class="mes-team-state">{escape(right)}</span>' if right else "") + "</span>")
    mode_label = config.MODES[mode].label if mode in config.MODES else ""
    title = "Agent team" + (f" · {mode_label}" if mode_label else "")
    team_line = f'<span class="mes-team-label">🧩 {escape(team)}</span>' if team else ""
    return (f'<div class="mes-team mes-team-rail"><div class="mes-team-head"><span class="mes-card-title">'
            f'{escape(title)}</span>{team_line}</div><div class="mes-team-chips">{"".join(chips)}</div></div>')


def status_html(states, mode=None, team=None, live=False):
    """Compact card of the agent team: one row per agent, the evidence agents side by side."""

    def cell(agent, compact=False):
        state, secs = states.get(agent, ("waiting", None))
        name = SHORT.get(agent, TITLES[agent]) if compact else TITLES[agent]
        right = _fmt_seconds(secs) if state == "done" and secs is not None else STATE_TEXT.get(state, state)
        return (f'<div class="mes-team-agent mes-team-{state}{" mes-team-ev" if compact else ""}">'
                f'<span class="mes-team-dot"></span><span class="mes-team-name">{ICONS[agent]} {escape(name)}'
                f'</span><span class="mes-team-state">{escape(right)}</span></div>')

    evidence = [a for a in states if a in EVIDENCE]
    rows = [cell("planner")] if "planner" in states else []
    if evidence:
        rows.append(f'<div class="mes-team-fan">{"".join(cell(a, True) for a in evidence)}</div>')
    rows += [cell(a) for a in ("synthesizer", "grounding", "safechecker") if a in states]
    mode_label = config.MODES[mode].label if mode in config.MODES else ""
    title = "Agent team" + (f" · {mode_label}" if mode_label else "")
    team_line = f'<div class="mes-team-label">🧩 {escape(team)}</div>' if team else ""
    return (f'<div class="mes-team{" mes-team-live" if live else ""}"><div class="mes-card-title">{escape(title)}'
            f'</div>{team_line}<div class="mes-team-flow">{"".join(rows)}</div></div>')
