"""
MES Agent Layer — session state, agent specifications, and agent building.

Agents:
  - SupervisorAgent       → session.result (primary tool: mes_answer_question, the full pipeline)
  - LiteratureMinerAgent  → session.literature / session.trials (PubMed, full text, PDFs, ClinicalTrials.gov)
  - KGSpecialistAgent     → session.kg_items (iBKH structured + semantic retrieval)
  - StatisticianAgent     → session.study_table (structured extraction of designs and effect estimates)
  - SafeCheckerAgent      → session.result.verification (LLM claim check on top of number grounding)

The Supervisor's coarse-grained tool runs the whole pipeline
(``tools/pipeline_tools.py``); the fine-grained tools of the other agents allow
custom agent workflows (see ``workflow.run_stepwise_pipeline``).
"""

import json
import os
from dataclasses import asdict

from agents import function_tool, set_tracing_disabled

# Agents SDK traces go to the OpenAI platform (questions, tool outputs): off unless MES_AGENT_TRACING=1
if os.environ.get("MES_AGENT_TRACING", "0").strip().lower() not in ("1", "true", "yes", "on"):
    set_tracing_disabled(True)

from mes import config
from mes.context import MESContext
from mes.llm_zoo import create_agent_model

# Agent-level LLM: Bedrock Claude (via LiteLLM) or LM Studio, per MES_LLM_PROVIDER
_AGENT_MODEL = create_agent_model()

from mes.agent_framework import (
    AgentSpec,
    AgentCategory,
    build_agent_and_caller,
    register_agent_spec,
)

from mes.tools import literature_tools, trials_tools
from mes.tools.ibkh_tools import get_kg, kg_status
from mes.tools.pipeline_tools import _assign_ids, answer_question, conversation_turn
from mes.tools.planner_tools import plan_query
from mes.tools.ranking_tools import encode, expanded_query
from mes.tools.result_schema import MESResult
from mes.tools.synthesis_tools import (PreambleGate, extract_study_table, parse_evidence_table, postprocess_answer,
                                       stream_synthesis, verify_claims)


# ===================================================================
# Session — mutable shared state managed by the agents
# ===================================================================

session = MESContext()


def _question(question=""):
    q = (question or session.question).strip()
    if not q:
        raise ValueError("No question: pass one or set session.question.")
    session.question = q
    return q


def _ensure_plan():
    if session.plan is None:
        session.plan = plan_query(_question(), session.llm, history=session.history,
                                  facets=config.get_mode(session.mode).facets)
        session.query_vector = encode([session.plan.query], is_query=True)[0]
    return session.plan


def _brief(items, n=12):
    return [{"title": it.title[:160], "design": it.design, "year": it.year, "source": it.origin,
             "id": next(iter(it.identifiers.values()), "")} for it in items[:n]]


def _references(result):
    by_id = result.evidence_by_id()
    return [{"n": n, "id": i, "title": by_id[i].title, "url": by_id[i].url or by_id[i].venue}
            for n, i in enumerate(result.citation_order, start=1)]


# ===================================================================
# Supervisor tools
# ===================================================================

@function_tool
def mes_answer_question(question: str = "", mode: str = "", sources: str = "") -> str:
    """Answer the clinical question with the full MES pipeline (plan, retrieval from all sources,
    evidence ranking, cited synthesis, number grounding). This is the primary tool.
    Args:
        question: The clinical question. If empty, uses the session question.
        mode: osler (quick answer), sackett (weigh the evidence) or snow (comprehensive survey).
            If empty, uses the session mode.
        sources: Comma-separated subset of literature, trials, kg, rwd (MIMIC-IV real-world data).
            If empty, uses the session sources.
    """
    q = _question(question)
    session.mode = mode or session.mode
    if sources:
        session.sources = [s.strip() for s in sources.split(",") if s.strip()]
    result = answer_question(q, session.llm, sources=session.sources, mode=session.mode,
                             uploads=session.uploads, history=session.history)
    session.result = result
    session.history.append(conversation_turn(result))
    return json.dumps({"answer": result.answer_display, "references": _references(result),
                       "checks": result.verification.get("counts", {}),
                       "evidence_counts": {t: sum(1 for e in result.evidence if e.source_type == t)
                                           for t in ("literature", "trial", "kg", "rwd")}}, default=str)


@function_tool
def mes_plan_query(question: str = "") -> str:
    """Decompose the clinical question (type, PICO concepts, synonyms) and build the search queries.
    Args:
        question: The clinical question. If empty, uses the session question.
    """
    if question:
        session.question = question
        session.plan = None
    plan = _ensure_plan()
    return json.dumps({"question_type": plan.question_type, "pico": plan.pico, "pubmed_query": plan.pubmed_query,
                       "trial_terms": plan.trial_terms, "entities": [e["name"] for e in plan.entities]})


@function_tool
def mes_synthesize() -> str:
    """Write the cited answer from the evidence collected by the other agents (literature, trials, KG)."""
    plan = _ensure_plan()
    mode_cfg = config.get_mode(session.mode)
    evidence = _assign_ids({"literature": session.literature, "trial": session.trials, "kg": session.kg_items,
                            "rwd": session.rwd_items})
    if not evidence:
        return json.dumps({"error": "No evidence collected yet. Run the LiteratureMiner / KGSpecialist / "
                                    "RWDAnalyst tools first."})
    for item in evidence:
        for study in session.rwd_studies:
            if item.source_type == "rwd" and item.meta.get("study_key") == study.get("key"):
                study["evidence_id"] = item.id
    by_id = {it.id: it for it in evidence}
    gate = PreambleGate()
    for kind, text in stream_synthesis(plan, evidence, session.llm, mode_cfg, session.sources,
                                       int(config.CONTEXT_BUDGET * mode_cfg.budget_scale), session.history):
        if kind == "text":
            gate.feed(text)
            if gate.repeated:
                break
    markdown, display, order, sections, takeaways, grounding = postprocess_answer(gate.released or gate.buffer, by_id)
    session.result = MESResult(
        question=session.question, mode=mode_cfg.key, sources=session.sources, plan=asdict(plan), evidence=evidence,
        answer_markdown=markdown, answer_display=display, sections=sections, takeaways=takeaways,
        citation_order=order, verification=grounding,
        study_table=session.study_table or parse_evidence_table(sections.get("Evidence table", "")),
        llm=session.llm.label, history=session.history, rwd_studies=session.rwd_studies)
    return json.dumps({"answer": display, "references": _references(session.result)}, default=str)


# ===================================================================
# LiteratureMiner tools
# ===================================================================

@function_tool
def lit_search(max_articles: int = 0) -> str:
    """Search PubMed for the session question, rank the articles and read open-access full texts.
    Args:
        max_articles: Number of articles to keep (0 = mode default).
    """
    plan = _ensure_plan()
    mode_cfg = config.get_mode(session.mode)
    items, stats = literature_tools.search_pubmed(plan, mode_cfg)
    selected = literature_tools.rank_documents(items, plan.query, session.query_vector,
                                               max_articles or mode_cfg.literature_max, mode_cfg.passages_per_doc,
                                               lexical_query=expanded_query(plan))
    literature_tools.add_fulltext(selected, plan.query, session.query_vector, mode_cfg.fulltext_max,
                                  mode_cfg.passages_per_doc)
    session.literature = [it for it in session.literature if it.origin.endswith("PDF")] + selected
    return json.dumps({"n_found": stats["n_found"], "kept": _brief(selected)}, default=str)


@function_tool
def lit_search_pdfs() -> str:
    """Search the local PDF library and the PDFs uploaded for this question."""
    plan = _ensure_plan()
    library = literature_tools.get_pdf_library()
    found = library.search(plan.query, session.query_vector, k=3)
    found += literature_tools.search_uploaded_pdfs(session.uploads, plan.query, session.query_vector, k=4)
    session.literature = [it for it in session.literature if not it.origin.endswith("PDF")] + found
    return json.dumps({"kept": _brief(found)}, default=str)


@function_tool
def trials_search() -> str:
    """Search ClinicalTrials.gov for the session question (including posted results)."""
    plan = _ensure_plan()
    items, stats = trials_tools.search_trials(plan, config.get_mode(session.mode), plan.query, session.query_vector)
    session.trials = items
    return json.dumps({"n_found": stats["n_found"], "kept": _brief(items)}, default=str)


# ===================================================================
# KG-Specialist tools
# ===================================================================

@function_tool
def kg_retrieve() -> str:
    """Retrieve iBKH relations for the session question: entity linking, direct relations and
    2-hop paths between the question's entities, semantic neighbours, with reliability labels."""
    plan = _ensure_plan()
    graph = get_kg()
    if graph is None:
        return json.dumps({"error": kg_status()["error"]})
    items, info = graph.retrieve(plan, plan.query, session.query_vector, k=config.get_mode(session.mode).kg_max)
    session.kg_items, session.kg_info = items, info
    return json.dumps({"links": {k: v["name"] for k, v in info["links"].items()},
                       "relations": [{"text": it.title, "reliability": it.meta["reliability"]} for it in items]})


# ===================================================================
# RWD-Analyst tools
# ===================================================================

@function_tool
def rwd_analyze(design_json: str | None = None) -> str:
    """Answer the session question with real-world data (MIMIC-IV): design an observational study (target trial
    emulation, adjusted association, rate, distribution, utilization, survival, trend, prognostic model or an
    exploratory aggregate query), compile it to SQL, run it and return the estimates with their diagnostics.
    Args:
        design_json: Optional edited study design (the "spec" returned by a previous call) to re-run without
            redesigning. Leave empty to design the study for the session question.
    """
    from mes.tools import rwd_tools
    if "rwd" not in session.sources:
        return json.dumps({"skipped": "real-world data is not among the selected sources"})
    design_json = (design_json or "").strip()        # small models send null for "no design"
    if not design_json and session.rwd_studies:
        return _rwd_summary()                     # already analysed in this run (small models call tools again)
    plan = _ensure_plan()
    try:
        spec = json.loads(design_json) if design_json else None
        if isinstance(spec, dict) and isinstance(spec.get("spec"), dict):
            spec = spec["spec"]                   # a whole study record was passed back
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("design_json must be a JSON object (the 'spec' of a previous result)")
        items, stats = rwd_tools.search_rwd(plan, config.get_mode(session.mode), plan.query, session.query_vector,
                                            llm=session.llm, spec_override=spec)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:400]}",
                           "hint": "call rwd_analyze with an empty design_json to design the study for the question"
                           + (" (the earlier result is kept)" if session.rwd_studies else "")})
    if stats.get("available") is False:
        return json.dumps({"error": stats.get("error")})
    session.rwd_items, session.rwd_studies = items, stats.get("studies", [])
    return _rwd_summary()


def _rwd_summary():
    out = []
    for study in session.rwd_studies:
        res = study.get("result") or {}
        out.append({"status": study.get("status"), "spec": study.get("spec"), "n": res.get("n"),
                    "attrition": res.get("attrition"),
                    "estimates": [e for e in res.get("estimates", []) if not e.get("sensitivity")][:8],
                    "warnings": res.get("warnings"), "sql": (study.get("sql") or "")[:1500]})
    return json.dumps({"studies": out}, default=str)


# ===================================================================
# Statistician and SafeChecker tools
# ===================================================================

@function_tool
def mes_extract_studies() -> str:
    """Tabulate design, population, sample size, comparison, outcome and effect estimate of the
    collected studies and trials (numbers are checked against the source text)."""
    _ensure_plan()
    evidence = _assign_ids({"literature": session.literature, "trial": session.trials, "kg": [], "rwd": []})
    session.study_table = extract_study_table(session.question, evidence, session.llm)
    return json.dumps({"rows": session.study_table})


@function_tool
def mes_verify() -> str:
    """Check the cited claims of the current answer against their sources (claims with numbers first)."""
    if session.result is None:
        return json.dumps({"error": "No answer yet. Run mes_answer_question or mes_synthesize first."})
    session.result.verification = verify_claims(session.result.verification, session.result.evidence_by_id(),
                                                session.llm)
    return json.dumps({"counts": session.result.verification.get("counts", {})})


# ===================================================================
# Agent Specs
# ===================================================================

_SESSION_NOTE = ("The question, mode and sources are already stored in the session: call your tools with "
                 "empty arguments unless you need to override them. Do not answer from memory.")

SUPERVISOR_SPEC = register_agent_spec(AgentSpec(
    name="SupervisorAgent",
    instructions=(
        "You are the Supervisor of MES, a multi-agent medical evidence synthesis system.\n\n"
        "Your PRIMARY tool is mes_answer_question: it runs the full pipeline — question planning, retrieval from\n"
        "PubMed (abstracts and open-access full text), local and uploaded PDFs, ClinicalTrials.gov and the iBKH\n"
        "knowledge graph, evidence ranking, a cited synthesis and number grounding. The control flow is code.\n\n"
        "Standard workflow:\n"
        "  1. Call mes_answer_question once.\n"
        "  2. Return its 'answer' text verbatim, followed by the numbered 'references'.\n\n"
        "Fine-grained tools for custom workflows: mes_plan_query (decompose the question) and mes_synthesize\n"
        "(write the cited answer from evidence collected by the other agents).\n\n" + _SESSION_NOTE
    ),
    category=AgentCategory.APPLICATION,
    tools=[mes_answer_question, mes_plan_query, mes_synthesize],
    model=_AGENT_MODEL,
    description="Plans the question and returns a cited evidence synthesis from literature, trials and iBKH.",
    task_prompt="Answer the clinical question in the session: call mes_answer_question, then return its answer "
                "verbatim with the numbered references.",
    session_output_key="supervisor_report",
))

LITERATURE_SPEC = register_agent_spec(AgentSpec(
    name="LiteratureMinerAgent",
    instructions=(
        "You are the LiteratureMiner of MES. You collect published evidence for the session question:\n"
        "  - lit_search: PubMed (relevance and high-evidence queries), ranking, open-access full text\n"
        "  - lit_search_pdfs: local PDF library and PDFs uploaded by the user\n"
        "  - trials_search: ClinicalTrials.gov registrations and posted results\n"
        "Call each tool once and summarize what was found (number of articles, main designs, trials with results).\n\n"
        + _SESSION_NOTE
    ),
    category=AgentCategory.ROUTING,
    tools=[lit_search, lit_search_pdfs, trials_search],
    model=_AGENT_MODEL,
    description="Retrieves and ranks PubMed articles, open-access full texts, PDFs and ClinicalTrials.gov records.",
    task_prompt="Collect the literature and trial evidence for the session question: call lit_search, "
                "lit_search_pdfs and trials_search, then summarize what was found.",
    session_output_key="literature_report",
))

KG_SPEC = register_agent_spec(AgentSpec(
    name="KGSpecialistAgent",
    instructions=(
        "You are the KG-Specialist of MES. kg_retrieve links the question's entities to the iBKH knowledge graph\n"
        "and returns direct relations, 2-hop paths and semantic neighbours with reliability labels.\n"
        "Call it once and summarize the relations. KG relations are mechanistic or contextual information, not\n"
        "evidence of a clinical effect — say so.\n\n" + _SESSION_NOTE
    ),
    category=AgentCategory.FOUNDATIONAL,
    tools=[kg_retrieve],
    model=_AGENT_MODEL,
    description="Structured and semantic retrieval from the iBKH biomedical knowledge graph.",
    task_prompt="Retrieve the iBKH relations for the session question with kg_retrieve and summarize them.",
    session_output_key="kg_report",
))

RWD_SPEC = register_agent_spec(AgentSpec(
    name="RWDAnalystAgent",
    instructions=(
        "You are the RWD-Analyst of MES. rwd_analyze answers the session question with real-world data from\n"
        "MIMIC-IV: it designs an observational study (no code is written by you), compiles it to SQL, runs it and\n"
        "returns the cohort size, the estimates with 95% CIs and the diagnostics. Call it once (empty arguments)\n"
        "and summarize the design, the main estimate and the warnings. These are observational, single-center\n"
        "results — say so.\n\n" + _SESSION_NOTE
    ),
    category=AgentCategory.FOUNDATIONAL,
    tools=[rwd_analyze],
    model=_AGENT_MODEL,
    description="Designs and runs real-world-data analyses on MIMIC-IV (target trial emulation, rates, associations).",
    task_prompt="Analyse the session question on real-world data with rwd_analyze and summarize the result.",
    session_output_key="rwd_report",
))

STATISTICIAN_SPEC = register_agent_spec(AgentSpec(
    name="StatisticianAgent",
    instructions=(
        "You are the Statistician of MES. mes_extract_studies tabulates design, population, sample size,\n"
        "comparison, outcome and effect estimate of the collected studies. Call it once and report the table.\n\n"
        + _SESSION_NOTE
    ),
    category=AgentCategory.FOUNDATIONAL,
    tools=[mes_extract_studies],
    model=_AGENT_MODEL,
    description="Structured extraction of study designs and effect estimates.",
    task_prompt="Tabulate the collected studies with mes_extract_studies and report the table.",
    session_output_key="statistician_report",
))

SAFECHECKER_SPEC = register_agent_spec(AgentSpec(
    name="SafeCheckerAgent",
    instructions=(
        "You are the SafeChecker of MES. mes_verify checks the cited claims of the current answer against their\n"
        "sources. Call it once and report how many claims are verified, partially supported or unsupported.\n\n"
        + _SESSION_NOTE
    ),
    category=AgentCategory.FOUNDATIONAL,
    tools=[mes_verify],
    model=_AGENT_MODEL,
    description="Claim-level check of the answer against the cited evidence.",
    task_prompt="Check the current answer with mes_verify and report the result.",
    session_output_key="safechecker_report",
))


# ===================================================================
# Build agents and callers from specs
# ===================================================================

ALL_SPECS = [SUPERVISOR_SPEC, LITERATURE_SPEC, KG_SPEC, RWD_SPEC, STATISTICIAN_SPEC, SAFECHECKER_SPEC]

agents = {}
agent_callers = []
for spec in ALL_SPECS:
    agent, caller = build_agent_and_caller(spec, session)
    agents[spec.name] = agent
    agent_callers.append(caller)
callers = {spec.name: caller for spec, caller in zip(ALL_SPECS, agent_callers)}


def set_agent_model(model):
    """Point every MES agent at ``model`` (a fresh model per run avoids reusing a closed event loop)."""
    for agent in agents.values():
        agent.model = model
