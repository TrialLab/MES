"""
MES Orchestrator & Workflow — evidence synthesis from literature, ClinicalTrials.gov and iBKH.

Two ways to run a question (same evidence pipeline underneath):

  * ``answer_question`` — the MES pipeline, streamed events (used by the UI and the CLI).
  * ``run_mes_pipeline`` — agent layer: the orchestrator calls the SupervisorAgent, whose primary tool
    runs the same pipeline; ``run_stepwise_pipeline`` instead lets the specialist agents
    (LiteratureMiner → KGSpecialist → Supervisor synthesis → SafeChecker) collect evidence step by step.

CLI:
    python -m mes.workflow "Does metformin reduce mortality in ICU patients with type 2 diabetes?" \
        [--mode fast|balanced|deep] [--sources literature,trials,kg] [--agents]
"""

import argparse
import asyncio
import sys
from difflib import get_close_matches

from agents import Agent, ModelSettings, RunConfig, Runner, ToolsToFinalOutputResult
from agents.exceptions import MaxTurnsExceeded

from mes import config, mes_agents
from mes.llm_zoo import create_agent_model, create_llm, get_provider
from mes.mes_agents import _AGENT_MODEL, callers, session, set_agent_model
from mes.tools import pipeline_tools


# ===================================================================
# Orchestrators
# ===================================================================

def _final_once_answered(context, tool_results):
    """
    The Supervisor's answer is the final output as is (no extra turn that re-types, and may alter, it).
    If the Supervisor returned without an answer (e.g. a tool error), the orchestrator gets another turn to retry.
    """
    if tool_results and session.result is not None:
        return ToolsToFinalOutputResult(is_final_output=True, final_output=tool_results[-1].output)
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


def _final_once_checked(context, tool_results):
    """Stepwise run ends when the SafeChecker has checked an existing answer; the output is Supervisor + SafeChecker."""
    if session.result is not None and any(r.tool.name == "call_SafeCheckerAgent" for r in tool_results):
        report = "\n\n".join(str(text) for text in (session.supervisor_report, session.safechecker_report) if text)
        return ToolsToFinalOutputResult(is_final_output=True, final_output=report)
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


orchestrator = Agent(
    name="MESOrchestrator",
    instructions="""You are the orchestrator of MES (Multi-agent Evidence Synthesis).

Call call_SupervisorAgent exactly once. It answers the clinical question stored in the session with cited
evidence from PubMed, open-access full text, PDFs, ClinicalTrials.gov and the iBKH knowledge graph.
Then return the Supervisor's answer and references unchanged. Do not add facts of your own.""",
    tools=[callers["SupervisorAgent"]],
    tool_use_behavior=_final_once_answered,
    model=_AGENT_MODEL,
)

STEPWISE_ORDER = ["LiteratureMinerAgent", "KGSpecialistAgent", "RWDAnalystAgent", "SupervisorAgent",
                  "SafeCheckerAgent"]

stepwise_orchestrator = Agent(
    name="MESStepwiseOrchestrator",
    instructions="""You are the orchestrator of MES (Multi-agent Evidence Synthesis).

Call these agents in order, each exactly once:
1. call_LiteratureMinerAgent — collects PubMed articles, full texts, PDFs and ClinicalTrials.gov records
2. call_KGSpecialistAgent — collects iBKH knowledge-graph relations
3. call_RWDAnalystAgent — analyses MIMIC-IV real-world data (it skips itself when real-world data is not selected)
4. call_SupervisorAgent — writes the cited answer from the collected evidence (it must use mes_synthesize)
5. call_SafeCheckerAgent — checks the cited claims

Then return the Supervisor's answer and references unchanged, followed by the SafeChecker summary.
Do NOT skip any agent. Do not add facts of your own.""",
    tools=[callers[name] for name in STEPWISE_ORDER],
    tool_use_behavior=_final_once_checked,
    model=_AGENT_MODEL,
)

ORCHESTRATOR_TOOLS = sorted({t.name for agent in (orchestrator, stepwise_orchestrator) for t in agent.tools})


def _tool_error_message(args):
    """Answer a misspelled tool call (small models write e.g. "SupervisingAgent") with the closest real name."""
    if args.kind != "tool_not_found":
        return None
    closest = get_close_matches(args.tool_name, ORCHESTRATOR_TOOLS, n=1, cutoff=0)
    return (f"Tool '{args.tool_name}' does not exist. Call `{closest[0]}` (exact name). "
            f"Available tools: {', '.join(ORCHESTRATOR_TOOLS)}.")


RUN_CONFIG = RunConfig(tool_not_found_behavior="return_error_to_model", tool_error_formatter=_tool_error_message)


def _tool_settings():
    """
    LM Studio models may answer in prose instead of calling the next agent, so every agent there must call a
    tool; the orchestrators keep that requirement until their stop condition above is met (bounded by
    ``max_turns``). Claude on Bedrock calls tools reliably and keeps the SDK defaults (forcing a tool call there
    is incompatible with its thinking).
    """
    forced = get_provider() == "lmstudio"
    settings = ModelSettings(tool_choice="required") if forced else ModelSettings()
    for agent in mes_agents.agents.values():
        agent.model_settings = settings
    for agent in (orchestrator, stepwise_orchestrator):
        agent.model_settings = settings
        agent.reset_tool_choice = not forced


def _prepare(question, mode, sources, llm_model, uploads, history, keep_history):
    llm = session.llm if (keep_history and session.llm is not None and llm_model is None) else create_llm(llm_model)
    previous = list(session.history) if keep_history else []
    session.reset(question=question, mode=config.get_mode(mode).key, sources=sources,
                  uploads=uploads, history=history if history is not None else previous)
    session.llm = llm
    # A fresh agent model per run: its async client is bound to the event loop of this run
    model = create_agent_model()
    set_agent_model(model)
    orchestrator.model = stepwise_orchestrator.model = model
    _tool_settings()
    return model


async def _close(model):
    """Close the run's HTTP client inside its own event loop (otherwise it is collected after the loop closed)."""
    client = getattr(model, "_client", None)
    if client is not None and hasattr(client, "close"):
        try:
            await client.close()
        except Exception:
            pass


# ===================================================================
# Runners
# ===================================================================

async def _run_agents(agent, question, max_turns):
    """The orchestrator's final text. A local model that must call a tool every turn may keep going after the
    answer is ready: when the turns run out with an answer in the session, that answer stands."""
    try:
        run = await Runner.run(agent, f"Answer the clinical question in the session: {question}", max_turns=max_turns,
                               run_config=RUN_CONFIG)
        return run.final_output
    except MaxTurnsExceeded:
        if session.result is None:
            raise
        return f"(stopped after {max_turns} turns; the answer was ready)"


async def run_mes_pipeline(question, mode="osler", sources=None, llm_model=None, uploads=None, history=None,
                           keep_history=False, max_turns=12):
    """
    Agent layer: orchestrator → SupervisorAgent → full MES pipeline.

    Returns a dict with the structured ``result`` (``MESResult``), the numbered ``answer`` and the
    orchestrator's final text. ``keep_history=True`` treats the question as a follow-up of the previous run.
    """
    model = _prepare(question, mode, sources, llm_model, uploads, history, keep_history)
    try:
        report = await _run_agents(orchestrator, question, max_turns)
    finally:
        await _close(model)
    result = session.result
    return {"result": result, "answer": result.answer_display if result else "",
            "orchestrator_report": report}


async def run_stepwise_pipeline(question, mode="osler", sources=None, llm_model=None, uploads=None,
                                history=None, max_turns=20):
    """Agent layer, step by step: LiteratureMiner → KGSpecialist → Supervisor (mes_synthesize) → SafeChecker."""
    model = _prepare(question, mode, sources, llm_model, uploads, history, keep_history=False)
    try:
        report = await _run_agents(stepwise_orchestrator, question, max_turns)
    finally:
        await _close(model)
    result = session.result
    return {"result": result, "answer": result.answer_display if result else "",
            "orchestrator_report": report,
            "agent_reports": {k: getattr(session, k) for k in ("literature_report", "kg_report", "rwd_report",
                                                               "supervisor_report", "safechecker_report")}}


def answer_question(question, mode="osler", sources=None, llm=None, uploads=None, history=None, use_cache=True,
                    verify=None, on_event=None):
    """The MES pipeline without the orchestrator loop: the fastest way to answer one question."""
    return pipeline_tools.answer_question(question, llm or create_llm(), sources=sources or pipeline_tools.DEFAULT_SOURCES,
                                          mode=mode, uploads=uploads, history=history, use_cache=use_cache,
                                          verify=verify, on_event=on_event)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="MES evidence synthesis")
    parser.add_argument("question")
    parser.add_argument("--mode", default="osler", type=config.mode_key,
                        choices=list(config.MODES), help="osler|sackett|snow, or fast|balanced|deep")
    parser.add_argument("--sources", default=",".join(pipeline_tools.DEFAULT_SOURCES))
    parser.add_argument("--agents", action="store_true", help="run through the agent layer (orchestrator)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    if args.agents:
        out = asyncio.run(run_mes_pipeline(args.question, mode=args.mode, sources=sources))
        print(out["answer"] or out["orchestrator_report"])
        return

    def show(event):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] in ("stage", "log") and event.get("status") != "running":
            print(f"  · {event.get('message', '')}", file=sys.stderr)

    result = answer_question(args.question, mode=args.mode, sources=sources, use_cache=not args.no_cache,
                             on_event=show)
    by_id = result.evidence_by_id()
    # the streamed text cites evidence IDs ([L1], [T2] …); the final answer numbers them in this order
    print("\n\nReferences")
    for n, i in enumerate(result.citation_order, start=1):
        print(f"[{n}] [{i}] {by_id[i].title} — {by_id[i].url or by_id[i].venue}")


if __name__ == "__main__":
    main()
