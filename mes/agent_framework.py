"""
Agent-layer helpers: declarative agent specifications turned into OpenAI Agents SDK ``Agent`` objects, plus the
``@function_tool`` an orchestrator calls to run each agent.

    spec = register_agent_spec(AgentSpec(name="KGSpecialistAgent", instructions="…",
                                         category=AgentCategory.FOUNDATIONAL, tools=[kg_retrieve], model=model,
                                         task_prompt="Retrieve …", session_output_key="kg_report"))
    agent, caller = build_agent_and_caller(spec, session)   # caller: a tool of the orchestrator agent

Agent categories:
  • Routing       — data extraction and preprocessing
  • Foundational  — reusable reasoning, retrieval and analysis
  • Application   — the task-specific agent tied to a workflow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner, function_tool


class AgentCategory(str, Enum):
    ROUTING = "routing"
    FOUNDATIONAL = "foundational"
    APPLICATION = "application"


@dataclass
class PipelineContext:
    """Mutable shared state of one pipeline run (subclassed by ``mes.context.MESContext``)."""

    llm: Any = None                                              # the tool-layer LLM (llm_zoo.LLMZoo)
    discussion_log: List[Dict[str, str]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)        # open-ended values

    def set(self, key: str, value: Any) -> None:
        """Store a value: a field of the context, else an extra."""
        if hasattr(self, key) and key != "extras":
            setattr(self, key, value)
        else:
            self.extras[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key) and key != "extras":
            return getattr(self, key)
        return self.extras.get(key, default)


@dataclass
class AgentSpec:
    """
    One agent: its name, instructions and tools (``@function_tool`` objects), and how an orchestrator runs it —
    ``task_prompt`` is the message the agent receives, ``session_output_key`` the session attribute its final
    output is written to.
    """

    name: str
    instructions: str
    category: AgentCategory
    tools: List[Any] = field(default_factory=list)
    model: Any = None                       # a model name or an Agents SDK model object (llm_zoo.create_agent_model)
    description: str = ""
    task_prompt: str = ""
    session_output_key: str = ""
    handoffs: List[Any] = field(default_factory=list)
    output_type: Optional[Any] = None


# name → {"obj": AgentSpec, "metadata": {...}}
agent_registry: Dict[str, Dict[str, Any]] = {}


def register_agent_spec(spec: AgentSpec) -> AgentSpec:
    """Register ``spec`` by its name (once) and return it."""
    if spec.name in agent_registry:
        raise KeyError(f"agent '{spec.name}' is already registered")
    agent_registry[spec.name] = {"obj": spec, "metadata": {"category": spec.category.value,
                                                           "description": spec.description}}
    return spec


def build_agent(spec: AgentSpec) -> Agent:
    """The Agents SDK ``Agent`` of ``spec``."""
    kwargs: Dict[str, Any] = dict(name=spec.name, instructions=spec.instructions, model=spec.model,
                                  tools=list(spec.tools), handoffs=list(spec.handoffs))
    if spec.output_type is not None:
        kwargs["output_type"] = spec.output_type
    return Agent(**kwargs)


def build_agent_caller(spec: AgentSpec, agent: Agent, session: Any) -> Any:
    """
    A ``@function_tool`` that runs ``agent`` on ``spec.task_prompt`` (``Runner.run``), writes its final output to
    ``session.<spec.session_output_key>`` and returns it to the orchestrator.
    """
    output_key = spec.session_output_key
    task_prompt = spec.task_prompt or "Execute your task now."
    agent_name = spec.name

    async def caller() -> str:
        print(f"\n>>> Orchestrator → {agent_name}")
        result = await Runner.run(agent, task_prompt)
        if output_key:
            setattr(session, output_key, result.final_output)
        return result.final_output

    caller.__name__ = f"call_{agent_name}"
    caller.__doc__ = f"Invoke the {agent_name} agent. {spec.description}"
    return function_tool(caller)


def build_agent_and_caller(spec: AgentSpec, session: Any) -> Tuple[Agent, Any]:
    """Both the agent of ``spec`` and its orchestrator caller: ``(agent, caller_tool)``."""
    agent = build_agent(spec)
    return agent, build_agent_caller(spec, agent, session)
