"""MES pipeline context — shared state for a single evidence-synthesis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mes.agent_framework import PipelineContext


@dataclass
class MESContext(PipelineContext):
    """Mutable shared state of the MES agents (the agent-layer ``session``)."""

    # ---- request ---------------------------------------------------------------
    question: str = ""
    mode: str = "osler"
    sources: List[str] = field(default_factory=lambda: ["literature", "trials", "kg", "rwd"])
    uploads: List[Any] = field(default_factory=list)          # [(file name, PDF bytes)]
    history: List[Dict[str, str]] = field(default_factory=list)

    # ---- fine-grained tool outputs ------------------------------------------------
    plan: Optional[Any] = None                                  # QueryPlan
    query_vector: Optional[Any] = None
    literature: List[Any] = field(default_factory=list)        # EvidenceItem
    trials: List[Any] = field(default_factory=list)
    kg_items: List[Any] = field(default_factory=list)
    kg_info: Dict[str, Any] = field(default_factory=dict)
    rwd_items: List[Any] = field(default_factory=list)         # EvidenceItem (prefix R)
    rwd_studies: List[Dict[str, Any]] = field(default_factory=list)

    # ---- results -----------------------------------------------------------------
    result: Optional[Any] = None                                # MESResult
    study_table: List[Dict[str, Any]] = field(default_factory=list)

    # ---- agent outputs (set by the orchestrator callers) ------------------------------
    supervisor_report: str = ""
    literature_report: str = ""
    kg_report: str = ""
    rwd_report: str = ""
    statistician_report: str = ""
    safechecker_report: str = ""

    def reset(self, question="", mode="osler", sources=None, uploads=None, history=None):
        """Start a new question, keeping the LLM."""
        llm = self.llm
        self.__init__()
        self.llm = llm
        self.question, self.mode = question, mode
        self.sources = list(sources or ["literature", "trials", "kg", "rwd"])
        self.uploads, self.history = list(uploads or []), list(history or [])
