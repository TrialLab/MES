"""
MES result schema — standard structures shared by the evidence sources, the
synthesis step and the UI.

Evidence items carry a stable ID with a source prefix (``L`` literature,
``T`` ClinicalTrials.gov, ``K`` knowledge graph, ``R`` real-world data). The
synthesis call cites these IDs; the UI renumbers them [1], [2], ... in order of
first citation. ``MESResult.to_dict()`` is the JSON contract read by the UI.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

SOURCE_TYPES = {
    "literature": {"prefix": "L", "label": "Literature"},
    "trial": {"prefix": "T", "label": "ClinicalTrials.gov"},
    "kg": {"prefix": "K", "label": "Knowledge graph (iBKH)"},
    "rwd": {"prefix": "R", "label": "Real-world data"},
}
PREFIX_TO_TYPE = {v["prefix"]: k for k, v in SOURCE_TYPES.items()}


@dataclass
class Passage:
    text: str
    section: str = ""      # e.g. "Abstract", "Results", "Table 2", "Primary outcome"
    score: float = 0.0


@dataclass
class EvidenceItem:
    id: str                               # "L1", "T3", "K2" — stable within one answer
    source_type: str                      # literature | trial | kg | rwd
    title: str
    url: str = ""
    year: Optional[int] = None
    venue: str = ""                       # journal, registry or "iBKH"
    design: str = ""                      # e.g. "Meta-analysis", "Randomized controlled trial", "Curated relation"
    evidence_level: int = 5               # 1 = strongest design ... 5 = weakest / contextual
    origin: str = ""                      # e.g. "PubMed", "Europe PMC full text", "Local PDF", "iBKH"
    identifiers: Dict[str, str] = field(default_factory=dict)   # pmid, doi, pmcid, nct, file
    passages: List[Passage] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)          # source-specific details
    score: float = 0.0
    full_text: bool = False

    @property
    def label(self):
        return SOURCE_TYPES.get(self.source_type, {}).get("label", self.source_type)

    def text_blob(self):
        """All text of the item (title, passages, structured details) for grounding checks."""
        parts = [self.title] + [p.text for p in self.passages] + [self.meta.get("abstract") or ""]
        parts += [json.dumps(self.meta.get(k), default=str) for k in ("results", "extraction", "edges") if self.meta.get(k)]
        return "\n".join(parts)

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in dict(d).items() if k in EvidenceItem.__dataclass_fields__}
        d["passages"] = [Passage(**{k: v for k, v in p.items() if k in Passage.__dataclass_fields__})
                         for p in d.get("passages", [])]
        return EvidenceItem(**d)


@dataclass
class QueryPlan:
    question: str
    standalone_question: str = ""
    question_type: str = "therapy"
    pico: Dict[str, str] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)    # {"name", "type", "synonyms"}
    pubmed_query: str = ""
    pubmed_query_broad: str = ""
    pubmed_tiers: List[str] = field(default_factory=list)           # queries from most to least specific
    trial_terms: Dict[str, str] = field(default_factory=dict)       # condition, intervention, outcome
    facets: List[Dict[str, str]] = field(default_factory=list)      # {"name", "question", "pubmed_query"}
    language: str = "English"
    fallback: bool = False                                           # True when the LLM plan could not be parsed
    error: str = ""                                                  # LLM backend error of the planner call
    in_scope: bool = True                                            # False: not a health / biomedical question
    follow_ups: List[str] = field(default_factory=list)              # related questions suggested under the answer
    rwd_previous: str = ""                                           # the MIMIC-IV design of the turn before (a
                                                                     # follow-up refines it)

    @property
    def query(self):
        return self.standalone_question or self.question


@dataclass
class MESResult:
    question: str
    mode: str
    sources: List[str]
    plan: Dict[str, Any]
    evidence: List[EvidenceItem]
    answer_markdown: str = ""                          # raw synthesis output with source IDs ([L1], [K2])
    answer_display: str = ""                           # renumbered citations ([1], [2])
    sections: Dict[str, str] = field(default_factory=dict)
    takeaways: List[str] = field(default_factory=list)
    citation_order: List[str] = field(default_factory=list)   # display index i+1 -> evidence ID
    verification: Dict[str, Any] = field(default_factory=dict)
    study_table: List[Dict[str, Any]] = field(default_factory=list)
    source_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # per-source counts, errors, timings
    literature: List[str] = field(default_factory=list)  # parts of the literature source used (empty: all of them)
    timings: Dict[str, float] = field(default_factory=dict)
    llm: str = ""
    n_llm_calls: int = 0
    history: List[Dict[str, str]] = field(default_factory=list)
    rwd_studies: List[Dict[str, Any]] = field(default_factory=list)  # RWD study records (spec, SQL, results)
    # an answer written with a protocol the user edited (tools/protocol_tools.py): version, changes, what ran again;
    # empty for the protocol MES made (version 1)
    protocol: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    version: str = SCHEMA_VERSION
    from_cache: bool = False
    notice: str = ""                                   # MES did not search (not a question, out of scope): why

    def evidence_by_id(self):
        return {e.id: e for e in self.evidence}

    def cited_evidence(self):
        by_id = self.evidence_by_id()
        return [by_id[i] for i in self.citation_order if i in by_id]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in dict(d).items() if k in MESResult.__dataclass_fields__}   # other versions' fields
        d["evidence"] = [EvidenceItem.from_dict(e) for e in d.get("evidence", [])]
        return MESResult(**d)


def save_result(result, path):
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return path


def load_result(path):
    with open(path) as f:
        return MESResult.from_dict(json.load(f))
