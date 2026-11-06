"""
MES configuration — settings from environment variables, ``.env.<hostname>`` and ``.env`` (repository root).

Precedence: real environment variables > ``.env.<hostname>`` (this machine only) > ``.env``.

LLM backend
    MES_LLM_PROVIDER        openai | lmstudio | bedrock            (default: openai)
    OPENAI_API_KEY          your OpenAI API key (OPENAI_ORGANIZATION, OPENAI_PROJECT optional)
    MES_OPENAI_MODEL        OpenAI model                           (default: gpt-5.4-nano)
    OPENAI_BASE_URL         optional: another OpenAI-compatible endpoint for the openai backend
    LMSTUDIO_BASE_URL       LM Studio OpenAI-compatible endpoint   (default: http://localhost:1234/v1)
    LMSTUDIO_MODEL          model identifier loaded in LM Studio   (default: google/gemma-4-e4b)
    MES_BEDROCK_MODEL_ID    Bedrock model / inference profile id   (default: us.anthropic.claude-opus-5)
    MES_BEDROCK_PLANNER_MODEL_ID  optional faster Bedrock model for the query planner (default: same model)
    AWS_PROFILE / AWS_REGION_NAME  standard AWS credentials for Bedrock (any source of the AWS credential chain)
    MES_AWS_SSO_LOGIN       1 = run `aws sso login` by itself when the SSO session of AWS_PROFILE has expired
                            (default: 0 — MES stops with a message saying how to log in again)

Evidence sources
    MES_DATA_DIR            folder of the local stores                          (default: data)
    MES_IBKH_DIR            built iBKH store (scripts/build_ibkh_kg.py)        (default: data/ibkh)
    MES_IBKH_SOURCE         iBKH triplet text file used by the build script
    MES_PDF_DIR             local PDF library searched for every question      (default: data/pdfs)
    MES_MIMIC_SOURCE        MIMIC-IV CSV release (hosp/, icu/)                 (default: data/mimic-iv-2.2)
    MES_MIMIC_DIR           RWD analysis store built from it                   (default: data/mimic)
    MES_RWD_ANALYST         code-writing data analyst: auto | on | off         (default: auto = on with Bedrock)
    MES_NCBI_API_KEY        optional NCBI key (10 instead of 3 requests/s)
    MES_CONTACT_EMAIL       contact e-mail for NCBI / Unpaywall (Unpaywall is skipped without it)

Retrieval and synthesis
    MES_ENCODER             sentence-transformers model for ranking and KG linking
    MES_CONTEXT_BUDGET      characters of evidence passed to the synthesis call (default: 24000)
    MES_HTTP_CACHE_TTL_HOURS  lifetime of cached API responses             (default: 72)
    MES_SYSTEM_CERTS        1 = use the OS trust store for HTTPS (needed behind TLS-inspecting proxies)
    MES_CACHE_DIR           answer, HTTP and full-text caches, saved conversations (default: .cache)
"""

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

MES_DIR = Path(__file__).resolve().parent
# The repository: .env, data/ and .cache/ live here
ROOT = MES_DIR.parent
HOSTNAME = socket.gethostname().split(".")[0]

# Machine-specific settings first (.env.<hostname>), then .env: when the folder is shared between computers, a
# per-host file lets each one pick its own LLM backend without editing .env.
ENV_FILES = [f for f in (ROOT / f".env.{HOSTNAME}", ROOT / ".env") if f.exists()]
for _env_file in ENV_FILES:
    load_dotenv(_env_file, override=False)


def _path(name, default):
    """A folder or file setting: ``~`` is expanded, and a relative path is relative to the repository root (not to the
    folder MES is started from), so ``.env`` can say e.g. ``MES_CACHE_DIR=local/cache`` on every computer."""
    value = os.environ.get(name, "").strip()
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else ROOT / path


def _creatable(path):
    """``path`` is a folder that exists or can be created (its nearest existing parent is a writable folder)."""
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK)


def _cache_dir():
    """MES_CACHE_DIR, or the repository's .cache when that folder cannot be used — e.g. an absolute path written on
    another computer: the caches and saved conversations then go to .cache instead of failing the page."""
    path, fallback = _path("MES_CACHE_DIR", ROOT / ".cache"), ROOT / ".cache"
    if path != fallback and not _creatable(path):
        print(f"[MES] MES_CACHE_DIR={path} cannot be created on this computer; using {fallback} instead "
              "(a relative path in .env, e.g. MES_CACHE_DIR=local/cache, works on every computer)", file=sys.stderr)
        return fallback
    return path


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name, default):
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


DATA_DIR = _path("MES_DATA_DIR", ROOT / "data")
if os.environ.get("MES_DATA_DIR", "").strip() and not DATA_DIR.exists():
    print(f"[MES] MES_DATA_DIR={DATA_DIR} does not exist on this computer: the local stores are not found",
          file=sys.stderr)
CACHE_DIR = _cache_dir()
IBKH_DIR = _path("MES_IBKH_DIR", DATA_DIR / "ibkh")
# The iBKH triplet text the build script reads (scripts/build_ibkh_kg.py); only needed to rebuild the store
IBKH_SOURCE = _path("MES_IBKH_SOURCE", DATA_DIR / "ibkh_source" / "whole_triplets_text.tsv")
PDF_DIR = _path("MES_PDF_DIR", DATA_DIR / "pdfs")
# Real-world data: the MIMIC-IV CSV release and the analysis store built from it (scripts/build_mimic_db.py)
MIMIC_SOURCE = _path("MES_MIMIC_SOURCE", DATA_DIR / "mimic-iv-2.2")
MIMIC_DIR = _path("MES_MIMIC_DIR", DATA_DIR / "mimic")
# Claude data analyst for MIMIC-IV questions the study templates cannot answer: auto (on with Claude on Bedrock),
# on (any model, e.g. to try it with LM Studio), off
RWD_ANALYST = os.environ.get("MES_RWD_ANALYST", "auto").strip().lower()
# Editing a protocol and re-running it (🔒 switches of the answer page, locked by default): on (every protocol),
# no_code (the analyst's Python code stays locked, e.g. on a shared server), off (every protocol stays locked)
PROTOCOL_EDIT = os.environ.get("MES_PROTOCOL_EDIT", "on").strip().lower()

ENCODER = os.environ.get("MES_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
CONTEXT_BUDGET = _int("MES_CONTEXT_BUDGET", 24000)
NCBI_API_KEY = os.environ.get("MES_NCBI_API_KEY", "").strip()
CONTACT_EMAIL = os.environ.get("MES_CONTACT_EMAIL", "").strip()
HTTP_CACHE_TTL_HOURS = _int("MES_HTTP_CACHE_TTL_HOURS", 72)
SYSTEM_CERTS = _bool("MES_SYSTEM_CERTS", True)


# ---------------------------------------------------------------------------
# Answer modes (depths). The keys — osler, sackett, snow — stay in caches, saved conversations and the command
# line; what the user reads is the label: Fast, Balanced, Deep.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    tagline: str
    pubmed_max: int          # PMIDs fetched per PubMed query
    literature_max: int      # literature items kept for synthesis
    fulltext_max: int        # open-access full texts fetched among the kept items
    passages_per_doc: int    # passages kept per document
    trials_max: int          # ClinicalTrials.gov records kept
    kg_max: int              # KG triplets / paths kept
    budget_scale: float      # multiplier of CONTEXT_BUDGET
    synthesis_tokens: int    # max output tokens of the synthesis call
    reasoning: bool          # let thinking models reason before answering
    extract_studies: bool    # separate batched study-extraction pass (Sackett/Snow write an
                             # evidence table inside the synthesis instead, which is faster)
    verify_claims: bool      # batched LLM claim check (opt-in; number grounding always runs)
    facets: bool             # Snow: search each facet of the question separately
    completeness_round: bool  # Snow: one targeted follow-up retrieval if a gap is found
    # MIMIC-IV analyst (Claude writes and runs its own analysis code; tools/rwd/analyst.py)
    analyst_steps: int = 8        # model turns (explore / code / submit)
    analyst_seconds: int = 90     # wall-clock budget of the whole analysis
    analyst_effort: str = "low"   # thinking effort of the analyst's turns
    analyst_revisions: int = 0    # rounds in which the analyst fixes problems found by the reviewer
    analyst_sensitivity: bool = False  # ask for at least one sensitivity analysis


MODES = {
    "osler": ModeConfig(
        key="osler", label="Fast", tagline="Quick answer",
        pubmed_max=40, literature_max=8, fulltext_max=3, passages_per_doc=2,
        trials_max=4, kg_max=8, budget_scale=0.75, synthesis_tokens=1800, reasoning=False,
        extract_studies=False, verify_claims=False, facets=False, completeness_round=False,
        analyst_steps=10, analyst_seconds=90, analyst_effort="low", analyst_revisions=0,
    ),
    "sackett": ModeConfig(
        key="sackett", label="Balanced", tagline="Weigh the evidence",
        pubmed_max=80, literature_max=12, fulltext_max=6, passages_per_doc=3,
        trials_max=6, kg_max=10, budget_scale=1.3, synthesis_tokens=3000, reasoning=False,
        extract_studies=False, verify_claims=False, facets=False, completeness_round=False,
        analyst_steps=14, analyst_seconds=210, analyst_effort="medium", analyst_revisions=1,
    ),
    "snow": ModeConfig(
        key="snow", label="Deep", tagline="Comprehensive survey",
        pubmed_max=60, literature_max=16, fulltext_max=8, passages_per_doc=3,
        trials_max=8, kg_max=12, budget_scale=1.7, synthesis_tokens=4000, reasoning=False,
        extract_studies=False, verify_claims=False, facets=True, completeness_round=True,
        analyst_steps=22, analyst_seconds=420, analyst_effort="high", analyst_revisions=2, analyst_sensitivity=True,
    ),
}
DEFAULT_MODE = "osler"
# A mode can also be named by its label ("fast", "balanced", "deep"), e.g. on the command line
MODE_ALIASES = {m.label.lower(): k for k, m in MODES.items()}


def mode_key(mode):
    """The key of a mode named by its key or its label, case-insensitive (anything else is returned as it is)."""
    key = str(mode or DEFAULT_MODE).strip().lower()
    return MODE_ALIASES.get(key, key)


def get_mode(mode):
    """Mode config by key or label (case-insensitive); defaults to Fast (osler)."""
    return MODES.get(mode_key(mode), MODES[DEFAULT_MODE])


# ---------------------------------------------------------------------------
# HTTPS trust store
# ---------------------------------------------------------------------------

_certs_injected = False


def ensure_system_certs():
    """Verify HTTPS with the OS trust store (fixes CERTIFICATE_VERIFY_FAILED behind TLS-inspecting proxies)."""
    global _certs_injected
    if _certs_injected or not SYSTEM_CERTS:
        return
    try:
        import truststore
        truststore.inject_into_ssl()
        _certs_injected = True
    except ImportError:
        pass


ensure_system_certs()
