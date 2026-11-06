#!/usr/bin/env bash
# Start MES (the Streamlit app in mes/app.py) at http://localhost:8501.
#   * finds the Python environment: the active conda env, <repository>/.venv, or a conda env named "mes"
#   * file watcher off: Streamlit's module watcher stalls on torch/transformers lazy imports and slows answers
#   * Streamlit usage statistics off: nothing leaves the machine except the configured LLM and literature APIs
#   * reports, before starting, the LLM backend configured in .env and which local stores are there (iBKH, MIMIC-IV);
#     a missing store only turns its source off — the literature and ClinicalTrials.gov work without them
# Usage:  ./run.sh [--open] [--lan] [--server.port 8501] [other streamlit options]
#   --open       open the page in the browser once it is ready
#   --lan        also reachable from other computers of the network (default: this computer only — the page has
#                no login, and MIMIC-IV results may only be shown to people covered by its data use agreement)
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"

open_browser=""; port="8501"; address="localhost"; args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --open) open_browser=1; shift ;;
    --lan) address=""; shift ;;
    --server.address) address=""; args+=("$1" "$2"); shift 2 ;;
    --server.address=*) address=""; args+=("$1"); shift ;;
    --server.port) port="$2"; args+=("$1" "$2"); shift 2 ;;
    --server.port=*) port="${1#*=}"; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

py=""
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ] && "${CONDA_PREFIX}/bin/python" -c "import streamlit" 2>/dev/null; then
  py="${CONDA_PREFIX}/bin/python"
elif [ -x "${root}/.venv/bin/python" ]; then
  py="${root}/.venv/bin/python"
else
  for base in "$(conda info --base 2>/dev/null || true)" "$HOME/miniconda3" "$HOME/anaconda3" /opt/anaconda3 \
              /opt/miniconda3 /opt/homebrew/Caskroom/miniconda/base; do
    if [ -n "$base" ] && [ -x "$base/envs/mes/bin/python" ]; then py="$base/envs/mes/bin/python"; break; fi
  done
fi
[ -n "$py" ] || py="$(command -v python3 || command -v python)"
"$py" -c "import streamlit" 2>/dev/null || {
  echo "[MES] Streamlit is not installed for $py — run: bash mes/scripts/install.sh"; exit 1; }

cd "$root" && "$py" -W ignore - <<'PY' || true
import os, sys
sys.path[:0] = [os.path.abspath(".")]
from mes import config
from mes.llm_zoo import bedrock_model_id, get_provider, lmstudio_base_url, lmstudio_model, openai_model
if get_provider() == "openai":
    has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
    print(f"[MES] OpenAI · {openai_model()}: " + ("API key set" if has_key else "NO API KEY — set OPENAI_API_KEY in .env"))
elif get_provider() == "lmstudio":
    import requests
    try:
        ids = [m["id"] for m in requests.get(lmstudio_base_url().rstrip("/") + "/models", timeout=2).json()["data"]]
        loaded = lmstudio_model() in ids
        print(f"[MES] LM Studio {lmstudio_base_url()} · {lmstudio_model()}: {'loaded' if loaded else 'NOT loaded'}")
        if not loaded:
            print(f"[MES]   run: lms load {lmstudio_model()} -c 32768 --parallel 4 -y")
    except Exception:
        print(f"[MES] LM Studio not reachable at {lmstudio_base_url()} — run: lms server start")
else:
    print(f"[MES] Bedrock · {bedrock_model_id()} (AWS credentials from the standard AWS chain"
          + (f", profile {os.environ['AWS_PROFILE']}" if os.environ.get("AWS_PROFILE") else "") + ")")
if not (config.IBKH_DIR / "ibkh.duckdb").exists():
    print(f"[MES] iBKH store not found in {config.IBKH_DIR} — the knowledge-graph source is off "
          "(docs/DATA.md: download or build it)")
if not ((config.MIMIC_DIR / "catalog.json").exists() and (config.MIMIC_DIR / "mes" / "icu_stays.parquet").exists()):
    print(f"[MES] MIMIC-IV store not found in {config.MIMIC_DIR} (needs catalog.json and mes/icu_stays.parquet) — "
          "the real-world data source is off (docs/DATA.md: build it, or copy a built store there)")
PY

if [ -n "$open_browser" ]; then
  ( for _ in $(seq 1 120); do
      if curl -s -m 1 "http://127.0.0.1:${port}/_stcore/health" >/dev/null 2>&1; then
        open "http://localhost:${port}" 2>/dev/null || xdg-open "http://localhost:${port}" 2>/dev/null; break
      fi
      sleep 1
    done ) &
fi
[ -n "$address" ] && args+=(--server.address "$address")
cd "$root"
exec "$py" -m streamlit run mes/app.py --server.fileWatcherType none --server.headless true \
     --browser.gatherUsageStats false ${args[@]+"${args[@]}"}
