#!/bin/bash
# MES — set up a computer: Python environment, packages, settings, model download, data and LLM checks, tests.
# Run from anywhere; safe to run again (it only adds what is missing).
#
#   bash mes/scripts/install.sh                  a virtual environment in <repository>/.venv (needs python3.11)
#   bash mes/scripts/install.sh --conda [NAME]   a conda env instead (default name: mes; created if missing)
#   options: --build-ibkh   build the iBKH store if it is missing (needs MES_IBKH_SOURCE, ~1 min)
#            --build-mimic  build the MIMIC-IV store if it is missing (needs the MIMIC-IV 2.2 CSVs, ~5 min, 6 GB)
#            --no-tests     skip the offline tests at the end
# Behind a TLS-inspecting proxy, conda and pip need the OS trust store: the script sets CONDA_SSL_VERIFY=truststore
# and uses pip's truststore feature.

ENV_NAME=""; BUILD_IBKH=""; BUILD_MIMIC=""; TESTS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --conda) if [ -n "${2:-}" ] && [ "${2#--}" = "$2" ]; then ENV_NAME="$2"; shift 2; else ENV_NAME="mes"; shift; fi ;;
    --build-ibkh) BUILD_IBKH=1; shift ;;
    --build-mimic) BUILD_MIMIC=1; shift ;;
    --no-tests) TESTS=""; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok() { printf '   \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$1"; }
fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1"; exit 1; }

MES="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$MES")"

say "1/7 Repository"
[ -f "$MES/config.py" ] && [ -f "$ROOT/requirements.txt" ] || fail "run this script from a complete MES repository"
ok "$ROOT"

say "2/7 Python environment"
export CONDA_SSL_VERIFY=truststore
if [ -z "$ENV_NAME" ]; then
  BASE=$(command -v python3.11 || true)
  [ -n "$BASE" ] || fail "python3.11 not found (install it, or use --conda)"
  [ -x "$ROOT/.venv/bin/python" ] || "$BASE" -m venv "$ROOT/.venv" || fail "could not create $ROOT/.venv"
  PY="$ROOT/.venv/bin/python"
else
  command -v conda >/dev/null 2>&1 || fail "conda not found: install Miniconda (https://docs.conda.io) or run without --conda"
  CONDA_BASE=$(conda info --base)
  if [ ! -x "$CONDA_BASE/envs/$ENV_NAME/bin/python" ]; then
    echo "   creating conda env $ENV_NAME (Python 3.11) …"
    conda create -n "$ENV_NAME" python=3.11 -y -q || fail "conda create failed (behind a proxy: conda config --set ssl_verify truststore)"
  fi
  PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi
VERSION=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ok "$PY (Python $VERSION)"
[ "$VERSION" = "3.11" ] || warn "MES is tested with Python 3.11"

say "3/7 Packages"
"$PY" -m pip install -q --use-feature=truststore --upgrade pip >/dev/null 2>&1
if "$PY" -m pip install -q --use-feature=truststore -r "$ROOT/requirements-lock.txt"; then
  ok "tested versions installed (requirements-lock.txt)"
else
  warn "the tested versions are not available here; installing the newest compatible ones (requirements.txt)"
  "$PY" -m pip install -q --use-feature=truststore -r "$ROOT/requirements.txt" || fail "pip install failed (see above)"
fi

say "4/7 Settings (.env)"
if [ -f "$ROOT/.env" ]; then
  ok ".env present"
else
  cp "$ROOT/.env.example" "$ROOT/.env"
  warn "created .env from .env.example — add your OPENAI_API_KEY, or choose another backend (MES_LLM_PROVIDER)"
fi
HOST=$("$PY" -c "import socket; print(socket.gethostname().split('.')[0])")
[ -f "$ROOT/.env.$HOST" ] && ok "machine-specific settings: .env.$HOST"

say "5/7 Sentence encoder (downloaded once from Hugging Face, ~90 MB)"
cd "$ROOT" || fail "cannot enter $ROOT"
if "$PY" -W ignore -c "from mes.tools.ranking_tools import encode; encode(['test'])" >/dev/null 2>&1; then
  ok "encoder ready"
else
  warn "encoder download failed — behind a TLS-inspecting proxy set MES_SYSTEM_CERTS=1 in .env and run again"
fi

say "6/7 Data and LLM backend"
"$PY" -W ignore - "$BUILD_IBKH" "$BUILD_MIMIC" <<'EOF'
import subprocess, sys
from mes import config
build_ibkh, build_mimic = sys.argv[1] == "1", sys.argv[2] == "1"
ibkh = config.IBKH_DIR / "ibkh.duckdb"
if ibkh.exists():
    print("   ✓ iBKH store:", config.IBKH_DIR)
elif build_ibkh and config.IBKH_SOURCE.exists():
    subprocess.run([sys.executable, "-m", "mes.scripts.build_ibkh_kg"], check=False)
else:
    print(f"   ! iBKH store missing ({ibkh}): optional, the knowledge-graph source stays off — see docs/DATA.md "
          f"(download the store, or build it from {config.IBKH_SOURCE} with --build-ibkh)")
store = config.MIMIC_DIR / "catalog.json"
raw = config.MIMIC_SOURCE / "hosp" / "admissions.csv.gz"
if store.exists():
    print("   ✓ MIMIC-IV store:", config.MIMIC_DIR)
elif build_mimic and raw.exists():
    subprocess.run([sys.executable, "-m", "mes.scripts.build_mimic_db"], check=False)
else:
    print(f"   ! MIMIC-IV store missing ({config.MIMIC_DIR}): optional, the real-world data source stays off — put "
          f"your PhysioNet copy of the MIMIC-IV 2.2 CSVs in {config.MIMIC_SOURCE} and run again with --build-mimic "
          "(docs/DATA.md)")
from mes.llm_zoo import bedrock_model_id, get_provider, lmstudio_base_url, lmstudio_model, openai_model
import os
if get_provider() == "openai":
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        print(f"   ✓ OpenAI {openai_model()}: API key set")
    else:
        print(f"   ! OpenAI {openai_model()}: no API key — set OPENAI_API_KEY in .env")
elif get_provider() == "lmstudio":
    import requests
    try:
        ids = [m["id"] for m in requests.get(lmstudio_base_url().rstrip("/") + "/models", timeout=3).json()["data"]]
        print(f"   {'✓' if lmstudio_model() in ids else '!'} LM Studio {lmstudio_base_url()}: "
              f"{lmstudio_model()} {'loaded' if lmstudio_model() in ids else 'NOT loaded (lms load ...)'}")
    except Exception:
        print(f"   ! LM Studio not reachable at {lmstudio_base_url()} — start it: lms server start")
else:
    import os
    try:
        import boto3
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None)
        session.client("sts", region_name=os.environ.get("AWS_REGION_NAME", "us-east-1")).get_caller_identity()
        print(f"   ✓ Bedrock {bedrock_model_id()}: AWS credentials found")
    except Exception as e:
        print(f"   ! Bedrock {bedrock_model_id()}: no usable AWS credentials ({type(e).__name__}) — configure them "
              "(for an SSO profile: aws sso login --profile <profile>, and AWS_PROFILE in .env)")
EOF

say "7/7 Offline tests"
if [ -n "$TESTS" ]; then
  "$PY" -W ignore -m pytest "$MES/tests" -q -p no:cacheprovider 2>&1 | grep -E "[0-9]+ (passed|failed)|error" | tail -3
else
  echo "   skipped"
fi

printf '\n\033[32m✓ MES is installed.\033[0m Start it with:\n'
echo "   $ROOT/run.sh --open          # MES at http://localhost:8501 (macOS: or double-click MES.command)"
echo "   another port: add --server.port 8502"
