"""
Sandboxed Python sessions for the MIMIC-IV analyst (code written by the language model).

Each session is a separate Python process (``sandbox_worker.py``) that keeps its variables between cells, like a
notebook, and is killed when the analysis ends. Layers of protection, each enough to stop an accidental mistake:

1. Code policy (``check_code``, here and again in the worker): imports only from an allowlist of analysis
   libraries; no ``open`` / ``exec`` / ``eval`` / ``__import__``, no double-underscore attributes, no file reading or
   writing functions (``pd.read_csv``, ``df.to_csv``, ``np.save`` …).
2. Operating system (macOS ``sandbox-exec``): no network, no file writes outside the session's temporary folder, no
   reading of credentials (~/.aws, ~/.ssh, keychains). Elsewhere the process runs without this layer.
3. Process limits: a time limit per cell and a memory limit; the session is restarted when one is exceeded.
4. Data: the worker's database connection is read-only and limited to the MIMIC-IV store; the environment passed
   to the worker carries no credentials.
5. Output: what a cell prints goes to the language model, so it is truncated and identifiers (subject / admission /
   stay ids) and dates are masked; the final result goes through ``submit()``, which only accepts aggregates.
"""

import ast
import atexit
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path

from mes import config

logger = logging.getLogger(__name__)

# the repository root (package `mes`): the worker's working directory and import path
ROOT_DIR = Path(__file__).resolve().parents[3]

# Top-level modules the analysis code may import
ALLOWED_MODULES = {
    "numpy", "pandas", "scipy", "statsmodels", "lifelines", "sklearn", "patsy",
    "math", "statistics", "random", "itertools", "functools", "collections", "re", "json", "datetime", "dataclasses",
    "typing", "warnings", "textwrap", "string", "decimal", "fractions", "heapq", "bisect", "operator", "copy",
}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
                "help", "exit", "quit", "getattr", "setattr", "delattr", "memoryview", "__builtins__"}
# Attribute calls that read or write files, start processes or touch the file system
BANNED_ATTRIBUTES = re.compile(
    r"^(read_\w+|to_(csv|parquet|pickle|excel|json|sql|hdf|feather|stata|orc|clipboard|latex|xml|html)|"
    r"save|savez|savez_compressed|savetxt|load|loadtxt|genfromtxt|fromfile|tofile|memmap|savefig|"
    r"system|popen|spawn\w*|fork|exec\w*|remove|unlink|rmdir|rmtree|chmod|chown|kill|connect|write_\w+|"
    r"from_csv|from_file|open\w*)$")

MAX_OUTPUT = 6000                 # characters of printed output returned to the model per cell
_ID = re.compile(r"(?<![\d.,])[123]\d{7}(?![\d.,])")             # subject_id 1…, hadm_id 2…, stay_id 3… (8 digits)
_DATE = re.compile(r"\b2[0-2]\d\d-[01]\d-[0-3]\d(?:[ T][0-2]\d:[0-5]\d(?::[0-5]\d(?:\.\d+)?)?)?")


def check_code(code):
    """Problems of ``code`` under the sandbox policy (empty list: allowed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                problems.append(f"import from '{node.module}' is not allowed")
            elif any(a.name == "*" for a in node.names):
                problems.append("'import *' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"'{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"attribute '{node.attr}' is not allowed")
            elif BANNED_ATTRIBUTES.match(node.attr):
                problems.append(f"'.{node.attr}' is not allowed (no file, process or network access)")
    seen, unique = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:5]


def mask_output(text, limit=MAX_OUTPUT):
    """Printed output as it may reach the language model: identifiers and dates masked, length limited."""
    text = _DATE.sub("[date]", _ID.sub("[id]", text or ""))
    if len(text) > limit:
        head, tail = int(limit * 0.75), int(limit * 0.2)
        text = f"{text[:head]}\n… [{len(text) - head - tail:,} characters omitted] …\n{text[-tail:]}"
    return text


def os_sandbox_available():
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def _profile(tmpdir):
    home = str(Path.home())
    tmp = [str(Path(tmpdir)), str(Path(tmpdir).resolve())]
    writable = " ".join(f'(subpath "{p}")' for p in dict.fromkeys(tmp))
    secrets = " ".join(f'(subpath "{home}/{d}")' for d in (".aws", ".ssh", ".gnupg", "Library/Keychains", ".netrc",
                                                           ".config/gcloud", ".azure", ".kube", ".docker"))
    return ("(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n"
            f'(allow file-write* {writable} (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") '
            '(regex #"^/dev/fd/") (regex #"^/dev/tty"))\n'
            f"(deny file-read* {secrets})\n")


def _memory_limit_mb():
    configured = os.environ.get("MES_ANALYST_MEMORY_MB")
    if configured:
        return int(configured)
    try:
        import psutil
        return int(min(8192, psutil.virtual_memory().total / 2**20 * 0.25))
    except Exception:
        return 6144


_live = weakref.WeakSet()


@atexit.register
def _close_all():
    for box in list(_live):
        box.close()


class Sandbox:
    """A persistent sandboxed Python session over the MIMIC-IV store."""

    def __init__(self, root=None, cell_timeout=90, memory_mb=None):
        self.root = Path(root or config.MIMIC_DIR)
        self.cell_timeout = cell_timeout
        self.memory_mb = memory_mb or _memory_limit_mb()
        self.proc = None
        self.tmpdir = None
        self._lines = None
        self._lock = threading.Lock()
        self.os_sandbox = os_sandbox_available() and os.environ.get("MES_ANALYST_OS_SANDBOX", "1") != "0"
        self.restarts = 0
        _live.add(self)

    # ------------------------------------------------------------------ process
    def start(self):
        """Start the worker (returns at once; the worker loads its libraries in the background)."""
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.tmpdir = tempfile.mkdtemp(prefix="mes_analyst_")
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(Path.home()), "TMPDIR": self.tmpdir,
                   "PYTHONPATH": str(ROOT_DIR), "PYTHONHASHSEED": "0",
                   "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                   "MES_RWD_THREADS": os.environ.get("MES_RWD_THREADS", "4"),
                   "MES_RWD_MEMORY": f"{max(1, self.memory_mb // 2560)}GB", "LANG": "en_US.UTF-8"}
            cmd = [sys.executable, "-u", "-m", "mes.tools.rwd.sandbox_worker", str(self.root), self.tmpdir]
            if self.os_sandbox:
                cmd = ["sandbox-exec", "-p", _profile(self.tmpdir)] + cmd
            self._stderr = open(Path(self.tmpdir) / "worker.log", "w")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                         cwd=str(ROOT_DIR), env=env, text=True, bufsize=1,
                                         start_new_session=True)
            self._lines = queue.Queue()
            threading.Thread(target=self._read, args=(self.proc, self._lines), daemon=True).start()

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _kill(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                pass
        self._kill()
        if getattr(self, "_stderr", None):
            self._stderr.close()
            self._stderr = None
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def worker_log(self):
        try:
            return (Path(self.tmpdir) / "worker.log").read_text()[-2000:] if self.tmpdir else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ cells
    def run(self, code, timeout=None):
        """Run a cell. Returns {ok, stdout (masked), error, submitted, seconds, restarted}."""
        problems = check_code(code)
        if problems:
            return {"ok": False, "stdout": "", "error": "Code not run — " + "; ".join(problems), "submitted": None,
                    "seconds": 0.0, "policy": True}
        timeout = timeout or self.cell_timeout
        restarted = self.proc is None or self.proc.poll() is not None
        if restarted:
            self.start()
        t = time.time()
        try:
            self.proc.stdin.write(json.dumps({"op": "run", "code": code}) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            self._kill()
            return {"ok": False, "stdout": "", "error": f"the Python session stopped ({e}); it will be restarted",
                    "submitted": None, "seconds": 0.0, "restarted": True, "killed": True}
        reply, reason = None, ""
        deadline = t + timeout
        while reply is None:
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                line = ""
            if line is None:
                reason = "the Python session crashed" + (f": {self.worker_log()[-400:]}" if self.worker_log() else "")
                break
            if line:
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
            if time.time() > deadline:
                reason = f"the cell ran longer than {timeout:.0f} s"
                break
            rss = self._rss_mb()
            if rss and rss > self.memory_mb:
                reason = f"the cell used more than {self.memory_mb / 1024:.1f} GB of memory"
                break
        if reply is None:
            self._kill()
            self.restarts += 1
            return {"ok": False, "stdout": "", "submitted": None, "seconds": round(time.time() - t, 2),
                    "error": f"{reason}; the session was restarted and its variables are lost — use smaller "
                             "intermediate results (aggregate in SQL) and re-create what you need",
                    "restarted": True, "killed": True}
        reply["stdout"] = mask_output(reply.get("stdout", ""))
        if reply.get("error"):
            reply["error"] = mask_output(reply["error"], 2500)
        reply["seconds"] = round(time.time() - t, 2)
        reply["restarted"] = restarted and self.restarts > 0
        return reply

    def _rss_mb(self):
        try:
            import psutil
            proc = psutil.Process(self.proc.pid)
            return sum(p.memory_info().rss for p in [proc] + proc.children(recursive=True)) / 2**20
        except Exception:
            return None
