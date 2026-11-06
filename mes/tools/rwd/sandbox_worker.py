"""
Worker process of the analyst sandbox (started by sandbox.py; not imported by MES itself).

    python -m mes.tools.rwd.sandbox_worker <store folder> <temporary folder>

Protocol: one JSON request per line on stdin ({"op": "run", "code": ...} or {"op": "close"}), one JSON reply per line
on the original stdout. Cells share one namespace (variables persist), like a notebook. What a cell prints is
captured and returned; library output written directly to the process's stdout goes to /dev/null so it cannot
disturb the protocol.
"""

import ast
import builtins
import contextlib
import io
import json
import os
import random
import re
import sys
import time
import traceback
import warnings


ORDERED = re.compile(r"\border\s+by\b", re.I)
ID_COLUMNS = ("subject_id", "hadm_id", "stay_id")


def _stable(df, query):
    """Rows of a query without ORDER BY come back in a thread-dependent order (joins, GROUP BY), which moves the
    numbers of bootstraps and fitted models a little; sorted rows make a re-run session give the same result."""
    if ORDERED.search(query) or not 1 < len(df) <= 2_000_000:
        return df
    ids = [c for c in ID_COLUMNS if c in df.columns]
    for keys in (ids + [c for c in df.columns if c not in ids], ids):
        try:
            return df.sort_values(keys, kind="mergesort", na_position="last", ignore_index=True) if keys else df
        except Exception:
            continue
    return df


def _namespace(root, state):
    import numpy as np
    import pandas as pd

    from mes.tools.rwd import library as L
    from mes.tools.rwd import stats
    from mes.tools.rwd.compiler import Compiler
    from mes.tools.rwd.concepts import ConceptIndex, code_predicate
    from mes.tools.rwd.db import Store
    from mes.tools.rwd.result_check import normalize_result
    from mes.tools.rwd.sandbox import ALLOWED_MODULES, BANNED_NAMES
    from mes.tools.rwd.spec import StudySpec, validate

    np.random.seed(0)
    random.seed(0)
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 160)
    store = Store(root)
    index = ConceptIndex(store)

    def sql(query, params=None):
        """Run a read-only DuckDB query on MIMIC-IV and return a pandas DataFrame (rows stay in this session)."""
        return _stable(store.query(query, params, timeout=180), query)

    def cohort(spec):
        """Cohort of a MES study design (dict): .df (one row per patient), .attrition, .sql. See the instructions."""
        s = StudySpec.from_dict(spec)
        errors, _warnings = validate(s, index)
        if errors:
            raise ValueError("invalid cohort spec: " + "; ".join(errors))
        return Compiler(index, store).build(s)

    def drug_names(concept):
        """Medication names (mes.drugs.drug) matched by a drug class or ingredient, e.g. 'drugclass:statin'."""
        return index.drug_names_for(concept if ":" in concept else f"drug:{concept}")

    def show(obj, n=30):
        """Print a DataFrame (first n rows) or any object."""
        if hasattr(obj, "head") and hasattr(obj, "to_string"):
            print(obj.head(n).to_string())
            if len(obj) > n:
                print(f"… {len(obj) - n:,} more rows")
        else:
            print(obj)

    def submit(result):
        """Submit the final result (dict, see the result format). Replaces an earlier submission."""
        clean = normalize_result(result)
        state["submitted"] = clean
        state["submitted_now"] = clean
        if clean.get("feasible") is False:
            print("✓ submitted: not answerable with MIMIC-IV")
        else:
            print(f"✓ result submitted: {len(clean['estimates'])} estimate(s), {len(clean['tables'])} table(s)")

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(f"import of '{name}' is not allowed in the analysis sandbox")
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in vars(builtins).items() if k not in BANNED_NAMES}
    safe["__import__"] = guarded_import
    return {"__builtins__": safe, "__name__": "__analysis__", "np": np, "pd": pd, "sql": sql, "cohort": cohort,
            "stats": stats, "L": L, "code_predicate": code_predicate, "drug_names": drug_names, "show": show,
            "submit": submit}


def _value_text(value):
    if hasattr(value, "to_string") and hasattr(value, "head"):
        text = value.head(40).to_string()
        return text + (f"\n… {len(value) - 40:,} more rows" if len(value) > 40 else "")
    return repr(value)[:4000]


def _error_text(exc, code):
    lines = code.splitlines()
    frames = traceback.extract_tb(exc.__traceback__)
    out = []
    for f in [f for f in frames if f.filename == "<cell>"][-3:]:
        src = lines[f.lineno - 1].strip() if f.lineno and f.lineno <= len(lines) else ""
        out.append(f"  cell line {f.lineno}: {src}")
    last = frames[-1] if frames else None
    if last is not None and last.filename != "<cell>":
        out.append(f"  in {os.path.basename(last.filename)}, {last.name}()")
    message = str(exc)
    if len(message) > 1500:
        message = message[:1500] + " …"
    return "\n".join(["Traceback (most recent call last):"] + out + [f"{type(exc).__name__}: {message}"])


def run_cell(ns, state, code):
    from mes.tools.rwd.sandbox import check_code
    problems = check_code(code)
    if problems:
        return {"ok": False, "stdout": "", "error": "Code not run — " + "; ".join(problems), "submitted": None}
    tree = ast.parse(code)
    state["submitted_now"] = None
    buffer = io.StringIO()
    ok, error = True, ""
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer), warnings.catch_warnings():
        warnings.simplefilter("once")
        try:
            if tree.body and isinstance(tree.body[-1], ast.Expr):     # show the value of a final expression
                body = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(body, "<cell>", "exec"), ns)
                value = eval(compile(ast.Expression(tree.body[-1].value), "<cell>", "eval"), ns)
                if value is not None:
                    print(_value_text(value))
            else:
                exec(compile(tree, "<cell>", "exec"), ns)
        except Exception as e:
            ok, error = False, _error_text(e, code)
    return {"ok": ok, "stdout": buffer.getvalue(), "error": error, "submitted": state.get("submitted_now")}


def main():
    root, tmpdir = sys.argv[1], sys.argv[2]
    os.chdir(tmpdir)
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    state = {"submitted": None, "submitted_now": None}
    ns = _namespace(root, state)
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("op") == "close":
            break
        if request.get("op") == "run":
            t = time.time()
            reply = run_cell(ns, state, request.get("code", ""))
            reply["seconds"] = round(time.time() - t, 3)
            proto.write(json.dumps(reply, default=str) + "\n")


if __name__ == "__main__":
    main()
