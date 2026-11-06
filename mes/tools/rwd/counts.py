"""
Live patient counts of a study design, for the cohort builder under a study card: the flow of patients through the
design's criteria exactly as its analysis counts them (``Compiler.build`` with only the columns the attrition needs),
without the analysis. A design counted before is answered from a small cache, so switching a criterion off and on
again is instant. Aggregates only; counts from 1 to 9 are shown as "<10" by the page (``shown``).

    flow(spec_dict) -> {"steps": [{"id", "label", "n"}], "groups": {"treated", "control"} | None, "n",
                        "errors", "warnings", "seconds"}
"""

import json
import threading
import time
from collections import OrderedDict

from mes.tools.rwd import library as L

SMALL = 10                  # counts below this are not shown (small cells)
MAX_CACHE = 160
_cache = OrderedDict()
_typical = {}
_lock = threading.Lock()


def shown(n):
    """A count as the page shows it: None for 1–9 patients."""
    return None if isinstance(n, int) and 0 < n < SMALL else n


def _key(store, spec_dict):
    return (str(store.root), store.catalog().get("built_at"), json.dumps(spec_dict, sort_keys=True, default=str))


def flow(spec_dict, index=None, store=None, timeout=60):
    """The flow of patients of a design (a ``StudySpec.to_dict()``); ``errors`` when it is not valid or did not run."""
    from mes.tools.rwd.compiler import Compiler
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.db import get_store
    from mes.tools.rwd.spec import StudySpec, validate
    store = store or get_store()
    if not store.available:
        return {"steps": [], "groups": None, "n": None, "warnings": [], "seconds": 0,
                "errors": ["the MIMIC-IV store is not built on this computer"]}
    key = _key(store, spec_dict)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return dict(_cache[key], cached=True)
    t = time.time()
    out = {"steps": [], "groups": None, "n": None, "warnings": [], "errors": [], "seconds": 0}
    try:
        spec = StudySpec.from_dict(spec_dict)
        errors, warnings = validate(spec, index or get_index())
    except Exception as e:                       # not a design at all (a damaged record)
        return {**out, "errors": [f"not a study design: {type(e).__name__}: {e}"]}
    out["warnings"] = list(warnings)
    if errors:
        out["errors"] = list(errors)
    else:
        try:
            cohort = Compiler(index or get_index(), store).build(spec, timeout=timeout, counts_only=True)
        except TimeoutError as e:
            return {**out, "errors": [f"the count did not finish: {e}"]}
        except Exception as e:
            return {**out, "errors": [f"the count did not run: {type(e).__name__}: {e}"]}
        out["steps"] = [{"id": i, "label": str(label), "n": int(n)}
                        for i, (label, n) in zip(cohort.step_ids, cohort.attrition)]
        out["n"] = int(len(cohort.df))
        if spec.analysis == "compare" and "treated" in cohort.df:
            treated = int(cohort.df["treated"].sum())
            out["groups"] = {"treated": treated, "control": out["n"] - treated}
    out["seconds"] = round(time.time() - t, 3)
    with _lock:
        _cache[key] = out
        while len(_cache) > MAX_CACHE:
            _cache.popitem(last=False)
    return dict(out)


def typical(base, variable, store=None):
    """The median of a numeric variable in the population (a new criterion's first threshold), or None."""
    from mes.tools.rwd.db import get_store
    if variable not in L.VARIABLES or L.VARIABLES[variable][1] != "numeric" or base not in L.BASES:
        return None
    store = store or get_store()
    key = (str(store.root), base, variable)
    if key not in _typical:
        b = L.BASES[base]
        try:
            v = store.query(f"SELECT median({variable}) AS v FROM {b['table']} WHERE {b['where']}").v[0]
            _typical[key] = None if v is None or v != v else float(v)
        except Exception:
            _typical[key] = None
    return _typical[key]


def clear():
    """Forget the counted designs (tests; a rebuilt store is recognised by its build time anyway)."""
    with _lock:
        _cache.clear()
        _typical.clear()
