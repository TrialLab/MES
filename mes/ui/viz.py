"""
MES charts as one Streamlit custom component (``st.components.v2``, no iframe): the study card of a real-world data
analysis, the answer at a glance, the map of the evidence and the knowledge-graph network, drawn in the browser from
a spec built in Python (ui/study_card.py, ui/answer_viz.py, ui/cohort_builder.py). The JavaScript is ui/web/viz_*.js,
joined in the order below into one module; the styles are ui/web/viz*.css, in the WCM palette, light and dark.

    viz.show("study", spec, key=...)       # a view: study | blocks | glance | landscape | network

What the browser keeps between reruns (the open tab, toggles, a dragged range) lives under the key. Interactions stay
in the browser: a chart never reruns the page. New chart types: a function in a viz_*.js file, its name in
viz_study.js ``CHARTS``, and a spec builder in Python.
"""

import json
import math
from pathlib import Path

import streamlit as st

from mes.ui.composer import is_dark

WEB = Path(__file__).parent / "web"
JS_FILES = ("viz_core.js", "viz_rows.js", "viz_km.js", "viz_balance.js", "viz_flow.js", "viz_plots.js", "viz_cohort.js",
            "viz_similar.js", "viz_patterns.js", "viz_study.js", "viz_answer.js", "viz_main.js")
CSS_FILES = ("viz.css", "viz_cohort.css", "viz_similar.css", "viz_patterns.css")
VIEWS = ("study", "blocks", "glance", "landscape", "network")
MAX_TEXT = 4000


def _js():
    return "\n".join((WEB / name).read_text(encoding="utf-8") for name in JS_FILES)


_component = st.components.v2.component("mes_viz", html='<div class="mv"></div>',
                                        css="\n".join((WEB / name).read_text(encoding="utf-8") for name in CSS_FILES),
                                        js=_js())


def clean(obj, depth=0):
    """A JSON-safe copy for the browser: NaN and infinities → None, numpy numbers → numbers, long texts cut."""
    if depth > 40:
        return None
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_TEXT else obj[:MAX_TEXT - 1] + "…"
    if isinstance(obj, dict):
        return {str(k): clean(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v, depth + 1) for v in obj]
    item = getattr(obj, "item", None)          # numpy scalars
    if callable(item):
        try:
            return clean(item(), depth + 1)
        except Exception:
            return None
    tolist = getattr(obj, "tolist", None)      # numpy arrays
    if callable(tolist):
        return clean(tolist(), depth + 1)
    return str(obj)


def show(view, spec, key):
    """Draw a view of ``spec`` (nothing when the spec is empty)."""
    if not spec or view not in VIEWS:
        return
    data = {"key": key, "dark": is_dark(), "view": view, "spec": clean(spec)}
    json.dumps(data)                           # what reaches the browser must be JSON
    _component(key=key, data=data, height="content")
