"""
The canvas as a Streamlit custom component (``st.components.v2``, no iframe).

``register(name, domain_js)`` builds one component per use case: ``graph.js`` (the generic graph) + the use case's own
JS (its compiler and the two texts the canvas cannot work out, as ``Canvas = {graph, view}``) + ``canvas.js`` (the
builder's interface, which reads ``Canvas``), with ``canvas.css``. The browser keeps the draft, with undo / redo, and
sends an action when a button is pressed::

    {"type": "apply" | "run" | "save" | "delete", "graph", "question", "id", "at"}
"""

from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent
HTML = '<div class="tb"></div>'


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _noop():
    pass


def is_dark():
    """Streamlit draws the page in its dark theme (the browser reports the theme with every run)."""
    try:
        return getattr(st.context.theme, "type", None) == "dark"
    except Exception:
        return False


def register(name, domain_js):
    """A canvas component for one use case; returns ``canvas(key, **data) -> action | None``."""
    js = "\n".join([_read(_DIR / "graph.js"), _read(domain_js), _read(_DIR / "canvas.js")])
    component = st.components.v2.component(name, html=HTML, css=_read(_DIR / "canvas.css"), js=js)

    def canvas(key, **data):
        result = component(key=key, data={"key": key, **data}, on_action_change=_noop, height="content")
        return getattr(result, "action", None)

    return canvas
