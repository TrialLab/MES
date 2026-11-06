"""
Harness app for ``test_composer.py``: renders one of the question boxes with the custom component
replaced by a stub, so the Python side can be tested without a browser.

The test sets, before running:

    site            "probe" | "landing" | "followup"
    submit_payload  what the stub hands back as the sent question (None = nothing was sent)
    edited_payload  what it hands back as an edit (None = the text was not touched)
    action_payload  what it hands back as a menu action (None = nothing was chosen)
    depth_payload   what it hands back as a depth change (None = the depth was not touched)

and reads afterwards:

    composer_calls  the ``data`` dict of every ``composer()`` call, in order, across reruns
    probe_result    the return value of the "probe" call

Not named ``test_*`` on purpose: pytest must not collect it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # repository root

import streamlit as st                                                 # noqa: E402

from mes.ui import composer as composer_module                          # noqa: E402
from mes.ui import warmup                                              # noqa: E402

# The modes the "probe" site offers: page.py's _mode_menu() shape, plus a heading and a trailing action
# (what a task picker would add) — one call that exercises every kind of menu entry.
PROBE_MODES = [{"key": "", "kind": "group", "label": "Answer a question"},
               {"key": "osler", "label": "Fast", "desc": "Quick", "time": "~30 s"},
               {"key": "sackett", "label": "Balanced", "desc": "Weigh", "time": "~1 min"},
               {"key": "snow", "label": "Deep", "desc": "Survey", "time": "2–3 min"},
               {"key": "new_team", "kind": "action", "label": "Build a team…", "desc": "", "time": ""}]

PROBE_DEPTHS = [{"key": "fast", "label": "Fast", "desc": "One round", "time": "~30 s"},
                {"key": "deep", "label": "Deep", "desc": "Every facet", "time": "2–3 min"}]


def _stub(key, data, **kwargs):
    """Stand in for the component: record what Python handed over, give back the browser's triggers."""
    st.session_state.setdefault("composer_calls", []).append(dict(data))
    return SimpleNamespace(submit=st.session_state.get("submit_payload"),
                           edited=st.session_state.get("edited_payload"),
                           action=st.session_state.get("action_payload"),
                           depth=st.session_state.get("depth_payload"))


composer_module._component = _stub
warmup.start = lambda: None            # no sentence encoder in a test


site = st.session_state.get("site", "probe")

if site == "probe":
    # Every argument at a non-default value, so the data contract can be read off one call
    st.session_state.probe_result = composer_module.composer(
        "probe_box", mode="sackett", modes=PROBE_MODES, placeholder="Ask a medical question…",
        value="seeded question", nonce="nonce-1", disabled=True, autofocus=True, compact=True,
        send_title="Go", default_question="A question in grey", routed="seeded question", keep=True,
        label="Task", depth="deep", depths=PROBE_DEPTHS, depth_label="Depth")

else:
    from mes.ui import page
    page._init_state()
    if site == "followup" and not st.session_state.mes_thread:
        # A finished conversation is the cheapest way to reach the follow-up box: a turn that
        # failed renders its error and the box, and starts nothing.
        turn = page._new_turn("The first question", "sackett", ["literature"], False, True, [])
        turn.update(status="error", error="stopped by the test harness")
        st.session_state.mes_thread = [turn]
    page.render()
