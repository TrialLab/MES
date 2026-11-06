"""
Agent team builder of the MES page: assemble the team that answers a question on a canvas — which agents take part,
their tools and the depth — then use it in the question box.

The canvas itself (``team_canvas``) knows nothing of MES; MES supplies the rules and the catalogue of ``teams.py``,
the compiler of ``team_graph.js``, and this view. The browser keeps the draft (with undo / redo) and
sends an action when a button is pressed:

    {"type": "back" | "apply" | "save" | "delete", "graph", "id", "at"}

``render()`` handles saving and deleting itself and returns the actions the page must carry out
({"type": "back" | "apply", "team"}).
"""

import uuid
from pathlib import Path

import streamlit as st

from mes.ui.team_canvas.component import register
from mes.tools import rwd_tools
from mes.tools.rwd import analyst as rwd_analyst
from mes.ui import teams
from mes.ui.composer import is_dark

canvas = register("mes_team_builder", Path(__file__).parent / "team_graph.js")


def availability():
    """What this computer can run: the real-world data store and the Claude analyst (with the reasons when not), and
    the name of the connected database (the RWD Analyst's chip)."""
    rwd_ok = bool(rwd_tools.available())
    if rwd_ok:
        analyst_ok, reason = rwd_analyst.status()
    else:
        analyst_ok, reason = False, f"the {rwd_tools.DATABASE} store is not built"
    return {"rwd": rwd_ok, "analyst": bool(analyst_ok), "database": rwd_tools.DATABASE,
            "rwd_reason": "" if rwd_ok else f"The {rwd_tools.DATABASE} store is not built on this computer "
                                            "(python -m mes.scripts.build_mimic_db).",
            "analyst_reason": reason or ""}


def team_record(graph, compiled, team_id=None):
    """The team as the page keeps it: name, graph and the settings of a run."""
    return {"id": team_id, "name": graph["name"], "graph": graph,
            "config": {k: compiled[k] for k in ("sources", "mode", "verify", "rwd_engine", "use_cache")}}


def flash(text, icon):
    """A message shown (as a toast) in the next run, wherever the page goes."""
    st.session_state.mes_team_flash = (text, icon)


def show_flash():
    message = st.session_state.pop("mes_team_flash", None)
    if isinstance(message, (list, tuple)) and len(message) == 2:
        st.toast(message[0], icon=message[1])


def _handle(action, av, default_question):
    """Carry out a browser action; returns an action for the page or None."""
    if not isinstance(action, dict) or not action.get("at") or action.get("at") == st.session_state.get("mes_team_seen"):
        return None
    st.session_state.mes_team_seen = action.get("at")
    kind = action.get("type")
    graph = teams.normalize(action.get("graph"))
    if kind == "save":
        try:
            teams.save(graph)
            flash(f"Saved the team “{graph['name']}” — it is under Templates.", "💾")
        except OSError as e:
            flash(f"Could not save the team: {e}", "⚠️")
        st.rerun()
    if kind == "delete":
        removed = teams.delete(action.get("id"))
        flash("Saved team deleted." if removed else "That saved team no longer exists.", "🗑️")
        st.rerun()
    if kind == "back":
        return {"type": "back"}
    if kind == "apply":
        compiled = teams.compile_team(graph, av)
        if not compiled["ok"]:
            st.session_state.mes_team_problems = [m["text"] for m in compiled["messages"] if m["level"] == "error"]
            st.rerun()
        return {"type": "apply", "team": team_record(graph, compiled)}
    return None


def render(default_question=""):
    """The builder view (inside the MES page); returns an action for the page or None."""
    st.markdown('<div class="mes-kicker">Multi-agent Evidence Synthesis</div>'
                '<div class="mes-question" style="margin-bottom:0">Agent team builder</div>'
                '<div class="mes-meta" style="margin-top:0.2rem">Drag agents onto the canvas, choose their tools '
                'and the depth, then use the team in the question box. '
                '🔒 settings are fixed in this version of the pipeline.</div>', unsafe_allow_html=True)
    av = availability()
    st.session_state.setdefault("mes_team_nonce", uuid.uuid4().hex)
    active = st.session_state.get("mes_team")
    graph = active["graph"] if isinstance(active, dict) and active.get("graph") else teams.default_graph()
    action = canvas("mes_team_canvas", rules=teams.RULES, catalog=teams.catalog(), templates=teams.all_templates(),
                    graph=teams.normalize(graph), nonce=st.session_state.mes_team_nonce, availability=av,
                    dark=is_dark(), default_question=default_question, model=teams.model_labels()[0],
                    active={"name": active["name"], "graph": teams.normalize(active["graph"])}
                    if isinstance(active, dict) and active.get("graph") else None)
    problems = st.session_state.pop("mes_team_problems", None)
    if problems:
        st.error("The team cannot run yet: " + " ".join(problems))
    return _handle(action, av, default_question)
