"""
MES page: the patient explorer under the card of a "patients like this" study. The described patient as fields — the
value of each characteristic, how much it weighs, whether it must match, characteristics added or removed — and how
many similar patients to keep; after every change the similar patients are found again and their outcome, treatments
and profile redrawn (tools/rwd/similar.py ``preview``: the cohort is built once, the patients ranked again in
milliseconds). Nothing runs again until the user asks: the edited patient then answers the question as a new
version, behind a 🔒 switch like the other protocol editors. A fragment: a change reruns only the explorer.
"""

import logging

import streamlit as st

from mes.tools import rwd_tools
from mes.tools.rwd import library as L
from mes.tools.rwd import similar as SIM
from mes.ui import components as C
from mes.ui import protocol

logger = logging.getLogger(__name__)

WEIGHTS = (0.5, 1.0, 2.0, 3.0)
K_OPTIONS = (30, 50, 100, 150, 200, 300, 500, 1000, 2000)


def render(study, turn, on_edit):
    """The explorer's entry under the card of a "patients like this" study (nothing for other studies)."""
    spec = study.get("spec") if isinstance(study.get("spec"), dict) else {}
    if spec.get("analysis") != "similar" or study.get("status") != "done":
        return
    key = f"mes_px_{turn['id']}_{study.get('key')}"
    if not rwd_tools.available():
        st.button("Patient explorer", key=f"{key}_locked", disabled=True, icon=":material/lock:",
                  help=f"🔒 {rwd_tools.unavailable_reason()}")
        return
    _explorer(key, study, on_edit)


def _toggle(key):
    st.session_state[f"{key}_open"] = not st.session_state.get(f"{key}_open")


def _rows(spec, generation=0):
    out = []
    for i, f in enumerate(spec.get("profile") or []):
        if isinstance(f, dict) and str(f.get("concept") or "").startswith("var:"):
            out.append({"uid": f"g{generation}_p{i}", "concept": f["concept"], "value": f.get("value"),
                        "weight": float(f.get("weight") or 1.0), "must": bool(f.get("must")),
                        "tolerance": f.get("tolerance")})
    return out


def _fresh(spec, generation=0):
    try:
        k = int(spec.get("k") or SIM.DEFAULT_K)
    except (TypeError, ValueError):
        k = SIM.DEFAULT_K
    return {"rows": _rows(spec, generation), "k": k, "generation": generation, "added": 0}


def _state(key, spec):
    state = st.session_state.get(f"{key}_state")
    if not isinstance(state, dict) or not isinstance(state.get("rows"), list):
        state = st.session_state[f"{key}_state"] = _fresh(spec)
    return state


def _reset(key, spec):
    old = st.session_state.get(f"{key}_state") or {}
    st.session_state[f"{key}_state"] = _fresh(spec, int(old.get("generation", 0)) + 1)


def _remove(key, uid):
    state = st.session_state.get(f"{key}_state") or {}
    state["rows"] = [r for r in state.get("rows", []) if r["uid"] != uid]


def _add(key, base):
    state = st.session_state.get(f"{key}_state") or {}
    name = st.session_state.get(f"{key}_new")
    if name not in L.variables_for(base) or any(r["concept"] == f"var:{name}" for r in state.get("rows", [])):
        return
    from mes.tools.rwd import counts
    kind = L.VARIABLES[name][1]
    value = counts.typical(base, name) if kind == "numeric" else 1 if kind == "binary" else \
        next(iter(_levels(base, name)), None)
    if value is None:
        value = 0.0
    state["added"] = int(state.get("added", 0)) + 1
    state.setdefault("rows", []).append({"uid": f"g{state.get('generation', 0)}_n{state['added']}",
                                         "concept": f"var:{name}", "value": value, "weight": 1.0, "must": False,
                                         "tolerance": None})
    st.session_state[f"{key}_new"] = None


def _levels(base, name):
    from mes.tools.rwd.concepts import get_index
    try:
        return list(get_index().levels(base, name))
    except Exception:
        return []


def draft_of(spec, state):
    """The design with the edited patient (and number of similar patients)."""
    import copy
    d = copy.deepcopy(spec)
    d["profile"] = [{"concept": r["concept"], "value": r["value"], "weight": r["weight"], "must": r["must"],
                     "tolerance": r.get("tolerance")} for r in state["rows"]]
    d["k"] = int(state["k"])
    return d


@st.fragment
def _explorer(key, study, on_edit):
    is_open = bool(st.session_state.get(f"{key}_open"))
    st.button("Close the patient explorer" if is_open else "Patient explorer — change the patient, see similar "
              "patients live", key=f"{key}_toggle", on_click=_toggle, args=(key,), icon=":material/person_search:",
              help="Change the patient's values, how much each characteristic weighs, which must match, how many similar "
                   f"patients to keep: the similar patients are found again on {rwd_tools.database_of(study)} after each "
                   "change.")
    if not is_open:
        return
    try:
        _panel(key, study, on_edit)
    except Exception as e:                   # the explorer never breaks the page
        logger.exception("The patient explorer could not be drawn")
        st.warning(f"The patient explorer could not be drawn: {C.md(f'{type(e).__name__}: {e}', 300)}")


# ============================================================================ the panel
def _panel(key, study, on_edit):
    spec = study.get("spec") or {}
    base = spec.get("base") if spec.get("base") in L.BASES else "icu"
    state = _state(key, spec)
    with st.container(border=True, key=f"{key}_box"):
        st.markdown('<div class="mes-cb-title">Patient explorer</div><div class="mes-cb-sub">Change the patient and the '
                    f'similar patients are found again on {C.esc(rwd_tools.database_of(study))} — the same ranking the '
                    'study used. The study itself runs again only when you choose to.</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1.25], gap="medium")
        with left:
            st.markdown('<div class="mes-cb-kicker">The patient</div>', unsafe_allow_html=True)
            if not state["rows"]:
                st.caption("No characteristic — add one below.")
            for r in state["rows"]:
                _row(key, r, base)
            _adder(key, base, state)
            options = sorted(set(K_OPTIONS) | {int(state["k"])})
            state["k"] = st.select_slider("Similar patients to keep", options, value=int(state["k"]), key=f"{key}_k",
                                          help="The nearest stays by the weighted distance; at least 30 (small groups "
                                               "are not shown)")
        draft = draft_of(spec, state)
        with right:
            ready = _preview(key, study, draft)
        _footer(key, study, spec, state, draft, ready, on_edit)


def _row(key, r, base):
    """One characteristic as a card: its name and details with the options (⋯) on top, its value below — the value
    has the card's whole width, so long names, yes / no and lists fit a narrow column."""
    uid = f"{key}_{r['uid']}"
    name = r["concept"][4:]
    label, kind, unit = (L.VARIABLES[name][0], L.VARIABLES[name][1], L.VARIABLES[name][2]) if name in L.VARIABLES \
        else (name, "numeric", "")
    with st.container(key=f"{uid}_feat_{kind}"):
        c1, c2 = st.columns([4, 0.7], vertical_alignment="center")
        head = c1.empty()                    # the name and its details, drawn once the options are read
        with c2:
            with st.popover("More options", icon=":material/more_horiz:", help="Weight, must match, remove"):
                r["weight"] = st.select_slider("Weight", WEIGHTS, value=r["weight"] if r["weight"] in WEIGHTS else 1.0,
                                               key=f"{uid}_w", format_func=lambda w: f"×{w:g}",
                                               help="How much this characteristic counts in the distance")
                r["must"] = st.toggle("Must match", value=bool(r.get("must")), key=f"{uid}_must",
                                      help="Only stays with the same value (a number: within half a robust SD) are "
                                           "compared")
                st.button("Remove", key=f"{uid}_rm", on_click=_remove, args=(key, r["uid"]), icon=":material/delete:",
                          width="stretch")
        if kind == "numeric":
            value = r.get("value") if isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool) \
                else 0.0
            r["value"] = st.number_input(label, value=float(value), key=f"{uid}_value", label_visibility="collapsed",
                                         step=1.0 if float(value).is_integer() and abs(value) >= 5 else 0.1, format="%g")
        elif kind == "binary":
            yes = r.get("value") in (1, True, "1", "yes", "true")
            choice = st.segmented_control(label, ["yes", "no"], default="yes" if yes else "no", key=f"{uid}_value",
                                          label_visibility="collapsed")
            r["value"] = 1 if (choice or ("yes" if yes else "no")) == "yes" else 0
        else:
            levels = _levels(base, name)
            if r.get("value") not in (None, "") and str(r["value"]) not in levels:
                levels.append(str(r["value"]))
            if levels:
                current = str(r.get("value")) if str(r.get("value")) in levels else levels[0]
                r["value"] = st.selectbox(label, levels, index=levels.index(current), key=f"{uid}_value",
                                          label_visibility="collapsed")
        details = " · ".join(m for m in (unit, f"weight ×{r['weight']:g}" if r["weight"] != 1 else "",
                                         "must match" if r.get("must") else "") if m)
        head.markdown(f'<div class="mes-cb-head"><span class="mes-cb-name">{C.esc(C.keep_units(label))}</span>'
                      + (f'<span class="mes-cb-meta">{C.esc(details)}</span>' if details else "") + "</div>",
                      unsafe_allow_html=True)


def _adder(key, base, state):
    used = {r["concept"][4:] for r in state["rows"]}
    options = [v for v in L.variables_for(base) if v not in used and v not in ("anchor_year_group",)]
    if not options:
        return
    with st.container(key=f"{key}_adder"):
        c1, c2 = st.columns([2.2, 1], vertical_alignment="bottom")
        c1.selectbox("Add a characteristic", options, key=f"{key}_new", format_func=lambda v: L.VARIABLES[v][0],
                     index=None, placeholder="Score, lab, treatment…")
        c2.button("Add", key=f"{key}_addgo", type="primary", icon=":material/add:", width="stretch", on_click=_add,
                  args=(key, base), disabled=not st.session_state.get(f"{key}_new"))


def _preview(key, study, draft):
    """The similar patients of the edited patient (the card's numbers, Results and Profile); True when found."""
    from mes.ui import study_card, viz
    res, errors = SIM.preview(draft)
    st.markdown('<div class="mes-cb-kicker">Similar patients — now</div>', unsafe_allow_html=True)
    if errors:
        st.warning("Not found — " + C.md("; ".join(errors), 500))
        return False
    record = {**study, "key": f"{study.get('key')}-draft", "spec": draft, "result": res, "definitions": []}
    card = study_card.card(type("Draft", (), {"citation_order": []})(), record)
    if not card:
        st.caption("The similar patients could not be shown.")
        return False
    card["tabs"] = [t for t in card["tabs"] if t["id"] in ("results", "profile")]
    card["subtitle"], card["cite"], card["seconds"] = "preview — not yet in the answer", "draft", None
    viz.show("study", card, key=f"{key}_preview")
    return bool(res and res.get("estimates"))


def _footer(key, study, spec, state, draft, ready, on_edit):
    from mes.tools.protocol_tools import design_changes
    try:
        changes = design_changes(spec, draft)
    except Exception:
        changes = []
    if changes:
        st.markdown('<div class="mes-cb-changes">' + "".join(
            f'<span><b>{C.esc(c["what"])}</b> {C.esc(c["before"])} → {C.esc(c["after"])}</span>' for c in changes[:4])
            + "</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.15, 0.85, 1.6], vertical_alignment="center")
    c1.button("Back to the study's patient", key=f"{key}_reset", on_click=_reset, args=(key, spec),
              icon=":material/restart_alt:", disabled=not changes, width="stretch")
    why = "" if on_edit is not None else "This answer cannot be edited here (it is still running)."
    why = why or rwd_tools.editable(study, "design")
    with c2:
        unlocked = protocol.lock(f"{key}_lock", "design", ready=not why, why=why)
    go = bool(unlocked and changes and ready)
    hint = ("Unlock the switch first" if not unlocked else "Change the patient first" if not changes else
            "Similar patients must be found first" if not ready else
            "Runs the study with this patient: the answer is written again as a new version below; this one stays.")
    if c3.button("Run the study with this patient", type="primary", key=f"{key}_run", disabled=not go,
                 icon=":material/play_arrow:", width="stretch", help=hint):
        protocol.close()
        st.session_state[f"{key}_open"] = False
        on_edit({"part": "rwd", "study": study.get("key"), "kind": "design", "spec": draft})
        st.rerun()
