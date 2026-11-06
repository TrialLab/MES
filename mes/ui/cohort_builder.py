"""
MES page: the cohort builder under the card of a template study. The criteria of its design as switches and fields
(inclusion, exclusion, the treatment groups, the grace period, new users) and the flow of patients counted again on
the study's database after every change, beside the study's own (tools/rwd/counts.py: the analysis's cohort query without the
analysis). Nothing runs again until the user asks: the edited design then answers the question as a new version,
behind a 🔒 switch like the other protocol editors. A fragment: a change reruns only the builder.
"""

import logging

import streamlit as st

from mes.tools import rwd_tools
from mes.tools.rwd import library as L
from mes.ui import cohort_state as CS
from mes.ui import components as C
from mes.ui import protocol

logger = logging.getLogger(__name__)

LOCKED = {"analyst": "A custom analysis defines its cohort in its code: edit the code in the analysis record.",
          "explore": "An exploratory query has no criteria: edit its SQL in the study record.",
          "unfinished": "Only a finished study can be refined.",
          "catalog": "A data catalog lists what the database records: ask one of its questions to get a study to refine."}


def unavailable(study):
    """Why the builder cannot refine ``study`` ("" when it can)."""
    spec = study.get("spec") if isinstance(study.get("spec"), dict) else {}
    if study.get("kind") == "analyst":
        return LOCKED["analyst"]
    if spec.get("analysis") == "explore":
        return LOCKED["explore"]
    if spec.get("analysis") == "catalog":
        return LOCKED["catalog"]
    if study.get("status") != "done" or not spec.get("analysis"):
        return LOCKED["unfinished"]
    if not rwd_tools.available():
        return rwd_tools.unavailable_reason()
    return ""


def render(study, turn, on_edit):
    """The builder's entry under a study card (🔒 where it cannot work), and the builder while it is open."""
    key = f"mes_cb_{turn['id']}_{study.get('key')}"
    why = unavailable(study)
    if why:
        st.button("Cohort builder", key=f"{key}_locked", disabled=True, icon=":material/lock:", help=f"🔒 {why}")
        return
    _builder(key, study, on_edit)


def _toggle(key):
    st.session_state[f"{key}_open"] = not st.session_state.get(f"{key}_open")


def _fresh(spec, generation=0):
    rows = CS.rows_of(spec)
    for r in rows:                       # new widget keys after a reset: the fields show the study's values again
        r["uid"] = f"g{generation}_{r['origin']}"
    return {"rows": rows, "grace": float(spec.get("grace_h") or 24), "new_user": spec.get("new_user", True) is not False,
            "generation": generation, "added": 0, "hits": {}}


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


def _add(key, base, role_key, choice_key):
    state = st.session_state.get(f"{key}_state") or {}
    hit = st.session_state.get(choice_key)
    if not hit or not isinstance(hit, tuple):
        return
    concept, label = hit
    role = {"Inclusion": "include", "Exclusion": "exclude"}.get(st.session_state.get(role_key), "include")
    from mes.tools.rwd import counts
    typical = counts.typical(base, concept[4:]) if concept.startswith("var:") else None
    state["added"] = int(state.get("added", 0)) + 1
    uid = f"g{state.get('generation', 0)}_n{state['added']}"
    state.setdefault("rows", []).append(CS.new_row(concept, label, role, uid, base, typical))
    st.session_state[f"{key}_term"] = ""


@st.fragment
def _builder(key, study, on_edit):
    is_open = bool(st.session_state.get(f"{key}_open"))
    st.button("Close the cohort builder" if is_open else "Cohort builder — change the criteria, see the patients live",
              key=f"{key}_toggle", on_click=_toggle, args=(key,), icon=":material/tune:",
              help="Switch criteria off, change thresholds and windows, add criteria: the flow of patients is counted "
                   f"again on {rwd_tools.database_of(study)} after each change. Nothing runs again until you choose to.")
    if not is_open:
        return
    try:
        _panel(key, study, on_edit)
    except Exception as e:                   # the builder never breaks the page
        logger.exception("The cohort builder could not be drawn")
        st.warning(f"The cohort builder could not be drawn: {C.md(f'{type(e).__name__}: {e}', 300)}")


# ============================================================================ the panel
def _panel(key, study, on_edit):
    spec = study.get("spec") or {}
    state = _state(key, spec)
    with st.container(border=True, key=f"{key}_box"):
        st.markdown('<div class="mes-cb-title">Cohort builder</div><div class="mes-cb-sub">Every change counts the '
                    f'patients again on {C.esc(rwd_tools.database_of(study))} — the same cohort query the study ran, '
                    'without its analysis. The study itself runs again only when you choose to.</div>',
                    unsafe_allow_html=True)
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            _criteria(key, spec, state)
            _adder(key, spec)
        draft = CS.draft_of(spec, state["rows"], state.get("grace") if spec.get("analysis") == "compare" else None,
                            state.get("new_user") if spec.get("analysis") == "compare" else None)
        with right:
            counted = _flow(key, spec, state, draft)
        _footer(key, study, spec, state, draft, counted, on_edit)


def _criteria(key, spec, state):
    rows = state["rows"]
    for role, title, empty in (("include", "Who is included", "Everyone in the population — add a criterion below."),
                               ("exclude", "Who is excluded", "Nobody is excluded.")):
        st.markdown(f'<div class="mes-cb-kicker">{title}</div>', unsafe_allow_html=True)
        mine = [r for r in rows if r["role"] == role]
        if not mine:
            st.caption(empty)
        for r in mine:
            _row(key, spec, r)
    treatments = [r for r in rows if r["role"] in ("exposure", "comparator")]
    compare = spec.get("analysis") == "compare"
    if treatments:
        st.markdown(f'<div class="mes-cb-kicker">{"Treatment groups" if compare else "Treatment / exposure"}</div>',
                    unsafe_allow_html=True)
        for r in treatments:
            _row(key, spec, r)
    if compare:                              # one above the other: neither label is cut in a narrow column
        options = CS.grace_options(state.get("grace"))
        state["grace"] = st.select_slider("Grace period — the first dose within", options, value=float(state["grace"])
                                          if float(state["grace"]) in options else 24.0, key=f"{key}_grace",
                                          format_func=lambda v: f"{v:g} h",
                                          help="Patients treated within this time are the treated group; the "
                                               "landmark (time zero of follow-up) is its end.")
        state["new_user"] = st.toggle("New users only", value=bool(state.get("new_user", True)), key=f"{key}_new",
                                      help="Exclude patients given either drug in the 30 days before time zero")
    elif treatments:
        st.caption("Here the treatment / exposure is what the analysis measures, not a filter of the cohort: the "
                   "counts do not change with it.")


def _row(key, spec, r):
    uid = f"{key}_{r['uid']}"
    kind = CS.kind_of(r["concept"])
    base = spec.get("base") if spec.get("base") in L.BASES else "icu"
    removable = r["role"] in ("include", "exclude")
    with st.container(key=f"{uid}_card_{r['role']}"):
        c1, c4 = st.columns([4.6, 0.6], vertical_alignment="center")        # the switch and its options (⋯) on top
        text = CS.text({**r, "op": "", "value": None, "negate": False})
        if r["role"] != "exposure":
            prefix = "vs " if r["role"] == "comparator" else ""
            r["on"] = c1.toggle(prefix + C.keep_units(C.md(text, 110)), value=bool(r.get("on", True)), key=f"{uid}_on",
                                help="Switch the criterion off to see the cohort without it" if removable else
                                "Off: the comparison group is everyone without the treatment")
        head = c1.empty()                    # (the exposure's name and) the details, once the options are read
        live = bool(r.get("on", True))
        with c4:
            with st.popover("More options", icon=":material/more_horiz:", help="Window, “without”, role, remove"):
                _more(key, uid, r, spec, kind, removable)
        if kind in ("numeric", "measure", "category"):          # the comparison and its value, below the name
            c2, c3 = st.columns([1, 1.5], vertical_alignment="center")
        if kind in ("numeric", "measure"):
            ops = list(CS.NUMERIC_OPS) + ([""] if kind == "measure" else [])
            op = r["op"] if r["op"] in ops else ops[0]
            r["op"] = c2.selectbox("Comparison", ops, index=ops.index(op), key=f"{uid}_op", disabled=not live,
                                   format_func=lambda o: CS.OP_TEXT.get(o, o), label_visibility="collapsed")
            _value_fields(c3, uid, r, base, live)
        elif kind == "category":
            levels = _levels(base, r["concept"][4:], r.get("value"))
            op = r["op"] if r["op"] in ("=", "!=") else "="
            r["op"] = c2.selectbox("Comparison", ["=", "!="], index=["=", "!="].index(op), key=f"{uid}_op",
                                   disabled=not live, format_func=lambda o: CS.OP_TEXT[o], label_visibility="collapsed")
            if levels:
                value = r.get("value") if r.get("value") in levels else levels[0]
                r["value"] = c3.selectbox("Value", levels, index=levels.index(value), key=f"{uid}_value",
                                          disabled=not live, label_visibility="collapsed")
        meta = [CS.KIND_TEXT.get(r["concept"].split(":")[0]) or ("variable" if r["concept"].startswith("var:") else ""),
                _unit(r["concept"]), _window_text(r, spec)]
        name = text if len(text) <= 110 else text[:109].rstrip() + "…"
        details = C.esc(" · ".join(m for m in meta if m)) + (" · <b>without</b>" if r.get("negate") else "")
        head.markdown((f'<div class="mes-cb-name">{C.esc(C.keep_units(name))}</div>' if r["role"] == "exposure" else "")
                      + (f'<div class="mes-cb-meta">{details.removeprefix(" · ")}</div>' if details else ""),
                      unsafe_allow_html=True)


def _value_fields(col, uid, r, base, live):
    if r["op"] == "between":
        value = r.get("value")
        if not (isinstance(value, list) and len(value) == 2):      # from a single threshold: it and twice it
            v = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
            value = [v, v * 2 if v > 0 else v + 1]
        a, b = col.columns(2)
        lo = a.number_input("From", value=float(value[0]), key=f"{uid}_lo", disabled=not live, label_visibility="collapsed")
        hi = b.number_input("To", value=float(value[1]), key=f"{uid}_hi", disabled=not live, label_visibility="collapsed")
        r["value"] = sorted([lo, hi])                                # a range typed backwards is put in order
    elif r["op"]:
        value = r.get("value")
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            value = value[0]                                         # from a range: its lower end
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            from mes.tools.rwd import counts
            value = counts.typical(base, r["concept"][4:]) if r["concept"].startswith("var:") else None
        r["value"] = col.number_input("Value", value=float(value or 0.0), key=f"{uid}_value", disabled=not live,
                                      label_visibility="collapsed")
    else:
        r["value"] = None
        col.caption("in the window")


def _more(key, uid, r, spec, kind, removable):
    concept_kind = r["concept"].split(":")[0]
    compare_treatment = spec.get("analysis") == "compare" and r["role"] in ("exposure", "comparator") \
        and concept_kind in ("drugclass", "drug", "px9", "px10")
    if concept_kind in CS.WINDOWED and not compare_treatment:
        window = r.get("window_h") or [-72.0, 24.0]
        st.caption("Window — hours from time zero (" + L.BASES.get(spec.get("base"), L.BASES["icu"])["t0_label"] + ")")
        a, b = st.columns(2)
        lo = a.number_input("From (h)", value=float(window[0]), step=6.0, key=f"{uid}_wlo")
        open_end = window[1] is None
        to_end = st.checkbox("…to the end of the stay", value=open_end, key=f"{uid}_wend")
        hi = b.number_input("To (h)", value=float(window[1] if window[1] is not None else 24.0), step=6.0,
                            key=f"{uid}_whi", disabled=to_end)
        r["window_h"] = [lo, None if to_end else hi]
    elif compare_treatment:
        st.caption("The first dose within the grace period (set below the treatments).")
    if removable:
        r["negate"] = st.toggle("Without (the criterion is absent)", value=bool(r.get("negate")), key=f"{uid}_neg")
    if removable:
        roles = ["include", "exclude"]
        r["role"] = st.selectbox("Use as", roles, index=roles.index(r["role"]), key=f"{uid}_role",
                                 format_func=lambda x: "Inclusion" if x == "include" else "Exclusion")
        st.button("Remove this criterion", key=f"{uid}_rm", on_click=_remove, args=(key, r["uid"]),
                  icon=":material/delete:", width="stretch")


def _levels(base, variable, current):
    from mes.tools.rwd.concepts import get_index
    try:
        levels = list(get_index().levels(base, variable))
    except Exception:
        levels = []
    if current not in (None, "") and str(current) not in levels:
        levels.append(str(current))
    return levels


def _unit(concept):
    kind, _, key = str(concept).partition(":")
    return L.VARIABLES[key][2] if kind == "var" and key in L.VARIABLES else ""


def _window_text(r, spec):
    window = r.get("window_h")
    t0 = L.BASES.get(spec.get("base"), L.BASES["icu"])["t0_label"]
    if spec.get("analysis") == "compare" and r["role"] in ("exposure", "comparator") and \
            r["concept"].split(":")[0] in ("drugclass", "drug", "px9", "px10"):
        return "first dose within the grace period"
    if not window:
        return "any time in the record" if r["concept"].split(":")[0] in ("phe", "dx9", "dx10") else ""
    lo, hi = window
    fmt = lambda v: f"{float(v):+g} h"
    return f"{fmt(lo)} to " + ("the end of the stay" if hi is None else fmt(hi)) + f" around {t0}"


def _adder(key, spec):
    """Find a concept of the connected database and add it as an inclusion or exclusion criterion."""
    base = spec.get("base") if spec.get("base") in L.BASES else "icu"
    with st.container(key=f"{key}_adder"):
        term = st.text_input("Add a criterion", key=f"{key}_term", placeholder=f"Search {rwd_tools.DATABASE}: a "
                             "condition, drug, lab or score", icon=":material/search:",
                             help="For example vasopressin, CKD, lactate, SOFA")
        if not term.strip():
            return
        state = st.session_state[f"{key}_state"]
        hits = state.setdefault("hits", {})
        if term not in hits:
            from mes.tools.rwd.concepts import get_index
            try:
                found = get_index().search(term, k=8)
            except Exception as e:
                st.caption(f"The search did not run: {C.md(f'{type(e).__name__}: {e}', 200)}")
                return
            variables = L.variables_for(base)
            hits[term] = [(c.concept_id, c.label, c.kind, c.n_patients) for c in found
                          if not c.concept_id.startswith("var:") or c.concept_id[4:] in variables]
            for old in list(hits)[:-12]:               # the last searches only
                hits.pop(old, None)
        options = [(c, label) for c, label, _k, _n in hits[term]]
        if not options:
            st.caption(f"Nothing in {rwd_tools.DATABASE} matches this term.")
            return
        meta = {(c, label): (kind, n) for c, label, kind, n in hits[term]}
        c1, c2, c3 = st.columns([2.3, 1.1, 0.8], vertical_alignment="bottom")
        c1.selectbox("Concept", options, key=f"{key}_choice", format_func=lambda o: _hit_text(o, meta.get(o)))
        c2.segmented_control("As", ["Inclusion", "Exclusion"], default="Inclusion", key=f"{key}_as")
        c3.button("Add", key=f"{key}_addgo", type="primary", icon=":material/add:", width="stretch", on_click=_add,
                  args=(key, base, f"{key}_as", f"{key}_choice"))


def _hit_text(option, meta):
    concept, label = option
    kind, n = meta or ("", None)
    bits = [CS.KIND_TEXT.get(kind) or ("variable" if kind == "var" else kind)]
    if n:
        bits.append(f"{n:,} patients")
    return f"{label} — {' · '.join(b for b in bits if b)}"


def _small(n):
    return isinstance(n, int) and 0 < n < 10


def _flow(key, spec, state, draft):
    """The draft's flow of patients beside the study's (a chart drawn in the browser); the counts or None."""
    from mes.tools.rwd import counts
    from mes.tools.rwd.analyses import MIN_GROUP
    from mes.ui import viz
    original = counts.flow(spec)
    now = counts.flow(draft)
    st.markdown('<div class="mes-cb-kicker">Patients — now and in the study</div>', unsafe_allow_html=True)
    if now.get("errors"):
        st.warning("Not counted — the design is not valid: " + C.md("; ".join(now["errors"]), 600))
        return None
    if original.get("errors"):
        st.caption("The study's own design could not be counted again here: " + C.md("; ".join(original["errors"]), 300))
    originals = {r["origin"]: r for r in CS.rows_of(spec)}
    steps, groups = CS.beside(original, now, state["rows"], originals)
    shown = [{**s, "n": counts.shown(s["n"]), "small": _small(s["n"]), "was": counts.shown(s["was"]),
              "was_small": _small(s["was"])} for s in steps]
    names = {"treated": CS.text(next((r for r in state["rows"] if r["role"] == "exposure"), {"concept": "treated"}))
             if any(r["role"] == "exposure" for r in state["rows"]) else "treated",
             "control": next((CS.text(r) for r in state["rows"] if r["role"] == "comparator" and r.get("on")),
                             "no treatment")}
    group_spec = {k: {"label": names[k], "n": counts.shown(v["n"]), "small": _small(v["n"]), "was": counts.shown(v["was"]),
                      "was_small": _small(v["was"])} for k, v in groups.items()} if groups else None
    unit = L.BASES.get(spec.get("base"), L.BASES["icu"])["unit"]
    n, was = now.get("n"), original.get("n")
    chart = {"type": "cohortflow", "steps": shown, "groups": group_spec,
             "final": {"n": counts.shown(n), "small": _small(n), "was": counts.shown(was), "was_small": _small(was),
                       "label": f"{unit}s in the analysis cohort"},
             "table": {"columns": [{"key": "step", "label": "Step"}, {"key": "now", "label": "Now", "numeric": True},
                                   {"key": "was", "label": "In the study", "numeric": True}],
                       "rows": [{"step": s["label"] + (" (switched off)" if s["state"] == "off" else ""),
                                 "now": "—" if s["n"] is None and not s["small"] else "<10" if s["small"] else f"{s['n']:,}",
                                 "was": "<10" if s["was_small"] else f"{s['was']:,}" if s["was"] is not None else "—"}
                                for s in shown]}}
    viz.show("blocks", {"blocks": [{"type": "chart", "id": "cohortflow", "chart": chart}]}, key=f"{key}_flow")
    notes = []
    if groups and min(v["n"] or 0 for v in groups.values()) < MIN_GROUP:
        notes.append(f"Fewer than {MIN_GROUP} patients in a group: the comparison could not be estimated.")
    elif n is not None and n < 30:
        notes.append("Fewer than 30 patients: most analyses cannot be estimated.")
    notes += [w for w in now.get("warnings") or [] if w]
    for note in notes[:3]:
        st.caption(f"⚠ {C.md(note, 300)}")
    st.caption(f"Counted on {rwd_tools.DATASET} in {now.get('seconds', 0):.2f} s" + (" (counted before)" if now.get("cached")
                                                                                   else ""))
    return now


def _footer(key, study, spec, state, draft, counted, on_edit):
    from mes.tools.protocol_tools import design_changes
    try:
        changes = design_changes(spec, draft)
    except Exception:
        changes = []
    if changes:
        st.markdown('<div class="mes-cb-changes">' + "".join(
            f'<span><b>{C.esc(c["what"])}</b> {C.esc(c["before"])} → {C.esc(c["after"])}</span>' for c in changes[:6])
            + "</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.15, 0.85, 1.6], vertical_alignment="center")
    c1.button("Back to the study's design", key=f"{key}_reset", on_click=_reset, args=(key, spec),
              icon=":material/restart_alt:", disabled=not changes and len(state["rows"]) == len(CS.rows_of(spec)),
              width="stretch")
    why = "" if on_edit is not None else "This answer cannot be edited here (it is still running)."
    why = why or rwd_tools.editable(study, "design")
    with c2:
        unlocked = protocol.lock(f"{key}_lock", "design", ready=not why, why=why)
    ready = bool(unlocked and changes and counted is not None and (counted.get("n") or 0) > 0)
    hint = ("Unlock the switch first" if not unlocked else "Change a criterion first" if not changes else
            "The design must count some patients" if not ready else
            "Runs the study with this cohort: the answer is written again as a new version below; this one stays.")
    if c3.button("Run the study with this cohort", type="primary", key=f"{key}_run", disabled=not ready,
                 icon=":material/play_arrow:", width="stretch", help=hint):
        protocol.close()
        st.session_state[f"{key}_open"] = False
        on_edit({"part": "rwd", "study": study.get("key"), "kind": "design", "spec": draft})
        st.rerun()
