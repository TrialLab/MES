"""
MES page: the protocol of an answer and its edits.

* ``entry``: the 🔒 switch every protocol editor sits behind — locked (read-only) by default; unlocked, its button
  opens the editor, and an edit answers the question again as a new version below (the edited answer stays). The
  open editor is kept in the session (one at a time) until it is dismissed or its edit runs.
  ``MES_PROTOCOL_EDIT`` (config.PROTOCOL_EDIT) can keep the code editor or every editor locked on a server.
* ``search_view`` / ``search_dialog``: the search protocol (concepts, PubMed and ClinicalTrials.gov queries, iBKH
  links) and its editor: concepts in a table, the queries built from them or written by hand.
* ``version_card``: an answer written with an edited protocol — its version, what changed, what ran again and what
  was kept, how the answer changed, the checks of the edited part and the versions before it.
"""

import pandas as pd
import streamlit as st

from mes import config
from mes.tools import protocol_tools as PT
from mes.tools.planner_tools import QUESTION_TYPES, ROLES
from mes.ui import components as C
from mes.ui.icons import icon

EDITOR = "mes_editor"          # the key of the open editor's entry
UNLOCK_HELP = ("Read-only by default. Unlock to edit this protocol and re-run it: the edited version is added below "
               "as a new answer and this one stays as it is.")
TYPE_LABELS = {"literature": "articles", "trial": "trial records", "kg": "knowledge-graph relations"}


def allowed(kind):
    """(True, "") when editors of ``kind`` ("search", "design", "sql", "code") may be unlocked on this server."""
    setting = config.PROTOCOL_EDIT
    if setting == "off":
        return False, "Protocol editing is locked on this MES server (MES_PROTOCOL_EDIT=off)."
    if setting == "no_code" and kind == "code":
        return False, "Editing analysis code is locked on this MES server (MES_PROTOCOL_EDIT=no_code)."
    return True, ""


def lock(key, kind, ready=True, why=""):
    """The 🔒 switch of a protocol editor; True when unlocked. ``ready`` False (with ``why``): it stays locked."""
    ok, reason = allowed(kind)
    if not ok or not ready:
        st.toggle("🔒 Locked", value=False, key=f"{key}_off", disabled=True, help=reason or why)
        return False
    on = bool(st.session_state.get(key))
    return st.toggle("🔓 Unlocked" if on else "🔒 Locked", key=key, help=UNLOCK_HELP)


def _open(key):
    st.session_state[EDITOR] = key


def close():
    """Close the open editor (its dialog is not drawn again)."""
    st.session_state.pop(EDITOR, None)


def entry(key, kind, label, ready=True, why=""):
    """A switch and the button of an editor side by side; True while the editor is open (the caller draws it)."""
    c1, c2 = st.columns([1, 2.4], vertical_alignment="center")
    with c1:
        unlocked = lock(key, kind, ready, why)
    with c2:
        st.button(f"✎ {label}", key=f"{key}_go", disabled=not unlocked, width="stretch", on_click=_open, args=(key,),
                  help="Opens the editor" if unlocked else (why or "Unlock the switch first"))
    return unlocked and st.session_state.get(EDITOR) == key


# ============================================================================ search protocol
def _concept_rows(plan):
    return [{"Role": c["role"], "Concept": c["label"], "Synonyms": "; ".join(c["synonyms"])}
            for c in PT.concepts_of(plan)]


def search_view(turn, result, on_edit):
    """The search protocol of ``result`` and its editor (``on_edit(edit)``: run an edit; None: no editing)."""
    plan = result.plan or {}
    pico = "; ".join(f"{k}: {C.md(v, 300)}" for k, v in (plan.get("pico") or {}).items() if v)
    st.markdown(f"**Question type:** {C.md(plan.get('question_type'))}" + (f" · **PICO:** {pico}" if pico else ""))
    rows = _concept_rows(plan)
    if rows:
        st.markdown(C.table_html(rows), unsafe_allow_html=True)
    elif plan.get("fallback"):
        st.caption("The question could not be planned into concepts: the queries are its keywords.")
    queries = (result.source_status.get("literature") or {}).get("queries", {})
    for name, q in queries.items():
        code = str(q).replace("`", "'")
        st.markdown(f"- PubMed ({C.md(name, 80)}): `{code}`")
    terms = {k: v for k, v in (plan.get("trial_terms") or {}).items() if v}
    if terms and "trials" in (result.sources or []):
        st.markdown("- ClinicalTrials.gov: " + " · ".join(f"{C.md(k, 20)} `{str(v).replace('`', chr(39))}`"
                                                          for k, v in terms.items()))
    links = (result.source_status.get("kg") or {}).get("links", {})
    if links:
        st.markdown("- iBKH entity links: " + ", ".join(f"{C.md(k, 80)} → {C.md(v['name'], 80)} ({v['method']})"
                                                         for k, v in links.items()))
    if on_edit is None:
        return
    searched = any(s in (result.sources or []) for s in ("literature", "trials", "kg"))
    if entry(f"mes_lock_{turn['id']}_search", "search", "Edit the search protocol and re-run", ready=searched,
             why="This answer searched no literature, trial or knowledge-graph source."):
        search_dialog(turn["id"], result, on_edit)
    st.caption("Only the sources the edit affects are searched again; the others are kept from this answer.")


@st.dialog("Edit the search protocol", width="large", on_dismiss=close)
def search_dialog(turn_id, result, on_edit):
    plan = result.plan or {}
    sources = result.sources or []
    key = f"mes_proto_{turn_id}"
    types = list(QUESTION_TYPES)
    qtype = st.selectbox("Question type", types, key=f"{key}_type",
                         index=types.index(plan.get("question_type")) if plan.get("question_type") in types else 0)
    st.markdown("**Concepts** — the queries are built from them: synonyms joined by OR, concepts by AND")
    frame = pd.DataFrame(_concept_rows(plan), columns=["Role", "Concept", "Synonyms"])
    table = st.data_editor(
        frame, num_rows="dynamic", hide_index=True, width="stretch", key=f"{key}_concepts",
        column_config={"Role": st.column_config.SelectboxColumn("Role", options=list(ROLES), required=True,
                                                                width="small"),
                       "Concept": st.column_config.TextColumn("Concept", max_chars=200),
                       "Synonyms": st.column_config.TextColumn("Synonyms (separated by ;)", max_chars=1000)})
    old = {c["label"].lower(): c["kind"] for c in PT.concepts_of(plan)}
    text = lambda v: v if isinstance(v, str) else ""          # an empty cell of a new row is None or NaN
    concepts = [{"role": r.get("Role"), "label": text(r.get("Concept")),
                 "kind": old.get(text(r.get("Concept")).strip().lower(), "other"),
                 "synonyms": text(r.get("Synonyms"))} for r in table.to_dict("records")]
    by_hand = st.toggle("Write the queries myself", key=f"{key}_hand",
                        help="Off: the PubMed and ClinicalTrials.gov queries are built from the concepts, as MES "
                             "builds them. On: they run as you write them (PubMed syntax).")
    queries = None
    if by_hand:
        main = st.text_area("PubMed query", plan.get("pubmed_query", ""), height=90, key=f"{key}_pubmed")
        relaxed = st.text_area("Relaxed PubMed queries — one per line, tried when the main query finds few articles",
                               "\n".join((plan.get("pubmed_tiers") or [])[1:]), height=80, key=f"{key}_relaxed")
        facets = {f["name"]: st.text_input(f"Facet query: {f['name']}", f.get("pubmed_query", ""),
                                           key=f"{key}_facet_{n}")
                  for n, f in enumerate(plan.get("facets") or []) if f.get("name")}
        terms = dict(plan.get("trial_terms") or {})
        trial_terms = dict(terms)                   # kept as they are when this answer did not search trials
        if "trials" in sources:
            fields = [k for k in PT.TRIAL_FIELDS if k != "term" or terms.get("term")]
            cols = st.columns(len(fields))
            trial_terms = {k: cols[i].text_input(f"ClinicalTrials.gov {k}", terms.get(k, ""), key=f"{key}_trial_{k}")
                           for i, k in enumerate(fields)}
        queries = {"pubmed_query": main, "relaxed": relaxed.splitlines(), "facets": facets,
                   "trial_terms": trial_terms}
    edit = {"part": "search", "concepts": concepts, "question_type": qtype, "queries": queries}
    new, changes, rerun, errors = PT.search_edit_plan(plan, edit, sources, result.question)
    if not by_hand:
        if "literature" in sources:
            st.markdown("PubMed query built from the concepts")
            st.code(new.pubmed_query or "—", language="text", wrap_lines=True)
        if "trials" in sources:
            st.caption("ClinicalTrials.gov: " + " · ".join(f"{k}: {v}" for k, v in new.trial_terms.items() if v))
    for e in errors:
        st.error(e)
    if not errors and not changes:
        st.caption("Change a concept, the question type or a query to re-run.")
    elif not errors:
        kept = PT._kept(result, rerun)
        st.info(f"{len(changes)} change{'s' if len(changes) > 1 else ''}. Searches again: {PT.names(rerun)}"
                + (f" · kept from this answer: {PT.names(kept)}" if kept else "")
                + ". Then the answer is written again from all of it.")
    if st.button("Re-run with this protocol", type="primary", disabled=bool(errors or not changes),
                 key=f"{key}_run"):
        close()
        on_edit(edit)
        st.rerun()


# ============================================================================ an edited version
def _cell(text, limit=500):
    text = " · ".join(line.strip() for line in str(text or "").splitlines() if line.strip()) or "—"
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _delta(protocol):
    """One line on how the answer changed: articles added / removed, cited sources, estimates."""
    cmp_ = protocol.get("comparison") or {}
    bits = []
    for t, d in (cmp_.get("sources") or {}).items():
        bits.append(f"{TYPE_LABELS.get(t, t)} +{d['n_added']} / −{d['n_removed']}")
    cited = cmp_.get("cited") or {}
    if cited and cited.get("before") != cited.get("after"):
        bits.append(f"cited sources {cited['before']} → {cited['after']}")
    for e in cmp_.get("estimates") or []:
        n = e.get("n_changed", 0)
        bits.append(f"RWD: {n} estimate{'s' if n != 1 else ''} changed" if n else "RWD estimates unchanged")
    for t in cmp_.get("tables") or []:
        bits.append(f"RWD query result: {t['before_rows']} → {t['after_rows']} rows" if t.get("changed")
                    else "RWD query result unchanged")
    return " · ".join(bits)


def trail_text(protocol):
    """The versions of an edited protocol: "v1 MES → v2 Search protocol (PubMed query) → v3 …"."""
    steps = ["v1 designed by MES"] + [f"v{t['version']} {t['label']}" + (f" ({t['summary']})" if t.get("summary")
                                                                          else "")
                                      for t in protocol.get("trail") or []]
    return " → ".join(steps)


def edit_title(protocol):
    """What was edited, in words: "Search protocol: PubMed query" or "SQL query of the RWD study “…”"."""
    if protocol.get("part") != "rwd":
        return f"{protocol.get('label', '')}: {protocol.get('summary', '')}"
    title = protocol.get("study_title") or ""
    title = title if len(title) <= 90 else title[:89].rstrip() + "…"
    what = f"{protocol.get('label', '')} of the RWD study" + (f" “{title}”" if title else "")
    summary = protocol.get("summary", "")
    return what + (f": {summary}" if summary and summary != protocol.get("label") else "")


def version_card(result, turn):
    """The card of an answer written with an edited protocol (nothing for MES's own protocol)."""
    p = result.protocol or {}
    if not p:
        return
    run = f"↻ ran again: {PT.names(p.get('rerun') or [])}" if p.get("rerun") else "↻ the answer was written again"
    kept = f" · ↺ kept from v{p.get('from_version', 1)}: {PT.names(p.get('kept') or [])}" if p.get("kept") else ""
    delta = _delta(p)
    st.markdown(
        f'<div class="mes-card mes-proto"><div class="mes-card-title">{icon("edit")} Protocol v{p.get("version", 2)} · edited by '
        f'you from v{p.get("from_version", 1)}</div><div class="mes-proto-what">{C.esc(edit_title(p))}</div>'
        f'<div class="mes-proto-run">{C.esc(run + kept)}</div>'
        + (f'<div class="mes-proto-delta">{C.esc(delta)}</div>' if delta else "") + "</div>", unsafe_allow_html=True)
    with st.expander(f"What changed in v{p.get('version', 2)}"):
        if p.get("changes"):
            st.markdown(C.table_html([{"What": c["what"], f"v{p.get('from_version', 1)}": _cell(c["before"]),
                                       f"v{p.get('version', 2)}": _cell(c["after"])} for c in p["changes"]]),
                        unsafe_allow_html=True)
        if p.get("diff"):
            st.code(p["diff"], language="diff", wrap_lines=True)
        cmp_ = p.get("comparison") or {}
        for t, d in (cmp_.get("sources") or {}).items():
            lines = [f"**{TYPE_LABELS.get(t, t).capitalize()}**: {d['before']} → {d['after']}"]
            lines += [f"- ＋ {C.md(x, 160)}" for x in d.get("added") or []]
            lines += [f"- − {C.md(x, 160)}" for x in d.get("removed") or []]
            more = d["n_added"] + d["n_removed"] - len(d.get("added") or []) - len(d.get("removed") or [])
            if more > 0:
                lines.append(f"- … and {more} more")
            st.markdown("\n".join(lines))
        for e in cmp_.get("estimates") or []:
            st.markdown(f"**{C.md(e['study'], 160)}** — {e.get('n_changed', 0)} estimate"
                        f"{'s' if e.get('n_changed', 0) != 1 else ''} changed, {e.get('n_same', 0)} the same")
            if e.get("changed"):
                st.markdown(C.table_html([{"Estimate": r["label"], f"v{p.get('from_version', 1)}": r["before"] or "—",
                                           f"v{p.get('version', 2)}": r["after"] or "—"} for r in e["changed"]]),
                            unsafe_allow_html=True)
        for t in cmp_.get("tables") or []:
            st.markdown(f"**{C.md(t['study'], 160)}** — query result " + (
                f"changed: {t['before_rows']} → {t['after_rows']} rows (shown in the study's Results)"
                if t.get("changed") else "unchanged"))
        for check in p.get("checks") or []:
            st.caption(f"ⓘ {C.md(check, 600)}")
        st.caption(f"Versions: {C.md(trail_text(p), 600)} · edited {C.md(p.get('edited_at', ''), 40)}")
