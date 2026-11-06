"""
MES page: trial vs real-world data under the card of a real-world data study — the baseline characteristics a trial
posted on ClinicalTrials.gov beside the same characteristics of the study's cohort (tools/rwd/trial_population.py):
how far the cohort resembles the trial's participants, so how far a result carries over between them. The trials of the answer
are offered first (those with posted results on top); any NCT number can be typed. A fragment: choosing and comparing
reruns only this panel.
"""

import logging
import re

import streamlit as st

from mes.tools import rwd_tools
from mes.tools.rwd import trial_population as TP
from mes.ui import components as C

logger = logging.getLogger(__name__)

# studies whose cohort is built from their design (the cohort builder's): a trial can be set beside it
ANALYSES = ("compare", "association", "survival", "rate", "trend", "distribution", "utilization", "prognostic",
            "characterize", "trajectory", "pathways", "top")
OTHER = "__other__"
TOOL = "Trial vs RWD"                  # the tool's name (its data: the study's database, by name)


def render(study, result, turn):
    """The comparison's entry under the card of ``study`` (nothing for studies without a designed cohort)."""
    spec = study.get("spec") if isinstance(study.get("spec"), dict) else {}
    if study.get("status") != "done" or study.get("kind") == "analyst" or spec.get("analysis") not in ANALYSES:
        return
    key = f"mes_tp_{turn['id']}_{study.get('key')}"
    if not rwd_tools.available():
        st.button(TOOL, key=f"{key}_locked", disabled=True, icon=":material/lock:",
                  help=f"🔒 {rwd_tools.unavailable_reason()}")
        return
    _fragment(key, study, trials_of(result))


def trials_of(result):
    """[(nct, title, has_results)] of the trials an answer cites or names — posted results first."""
    found = {}
    for item in getattr(result, "evidence", None) or []:
        get = item.get if isinstance(item, dict) else (lambda k, d=None, _i=item: getattr(_i, k, d))
        nct = str((get("identifiers") or {}).get("nct") or "").upper()
        if TP.NCT.fullmatch(nct) and nct not in found:
            meta = get("meta") or {}
            found[nct] = (nct, str(get("title") or ""), bool(meta.get("has_results")) if get("source_type") == "trial"
                          else None)
    for nct in TP.NCT.findall(f"{getattr(result, 'question', '')}"):
        found.setdefault(nct.upper(), (nct.upper(), "named in the question", None))
    return sorted(found.values(), key=lambda t: (t[2] is not True, t[2] is None))


def _toggle(key):
    st.session_state[f"{key}_open"] = not st.session_state.get(f"{key}_open")


@st.fragment
def _fragment(key, study, trials):
    is_open = bool(st.session_state.get(f"{key}_open"))
    st.button("Close the trial comparison" if is_open else f"{TOOL} — set these patients beside a trial's",
              key=f"{key}_toggle", on_click=_toggle, args=(key,), icon=":material/compare_arrows:",
              help="The baseline characteristics a trial posted on ClinicalTrials.gov beside the same characteristics "
                   f"of this study's {rwd_tools.database_of(study)} cohort, with standardized differences.")
    if not is_open:
        return
    try:
        _panel(key, study, trials)
    except Exception as e:                   # the comparison never breaks the page
        logger.exception("The trial comparison could not be drawn")
        st.warning(f"The trial comparison could not be drawn: {C.md(f'{type(e).__name__}: {e}', 300)}")


def _label(trials, option):
    if option == OTHER:
        return "Another trial — type its NCT number"
    hit = next((t for t in trials if t[0] == option), None)
    if hit is None:
        return str(option)                 # (not an option: shown as it is)
    nct, title, results = hit
    tail = " · results posted" if results is True else " · no posted results" if results is False else ""
    return f"{nct} — {title[:70]}{'…' if len(title) > 70 else ''}{tail}"


def _panel(key, study, trials):
    with st.container(border=True, key=f"{key}_box"):
        st.markdown(f'<div class="mes-cb-title">{TOOL}</div><div class="mes-cb-sub">How this study\'s '
                    'patients compare with a trial\'s participants: the baseline characteristics the trial posted on '
                    f'ClinicalTrials.gov beside the same characteristics of the {C.esc(rwd_tools.database_of(study))} '
                    'cohort. Large differences mean the trial\'s result may not carry over to these patients — or the '
                    'emulation to the trial.</div>', unsafe_allow_html=True)
        options = [t[0] for t in trials] + [OTHER]
        c1, c2 = st.columns([2.6, 1], vertical_alignment="bottom")
        choice = c1.selectbox("Trial", options, key=f"{key}_pick", format_func=lambda o: _label(trials, o))
        nct = choice
        if choice == OTHER:
            typed = c1.text_input("NCT number", key=f"{key}_nct", placeholder="NCT03434028",
                                  help="The trial's ClinicalTrials.gov number (NCT and 8 digits)")
            text = str(typed or "")
            m = TP.NCT.search(text) or TP.NCT.search(re.sub(r"[\s-]+", "", text))     # also "NCT 0343-4028"
            nct = m.group(0).upper() if m else ""
        results = st.session_state.setdefault(f"{key}_res", {})
        if c2.button("Compare", type="primary", key=f"{key}_go", icon=":material/play_arrow:", width="stretch",
                     disabled=not nct, help="Reads the trial's posted baseline and builds this study's cohort again"):
            with st.spinner(f"Reading {nct} on ClinicalTrials.gov and building the cohort…"):
                results[nct] = TP.run(study.get("spec") or {}, nct)
        res = results.get(nct) if nct else None
        if res is None:
            st.caption("Choose a trial and press Compare." if nct else "Type the trial's NCT number.")
        elif res.get("error"):
            st.warning(f"{C.md(res.get('nct', nct), 20)}: {C.md(res['error'], 300)}")
        else:
            from mes.ui import viz
            viz.show("study", card(res, study.get("dataset")), key=f"{key}_card_{nct}")


def card(res, dataset=None):
    """The study-card spec of a comparison (ui/viz.py draws it); ``dataset``: the study's (default: the connected one)."""
    dataset = dataset or rwd_tools.DATASET
    db = rwd_tools.database_of({"dataset": dataset})
    rows = res.get("rows") or []
    scored = sorted([r for r in rows if isinstance(r.get("smd"), (int, float))], key=lambda r: -abs(r["smd"]))
    differ = [r for r in scored if abs(r["smd"]) >= TP.MEANINGFUL]
    n_t, n_m = int(res.get("n_trial") or 0), int(res.get("n_cohort") or 0)
    trial = f"Trial (n = {n_t:,})"
    blocks = [{"type": "text", "text": res.get("summary", "")}]
    if scored:
        forest = [{"label": r["name"], "value": round(r["smd"], 3), "text": f"{r['smd']:+.2f}" + (" ≈" if r["approx"] else ""),
                   "textSub": f"{r['centers'][0]} vs {r['centers'][1]}" + (f" {r['unit']}" if r["unit"] else ""),
                   "full": f"{r['label']}: {db} {r['mimic']} vs trial {r['trial']} (SMD {r['smd']:+.2f})"}
                  for r in scored[:15]]
        blocks.append({"type": "chart", "id": "trial-smd", "title": f"Where the {db} patients differ from the trial's",
                       "subtitle": f"standardized difference, {db} − trial (|SMD| ≥ 0.1: a meaningful difference)",
                       "chart": {"type": "forest", "rows": forest, "nul": 0, "axis": "Standardized difference",
                                 "dir": ["higher in the trial", f"higher in {db}"]}})
    blocks.append({"type": "table", "id": "trial-table", "title": "Baseline characteristics",
                   "subtitle": "mean (SD), median (IQR) or % · ≈: from medians and IQRs, or without the trial's SD",
                   "smd": True, "columns": [{"key": "label", "label": "Characteristic"},
                                            {"key": "trial", "label": trial, "numeric": True},
                                            {"key": "mimic", "label": f"{db} (n = {n_m:,})", "numeric": True},
                                            {"key": "smd", "label": "SMD", "numeric": True, "smd": True}],
                   "rows": [{"label": r["label"], "trial": r["trial"], "mimic": r["mimic"],
                             "smd": round(r["smd"], 3) if isinstance(r.get("smd"), (int, float)) else "—"}
                            for r in rows]})
    notes = list(dict.fromkeys(r["note"] for r in rows if r.get("note")))
    if notes:
        blocks.append({"type": "notes", "title": "Definitions differ", "items": notes})
    other = res.get("other") or []
    tabs = [{"id": "results", "label": "Comparison", "blocks": blocks}]
    if other:
        tabs.append({"id": "other", "label": "Not compared", "count": len(other), "blocks": [{
            "type": "table", "id": "trial-other", "title": "Characteristics of the trial not compared",
            "columns": [{"key": "characteristic", "label": "Characteristic"}, {"key": "why", "label": "Why"}],
            "rows": [{"characteristic": o.get("characteristic"), "why": o.get("why")} for o in other]}]})
    return {"cite": "trial", "dataset": f"ClinicalTrials.gov · {dataset}", "design": "Trial vs real-world population",
            "n": n_m, "unit": "patients", "seconds": None,
            "title": f"{res.get('nct')}: {res.get('title') or 'trial'}",
            "subtitle": f"the trial's posted baseline (all arms) beside this study's {db} cohort",
            "badges": [], "tabs": tabs,
            "kpis": [{"label": "Trial participants", "value": f"{n_t:,}", "note": res.get("nct", "")},
                     {"label": f"{db} patients", "value": f"{n_m:,}", "note": "this study's cohort"},
                     {"label": "Compared", "value": f"{len(scored)}", "note": "characteristics both record"},
                     {"label": "Differ", "value": f"{len(differ)}", "tone": "primary" if differ else "",
                      "note": "|SMD| ≥ 0.1" + (f": {differ[0]['name']}" if differ else "")}]}
