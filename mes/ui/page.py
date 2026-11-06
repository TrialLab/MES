"""
MES page — the evidence-synthesis UI.

Entry point: ``render()`` (drawn by ``mes/app.py`` under the header).

* Landing: question box with the mode menu (ui/composer.py), evidence sources, options (PDFs, claim check,
  cache, real-world data analysis engine), examples.
* Run: live trace of every stage and the answer streamed token by token.
* Result: key takeaways, answer with numbered citation chips and grounding badges,
  evidence table (Balanced / Deep), conflicts and gaps, sources rail, follow-up questions.
* Protocols (ui/protocol.py, ui/rwd_panel.py): the search protocol and each real-world data study's protocol behind 🔒
  switches; an edit answers the question again as a new version at the end of the conversation (the edited answer
  stays), running only what the edit changes (tools/protocol_tools.py).
* Agent team builder (ui/team_builder.py): the team that answers — agents, their tools and the depth — on a canvas;
  the team's settings become the question box's, and the run shows the state of each agent.

Heavy resources (sentence encoder, iBKH store, PDF index) are loaded once per
process in a background thread as soon as the page opens.
"""

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

from mes import config
from mes.llm_zoo import bedrock_model_id, create_llm, get_provider, lmstudio_base_url, lmstudio_model, openai_model
from mes.tools import literature_tools, rwd_tools
from mes.tools.ibkh_tools import kg_status
from mes.tools.pipeline_tools import LITERATURE_PARTS, conversation_turn, iter_answer
from mes.tools.protocol_tools import iter_rerun
from mes.tools.rwd import DATABASES
from mes.tools.rwd import analyst as rwd_analyst
from mes.tools.rwd.report import no_traceback
from mes.tools.rwd_tools import ENGINES
from mes.ui import answer_viz, dom, exports, history, landing, protocol, rwd_panel, team_builder, teams, viz, warmup
from mes.ui import components as C
from mes.ui.composer import composer, new_nonce
from mes.ui.icons import icon
from mes.ui.theme import dark_scoped

_CSS = (Path(__file__).parent / "style.css").read_text()
_CSS_DARK = (Path(__file__).parent / "style_dark.css").read_text()
# the chrome of the MES page itself (Streamlit's widgets)
_PAGE_CSS = (Path(__file__).parent / "page.css").read_text() + (Path(__file__).parent / "rwd_tools.css").read_text()
_PAGE_CSS_DARK = (Path(__file__).parent / "page_dark.css").read_text() + \
    (Path(__file__).parent / "rwd_tools_dark.css").read_text()
_DARK = dark_scoped(_CSS_DARK + _PAGE_CSS_DARK)
logger = logging.getLogger(__name__)

EXAMPLES = [
    ("Therapy", "In type 2 diabetes, is semaglutide more effective for weight loss than empagliflozin?"),
    ("Etiology / Harm", "What is the risk of lactic acidosis in patients taking metformin, and how does it change "
                        "with chronic kidney disease severity?"),
    ("Prognosis", "Is a high C-reactive protein (CRP) level a prognostic factor for future cardiovascular events?"),
    ("Therapy", "Does sacubitril/valsartan reduce the risk of hospitalization in patients with heart failure (HFrEF)?"),
    ("Etiology / Harm", "Is childhood obesity a risk factor for developing type 2 diabetes in adulthood?"),
    ("Critical care", "For adults with septic shock and persistent tachycardia, does a short-acting beta-blocker "
                      "such as esmolol improve 28-day mortality?"),
    ("Real-world data", "In septic shock, is norepinephrine associated with lower 28-day mortality than dopamine?"),
    ("Real-world data", "What is the 28-day mortality of ICU patients with septic shock, by ICU type?"),
]

# Chat with the data: every kind of question real-world data answers (asked from the connected database alone)
DATA_EXAMPLES = [
    (":material/person_search:", "Patients like this", "A 68-year-old man with septic shock, lactate 6, creatinine 2.5, "
                                                        "on norepinephrine: how did similar patients do with vasopressin?"),
    (":material/compare_arrows:", "Compare treatments", "In septic shock, is early hydrocortisone associated with lower "
                                                         "28-day mortality?"),
    (":material/groups:", "What patients are like", "How do septic shock patients who die differ from those who survive?"),
    (":material/show_chart:", "Over time", "How does lactate change over the first 72 hours of septic shock in survivors "
                                           "and non-survivors?"),
    (":material/account_tree:", "Treatment pathways", "In septic shock, which vasopressor is started first, and which "
                                                      "follow?"),
    (":material/percent:", "How often", "What is the 28-day mortality of ICU patients with septic shock, by ICU type?"),
    (":material/insights:", "What predicts", "Which first-day characteristics predict in-hospital death in sepsis?"),
    (":material/database:", "What the data has", "Does the database record ECMO, vasopressin and lactate?"),
    (":material/format_list_numbered:", "The commonest", "What are the most commonly used antibiotics in sepsis?"),
    (":material/timer:", "Time on treatment", "How many hours of mechanical ventilation do patients with ARDS receive?"),
]

# Mode menu of the question box: (description, typical time)
MODE_MENU = {"osler": ("Direct answers from the strongest evidence", "~30 s"),
             "sackett": ("Weighs the evidence: evidence table and certainty", "~1 min"),
             "snow": ("Comprehensive survey of every facet, coverage check", "2–3 min")}
ANALYST_TIME = {"osler": "≤ 1½ min", "sackett": "≤ 3½ min", "snow": "≤ 7 min"}

MODE_HELP = {
    "osler": "Fast · quick answer from the strongest evidence (about half a minute)",
    "sackett": "Balanced · weighs the evidence: evidence table and certainty per outcome (about a minute)",
    "snow": "Deep · comprehensive survey: every facet searched, with a coverage check (1–2 minutes)",
}

# Shown in grey in the empty question box; sending the empty box asks it
DEFAULT_QUESTION = "Is apixaban superior to warfarin for stroke prevention in non-valvular atrial fibrillation?"

SOURCE_OPTIONS = {"literature": ":material/article: Literature", "trials": ":material/science: ClinicalTrials.gov",
                  "kg": ":material/hub: iBKH graph",
                  "rwd": ":material/database: Real-world data"}
# what Literature opens to on the source bar (pipeline_tools.LITERATURE_PARTS): name, what it is
LITERATURE_MENU = {"pubmed": ("PubMed", "abstracts; landmark trials first"),
                   "fulltext": ("Open-access full text", "Europe PMC · Unpaywall, for the articles PubMed finds"),
                   "pdfs": ("Your PDF library", "")}
# the status line of the connected real-world database (status menu, the landing's source card)
RWD_STATUS = f"{rwd_tools.DATABASE} (real-world data)"
# The sources of the default team ("MES standard", ui/teams.py): every evidence class of MES, real-world data included
# (it adds a study design and an analysis to every question); see default_sources()
DEFAULT_SOURCES = ["literature", "trials", "kg", "rwd"]
DEEPER_ICONS = {"sackett": ":material/balance:", "snow": ":material/travel_explore:"}

# Display names of the fixed answer headings for answers written in Chinese
HEADINGS_ZH = {"Key takeaways": "要点", "Answer": "回答", "Evidence table": "证据表", "Certainty of evidence": "证据确定性",
               "Evidence from published studies": "已发表研究的证据", "Clinical trials": "临床试验",
               "Knowledge-graph context": "知识图谱背景", "Real-world evidence": "真实世界证据",
               "Agreement, conflicts and gaps": "一致性、冲突与缺口", "Limitations": "局限性"}


HEADINGS_JA = {"Key takeaways": "要点", "Answer": "回答", "Evidence table": "エビデンス表",
               "Certainty of evidence": "エビデンスの確実性", "Evidence from published studies": "公表研究のエビデンス",
               "Clinical trials": "臨床試験", "Knowledge-graph context": "知識グラフの文脈",
               "Real-world evidence": "リアルワールドエビデンス", "Agreement, conflicts and gaps": "一致・不一致・ギャップ",
               "Limitations": "限界"}
HEADINGS_KO = {"Key takeaways": "핵심 요약", "Answer": "답변", "Evidence table": "근거 표",
               "Certainty of evidence": "근거의 확실성", "Evidence from published studies": "출판된 연구의 근거",
               "Clinical trials": "임상시험", "Knowledge-graph context": "지식 그래프 맥락",
               "Real-world evidence": "실사용 근거", "Agreement, conflicts and gaps": "일치, 상충과 공백",
               "Limitations": "제한점"}
HEADINGS = {"chinese": HEADINGS_ZH, "japanese": HEADINGS_JA, "korean": HEADINGS_KO}


def _heading(result, heading):
    language = str(result.plan.get("language", "")).lower()
    names = next((v for k, v in HEADINGS.items() if k in language), {})
    return names.get(heading, heading)
STAGE_LABELS = {"plan": "Question", "literature": "PubMed & full text", "pdfs": "PDFs", "trials": "ClinicalTrials.gov",
                "kg": "iBKH graph", "rwd": "Real-world data", "coverage": "Coverage check", "extract": "Extraction",
                "synthesis": "Synthesis", "verify": "Claim check", "cache": "Cache", "done": "Done"}


def _folder(path):
    """A folder as the page shows it: inside the repository relative to it (data/pdfs), elsewhere its full path."""
    try:
        return Path(path).resolve().relative_to(config.ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


@st.cache_data(ttl=15, show_spinner=False)
def _llm_health():
    """(ok, label, hint) for the configured LLM backend."""
    import os
    provider = get_provider()
    if provider == "openai":
        label = f"OpenAI · {openai_model()}"
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            return True, label, ""
        return False, label, "No OpenAI API key: set OPENAI_API_KEY in .env (see .env.example)"
    if provider == "lmstudio":
        label = f"LM Studio · {lmstudio_model()}"
        try:
            r = requests.get(f"{lmstudio_base_url().rstrip('/')}/models", timeout=2)
            ids = [m.get("id") for m in r.json().get("data", [])] if r.ok else []
            if lmstudio_model() in ids:
                try:        # after a crash LM Studio reloads the model with its default (often 4K) context
                    info = requests.get(f"{lmstudio_base_url().rstrip('/').rsplit('/v1', 1)[0]}/api/v0/models/"
                                        f"{lmstudio_model()}", timeout=2).json()
                    context = info.get("loaded_context_length")
                    if info.get("state") == "loaded" and context and context < 16000:
                        return False, label, (f"{lmstudio_model()} is loaded with only {context:,} tokens of context. "
                                              f"Run: `lms unload {lmstudio_model()}` and "
                                              f"`lms load {lmstudio_model()} -c 32768 -y`")
                except Exception:
                    pass
                return True, label, ""
            return False, label, (f"Model {lmstudio_model()} is not loaded. Run: "
                                  f"`lms load {lmstudio_model()} -c 32768 -y`")
        except Exception:
            return False, label, (f"LM Studio server not reachable at {lmstudio_base_url()}. Run: `lms server start` "
                                  f"and `lms load {lmstudio_model()} -c 32768 -y`")
    label = f"Bedrock · {bedrock_model_id()}"
    try:
        import boto3
        import os
        creds = boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None).get_credentials()
        return (creds is not None), label, "" if creds else ("No AWS credentials found. Configure them (for an SSO "
                                                             "profile: `aws sso login --profile <profile>`)")
    except Exception as e:
        return False, label, f"AWS session error: {e}. Log in again (for an SSO profile: `aws sso login --profile <profile>`)"


# ============================================================================ state
# Widgets of the landing page. While a conversation is shown they are not drawn, so Streamlit deletes their state
# and the browser would draw them empty when the landing page comes back (e.g. no evidence source selected): their
# values are kept under "_keep_<key>" and given back before they are drawn.
LANDING_WIDGETS = {"mes_sources": list(DEFAULT_SOURCES), "mes_verify": False, "mes_use_cache": True,
                   "mes_rwd_engine": "auto", "mes_lit": list(LITERATURE_PARTS)}


def _keep(key):
    st.session_state[f"_keep_{key}"] = st.session_state[key]


def _restore_landing_widgets():
    for key in LANDING_WIDGETS:
        st.session_state[key] = st.session_state[f"_keep_{key}"]


def default_sources():
    """DEFAULT_SOURCES as this computer can run them: real-world data only where its store is built."""
    return [s for s in DEFAULT_SOURCES if s != "rwd" or rwd_tools.available()]


def _init_state():
    defaults = {"mes_thread": [], "mes_mode": "osler", "mes_upload_key": 0, "mes_compose_value": None,
                "mes_compose_nonce": None, "mes_thread_id": None,
                **{f"_keep_{k}": (list(v) if isinstance(v, list) else v) for k, v in LANDING_WIDGETS.items()},
                "_keep_mes_sources": default_sources()}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _new_turn(question, mode, sources, verify, use_cache, uploads, rwd_engine="auto", team=None, literature=None):
    """A question to answer; ``literature``: the parts of the literature source (None: all of them)."""
    return {"id": uuid.uuid4().hex[:8], "question": str(question or "").strip()[:4000],
            "mode": config.get_mode(mode).key, "sources": list(sources or default_sources()), "verify": verify, "use_cache": use_cache,
            "uploads": uploads, "rwd_engine": rwd_engine, "team": team, "status": "pending", "log": [], "error": "",
            "result": None, "literature": list(literature) if literature is not None else None}


def _same_team(last, mode):
    """A follow-up keeps the team of the previous question while it is asked in the same mode."""
    return last.get("team") if config.get_mode(mode).key == last.get("mode") else None


def _team_suffix(turn):
    return f" · 🧩 {C.esc(turn['team'])}" if turn.get("team") else ""


def _mode_menu(sources, engine):
    """Entries of the question box's mode menu; with real-world data and the Claude analyst the time can be longer.
    ``level`` makes the menu an effort scale (animated bars and meter, see composer.py)."""
    analyst = "rwd" in (sources or []) and engine != "templates" and rwd_analyst.status()[0]
    return [{"key": k, "label": config.MODES[k].label, "desc": MODE_MENU[k][0], "level": level,
             "time": MODE_MENU[k][1] + (f" · RWD analyst {ANALYST_TIME[k]}" if analyst else "")}
            for level, k in enumerate(config.MODES)]


def _sent(submitted):
    """A question sent from a question box in this run (each one only once)."""
    if not submitted or not str(submitted.get("question", "")).strip():
        return None
    if submitted.get("at") == st.session_state.get("mes_last_sent"):
        return None
    st.session_state.mes_last_sent = submitted.get("at")
    return submitted


def _read_uploads():
    files = st.session_state.get(f"mes_uploads_{st.session_state.mes_upload_key}") or []
    return [(f.name, f.getvalue()) for f in files]


def _submit_landing(question, mode):
    mode = config.get_mode(mode).key
    st.session_state.mes_mode = mode
    team = st.session_state.get("mes_team")
    same = team and teams.matches(team["config"], st.session_state.mes_sources, mode, st.session_state.mes_verify,
                                  st.session_state.mes_use_cache, st.session_state.mes_rwd_engine)
    st.session_state.mes_thread.append(_new_turn(
        question, mode, st.session_state.mes_sources, st.session_state.mes_verify,
        st.session_state.mes_use_cache, _read_uploads(), st.session_state.mes_rwd_engine,
        team=team["name"] if same else None, literature=_literature_chosen()))


def _use_data_example(question):
    """A question for the data: asked from the real-world data alone (the other sources can be switched back on)."""
    st.session_state.mes_compose_value = question
    st.session_state.mes_compose_nonce = new_nonce()
    st.session_state.mes_sources = ["rwd"]
    _keep("mes_sources")


def _use_example(question, kind=""):
    st.session_state.mes_compose_value = question
    st.session_state.mes_compose_nonce = new_nonce()
    if kind == "Real-world data" and rwd_tools.available() and "rwd" not in st.session_state.mes_sources:
        st.session_state.mes_sources = list(st.session_state.mes_sources) + ["rwd"]
        _keep("mes_sources")


INTERRUPTED = ("The real-world data analysis was interrupted before it finished; the answer is written from the other "
               "sources.")


def _settle_thread():
    """Before the conversation is replaced: an answer shown while the real-world data were still analysed is kept (and
    saved)."""
    thread = st.session_state.get("mes_thread") or []
    changed = False
    for turn in thread:
        if turn.get("status") == "running" and turn.get("partial") is not None:
            if turn.get("cancel") is not None:
                turn["cancel"].set()
            turn.update(status="done", result=turn.pop("partial"), rwd_note=INTERRUPTED)
            changed = True
    if changed and st.session_state.get("mes_thread_id"):
        history.save(st.session_state.mes_thread_id, thread)


def _new_search():
    _settle_thread()
    protocol.close()
    st.session_state.mes_thread = []
    st.session_state.mes_thread_id = None
    st.session_state.mes_compose_value, st.session_state.mes_compose_nonce = "", new_nonce()
    st.session_state.mes_upload_key += 1
    st.session_state.pop("mes_followup_mode", None)


def _restart(turn, **changes):
    """Run ``turn`` again (a new attempt: the start view is drawn first, then the run begins)."""
    turn.update(status="pending", log=[], error="", result=None, primed=False, attempt=turn.get("attempt", 0) + 1,
                **changes)
    for key in ("partial", "rwd_note", "skip_rwd", "stopped"):
        turn.pop(key, None)


def _edit_protocol(index, edit, tested=None):
    """Answer the question of turn ``index`` again with an edited protocol: a new version at the end of the
    conversation, with the conversation before the edited question (the edited answer and its follow-ups stay).
    ``tested``: the editor's test run of the same SQL / code edit (used instead of running it again)."""
    thread = st.session_state.mes_thread
    parent = thread[index]
    root = parent.get("root") or parent["id"]
    versions = [int(t.get("version") or 1) for t in thread if (t.get("root") or t["id"]) == root]
    turn = _new_turn(parent["question"], parent["mode"], parent["sources"], parent["verify"], False, [],
                     parent.get("rwd_engine", "auto"), team=parent.get("team"), literature=parent.get("literature"))
    turn.update(edit=edit, parent=parent["id"], root=root, version=max(versions) + 1,
                history_until=parent.get("history_until", index), base=parent["result"], tested=tested)
    thread.append(turn)


def _base_result(turn):
    """The answer an edited version was made from (kept with the turn while it runs, else found in the thread)."""
    if turn.get("base") is not None:
        return turn["base"]
    parent = next((t for t in st.session_state.mes_thread if t["id"] == turn.get("parent")), None)
    return parent.get("result") if parent else None


def _kicker(index, turn):
    """"Question", "Follow-up" or "Question · protocol v2 (edited)" above a question."""
    kicker = "Follow-up" if turn.get("history_until", index) else "Question"
    return f"{kicker} · protocol v{turn['version']} (edited)" if turn.get("edit") else kicker


def _retry(index):
    turn = st.session_state.mes_thread[index]
    _restart(turn, use_cache=False)
    del st.session_state.mes_thread[index + 1:]


def _ask_related(question):
    """Ask a related question (suggested under the answer) as the next turn of the conversation."""
    thread = st.session_state.mes_thread
    if not thread:
        return
    last = thread[-1]
    mode = st.session_state.get("mes_followup_mode") or last["mode"]
    thread.append(_new_turn(question, mode, last["sources"], last["verify"], last["use_cache"], [],
                            last.get("rwd_engine", "auto"), team=_same_team(last, mode), literature=last.get("literature")))


def _related(index, turn, result):
    """Related questions (from the planner) under the last answer: one click asks them."""
    questions = [q for q in (result.plan.get("follow_ups") or []) if isinstance(q, str) and q.strip()][:3]
    if not questions or index != len(st.session_state.mes_thread) - 1:
        return
    st.markdown('<div class="mes-kicker" style="margin-top:0.9rem">Related</div>', unsafe_allow_html=True)
    for i, q in enumerate(questions):
        st.button(f"↳ {C.md(q, 200)}", key=f"mes_rel_{turn['id']}_{i}", on_click=_ask_related, args=(q,),
                  width="stretch")


def _feedback(index):
    """A thumbs up / down for an answer: kept with the conversation and logged for review."""
    thread = st.session_state.mes_thread
    if index >= len(thread):
        return
    turn = thread[index]
    value = st.session_state.get(f"mes_fb_{turn['id']}")
    turn["feedback"] = value
    result = turn.get("result")
    history.record_feedback({"rating": {0: "down", 1: "up"}.get(value, "cleared"), "question": turn["question"],
                             "mode": turn["mode"], "sources": turn["sources"],
                             "llm": getattr(result, "llm", ""), "created_at": getattr(result, "created_at", ""),
                             "conversation": st.session_state.get("mes_thread_id")})
    history.save(st.session_state.mes_thread_id, thread)


def _deepen(index, mode):
    """Ask the same question again in a deeper mode (the quicker answer stays above, collapsed)."""
    turn = st.session_state.mes_thread[index]
    deeper = _new_turn(turn["question"], mode, turn["sources"], turn["verify"], turn["use_cache"], turn["uploads"],
                       turn.get("rwd_engine", "auto"), literature=turn.get("literature"))
    deeper["history_until"] = turn.get("history_until", index)
    del st.session_state.mes_thread[index + 1:]
    st.session_state.mes_thread.append(deeper)
    st.session_state.mes_followup_mode = mode


# ============================================================================ landing
def _system_status():
    """(llm_ok, llm_hint, rows): one row per component — (name, state text, ok)."""
    ok, label, hint = _llm_health()
    rows = [("Language model", label, ok)]
    kg = kg_status() if warmup.ready() else None
    if kg is None:
        rows.append(("Search engine", "loading (first question only)…", True))
    else:
        rows.append(("Knowledge graph (iBKH)", f"{kg['meta']['n_edges'] / 1e6:.1f}M relations" if kg["available"]
                     else f"not available — {kg['error']}", kg["available"]))
    library = literature_tools.get_pdf_library()
    rows.append(("PDF library", f"{library.refresh()[0]} documents in `{_folder(config.PDF_DIR)}`", True))
    if rwd_tools.available():
        n = (rwd_tools.get_store().catalog().get("summary") or {}).get("adult_first_icu_stays")
        rows.append((RWD_STATUS, f"{int(n):,} adult ICU patients" if n else "ready", True))
        analyst_ok, reason = rwd_analyst.status()
        rows.append(("LLM analyst", "on" if analyst_ok else f"off — {reason}", True))
    else:
        rows.append((RWD_STATUS, rwd_tools.unavailable_reason(), False))
    for db in DATABASES.values():                     # the databases still to come: shown, not an issue
        if not db["supported"]:
            rows.append((f"{db['name']} (real-world data)", "not connected yet", None))
    return ok, hint, rows


def _status_button(status=None):
    """Compact system status: one popover button; details on demand."""
    ok, hint, rows = status or _system_status()
    problems = [r for r in rows if r[2] is False]
    label = ":green[●] Ready" if not problems else f":orange[●] {len(problems)} issue{'s' if len(problems) > 1 else ''}"
    with st.popover(label, width="stretch", help="System status"):
        st.markdown("**System status**")
        for name, text, good in rows:
            st.markdown(f"{'✓' if good else '○' if good is None else '⚠'} **{name}** — {text}")
    return ok, hint


def _open_thread(thread_id):
    """Reopen a saved conversation."""
    _settle_thread()
    protocol.close()
    try:
        turns = history.load(thread_id)
    except Exception as e:
        st.session_state.mes_history_error = f"Could not open the conversation: {e}"
        return
    st.session_state.mes_thread = turns
    st.session_state.mes_thread_id = thread_id
    st.session_state.mes_followup_mode = turns[-1]["mode"] if turns else "osler"
    st.session_state.mes_view = None                 # e.g. opened from the home page while the builder was open


def _delete_thread(thread_id):
    history.delete(thread_id)
    if st.session_state.get("mes_thread_id") == thread_id:
        st.session_state.mes_thread_id = None


def _recent_list(limit=6, key="land"):
    """Recently answered conversations: search, open or remove."""
    if st.session_state.get("mes_history_error"):
        st.warning(st.session_state.pop("mes_history_error"))
    total = history.count()
    if not total:
        if key == "thread":                    # the History menu: say so instead of an empty, greyed-out button
            st.caption("No saved conversations yet: answered questions are kept here.")
        return
    st.markdown('<div class="mes-kicker" style="margin-top:1.2rem">Recent</div>', unsafe_allow_html=True)
    query = ""
    if total > limit:
        query = st.text_input("Search the history", key=f"mes_hist_q_{key}", placeholder="Search the history…",
                              label_visibility="collapsed").strip()
    items = history.search(query, limit) if query else history.recent(limit)
    if query and not items:
        st.caption("No saved conversation matches.")
    for item in items:
        mode = config.get_mode(item["mode"]).label
        more = f" · {item['n_turns']} questions" if item["n_turns"] > 1 else ""
        c1, c2 = st.columns([14, 1], vertical_alignment="center")
        c1.button(f"{C.md(item['title'], 140)}  :gray[· {history.ago(item['updated_at'])} · {mode}{more}]",
                  icon=":material/chat_bubble:",
                  key=f"mes_hist_{key}_{item['id']}", on_click=_open_thread, args=(item["id"],), width="stretch")
        c2.button("✕", key=f"mes_hist_del_{key}_{item['id']}", on_click=_delete_thread, args=(item["id"],),
                  help="Remove from the history")
    # clearing asks once more, in place (a dialog would open under the History menu)
    if st.session_state.get("mes_confirm_clear") == key:
        n = history.count()
        st.caption(f"Delete all {n} saved conversation{'s' if n != 1 else ''}? This cannot be undone.")
        c1, c2 = st.columns(2)
        c1.button("Delete all", key=f"mes_hist_clear_yes_{key}", type="primary", width="stretch", on_click=_clear_history)
        c2.button("Cancel", key=f"mes_hist_clear_no_{key}", width="stretch", on_click=_confirm_clear, args=(None,))
    else:
        st.button("Clear the history…", key=f"mes_hist_clear_{key}", type="tertiary", on_click=_confirm_clear,
                  args=(key,))


def _confirm_clear(key):
    st.session_state.mes_confirm_clear = key


def _clear_history():
    history.clear()
    st.session_state.mes_confirm_clear = None


# ---------------------------------------------------------------------------- the source bar
def _literature_chosen():
    """The literature parts the source bar has on (full text only with PubMed, whose articles it reads)."""
    if "literature" not in (st.session_state.get("mes_sources") or []):
        return []
    parts = [p for p in LITERATURE_PARTS if p in (st.session_state.get("mes_lit") or LITERATURE_PARTS)]
    return [p for p in parts if p != "fulltext" or "pubmed" in parts]


def _set_sources():
    """A choice on the source bar → ``mes_sources`` (literature on while one of its parts is) and ``mes_lit``."""
    previous = st.session_state.get("mes_lit") or []
    lit = [p for p in LITERATURE_PARTS if st.session_state.get(f"mes_src_lit_{p}")]
    if "pubmed" not in previous:         # the full-text box was locked (it reads PubMed's articles): it keeps its choice
        lit = [p for p in LITERATURE_PARTS if p in lit and p != "fulltext" or p == "fulltext" and p in previous]
    on = [p for p in lit if p != "fulltext" or "pubmed" in lit]
    middle = st.session_state.get("mes_src_mid") or []
    st.session_state.mes_lit = lit
    st.session_state.mes_sources = (["literature"] if on else []) + [s for s in ("trials", "kg") if s in middle] \
        + (["rwd"] if st.session_state.get("mes_src_db_mimic") else [])
    _keep("mes_lit")
    _keep("mes_sources")


def _source_bar():
    """Literature ▾ (PubMed, its open-access full text, the PDF library) · ClinicalTrials.gov · iBKH graph ·
    Real-world data ▾ (MIMIC-IV; the databases not connected yet shown, locked). Its widgets are drawn from
    ``mes_sources`` / ``mes_lit`` on every run, so a team, an example or another page that sets them is shown."""
    sources = list(st.session_state.get("mes_sources") or [])
    lit = list(st.session_state.get("mes_lit") or LITERATURE_PARTS) if "literature" in sources else []
    chosen = _literature_chosen()
    for p in LITERATURE_PARTS:
        st.session_state[f"mes_src_lit_{p}"] = p in lit and (p != "fulltext" or "pubmed" in lit)
    st.session_state["mes_src_mid"] = [x for x in sources if x in ("trials", "kg")]
    for key in DATABASES:
        st.session_state[f"mes_src_db_{key}"] = key == "mimic" and "rwd" in sources
    with st.container(horizontal=True, key="mes_sources_bar", gap="small", vertical_alignment="center"):
        with st.container(key="mes_src_litbox", width="content"):
            label = "Literature" if len(chosen) in (0, len(LITERATURE_PARTS)) else \
                f"Literature · {len(chosen)} of {len(LITERATURE_PARTS)}"
            with st.popover(label, icon=":material/article:", key="mes_src_lit",
                            help="PubMed, open-access full text and your PDFs: choose which"):
                st.markdown('<div class="mes-src-menu">Literature</div>', unsafe_allow_html=True)
                n_pdfs = literature_tools.get_pdf_library().refresh()[0]
                for p, (name, about) in LITERATURE_MENU.items():
                    needs = p == "fulltext" and "pubmed" not in lit
                    st.checkbox(name, key=f"mes_src_lit_{p}", on_change=_set_sources, disabled=needs,
                                help="The full text of the articles PubMed finds: on with PubMed" if needs else None)
                    if p == "pdfs":
                        about = f"{n_pdfs} document{'s' if n_pdfs != 1 else ''} in {_folder(config.PDF_DIR)}"
                    st.caption(about + (" · needs PubMed" if needs else ""))
                st.markdown('<div class="mes-src-note">PDFs attached to a question (under Options) are always read.</div>',
                            unsafe_allow_html=True)
            if chosen:
                st.markdown('<i class="mes-src-on"></i>', unsafe_allow_html=True)
        st.pills("Evidence sources", options=["trials", "kg"], format_func=SOURCE_OPTIONS.get, selection_mode="multi",
                 key="mes_src_mid", label_visibility="collapsed", on_change=_set_sources)
        with st.container(key="mes_src_rwdbox", width="content"):
            with st.popover("Real-world data", icon=":material/database:", key="mes_src_rwd",
                            help="A new study on a real-world database, designed and run for the question"):
                st.markdown('<div class="mes-src-menu">Real-world data</div>', unsafe_allow_html=True)
                for key, db in DATABASES.items():
                    usable = db["supported"] and rwd_tools.available()
                    st.checkbox(db["name"] if db["supported"] else f"{db['name']} 🔒", key=f"mes_src_db_{key}",
                                on_change=_set_sources, disabled=not usable,
                                help=None if usable else ("Not connected yet: studies on it come in a later version"
                                                          if not db["supported"] else rwd_tools.unavailable_reason()))
                    state = "connected" if usable else "not connected yet" if not db["supported"] else "not built here"
                    st.caption(f"{db['about']} · {state}")
            if "rwd" in sources:
                st.markdown('<i class="mes-src-on"></i>', unsafe_allow_html=True)


def _options():
    """Attachments and settings of the next question (popover next to the sources)."""
    attached = _read_uploads()
    st.file_uploader("PDFs for this question", type=["pdf"], accept_multiple_files=True,
                     key=f"mes_uploads_{st.session_state.mes_upload_key}")
    st.toggle("Verify claims with a second LLM pass (slower)", key="mes_verify", on_change=_keep, args=("mes_verify",),
              help="Number grounding always runs; this adds an LLM check of each cited claim.")
    st.toggle("Reuse recent answers (cache)", key="mes_use_cache", on_change=_keep, args=("mes_use_cache",))
    ok, reason = rwd_analyst.status()
    st.radio("Real-world data analysis (when real-world data is selected)", options=list(ENGINES), key="mes_rwd_engine",
             on_change=_keep, args=("mes_rwd_engine",),
             format_func=lambda k: ENGINES[k] + ("" if ok or k == "templates" else " — not available"),
             help="Study templates are validated designs that run in seconds. The LLM analyst explores the data "
                  "and writes its own analysis code in a sandbox; its result is reviewed and re-run before it is used.")
    if not ok and st.session_state.mes_rwd_engine != "templates":
        st.caption(f"LLM analyst: {reason}. Questions use the study templates.")
    st.caption(f"Local PDF library folder (searched for every question): `{_folder(config.PDF_DIR)}`")
    return attached


# ============================================================================ agent team
def team_line_html(team, same, sources, mode):
    """One line about the team of a question box: its name, agents and mode (or how to build one)."""
    if not team:
        return (f'<div class="mes-team-line">{icon("team")} <b>Agent team</b> <span>· choose the agents, their tools and '
                'the depth on a canvas</span></div>')
    agents = teams.run_agents(sources if same else team["config"]["sources"], team["config"]["verify"])
    icons = "".join(f'<span class="mes-team-ico" title="{C.esc(teams.TITLES[a])}">{teams.ICONS[a]}</span>'
                    for a in agents)
    label = config.get_mode(team["config"]["mode"]).label
    changed = "" if same else '<span class="mes-team-changed">· settings changed</span>'
    return (f'<div class="mes-team-line">{icon("team")} <b>{C.esc(team["name"])}</b> {icons}<span>· {C.esc(label)}</span>'
            f'{changed}</div>')


def _open_builder(origin="mes"):
    st.session_state.mes_view = "builder"
    st.session_state.mes_builder_from = origin


def _forget_team():
    st.session_state.mes_team = None


def _team_row():
    """The agent team of the question box, and the way into the builder."""
    team = st.session_state.get("mes_team")
    same = bool(team) and teams.matches(team["config"], st.session_state.mes_sources, st.session_state.mes_mode,
                                        st.session_state.mes_verify, st.session_state.mes_use_cache,
                                        st.session_state.mes_rwd_engine)
    cols = st.columns([5, 1.6, 0.35] if team else [5, 1.6], vertical_alignment="center")
    cols[0].markdown(team_line_html(team, same, st.session_state.mes_sources, st.session_state.mes_mode),
                     unsafe_allow_html=True)
    cols[1].button("Agent team builder", key="mes_team_open", on_click=_open_builder, args=("mes",), icon=":material/account_tree:",
                   width="stretch", help="Build the team that answers: agents, tools and depth on a canvas")
    if team:
        cols[2].button("✕", key="mes_team_forget", on_click=_forget_team, help="Stop using this team (the settings "
                                                                                "stay as they are)")


def _apply_team(team):
    """The team's settings become those of the question box."""
    cfg = team["config"]
    st.session_state.mes_team = team
    st.session_state._keep_mes_sources = list(cfg["sources"])
    st.session_state._keep_mes_verify = bool(cfg["verify"])
    st.session_state._keep_mes_use_cache = bool(cfg["use_cache"])
    st.session_state._keep_mes_rwd_engine = cfg["rwd_engine"]
    st.session_state.mes_mode = cfg["mode"]


def _leave_builder():
    st.session_state.pop("mes_builder_from", None)
    st.session_state.mes_view = None
    st.rerun()


def _builder_view():
    """The builder in place of the page; "Use this team" sets the question boxes to the team, then (as "Back") the
    builder is left — the question is asked in the box it came from."""
    action = team_builder.render(DEFAULT_QUESTION)
    if not action:
        return
    if action["type"] == "apply":
        team = action["team"]
        _apply_team(team)
        team_builder.flash(f"The question box now uses the team “{team['name']}”.", "🧩")
    _leave_builder()


def _landing_stats(status):
    """What each source card says is connected on this computer: {source: (text, ok)} — names, no counts (the counts
    and the language model are in the status menu)."""
    _ok, _hint, rows = status
    by = {name: (text, good) for name, text, good in rows}
    caps = {"literature": ("PubMed · Europe PMC · Unpaywall", True), "trials": ("ClinicalTrials.gov, posted results", True)}
    kg = by.get("Knowledge graph (iBKH)")
    if kg:
        caps["kg"] = ("iBKH biomedical knowledge graph" if kg[1] else "knowledge graph off", kg[1])
    rwd = by.get(RWD_STATUS)
    if rwd:                                          # the databases side by side: connected, or not yet
        caps["rwd"] = [(db["name"] if rwd[1] or not db["supported"] else f"{db['name']} — store not built here",
                        bool(rwd[1]) if db["supported"] else None) for db in DATABASES.values()]
    pdfs = by.get("PDF library")
    if pdfs and pdfs[0][:1].isdigit() and not pdfs[0].startswith("0 "):
        caps["literature"] = (pdfs[0].split(" in ")[0] + " in your PDF library · PubMed", True)
    return caps


def _render_landing():
    status = _system_status()
    caps = _landing_stats(status)
    st.markdown(landing.hero(), unsafe_allow_html=True)
    _restore_landing_widgets()
    _, center, _ = st.columns([1, 7, 1])
    with center:
        sources = st.session_state.mes_sources or []
        res = composer("mes_composer", mode=st.session_state.mes_mode,
                       modes=_mode_menu(sources, st.session_state.mes_rwd_engine),
                       placeholder="Ask a medical question…", default_question=DEFAULT_QUESTION,
                       value=st.session_state.mes_compose_value,
                       nonce=st.session_state.mes_compose_nonce, disabled=not sources, autofocus=True,
                       send_title="Search (Enter)" if sources else "Choose at least one evidence source", variant="studio")
        sent = _sent(res["submit"])
        if sent:
            _submit_landing(sent["question"].strip(), sent.get("mode") or st.session_state.mes_mode)
            st.rerun()
        c1, c2, c3 = st.columns([5, 1.25, 1.05], vertical_alignment="center")
        with c1:
            _source_bar()
        with c2:
            attached = _read_uploads()
            with st.popover(f"{len(attached)} PDF · Options" if attached else "Options", width="stretch",
                            icon=":material/attach_file:" if attached else ":material/tune:"):
                _options()
        with c3:
            llm_ok, hint = _status_button(status)
        if not llm_ok:
            st.warning(hint)
        if not sources:
            st.caption("Choose at least one evidence source.")
        if "rwd" in sources and not rwd_tools.available():
            st.caption(rwd_tools.unavailable_reason())
        _team_row()

        st.markdown('<div class="mes-kicker" style="text-align:center;margin-top:1.6rem">Try an example</div>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (kind, question) in enumerate(EXAMPLES):
            with cols[i % 2]:
                st.button(f"**{kind}** {question}", key=f"mes_ex_{i}", on_click=_use_example,
                          args=(question, kind), width="stretch")
        _data_gallery()
        _recent_list()
        st.markdown('<div class="mes-kicker" style="text-align:center;margin-top:2rem">What MES answers from</div>'
                    + landing.capabilities(caps) + landing.how_it_works(), unsafe_allow_html=True)


def _data_gallery():
    """Chat with the data: one question of each kind real-world data answers; a click asks it from the connected
    database alone."""
    ready = rwd_tools.available()
    st.markdown('<div class="mes-kicker" style="text-align:center;margin-top:1.4rem">Chat with real-world data</div>'
                '<div class="mes-data-sub">Ask the data itself: patients like yours, treatment comparisons, what '
                'patients are like, values over time, treatment pathways, rates, predictors — each answer is a study '
                'you can open, refine and run again.</div>'
                + landing.databases([(db["name"], (ready if db["supported"] else None), db["about"])
                                     for db in DATABASES.values()]), unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon_name, kind, question) in enumerate(DATA_EXAMPLES):
        with cols[i % 2]:
            st.button(f"**{kind}** {question}", key=f"mes_dx_{i}", on_click=_use_data_example, args=(question,),
                      width="stretch", icon=icon_name, disabled=not ready,
                      help="Asks it from the real-world data alone (switch the other sources back on to add the "
                           "literature)" if ready
                      else f"🔒 {rwd_tools.unavailable_reason()}")


# ============================================================================ running a question
def _stream_view(text):
    """Escaped streaming markdown with provisional (unnumbered) citation chips and a typing caret."""
    safe = C.esc(text)
    safe = re.sub(r"(?m)^\s*##\s*(.+)$", r'\n<div class="mes-h">\1</div>\n', safe)    # HTML blocks end at a blank line
    safe = re.sub(r"\[([LTKR])(\d+)\]",
                  lambda m: f'<span class="mes-cite mes-cite-{C.TYPE_META[C.source_type(m.group(1))]["cls"]}">'
                            f'{m.group(1)}{m.group(2)}</span>', safe)
    return safe + '<span class="mes-caret">▍</span>'


RETRIEVAL_STAGES = ("literature", "pdfs", "trials", "kg", "rwd")
SEARCH_NAMES = {"literature": "PubMed", "trials": "ClinicalTrials.gov", "kg": "the iBKH graph",
                "rwd": "real-world data"}


def _scroll_to(anchor):
    """Bring the element with id ``anchor`` to the top of the page (Streamlit keeps the old scroll position)."""
    dom.scroll_to(anchor)


def _searching_label(sources, literature=None):
    names = [SEARCH_NAMES[s] if s != "literature" or "pubmed" in (literature or LITERATURE_PARTS) else "your PDFs"
             for s in sources if s in SEARCH_NAMES]
    joined = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] if names else "the sources"
    return f"Searching {joined}…"


def _reading_html(items, limit=8):
    """Compact list of the sources the answer is being written from (shown while it streams)."""
    rows = []
    for item in items[:limit]:
        meta = C.TYPE_META.get(item["type"], C.TYPE_META["literature"])
        detail = " · ".join(str(b) for b in (item.get("design"), item.get("venue"), item.get("year")) if b)
        rows.append(f'<div class="mes-reading-item"><span class="mes-cite mes-cite-{meta["cls"]}">{C.esc(item["id"])}'
                    f'</span> <span class="mes-reading-title">{C.esc(item["title"])}</span>'
                    f'<div class="mes-src-detail">{C.esc(detail)}</div></div>')
    more = f'<div class="mes-src-detail">+ {len(items) - limit} more</div>' if len(items) > limit else ""
    counts = {t: sum(1 for i in items if i["type"] == t) for t in C.TYPE_META}
    summary = " · ".join(f"{n} {C.TYPE_META[t]['label'].lower() if t != 'literature' else 'articles'}"
                         for t, n in counts.items() if n)
    return (f'<div class="mes-reading"><div class="mes-card-title">Reading {len(items)} sources</div>'
            f'<div class="mes-src-detail" style="margin-bottom:0.4rem">{C.esc(summary)}</div>{"".join(rows)}{more}</div>')


def _start_view(index, turn):
    left, _right = st.columns([2.3, 1], gap="large")
    with left:
        st.markdown(f'<div class="mes-kicker">{_kicker(index, turn)} · {config.get_mode(turn["mode"]).label} mode'
                    f'{_team_suffix(turn)}</div><div class="mes-question">{C.esc(turn["question"])}</div>',
                    unsafe_allow_html=True)
        st.status("Starting…", expanded=False)


# Stage lines in plain words ("{n} articles" instead of "8 items"); technical messages go to the search trace
DONE_TEXT = {"literature": "{n} articles", "pdfs": "{n} passages from your PDFs", "trials": "{n} trial records",
             "kg": "{n} knowledge-graph relations", "rwd": "study ready"}


def _stage_line(event):
    stage, status, message = event["stage"], event["status"], event.get("message") or ""
    count = re.match(r"(\d+) items", message)
    if status == "done" and stage in DONE_TEXT and count:
        message = DONE_TEXT[stage].format(n=count.group(1)) if stage != "rwd" or count.group(1) != "0" else \
            "no study"
    icon = {"done": "✓", "error": "⚠", "skipped": "–", "kept": "↺", "edited": "✎"}.get(status, "·")
    secs = f" <span class='mes-trace'>{event['elapsed']:.0f} s</span>" if event.get("elapsed") is not None else ""
    return f"{icon} **{STAGE_LABELS.get(stage, stage)}** — {C.esc(message)}{secs}"


# The steps of a real-world data study as the live card shows them: (label, the progress messages that mean it has begun)
TEMPLATE_STEPS = [("Design", ("designing the study", "correcting the study plan", "exploratory query", "explorer")),
                  ("Cohort", ("running the cohort query", "running the edited query")),
                  ("Analysis", ("after the criteria; estimating",)), ("Checks", ("analysis done",))]
ANALYST_STEPS = [("Explore and code", ("analyst explores", "exploring the data", "analyst · step", "edited analysis code")),
                 ("Review", ("independent review", "revising after review")), ("Reproduce", ("reproduction",))]


def _rwd_steps(state):
    """HTML of the steps of the running study: done, under way, still to come (they never go back)."""
    message = (state.get("message") or "").lower()
    analyst = state.get("analyst") or "analyst" in message
    state["analyst"] = analyst
    steps = (ANALYST_STEPS if analyst else TEMPLATE_STEPS) + [("Answer", ())]
    reached = next((i for i in range(len(steps) - 1, -1, -1) if any(m in message for m in steps[i][1])), 0)
    if analyst and "independent review and reproduction" in message:
        reached = 2                                # review and reproduction run at the same time
    if state.get("phase2"):
        reached = len(steps) - 1
    state["reached"] = reached = max(reached, state.get("reached", 0) if state.get("analyst_was") == analyst else 0)
    state["analyst_was"] = analyst
    items = []
    for i, (label, _m) in enumerate(steps):
        cls = "done" if i < reached else "running" if i == reached else "waiting"
        items.append(f'<span class="mes-step mes-team-{cls}" title="{C.esc(label)}"><span class="mes-team-dot"></span>'
                     f'<span class="mes-step-label">{C.esc(label)}</span></span>')
    return f'<div class="mes-steps-box"><div class="mes-steps">{"<i></i>".join(items)}</div></div>'


# "RWD: ", "RWD analyst (Claude …) · step 3: " (and "MIMIC-IV analyst" in answers saved before) at the start of a message
PROGRESS_PREFIX = re.compile(r"^(?:RWD|MIMIC-IV)(?: analyst)?(?: \([^)]*\))?(?: · step \d+)?:\s*")


def _rwd_progress(slots, state):
    """Steps, text and bar of the live real-world data card (progressive answers)."""
    if not slots:
        return
    elapsed = time.time() - state["start"]
    message = C.esc(PROGRESS_PREFIX.sub("", state.get("message") or ""))
    if state.get("phase2"):
        text, value = "Adding the real-world data results to the answer…", 1.0
    elif state.get("step"):
        text = (f"RWD analyst · step {state['step']} of {state['steps']} · {elapsed:.0f} s of about "
                f"{state['budget']} s<br><span class='mes-trace'>{message}</span>")
        value = min(0.97, elapsed / max(1, state["budget"]))
    else:
        text = f"Designing and running the study · {elapsed:.0f} s<br><span class='mes-trace'>{message}</span>"
        value = min(0.9, elapsed / 60)
    slots["text"].markdown(_rwd_steps(state) + text, unsafe_allow_html=True)
    slots["bar"].progress(value)


def _stop(index):
    turn = st.session_state.mes_thread[index]
    turn["stopped"] = True
    if turn.get("cancel") is not None:
        turn["cancel"].set()


def _skip_rwd(index):
    turn = st.session_state.mes_thread[index]
    turn["skip_rwd"] = True
    if turn.get("cancel") is not None:
        turn["cancel"].set()


def _run_turn(index, turn):
    protocol.close()                       # an editor left open belongs to an answer that is being replaced
    turn["status"] = "running"
    turn["cancel"] = threading.Event()
    turn.pop("partial", None)
    first = turn.get("history_until", index)
    history_turns = [conversation_turn(t["result"]) for t in st.session_state.mes_thread[:first]
                     if t["status"] == "done" and t["result"] is not None]
    anchor = f"mes-q-{turn['id']}"
    body = st.empty()                      # the running view; cleared when the first answer is ready
    first_answer = st.empty()              # the first answer (a fresh placeholder: refilling `body` with a different
    with body.container():                 # structure can leave the browser with a blank page)
        left, right = st.columns([2.3, 1], gap="large")
        with left:
            st.markdown(f'<div class="mes-kicker" id="{anchor}">{_kicker(index, turn)} · '
                        f'{config.get_mode(turn["mode"]).label} mode{_team_suffix(turn)}</div>'
                        f'<div class="mes-question">{C.esc(turn["question"])}</div>', unsafe_allow_html=True)
            _scroll_to(anchor)
            loading = "" if warmup.ready() else " (loading the search engine — first question only)"
            status = st.status(f"Understanding the question{loading}…", expanded=True)
            st.button("Stop", key=f"stop_{turn['id']}_{turn.get('attempt', 0)}", on_click=_stop, args=(index,),
                      icon=":material/stop_circle:",
                      type="tertiary", help="Stop this search")
            detail = st.empty()            # the latest technical message, one line
            answer = st.empty()
        with right:
            team_slot = st.empty()         # the state of each agent of the team
            reading = st.empty()
    tracker = teams.TeamTracker(turn["sources"], turn["verify"])

    def paint_team():
        team_slot.markdown(teams.status_html(tracker.states(), turn["mode"], turn.get("team"), live=True),
                           unsafe_allow_html=True)
    paint_team()
    mode_cfg = config.get_mode(turn["mode"])
    rwd = {"start": time.time(), "steps": mode_cfg.analyst_steps, "budget": mode_cfg.analyst_seconds}
    slots = None
    buffer, last_paint, thinking, n_sources = "", 0.0, False, 0
    try:
        llm = create_llm()
        if turn.get("edit"):
            base = _base_result(turn)
            if base is None:
                raise RuntimeError("the answer this version was edited from is no longer in the conversation")
            events = iter_rerun(turn["question"], llm, base, turn["edit"], version=turn.get("version"),
                                mode=turn["mode"], history=history_turns, verify=turn["verify"],
                                tested=turn.get("tested"), cancel=turn["cancel"])
        else:
            events = iter_answer(turn["question"], llm, sources=turn["sources"], mode=turn["mode"],
                                 uploads=turn["uploads"], history=history_turns, use_cache=turn["use_cache"],
                                 verify=turn["verify"], rwd_engine=turn.get("rwd_engine", "auto"), progressive=True,
                                 cancel=turn["cancel"], literature=turn.get("literature"))
        for event in events:
            kind, phase = event["type"], event.get("phase", 0)
            if kind == "stage":
                if slots is None and tracker.update(event):
                    paint_team()
                if event["stage"] == "rwd" and event["status"] == "running":
                    rwd["start"] = time.time()
                if slots is not None:      # the first answer is shown: only the real-world data card changes
                    if event["stage"] == "synthesis" and event["status"] == "running":
                        rwd["phase2"] = True
                        _rwd_progress(slots, rwd)
                    if event["status"] != "running":
                        turn["log"].append(_stage_line(event))
                    continue
                if event["status"] == "running":
                    if event["stage"] in RETRIEVAL_STAGES and turn.get("edit"):
                        status.update(label=f"Running the edited protocol (v{turn.get('version')})…", expanded=True)
                    elif event["stage"] in RETRIEVAL_STAGES:
                        status.update(label=_searching_label(turn["sources"], turn.get("literature")),
                                      expanded=True)
                    elif event["stage"] == "synthesis":
                        status.update(label=f"Writing the answer from {n_sources} sources…", expanded=True)
                    else:
                        status.update(label=f"{event['message']}…", expanded=True)
                else:
                    line = _stage_line(event)
                    turn["log"].append(line)
                    status.markdown(line, unsafe_allow_html=True)
            elif kind == "log":
                turn["log"].append(f"· {C.esc(event['message'])}")
                if event.get("stage") == "rwd":
                    rwd["message"] = event["message"]
                    step = re.search(r"(?:RWD|MIMIC-IV) analyst · step (\d+)", event["message"])
                    if step:
                        rwd["step"] = int(step.group(1))
                    _rwd_progress(slots, rwd)
                if slots is None:
                    detail.markdown(f"<span class='mes-trace'>{C.esc(event['message'][:160])}</span>",
                                    unsafe_allow_html=True)
            elif kind == "evidence" and phase < 2:
                n_sources = len(event["items"])
                if event["items"]:
                    reading.markdown(_reading_html(event["items"]), unsafe_allow_html=True)
            elif kind == "reasoning" and not thinking and phase < 2:
                thinking = True
                status.update(label="Thinking…")
            elif kind == "token" and phase < 2:
                if not buffer:
                    status.update(label=f"Writing the answer from {n_sources} sources…", expanded=False)
                    detail.empty()
                buffer += event["text"]
                if time.time() - last_paint > 0.12:
                    answer.markdown(_stream_view(buffer), unsafe_allow_html=True)
                    last_paint = time.time()
            elif kind == "error":
                turn["error"] = event["message"]
            elif kind == "result" and not event.get("final", True):
                # progressive: show the answer from the other sources while the real-world data are still analysed
                turn["partial"] = event["result"]
                body.empty()
                with first_answer.container():
                    slots = {}
                    _render_result(index, turn, result=event["result"], live=slots)
                _rwd_progress(slots, rwd)
            elif kind == "result":
                turn["result"] = event["result"]
    except Exception as e:
        logger.exception("The answer to %r failed", turn["question"][:200])
        turn["error"] = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    if turn["result"] is None and turn.get("partial") is not None:
        # the run ended after the first answer without its real-world data update: the first answer stands
        turn["result"] = turn["partial"]
        turn["rwd_note"] = ("The real-world data analysis could not be added" + (f" ({turn['error'][:200]})" if turn["error"]
                                                                          else "")
                            + "; the answer is written from the other sources.")
    elif turn.get("partial") is not None and turn["result"] is not turn["partial"]:
        st.toast("The answer now includes the real-world data analysis.", icon="📊")
    turn.pop("partial", None)
    if turn["result"] is not None:
        turn["status"] = "done"
        history.save(st.session_state.mes_thread_id, st.session_state.mes_thread)
    else:
        turn["status"] = "error"
        if slots is None:
            status.update(label="Could not answer", state="error")
    st.rerun()


# ============================================================================ result view
def _evidence_table(result, anchor):
    rows = result.study_table
    if not rows:
        return
    index = {i: n + 1 for n, i in enumerate(result.citation_order)}
    clean = lambda v: re.sub(r"\[([LTKR]\d+)\]", lambda m: f"[{index[m.group(1)]}]" if m.group(1) in index else "",
                             str(v))
    st.markdown(f'<div class="mes-h">{C.esc(_heading(result, "Evidence table"))}</div>', unsafe_allow_html=True)
    st.markdown(C.table_html([{k: clean(v) for k, v in row.items()} for row in rows]), unsafe_allow_html=True)


def _section(heading, body, result, by_id, anchor):
    sentences = result.verification.get("sentences", [])
    html_body = C.answer_html(body, by_id, result.citation_order, sentences, anchor)
    title = C.esc(_heading(result, heading))
    if heading in ("Agreement, conflicts and gaps",):
        st.markdown(f'<div class="mes-card mes-conflicts"><div class="mes-card-title">{icon("scale")} {title}</div>'
                    f"\n\n{html_body}\n\n</div>", unsafe_allow_html=True)
    elif heading == "Limitations":
        st.markdown(f'<div class="mes-card mes-limits"><div class="mes-card-title">{icon("info")} {title}</div>'
                    f"\n\n{html_body}\n\n</div>", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="mes-h">{title}</div>', unsafe_allow_html=True)
        st.markdown(html_body, unsafe_allow_html=True)


def _verification_note(result):
    v = result.verification
    counts = v.get("counts", {})
    flagged = sum(counts.get(k, 0) for k in ("unsupported_number", "uncited_number", "unsupported"))
    parts = [f"{v.get('n_numbers_grounded', 0)}/{v.get('n_numbers', 0)} numbers found in the cited sources",
             f"{int(round(100 * v.get('citation_coverage', 0)))}% of statements cited"]
    if v.get("invalid_citations"):
        parts.append(f"{len(v['invalid_citations'])} invalid citations removed")
    if v.get("repaired"):
        parts.append(f"{len(v['repaired'])} citations corrected to the source containing the numbers")
    if v.get("llm_verified"):
        parts.append(f"{counts.get('verified', 0)} claims verified by the LLM check")
    if flagged:
        many = flagged != 1
        parts.append(f"⚠ {flagged} statement{'s' if many else ''} flagged — check {'them' if many else 'it'} against "
                     "the sources")
    if v.get("error"):
        parts = ["⚠ the numbers could not be checked this time — compare them with the cited sources"]
    st.caption("Grounding: " + " · ".join(parts))


def _sources_rail(result, by_id, anchor):
    cited = result.citation_order
    groups = {}
    for n, evidence_id in enumerate(cited, start=1):
        if evidence_id in by_id:
            groups.setdefault(by_id[evidence_id].source_type, []).append((n, by_id[evidence_id]))
    reviewed = [e for e in result.evidence if e.id not in set(cited)]
    st.markdown(f"**Sources** <span class='mes-quiet'>({len(cited)} cited · {len(result.evidence)} reviewed)</span>",
                unsafe_allow_html=True)
    types = [t for t in ("literature", "trial", "kg", "rwd") if groups.get(t) or any(e.source_type == t for e in reviewed)]
    if not types:
        st.caption("No sources.")
        return
    turn_id = anchor[len("mes-"):]
    evidence_map = answer_viz.landscape(result, turn_id)
    tabs = st.tabs([f"{C.TYPE_META[t]['short']} {len(groups.get(t, []))}" for t in types] + (["Map"] if evidence_map else []))
    if evidence_map:
        with tabs[-1]:
            st.caption("Every study reviewed, by year and design: filled = cited, larger = more cited (trials: enrolment). "
                       "Hover a study to light up its citation; click it to find it in the answer.")
            viz.show("landscape", evidence_map, key=f"mes_map_{turn_id}")
    for tab, t in zip(tabs, types):
        with tab:
            for n, item in groups.get(t, []):
                st.markdown(_rail_item(C.source_card(n, item, f"{anchor}-{item.id}") + _passages_html(item)),
                            unsafe_allow_html=True)
            if t == "kg" and groups.get(t):
                net = answer_viz.network([item for _, item in groups[t]])
                if net:
                    st.caption("The relations as a network — drag an entity, hover it to see its relations.")
                    viz.show("network", net, key=f"mes_net_{turn_id}")
            others = [e for e in reviewed if e.source_type == t]
            if others:
                with st.expander(f"Also reviewed ({len(others)})"):
                    for item in others:
                        st.markdown(_rail_item(C.source_card("", item, f"{anchor}-{item.id}", cited=False)
                                               + _passages_html(item)), unsafe_allow_html=True)


def _rail_item(html_):
    """A source card of the rail. Streamlit pulls markdown up by 1rem (for a final paragraph's margin); the padding
    gives that back, so the last card of a tab is not cut off at the bottom."""
    return f'<div style="padding-bottom:1rem">{html_}</div>'


def _passages_html(item):
    if item.source_type == "kg" or not item.passages:
        return ""
    body = "".join(f'<div class="mes-passage"><b>{C.esc(p.section)}</b> — {C.esc(no_traceback(p.text))}</div>'
                   for p in item.passages)
    if item.meta.get("results"):
        body += "".join(f'<div class="mes-passage"><b>Posted results</b> — {C.esc(r)}</div>'
                        for r in item.meta["results"] if r not in body)
    return f'<details style="margin:-0.3rem 0 0.5rem 2.2rem"><summary style="font-size:0.72rem;color:#666666">' \
           f'Passages used</summary>{body}</details>'


def _live_card(index, turn, live):
    """Real-world data progress while the first answer is shown (progressive answers)."""
    with st.container(border=True):
        st.markdown(f'<div class="mes-card-title" style="margin:0">{icon("chart")} Real-world data analysis in progress</div>',
                    unsafe_allow_html=True)
        live["text"] = st.empty()
        live["bar"] = st.empty()
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.caption("The answer below is written from the other sources; it is rewritten with the real-world data results "
                   "when the analysis ends.")
        c2.button("Skip", key=f"skip_rwd_{turn['id']}", on_click=_skip_rwd, args=(index,), width="stretch",
                  help="Stop the real-world data analysis and keep this answer")


def _rwd_note(index, turn):
    st.info(turn["rwd_note"])
    if index == len(st.session_state.mes_thread) - 1:
        st.button("Run the real-world data analysis again", key=f"rwd_again_{turn['id']}", on_click=_retry, args=(index,),
                  icon=":material/replay:")


def _notice_view(index, turn, result):
    """A question MES did not search (not a question, not a health question): the reason, nothing else."""
    left, _right = st.columns([2.3, 1], gap="large")
    with left:
        st.markdown(f'<div class="mes-kicker">{_kicker(index, turn)}</div><div class="mes-question">'
                    f'{C.esc(turn["question"])}</div>'
                    f'<div class="mes-card mes-notice">{C.esc(result.notice)}</div>', unsafe_allow_html=True)


def _meta_html(result, turn, live):
    """The line under the question: depth (with its level bars), model, time, sources, cache, team."""
    order = list(config.MODES)
    level = order.index(result.mode) if result.mode in order else 0
    bars = "".join(f'<i class="{"on" if i <= level else ""}"></i>' for i in range(len(order)))
    first = result.timings.get("first_answer")
    total = result.timings.get("total", 0)
    time_text = f"{total:.0f} s" + (f" · first answer {first:.0f} s" if first and live is None and first < total else "")
    chips = [f'<span class="mes-chip mes-chip-mode"><span class="mes-lvl" aria-hidden="true">{bars}</span>'
             f'{C.esc(config.get_mode(result.mode).label)}</span>',
             f'<span class="mes-chip">⏱ <b>{C.esc(time_text)}</b></span>',
             f'<span class="mes-chip"><b>{len(result.evidence)}</b> sources reviewed</span>']
    if result.llm:
        chips.append(f'<span class="mes-chip">{C.esc(result.llm)}</span>')
    if result.from_cache:
        chips.append('<span class="mes-chip">↺ cached answer</span>')
    if turn.get("team"):
        chips.append(f'<span class="mes-chip">🧩 {C.esc(turn["team"])}</span>')
    return f'<div class="mes-meta">{"".join(chips)}</div>'


def _render_result(index, turn, result=None, live=None):
    """The answer of ``turn`` (``result``: a first answer while the real-world data are still analysed; ``live`` receives the
    slots of its progress card)."""
    result = result or turn["result"]
    if result.notice:
        _notice_view(index, turn, result)
        return
    by_id = result.evidence_by_id()
    anchor = f"mes-{turn['id']}"
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        q_anchor = f"mes-q-{turn['id']}"
        st.markdown(f'<div class="mes-kicker" id="{q_anchor}">{_kicker(index, turn)}</div>'
                    f'<div class="mes-question">{C.esc(turn["question"])}</div>', unsafe_allow_html=True)
        scroll = not turn.get("shown") and live is None    # once, when the answer is complete: start at the question
        if scroll:
            turn["shown"] = True
        # drawn on every run, scrolling only the first time: the elements below keep their place, so an expander the
        # reader opened (search protocol, study) stays open when a widget in it (a 🔒 switch) reruns the page. Not in
        # a first answer shown while the real-world analysis runs: the run view has this element already.
        if live is None:
            dom.scroll_to(q_anchor if scroll else "", key=f"mes_scroll_{q_anchor}")
        st.markdown(_meta_html(result, turn, live), unsafe_allow_html=True)
        upload_problems = (result.source_status.get("pdfs") or {}).get("upload_problems")
        if upload_problems:
            st.caption(f"⚠ {C.md(upload_problems, 400)}")
        protocol.version_card(result, turn)
        if live is not None:
            _live_card(index, turn, live)
        if not result.citation_order and not result.evidence:
            st.info("No evidence was found for this question in the selected sources.")
        if live is None:                       # the answer at a glance: its evidence, grounding and designs
            viz.show("glance", answer_viz.glance(result, turn["id"]), key=f"mes_glance_{turn['id']}")
        if result.takeaways:
            st.markdown(C.takeaways_html(result.takeaways, by_id, result.citation_order, anchor,
                                         title=_heading(result, "Key takeaways")), unsafe_allow_html=True)
        rwd_shown = False
        # an edited protocol becomes a new version at the end of the conversation: any finished answer can be edited
        on_edit = (lambda edit, tested=None: _edit_protocol(index, edit, tested)) if live is None else None
        # a question a data catalog suggests: asked as the next turn (only under the last answer)
        on_ask = _ask_related if live is None and index == len(st.session_state.mes_thread) - 1 else None
        for heading, body in result.sections.items():
            if heading == "Key takeaways":
                continue
            if heading == "Evidence table":
                _evidence_table(result, anchor)
                continue
            _section(heading, body, result, by_id, anchor)
            if heading == "Real-world evidence" and live is None:
                if result.rwd_studies:          # the study right under the text that reports it
                    rwd_panel.render(result, turn, on_edit, summary=True, on_ask=on_ask)
                if turn.get("rwd_note"):
                    _rwd_note(index, turn)
                rwd_shown = True
        if live is None and not rwd_shown:
            if result.rwd_studies:
                rwd_panel.render(result, turn, on_edit, summary=True, on_ask=on_ask)
            if turn.get("rwd_note"):
                _rwd_note(index, turn)

        counts = {t: sum(1 for i in result.citation_order if i in by_id and by_id[i].source_type == t) for t in C.TYPE_META}
        st.markdown(C.pill("Literature", counts["literature"], "lit", "literature")
                    + C.pill("Trials", counts["trial"], "trial", "trials")
                    + C.pill("Knowledge graph", counts["kg"], "kg", "kg") + C.pill("Real-world data", counts["rwd"], "rwd", "rwd"),
                    unsafe_allow_html=True)
        _verification_note(result)

        key = f"{turn['id']}{'_live' if live is not None else ''}"
        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.55], vertical_alignment="center")
        with c1:
            exports.copy_button(exports.plain_text(result), key=f"copy_{key}")
        with c2:
            with st.popover("Export", width="stretch", icon=":material/download:"):
                st.download_button("Report for printing / Word (.html)", exports.report_html(result),
                                   file_name=f"mes_{turn['id']}.html", mime="text/html", key=f"dl_html_{key}",
                                   width="stretch", on_click="ignore")
                st.download_button("Markdown (.md)", C.export_markdown(result), file_name=f"mes_{turn['id']}.md",
                                   mime="text/markdown", key=f"dl_md_{key}", width="stretch", on_click="ignore")
                st.download_button("All data (.json)", json.dumps(result.to_dict(), indent=2, default=str),
                                   file_name=f"mes_{turn['id']}.json", mime="application/json", key=f"dl_js_{key}",
                                   width="stretch", on_click="ignore")
                st.caption("Open the .html report in a browser and print it to PDF, or open it in Word.")
        with c3:
            last = index == len(st.session_state.mes_thread) - 1
            # an earlier answer is not re-run (its follow-ups would be dropped)
            st.button("Re-run", key=f"rerun_{key}", on_click=_retry, args=(index,), width="stretch", icon=":material/refresh:",
                      disabled=not last, help="Search again without the cache" if last else
                      "Only the last answer of a conversation can be re-run")
        if live is None:
            with c4:                           # was the answer helpful? (kept with the conversation, logged)
                st.feedback("thumbs", key=f"mes_fb_{turn['id']}", default=turn.get("feedback"),
                            on_change=_feedback, args=(index,))
        order = list(config.MODES)                      # osler → sackett → snow
        deeper = order[order.index(result.mode) + 1:] if result.mode in order else []
        if deeper and index == len(st.session_state.mes_thread) - 1 and live is None:
            cols = st.columns([1.2] + [2] * len(deeper))
            cols[0].markdown('<div class="mes-deeper-label">Go deeper</div>', unsafe_allow_html=True)
            for col, m in zip(cols[1:], deeper):
                col.button(f"{config.MODES[m].label} — {config.MODES[m].tagline.lower()}", icon=DEEPER_ICONS[m],
                           key=f"deep_{m}_{turn['id']}", on_click=_deepen, args=(index, m), width="stretch",
                           help=MODE_HELP[m])
        if live is None:
            _related(index, turn, result)
        version = f" · v{result.protocol['version']} (edited)" if result.protocol else ""
        with st.expander(f"Search protocol — concepts and queries{version}", icon=":material/manage_search:"):
            protocol.search_view(turn, result, on_edit)
        with st.expander("Search trace", icon=":material/timeline:"):
            for line in turn["log"]:
                st.markdown(line, unsafe_allow_html=True)
            st.caption("Timings (s): " + ", ".join(f"{k} {v}" for k, v in result.timings.items()))
    with right:
        st.markdown('<div class="mes-rail-marker"></div>', unsafe_allow_html=True)
        # the agents that answered, with their times, above the sources they found
        st.markdown(teams.rail_html(teams.result_states(result, turn.get("verify")), result.mode, turn.get("team")),
                    unsafe_allow_html=True)
        _sources_rail(result, by_id, anchor)


def _render_thread():
    thread = st.session_state.mes_thread
    if not st.session_state.get("mes_thread_id"):
        st.session_state.mes_thread_id = history.new_id()
    top = st.columns([3.2, 1, 1])
    with top[1]:
        with st.popover("History", width="stretch", icon=":material/history:"):
            _recent_list(limit=10, key="thread")
    with top[2]:
        st.button("New search", on_click=_new_search, width="stretch", icon=":material/add:")
    for index, turn in enumerate(thread):
        if turn["status"] == "running" and turn.get("partial") is not None:
            # the run stopped while the real-world data were still analysed (Skip, or another action): keep the first answer
            skipped = turn.pop("skip_rwd", False)
            if turn.get("cancel") is not None:
                turn["cancel"].set()
            turn.update(status="done", result=turn.pop("partial"),
                        rwd_note="The real-world data analysis was skipped; the answer is written from the other sources."
                        if skipped else INTERRUPTED)
            history.save(st.session_state.mes_thread_id, thread)
        if turn["status"] == "pending":
            if not turn.get("primed") and not st.session_state.get("mes_no_kick"):
                # draw the question first and let this run end (it clears the previous view), then start the run
                turn["primed"] = True
                _start_view(index, turn)
                dom.rerun_when_shown(f"{turn['id']}-{turn.get('attempt', 0)}")
                return
            _run_turn(index, turn)
            return
        if index < len(thread) - 1 and turn["status"] == "done":
            later = any(t.get("root") == turn["id"] for t in thread)
            version = f" · protocol v{turn['version']} (edited)" if turn.get("edit") else " · protocol v1" if later else ""
            with st.expander(f"Q: {C.md(turn['question'], 160)} · {config.get_mode(turn['mode']).label}{version}",
                             expanded=False):
                _render_result(index, turn)
            continue
        if turn["status"] == "done":
            _render_result(index, turn)
        elif turn["status"] == "running":
            if turn.get("stopped"):
                st.info(f"Stopped: “{C.md(turn['question'], 160)}”.")
            else:
                st.warning(f"The search for “{C.md(turn['question'], 160)}” was interrupted.")
            st.button("Retry", key=f"retry_{turn['id']}", on_click=_retry, args=(index,))
        else:
            st.error(f"Could not answer “{C.md(turn['question'], 160)}”: {C.md(turn['error'] or 'unknown error', 400)}")
            ok, _, hint = _llm_health()
            if not ok:
                st.info(hint)
            st.button("Retry", key=f"retry_{turn['id']}", on_click=_retry, args=(index,))

    # Follow-up box right under the answer. (A pinned st.chat_input would turn the page into a chat view that
    # keeps jumping to the bottom, while an evidence answer is read from the top.)
    last = thread[-1]
    st.session_state.setdefault("mes_followup_mode", last["mode"])
    res = composer(f"mes_followup_{len(thread)}", mode=st.session_state.mes_followup_mode,
                   modes=_mode_menu(last["sources"], last.get("rwd_engine", "auto")),
                   placeholder="Ask a follow-up…", compact=True, variant="studio")
    st.markdown('<div class="mes-disclaimer"><span class="mes-mark">M</span>MES is a research prototype for evidence '
                'synthesis, not a medical device. Check the cited sources before any clinical decision.</div>',
                unsafe_allow_html=True)
    sent = _sent(res["submit"])
    if sent:
        mode = sent.get("mode") or st.session_state.mes_followup_mode or last["mode"]
        st.session_state.mes_followup_mode = mode
        thread.append(_new_turn(sent["question"], mode, last["sources"], last["verify"], last["use_cache"], [],
                                last.get("rwd_engine", "auto"), team=_same_team(last, mode),
                                literature=last.get("literature")))
        st.rerun()


# ============================================================================ entry point
def render():
    warmup.start()
    _init_state()
    # the dark styles apply on a page drawn dark only (html.mes-dark, set by the theme probe: ui/theme.py)
    st.markdown(f"<style>{_CSS}{_PAGE_CSS}{_DARK}</style><div class=\"mes-page\"></div>", unsafe_allow_html=True)
    dom.theme_probe()
    team_builder.show_flash()
    # A question handed over by another page (text) or a test ({"question", "mode", "sources", "rwd_engine",
    # "verify", "use_cache", "team"})
    prefill = st.session_state.pop("mes_prefill", None)
    if prefill:
        st.session_state.mes_view = None                 # the question is shown, not the builder
    if prefill and not st.session_state.mes_thread:
        p = prefill if isinstance(prefill, dict) else {"question": prefill}
        st.session_state.mes_thread_id = None            # a new conversation
        st.session_state.mes_thread.append(_new_turn(p["question"], p.get("mode", "osler"),
                                                     p.get("sources", default_sources()),
                                                     bool(p.get("verify", False)), bool(p.get("use_cache", True)), [],
                                                     p.get("rwd_engine", "auto"), team=p.get("team"),
                                                     literature=p.get("literature")))
    # One root placeholder: a new view (e.g. a question that starts running) replaces the previous one at once,
    # instead of leaving its elements on screen, faded, until the long run finishes
    with st.empty().container():
        if st.session_state.get("mes_view") == "builder":
            _builder_view()
        elif st.session_state.mes_thread:
            _render_thread()
        else:
            _render_landing()
