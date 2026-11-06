"""
MES page: real-world data study panel — design, cohort, results, diagnostics, SQL or code, the cohort
builder (ui/cohort_builder.py), the trial comparison (ui/trial_compare.py) and the editors of a study's protocol behind 🔒 switches (ui/protocol.py): the design
of a template study, the query of an exploratory one, the code of a custom analysis. An edit is tested in the editor (SQL, code) and answers the
question again as a new version (tools/protocol_tools.py); only the edited study runs again.
"""

import html
import json

import pandas as pd
import streamlit as st

from mes import config
from mes.tools import rwd_tools
from mes.tools.rwd import library as L
from mes.tools.rwd.report import estimate_text, no_traceback
from mes.ui import design_rows, protocol
from mes.ui.components import md

ANALYSIS_LABELS = {"compare": "Treatment comparison (target trial emulation)", "association": "Adjusted association",
                   "rate": "Rate / proportion", "distribution": "Distribution", "utilization": "Treatment use",
                   "survival": "Survival curves", "trend": "Trend over time", "prognostic": "Prognostic model",
                   "explore": "Exploratory query", "custom": "Custom analysis (code written by the MES analyst)",
                   "similar": "Patients like this (similar-patient cohort)", "characterize": "Cohort characterization",
                   "trajectory": "Trajectory over time", "pathways": "Treatment pathways",
                   "top": "Commonest medications, diagnoses or procedures"}
VERDICT_BADGE = {"sound": ("✓ review: no problems found", "ok"), "minor_issues": ("◐ review: minor issues", "warn"),
                 "major_issues": ("✗ review: major problems", "bad"), "not_reviewed": ("review not done", "warn")}


COLUMN_NAMES = {"variable": "Subgroup", "level": "Level", "n": "Patients", "events": "Events", "rate": "Rate %",
                "lo": "95% CI low", "hi": "95% CI high", "subgroup": "Subgroup", "rd": "Risk difference",
                "median": "Median", "q1": "Q1", "q3": "Q3", "period": "Period", "quartile": "Quartile"}


def _frame(rows, readable=False):
    """Table for display: a column mixing numbers and suppressed counts ("<10") is shown as text; ``readable``:
    the column names of the template tables in words."""
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, str)).any() and df[col].map(lambda v: isinstance(v, (int, float))).any():
            df[col] = df[col].astype(str)
    return df.rename(columns={c: COLUMN_NAMES.get(str(c), c) for c in df.columns}) if readable else df


def _readable(text):
    """A warning without the Python traceback a stored study record may carry (only the error line stays)."""
    return no_traceback(text)


def _badge(text, kind):
    colors = {"ok": ("#15803D", "#F0FDF4", "#BBF7D0"), "warn": ("#B45309", "#FFFBEB", "#FDE68A"),
              "bad": ("#B91C1C", "#FEF2F2", "#FECACA"), "info": ("#0F766E", "#F0FDFA", "#99F6E4"),
              "edit": ("#CF4520", "#FBEEE9", "#F2CBBE")}[kind]
    return (f'<span style="display:inline-block;font-size:0.74rem;font-weight:600;border-radius:999px;padding:1px 9px;'
            f'margin:0 4px 4px 0;color:{colors[0]};background:{colors[1]};border:1px solid {colors[2]}">{text}</span>')


def _script(study, question):
    """The analysis code as a script with a header (runs in the MES sandbox, which provides sql(), cohort() …)."""
    header = [f"# MES custom analysis of {study.get('dataset') or rwd_tools.DATASET} — written by {study.get('llm', 'the analyst')}",
              f"# Question: {question}", f"# Data store built {study.get('data_built_at', '?')}; MES mode {study.get('mode')}",
              "# Runs in the MES analysis sandbox (mes/tools/rwd/sandbox.py), which provides np, pd, sql(), cohort(),",
              "# stats, L, code_predicate(), drug_names(), show() and submit().", ""]
    return "\n".join(header) + (study.get("code") or "") + "\n"


def _number(result, study):
    """The citation number of a study in the answer ([3]) or its source id (R1) when it is not cited."""
    evidence_id = study.get("evidence_id", "R")
    order = list(result.citation_order or [])
    return str(order.index(evidence_id) + 1) if evidence_id in order else evidence_id


def summary_card(result, study):
    """Compact card of a study: design, patients, main estimate and its checks."""
    res, status = study.get("result") or {}, study.get("status")
    number = html.escape(_number(result, study))
    if study.get("kind") == "analyst":
        design = "custom analysis (code written by the MES analyst)"
    else:
        design = ANALYSIS_LABELS.get((study.get("spec") or {}).get("analysis"), "analysis").lower()
    estimates = [e for e in res.get("estimates") or [] if not e.get("sensitivity")]
    main = next((e for e in estimates if e.get("primary")), estimates[0] if estimates else None)
    if status == "not_feasible":
        body = f"Not answerable with {html.escape(rwd_tools.database_of(study))}: " + html.escape(
            (study.get("spec") or {}).get("reason", ""))
    elif main is None:
        body = "No estimate: " + html.escape("; ".join(_readable(w) for w in res.get("warnings") or []
                                                       or (study.get("design_info") or {}).get("errors", []))[:240])
    else:
        body = f"<b>{html.escape(estimate_text(main))}</b>"
    badges = []
    edited = (study.get("edited") or {}).get("kind")
    if edited:
        badges.append(_badge(f"✎ {rwd_tools.EDIT_KINDS.get(edited, 'protocol')} edited by you", "edit"))
    if study.get("kind") == "analyst" and status == "done" and edited:
        badges.append(_badge("review not repeated after the edit", "warn"))
    elif study.get("kind") == "analyst" and status == "done":
        text, kind = VERDICT_BADGE.get((study.get("review") or {}).get("verdict"), VERDICT_BADGE["not_reviewed"])
        badges.append(_badge(text, kind))
        reproduced = (study.get("reproduction") or {}).get("reproduced")
        badges.append(_badge("✓ reproduced" if reproduced else "✗ not reproduced", "ok" if reproduced else "bad"))
    elif edited == "sql":
        badges.append(_badge("no independent check query", "warn"))
    warnings = res.get("warnings") or []
    if warnings and main is not None:
        badges.append(_badge(f"⚠ {len(warnings)} warning{'s' if len(warnings) > 1 else ''}", "warn"))
    patients = f" · {res['n']:,} patients" if res.get("n") else ""
    return (f'<div class="mes-rwd-card"><div class="mes-rwd-head"><span class="mes-cite mes-cite-rwd">{number}</span> '
            f'{html.escape(study.get("dataset") or rwd_tools.DATASET)} · {html.escape(design)}{patients}</div>'
            f'<div class="mes-rwd-main">{body}</div>'
            f'<div>{"".join(badges)}</div></div>')


def _record_label(study):
    if study.get("kind") == "analyst":
        return "Open the analysis record — code, steps, downloads, edit the code"
    return "Open the study record — SQL, downloads, edit the design"


def _template_record(study, result, turn, on_edit):
    """What the card does not show: the SQL, the downloads and the 🔒 editors of a template or exploratory study."""
    spec, res = study.get("spec") or {}, study.get("result") or {}
    if study.get("sql"):
        st.markdown("**SQL** · one DuckDB query of the cohort, compiled from the design")
        st.code(study["sql"], language="sql")
    if res.get("sql_check"):
        st.caption("Independent check query:")
        st.code(res["sql_check"], language="sql")
    if spec.get("analysis") == "explore" and on_edit is not None:
        why = rwd_tools.editable(study, "sql") or ("" if rwd_tools.available() else rwd_tools.unavailable_reason())
        if protocol.entry(f"mes_lock_{turn['id']}_sql_{study.get('key')}", "sql", "Edit the SQL and re-run",
                          ready=not why, why=why):
            sql_dialog(turn["id"], study, result, on_edit)
    elif spec.get("analysis") != "explore":
        st.caption("This SQL is compiled from the study design: to change it, edit the design.")
    c0, c1 = st.columns(2, vertical_alignment="center")
    with c0:
        from mes.ui import study_report
        st.download_button("Study report (.html)", study_report.report_html(result, study) or "",
                           file_name=f"mes_study_{study.get('key')}.html", mime="text/html", on_click="ignore",
                           key=f"rwd_rep_{turn['id']}_{study.get('key')}", width="stretch", icon=":material/description:",
                           help="Every tab of the study card with its charts — print it to PDF or open it in Word")
    with c1:
        st.download_button("Protocol and results (.json)", json.dumps(study, indent=2, default=str),
                           file_name=f"mes_rwd_{study.get('key')}.json", mime="application/json", on_click="ignore",
                           key=f"rwd_dl_{turn['id']}_{study.get('key')}", width="stretch", icon=":material/download:")
    if on_edit is not None:                  # the design editor on a row of its own: its 🔒 switch keeps its label
        why = rwd_tools.editable(study, "design") or ("" if rwd_tools.available() else rwd_tools.unavailable_reason())
        if protocol.entry(f"mes_lock_{turn['id']}_design_{study.get('key')}", "design", "Edit design and re-run",
                          ready=not why, why=why):
            edit_dialog(spec, on_edit, study.get("key"))


def _analyst_record(study, result, turn, on_edit):
    """The analyst's code (downloadable, 🔒 editable), every step with its code and output, the record."""
    with st.container():
        if study.get("code"):
            st.markdown(f"**Code** · written by {md(study.get('llm', 'the analyst'), 80)}, run in the MES sandbox")
            st.code(study["code"], language="python", line_numbers=True)
            st.download_button("analysis.py", _script(study, result.question), file_name=f"mes_analysis_{study['key']}.py",
                               mime="text/x-python", key=f"rwd_py_{turn['id']}_{study['key']}", icon=":material/code:")
            if on_edit is not None:
                why = rwd_tools.editable(study, "code") or ("" if rwd_tools.available() else
                                                           rwd_tools.unavailable_reason())
                if protocol.entry(f"mes_lock_{turn['id']}_code_{study['key']}", "code", "Edit the code and re-run",
                                  ready=not why, why=why):
                    code_dialog(turn["id"], study, result, on_edit)
        else:
            st.caption("No code ran successfully.")
        steps = study.get("steps") or []
        if steps:
            st.markdown("**Steps**")
            cells = iter(study.get("cells") or [])
            for n, step in enumerate(steps, start=1):
                st.markdown(f"**{n}. {md(step.get('action'))}** · {step.get('seconds', 0):.0f} s — "
                            f"{md(_readable(step.get('plan') or ''), 800)}")
                if step.get("action") == "python":
                    cell = next(cells, None)
                    if cell is not None:
                        with st.expander("code and output" + ("" if cell.get("ok") else " (error)")):
                            st.code(cell.get("code", ""), language="python")
                            out = (cell.get("stdout") or "") + (("\n" + cell["error"]) if cell.get("error") else "")
                            st.code(_readable(out.strip()) or "(nothing printed)", language="text")
                elif step.get("code"):
                    st.code(step["code"], language="sql" if step.get("action") == "sql" else "text")
        c1, c2 = st.columns(2)
        if study.get("status") == "done":
            from mes.ui import study_report
            c1.download_button("Study report (.html)", study_report.report_html(result, study) or "",
                               file_name=f"mes_study_{study.get('key')}.html", mime="text/html", on_click="ignore",
                               key=f"rwd_rep_{turn['id']}_{study.get('key')}", width="stretch",
                               icon=":material/description:")
        c2.download_button("Analysis record (.json)", json.dumps(study, indent=2, default=str),
                           file_name=f"mes_rwd_{study.get('key')}.json", mime="application/json", on_click="ignore",
                           key=f"rwd_dl_{turn['id']}_{study.get('key')}", width="stretch", icon=":material/download:")


def _unfinished(study, result, turn, on_edit):
    """A study without results: why (not feasible, no valid design, the analyst did not finish) and what can be done."""
    st.markdown(summary_card(result, study), unsafe_allow_html=True)
    if study.get("kind") == "analyst":
        with st.expander(_record_label(study), expanded=False, icon=":material/receipt_long:"):
            routed = (study.get("design_info") or {}).get("routed")
            if routed:
                st.caption(f"Why a custom analysis: {routed}.")
            if study.get("status") == "not_feasible":
                st.info(f"Not answerable with {rwd_tools.database_of(study)}: "
                        f"{(study.get('spec') or {}).get('reason', '')}")
            else:
                st.warning("The analyst did not finish: " + md(_readable("; ".join((study.get("design_info") or {})
                                                                             .get("errors", []))), 600))
            _analyst_record(study, result, turn, on_edit)
        return
    why = ("why this question cannot be answered with this data" if study.get("status") == "not_feasible"
           else "why no valid study design was made")
    with st.expander(f"{rwd_tools.database_of(study)} — {why}", expanded=False, icon=":material/info:"):
        spec = study.get("spec") or {}
        if study.get("status") == "not_feasible":
            st.info(f"Not answerable with {rwd_tools.database_of(study)}: {spec.get('reason', '')}")
        else:
            st.warning("No valid study design: " + md(_readable("; ".join((study.get("design_info") or {})
                                                                        .get("errors", []))), 600))
    if (study.get("spec") or {}).get("analysis") == "explore" and study.get("status") != "not_feasible":
        # an exploratory query that failed: its SQL can be fixed and run again (rwd_tools.editable), like an
        # analyst's code that did not finish
        with st.expander("Open the study record — the query, downloads, fix the SQL", expanded=False,
                         icon=":material/receipt_long:"):
            _template_record(study, result, turn, on_edit)


def render(result, turn, on_edit, summary=False, on_ask=None):
    """
    RWD panel for every study of ``result``: the study card (ui/study_card.py, drawn by ui/viz.py) and, under it, the
    record — SQL or code, steps, downloads and the 🔒 editors. ``on_edit(edit, tested=None)`` answers the question
    again with an edited study protocol (None: no editing); ``on_ask(question)`` asks a question a data catalog
    suggests (None: its questions are only listed). ``summary`` is kept for callers of older versions.
    """
    from mes.ui import cohort_builder, patient_explorer, study_card, trial_compare, viz
    for study in result.rwd_studies or []:
        spec = study_card.card(result, study) if study.get("status") == "done" else None
        if spec is None:
            _unfinished(study, result, turn, on_edit)
            continue
        viz.show("study", spec, key=f"mes_study_{turn['id']}_{study.get('key')}")
        if (study.get("spec") or {}).get("analysis") == "catalog":
            _ask_buttons(study, turn, on_ask)
            continue
        patient_explorer.render(study, turn, on_edit)            # patients like this: the patient, live
        cohort_builder.render(study, turn, on_edit)
        trial_compare.render(study, result, turn)                 # the cohort beside a trial's participants
        with st.expander(_record_label(study), expanded=False, icon=":material/receipt_long:"):
            if study.get("kind") == "analyst":
                routed = (study.get("design_info") or {}).get("routed")
                if routed:
                    st.caption(f"Why a custom analysis: {routed}.")
                _analyst_record(study, result, turn, on_edit)
            else:
                _template_record(study, result, turn, on_edit)


def _ask_buttons(study, turn, on_ask):
    """The questions a data catalog suggests, each one click away (as a follow-up of the conversation)."""
    questions = [x.get("question") for x in ((study.get("result") or {}).get("tables") or {}).get("suggestions") or []
                 if isinstance(x, dict) and x.get("question")]
    if not questions or on_ask is None:
        return
    st.markdown(f'<div class="mes-kicker" style="margin-top:0.4rem">Ask {html.escape(rwd_tools.database_of(study))}</div>',
                unsafe_allow_html=True)
    for i, q in enumerate(questions[:4]):
        st.button(md(q, 160), key=f"mes_ask_{turn['id']}_{study.get('key')}_{i}", on_click=on_ask, args=(q,),
                  icon=":material/database:", type="tertiary")


@st.dialog("Edit the study design", width="large", on_dismiss=protocol.close)
def edit_dialog(spec, on_edit, study_key):
    from mes.tools.protocol_tools import design_changes
    from mes.tools.rwd.concepts import get_index
    from mes.tools.rwd.spec import StudySpec, validate
    base = st.selectbox("Population", list(L.BASES), index=list(L.BASES).index(spec["base"]),
                        format_func=lambda k: L.BASES[k]["label"])
    analysis = st.selectbox("Analysis", list(L.ANALYSES), index=list(L.ANALYSES).index(spec["analysis"]),
                            format_func=lambda k: ANALYSIS_LABELS.get(k, k))
    outcomes = [""] + list(L.outcomes_for(base))
    outcome = st.selectbox("Outcome", outcomes, index=outcomes.index(spec.get("outcome", "")) if spec.get("outcome", "")
                           in outcomes else 0, format_func=lambda k: L.OUTCOMES[k][0] if k else "—")
    variables = list(L.variables_for(base))
    extra = {}                  # what the chosen analysis needs besides the fields every design has
    if analysis == "distribution":
        numeric = [v for v in variables if L.VARIABLES[v][1] == "numeric"]
        current = spec.get("variable") or ""
        options = numeric + ([current] if current and current not in numeric else [])
        c1, c2 = st.columns(2)
        chosen = c1.selectbox("Variable — its distribution: median, IQR, histogram", options,
                              index=options.index(current) if current in options else 0, key=f"mes_design_var_{study_key}",
                              format_func=lambda k: L.VARIABLES[k][0] if k in L.VARIABLES else get_index().label(k))
        concept = c2.text_input("…or a lab test / ICU item (its concept id, e.g. from the search below)", "",
                                key=f"mes_design_var_concept_{study_key}", placeholder="e.g. lab:50813").strip()
        extra["variable"] = concept.removeprefix("var:") or chosen    # the search lists variables as var:…
    elif analysis == "explore":
        extra["sql_question"] = st.text_input(
            "Question for the exploratory query (MES writes its SQL; to write the SQL yourself, edit it in the SQL tab)",
            spec.get("sql_question") or "", key=f"mes_design_question_{study_key}").strip()
    c1, c2 = st.columns(2)
    with c1:
        grace = st.number_input("Grace period (h, treatment comparison)", 0.0, 168.0, float(spec.get("grace_h", 24)))
        stratify = st.multiselect("Break down by", variables, default=[v for v in spec.get("stratify", [])
                                                                        if v in variables],
                                  format_func=lambda k: L.VARIABLES[k][0])
    with c2:
        covariates = st.multiselect("Adjust for (covariates)", variables,
                                    default=[v for v in spec.get("covariates", []) if v in variables],
                                    format_func=lambda k: L.VARIABLES[k][0])
    st.markdown("**Criteria** — one row each: who is included or excluded, the treatment (or exposure) and its "
                "comparator; windows in hours from time zero")
    as_json = st.toggle("Edit the criteria as JSON (advanced)", key=f"mes_design_json_{study_key}")
    criteria = {k: spec.get(k) for k in ("include", "exclude", "exposure", "comparator")}
    text, table = "", []
    if as_json:
        text = st.text_area("Criteria", json.dumps(criteria, indent=1, default=str), height=220,
                            label_visibility="collapsed", key=f"mes_design_criteria_{study_key}")
    else:
        rows = design_rows.rows_of(spec) or [{c: "" for c in design_rows.COLUMNS}]
        frame = pd.DataFrame(rows, columns=design_rows.COLUMNS)
        frame["Without"] = frame["Without"].map(lambda v: v is True)
        table = st.data_editor(
            frame, num_rows="dynamic", hide_index=True, width="stretch", key=f"mes_design_rows_{study_key}",
            column_config={
                "Role": st.column_config.SelectboxColumn("Role", options=list(design_rows.ROLE_KEYS), required=True),
                "Concept": st.column_config.TextColumn("Concept", help="A concept id: var:…, phe:…, drugclass:…, drug:…, "
                                                       "lab:…, item:…, dx10:… — find one below", max_chars=120),
                "Comparison": st.column_config.SelectboxColumn("Comparison", options=design_rows.CHOICES),
                "Value": st.column_config.TextColumn("Value", help="A number, two for “between” (2–4), or a category"),
                "From (h)": st.column_config.TextColumn("From (h)", help="Window start, hours from time zero"),
                "To (h)": st.column_config.TextColumn("To (h)", help="Window end (empty: to the end of the stay)"),
                "Without": st.column_config.CheckboxColumn("Without", help="The criterion is absent")}).to_dict("records")
        meanings = []
        for r in table:
            concept = str(r.get("Concept") or "").strip()
            if ":" in concept:
                try:
                    meanings.append(f"{concept} = {get_index().label(concept)}")
                except Exception:
                    meanings.append(concept)
        if meanings:
            st.caption("Concepts: " + " · ".join(md(m, 140) for m in meanings[:10]))
    term = st.text_input("Find a concept (condition, drug, lab, variable)", placeholder="e.g. vasopressin")
    if term:
        hits = get_index().search(term, k=8)
        st.dataframe(pd.DataFrame([{"concept": c.concept_id, "description": c.label,
                                    "patients": c.n_patients} for c in hits]), hide_index=True, width="stretch")
    st.caption("The edited version is added below as a new answer; only this study runs again, the other sources "
               "are kept.")
    if st.button("Re-run with this design", type="primary", key=f"mes_design_run_{study_key}"):
        try:
            errors = []
            if as_json:
                chosen_criteria = json.loads(text)
                if not isinstance(chosen_criteria, dict):
                    raise ValueError("the criteria must be a JSON object with include, exclude, exposure and comparator")
            else:
                chosen_criteria, errors = design_rows.criteria_of(table)
            if errors:
                raise ValueError("; ".join(errors))
            edited = {**spec, "base": base, "analysis": analysis, "outcome": outcome, "grace_h": grace,
                      "stratify": stratify, "covariates": covariates, **extra, **chosen_criteria}
            new = StudySpec.from_dict(edited)
            for c in new.include + new.exclude + [new.exposure, new.comparator]:
                if c is not None:
                    c.label = ""
            errors, _warnings = validate(new, get_index())
            if analysis == "distribution" and new.variable.split(":")[0] in ("lab", "item") \
                    and not get_index().is_known(new.variable):
                errors.append(f"“{new.variable}” is not a lab test or ICU item of {rwd_tools.DATABASE}: search it below and use "
                              "its concept id (lab:… or item:…)")
            if not errors and not design_changes(spec, new.to_dict()):
                errors = ["Nothing was changed in the study design."]
        except json.JSONDecodeError as e:
            errors = [f"The criteria are not valid JSON: {e}"]
        except ValueError as e:                    # what the table or the JSON cannot mean
            errors = [str(e)]
        except Exception as e:
            errors = [f"{type(e).__name__}: {e}"]
        if errors:
            for e in errors:
                st.error(e)
        else:
            protocol.close()
            on_edit({"part": "rwd", "study": study_key, "kind": "design", "spec": new.to_dict()})
            st.rerun()


def _tested(key, edit):
    """The editor's test run of exactly this edit (None if it was not tested or the text changed since)."""
    tested = st.session_state.get(key)
    return tested if tested and tested.get("edit_hash") == rwd_tools.edit_hash(edit) else None


def _run_test(key, study, edit, result, spinner):
    with st.spinner(spinner):
        try:
            record = rwd_tools.edited_study(study, edit, result.question, config.get_mode(result.mode))
        except Exception as e:
            record = {"status": "failed", "design_info": {"errors": [f"{type(e).__name__}: {e}"]}}
    record["edit_hash"] = rwd_tools.edit_hash(edit)
    st.session_state[key] = record


def _unchanged(before, after):
    return " ".join(str(before or "").strip().rstrip(";").split()) == " ".join(str(after or "").strip().rstrip(";").split())


@st.dialog("Edit the SQL query", width="large", on_dismiss=protocol.close)
def sql_dialog(turn_id, study, result, on_edit):
    key = f"mes_test_{turn_id}_sql_{study['key']}"
    st.caption(f"One read-only SELECT over the {rwd_tools.database_of(study)} tables (mes.*, mimiciv_hosp.*, "
               "mimiciv_icu.*, mimiciv_derived.*). The result must be aggregated: no patient, admission or stay "
               "identifiers, at most 200 rows; counts below 10 are shown as \"<10\".")
    sql = st.text_area("SQL", study.get("sql") or "", height=300, key=f"{key}_text", label_visibility="collapsed")
    edit = {"part": "rwd", "study": study["key"], "kind": "sql", "sql": sql}
    c1, c2 = st.columns(2)
    if c1.button("▶ Test the query", width="stretch", key=f"{key}_test"):
        _run_test(key, study, edit, result, f"Running the query on {rwd_tools.database_of(study)}…")
    tested = _tested(key, edit)
    if tested is not None:
        errors = (tested.get("design_info") or {}).get("errors") or []
        if tested.get("status") == "done":
            rows = ((tested.get("result") or {}).get("tables") or {}).get("result") or []
            st.success(f"The query ran: {len(rows)} row{'s' if len(rows) != 1 else ''}.")
            if rows:
                st.dataframe(_frame(rows), hide_index=True, width="stretch")
        for e in errors:
            st.error(_readable(e))
    elif st.session_state.get(key):
        st.caption("The query changed since the test: test it again.")
    unchanged = _unchanged(study.get("sql"), sql)
    if unchanged:
        st.caption("Change the query, test it, then re-run the answer with it.")
    ready = tested is not None and tested.get("status") == "done" and not unchanged
    if c2.button("Re-run the answer with this query", type="primary", disabled=not ready, width="stretch",
                 key=f"{key}_run", help=None if ready else "Test the changed query first"):
        st.session_state.pop(key, None)
        protocol.close()
        on_edit(edit, tested)
        st.rerun()


@st.dialog("Edit the analysis code", width="large", on_dismiss=protocol.close)
def code_dialog(turn_id, study, result, on_edit):
    key = f"mes_test_{turn_id}_code_{study['key']}"
    mode = config.get_mode(result.mode)
    st.caption("Python in the MES analysis sandbox: np, pd, sql(), cohort(), stats, L, code_predicate(), drug_names(), "
               "show() and submit(result). Imports from numpy, pandas, scipy, statsmodels, lifelines, sklearn and the "
               "standard library only; no files, network or processes. The code runs as one script in a fresh "
               "session; the numbers of the answer come from its submit().")
    code = st.text_area("Code", study.get("code") or "", height=460, key=f"{key}_text", label_visibility="collapsed")
    edit = {"part": "rwd", "study": study["key"], "kind": "code", "code": code}
    c1, c2 = st.columns(2)
    if c1.button("▶ Test in a fresh session", width="stretch", key=f"{key}_test"):
        _run_test(key, study, edit, result, f"Running the code in a fresh sandboxed session (at most "
                                            f"{max(90, min(300, 2 * mode.analyst_seconds))} s)…")
    tested = _tested(key, edit)
    if tested is not None:
        res = tested.get("result") or {}
        if tested.get("status") == "done":
            st.success(f"Submitted: {res.get('title', '')}")
            st.markdown("\n".join(f"- {md(estimate_text(e), 300)}" for e in (res.get("estimates") or [])[:8]))
        elif tested.get("status") == "not_feasible":
            st.info(f"Submitted as not answerable with {rwd_tools.database_of(study)}: "
                    f"{md((tested.get('spec') or {}).get('reason'), 400)}")
        for e in (tested.get("design_info") or {}).get("errors") or []:
            st.error(_readable(e))
        problems = [r for r in tested.get("code_audit") or [] if r.get("problem")]
        if problems:
            st.warning("Code check: " + "; ".join(f"{r['kind']} {r['written']}: {r['problem']}" for r in problems[:6]))
        out = ((tested.get("cells") or [{}])[0].get("stdout") or "").strip()
        if out:
            with st.expander("Printed output"):
                st.code(out[-3000:], language="text")
    elif st.session_state.get(key):
        st.caption("The code changed since the test: test it again.")
    unchanged = str(study.get("code") or "").strip() == str(code or "").strip()
    if unchanged:
        st.caption("Change the code, test it, then re-run the answer with it.")
    ready = tested is not None and tested.get("status") in ("done", "not_feasible") and not unchanged
    if c2.button("Re-run the answer with this code", type="primary", disabled=not ready, width="stretch",
                 key=f"{key}_run", help=None if ready else "Test the changed code first"):
        st.session_state.pop(key, None)
        protocol.close()
        on_edit(edit, tested)
        st.rerun()
