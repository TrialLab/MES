"""
MES pipeline — the evidence synthesis of one question (the Supervisor's coarse-grained tool).

    plan (1 LLM call) → parallel retrieval (literature, PDFs, trials, iBKH: no LLM; real-world data: its designer)
    → [Snow: facet coverage check + targeted retrieval] → evidence IDs
    → [batched study extraction: ModeConfig.extract_studies, off in every mode] → streamed synthesis (1 LLM call)
    → citation validation + number grounding → [batched LLM claim verification: the claim check option, off by
    default]

``iter_answer`` is a generator of events, all produced on the caller's thread
(worker threads only push progress messages to a queue), so a UI can render them
directly::

    {"type": "stage", "stage": ..., "status": "running" | "done" | "error" | "skipped", "message": ...}
    {"type": "log", "stage": ..., "message": ...}
    {"type": "evidence", "items": [...]}     # the sources the answer is written from (id, type, title, design, year)
    {"type": "reasoning", "text": ...}      # thinking tokens of reasoning models
    {"type": "token", "text": ..., "phase": 0 | 1 | 2}   # answer tokens
    {"type": "result", "result": MESResult, "final": bool}

Progressive answers (``progressive=True``, the UI): when only the real-world data analysis is still running after the
other sources finished, the answer is written from those sources first (a result event with ``final: False``,
tokens of phase 1), then written again with the MIMIC-IV results when the analysis ends (phase 2, ``final: True``).
``cancel`` (a ``threading.Event``) stops a running analysis; it is also set when the consumer stops reading.

``answer_question`` consumes the events and returns the ``MESResult``.
Results are cached (``.cache/results``) by question, sources, mode, model and conversation.
"""

import hashlib
import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from mes import config
from mes.tools import literature_tools, rwd_tools, trials_tools
from mes.tools.ibkh_tools import get_kg, kg_status
from mes.tools.planner_tools import plan_query
from mes.tools.ranking_tools import encode, expanded_query
from mes.tools.result_schema import SCHEMA_VERSION, SOURCE_TYPES, MESResult, QueryPlan, load_result, save_result
from mes.tools.synthesis_tools import (ANSWER_PREFILL, PreambleGate, answer_summary, extract_study_table,
                                       parse_evidence_table, postprocess_answer, stream_synthesis, verify_claims)

logger = logging.getLogger(__name__)

# Bump when retrieval, ranking or prompts change: cached answers of older versions are not reused
PIPELINE_VERSION = "2026-10-25.11"

AVAILABLE_SOURCES = ("literature", "trials", "kg", "rwd")
# every evidence class of MES; real-world data is skipped (stage "skipped") where its store is not built
DEFAULT_SOURCES = ("literature", "trials", "kg", "rwd")
# the parts of the literature source (all of them by default): PubMed, the open-access full text of the articles it
# finds, the PDF library on this computer — PDFs uploaded with a question are always read
LITERATURE_PARTS = ("pubmed", "fulltext", "pdfs")


def literature_parts(parts=None):
    """The literature parts asked for, in their order (None: all of them)."""
    return list(LITERATURE_PARTS) if parts is None else [p for p in LITERATURE_PARTS if p in parts]
SOURCE_LABELS = {"literature": "Literature (PubMed, full text, PDFs)", "trials": "ClinicalTrials.gov",
                 "kg": "Knowledge graph (iBKH)", "rwd": f"Real-world data ({rwd_tools.DATABASE})", "pdfs": "PDFs"}

NO_EVIDENCE_ANSWER = """## Key takeaways
- No relevant evidence was found in the selected sources for this question.

## Answer
MES could not find studies, trial records or knowledge-graph relations that address this question, so it does not
give an answer.{note} Try rephrasing the question with standard terms (drug, condition, outcome), or select more
evidence sources.
"""


NOT_A_QUESTION = ("Please type a medical question — for example: “Is apixaban superior to warfarin for stroke "
                  "prevention in atrial fibrillation?”")
OUT_OF_SCOPE = ("MES answers clinical and biomedical questions from the evidence (PubMed, ClinicalTrials.gov, the iBKH "
                "knowledge graph and real-world data). This question does not look like one, so MES did "
                "not search. Ask about a treatment, a test, a disease, its causes or its outcomes — for example: “Does "
                "empagliflozin slow the progression of chronic kidney disease?”")


NOT_A_QUESTION_ZH = "请输入一个医学问题，例如：“在非瓣膜性房颤中，阿哌沙班预防卒中是否优于华法林？”"
OUT_OF_SCOPE_ZH = ("MES 依据证据（PubMed、ClinicalTrials.gov、iBKH 知识图谱和真实世界数据）回答临床和生物医学问题。"
                   "这个问题看起来不属于此类，所以 MES 没有检索。可以问某种治疗、检查、疾病及其病因或结局，例如："
                   "“恩格列净能否延缓慢性肾病进展？”")


MAX_QUESTION = 4000            # characters of a question that are used (a pasted article would overflow the prompts)


def question_problem(question):
    """Why ``question`` cannot be searched (no words: "", "?", "123"), or None."""
    letters = re.findall(r"[^\W\d_]", question or "")
    cjk = any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in letters)
    return None if len(letters) >= 2 or (letters and cjk) else NOT_A_QUESTION


def _localized(text, language):
    if "chinese" in str(language or "").lower():
        return {NOT_A_QUESTION: NOT_A_QUESTION_ZH, OUT_OF_SCOPE: OUT_OF_SCOPE_ZH}.get(text, text)
    return text


def _notice(question, text, mode_cfg, sources, llm, t_start, plan=None):
    """Events of an answer that explains why MES did not search."""
    markdown = f"## Answer\n{text}\n"
    result = MESResult(question=question, mode=mode_cfg.key, sources=list(sources),
                       plan=asdict(plan or QueryPlan(question=question)), evidence=[], answer_markdown=markdown,
                       answer_display=markdown, sections={"Answer": text},
                       timings={"total": round(time.time() - t_start, 2)}, llm=llm.label, notice=text)
    yield {"type": "stage", "stage": "done", "status": "done", "message": "No search: " + text[:80]}
    yield {"type": "result", "result": result, "final": True}


# ============================================================================ sources
def _literature(plan, mode, query, qv, report, fulltext=True):
    items, stats = literature_tools.search_pubmed(plan, mode, report)
    lexical = expanded_query(plan)
    selected = literature_tools.rank_documents(items, query, qv, mode.literature_max, mode.passages_per_doc,
                                               facets=plan.facets if mode.facets else None, lexical_query=lexical)
    report(f"PubMed: kept {len(selected)} most relevant articles "
           f"({sum(1 for it in selected if it.evidence_level <= 2)} reviews/trials)")
    stats["n_fulltext"] = literature_tools.add_fulltext(selected, query, qv, mode.fulltext_max,
                                                        mode.passages_per_doc, report, lexical_query=lexical) \
        if fulltext else 0
    stats["n_selected"] = len(selected)
    return selected, stats


def _pdfs(uploads, query, qv, mode, report, library_too=True):
    items = []
    library = literature_tools.get_pdf_library()
    indexed, total = library.refresh() if library_too else (0, 0)
    if total:
        items += library.search(query, qv, k=3, passages_per_doc=mode.passages_per_doc)
        note = f" (indexing {indexed}/{total})" if library.indexing else ""
        report(f"Local PDF library: {len(items)} relevant of {indexed} documents{note}")
    problems = []
    if uploads:
        def note(message):
            problems.append(message)
            report(message)
        uploaded = literature_tools.search_uploaded_pdfs(uploads, query, qv, k=4, passages_per_doc=mode.passages_per_doc,
                                                         report=note)
        report(f"Uploaded PDFs: {len(uploaded)} of {len(uploads)} relevant")
        items += uploaded
    return items, {"n_selected": len(items), "library_files": total, "upload_problems": "; ".join(problems)}


def _kg(plan, mode, query, qv, report):
    graph = get_kg()
    if graph is None:
        raise RuntimeError(kg_status()["error"])
    return graph.retrieve(plan, query, qv, k=mode.kg_max, report=report)


def _snow_coverage(plan, items, mode, query, report):
    """Snow: facets without supporting articles get one targeted retrieval round (no LLM call)."""
    if not plan.facets:
        return []
    added = []
    covered = {it.meta.get("facet") for it in items}
    for facet in plan.facets:
        if facet["name"] in covered:
            continue
        facet_plan = QueryPlan(question=facet["question"] or query, standalone_question=facet["question"] or query,
                                pubmed_query=facet["pubmed_query"], pubmed_tiers=[facet["pubmed_query"]])
        found, _ = literature_tools.search_pubmed(facet_plan, mode)
        known = {it.identifiers.get("pmid") for it in items + added}
        found = [it for it in found if it.identifiers.get("pmid") not in known]
        fv = encode([facet_plan.query], is_query=True)[0]
        best = literature_tools.rank_documents(found, facet_plan.query, fv, 2, mode.passages_per_doc)
        for it in best:
            it.meta["facet"] = facet["name"]
        added += best
        report(f"Coverage check: facet '{facet['name']}' had no article; added {len(best)}")
    return added


def _assign_ids(groups):
    """Final evidence list with stable IDs L1.., T1.., K1.., R1.. (best first within each source)."""
    evidence = []
    for source_type, members in groups.items():
        prefix = SOURCE_TYPES[source_type]["prefix"]
        order = lambda it: (not it.meta.get("named"), not it.meta.get("landmark"), -it.score)
        for n, item in enumerate(sorted(members, key=order), start=1):
            item.id = f"{prefix}{n}"
            evidence.append(item)
    return evidence


# ============================================================================ cache
def _cache_key(question, sources, mode, llm, history, uploads, verify=False, rwd_engine="auto", literature=None):
    parts = literature_parts(literature)
    payload = {"q": " ".join(question.lower().split()), "sources": sorted(sources), "mode": mode, "llm": llm.label,
               "verify": bool(verify), **({"rwd_engine": rwd_engine} if "rwd" in sources else {}),
               **({"literature": parts} if "literature" in sources and parts != list(LITERATURE_PARTS) else {}),
               "history": [h.get("question", "") for h in history or []],
               "uploads": sorted(hashlib.sha1(data).hexdigest() for _, data in uploads or []),
               "v": SCHEMA_VERSION, "pipeline": PIPELINE_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _cache_path(key):
    return config.CACHE_DIR / "results" / f"{key}.json"


def cached_result(question, llm, sources=DEFAULT_SOURCES, mode="osler", history=None, uploads=None, verify=None,
                  rwd_engine="auto", literature=None):
    mode_cfg = config.get_mode(mode)
    verify = mode_cfg.verify_claims if verify is None else bool(verify)
    path = _cache_path(_cache_key(question, sources, mode_cfg.key, llm, history, uploads, verify,
                                     rwd_engine, literature))
    if path.exists() and time.time() - path.stat().st_mtime < config.HTTP_CACHE_TTL_HOURS * 3600:
        try:
            result = load_result(path)
            result.from_cache = True
            return result
        except Exception:
            return None
    return None


# ============================================================================ pipeline
def iter_answer(question, llm, sources=DEFAULT_SOURCES, mode="osler", uploads=None, history=None, use_cache=True,
                verify=None, rwd_spec=None, rwd_engine="auto", progressive=False, cancel=None, literature=None):
    """
    Answer ``question``; yields progress, token and result events (see module docstring).
    ``verify`` turns the LLM claim check on or off (default: the mode setting); ``rwd_spec`` re-runs the
    real-world data analysis with an edited study design (no cache); ``rwd_engine`` chooses how the real-world data
    are analysed: "auto", "templates" or "analyst" (``rwd_tools.ENGINES``); ``literature``: the parts of the
    literature source (``LITERATURE_PARTS``; None: all of them).
    """
    if rwd_spec is not None:
        use_cache = False
    mode_cfg = config.get_mode(mode)
    verify = mode_cfg.verify_claims if verify is None else bool(verify)
    sources = [s for s in AVAILABLE_SOURCES if s in (sources or DEFAULT_SOURCES)]
    parts = literature_parts(literature) or list(LITERATURE_PARTS)          # none chosen: all of them
    history = history or []
    t_start = time.time()
    timings = {}
    question = (question or "").strip()[:MAX_QUESTION]
    problem = question_problem(question)
    if problem:
        language = "Chinese" if re.search(r"[\u3000-\u303f\uff00-\uffef]", question or "") else "English"
        yield from _notice(question, _localized(problem, language), mode_cfg, sources, llm, t_start)
        return

    if use_cache:
        hit = cached_result(question, llm, sources, mode_cfg.key, history, uploads, verify, rwd_engine, parts)
        if hit is not None:
            yield {"type": "stage", "stage": "cache", "status": "done", "message": "Answer loaded from cache"}
            yield {"type": "result", "result": hit}
            return

    calls_before = llm.n_calls

    # ---- 1. plan ---------------------------------------------------------------
    yield {"type": "stage", "stage": "plan", "status": "running", "message": "Understanding the question"}
    t = time.time()
    plan = plan_query(question, llm, history=history, facets=mode_cfg.facets)
    timings["plan"] = round(time.time() - t, 2)
    previous = next((h.get("rwd_design") for h in reversed(history) if h.get("rwd_design")), "")
    if previous and "rwd" in sources:                # a follow-up refines the MIMIC-IV study of the conversation
        plan.rwd_previous = str(previous)[:900]
    if plan.error and is_backend_error(plan.error):   # LLM unreachable: stop before spending time on retrieval
        yield {"type": "stage", "stage": "plan", "status": "error", "message": plan.error[:300]}
        yield {"type": "error", "message": f"The LLM backend ({llm.label}) is not reachable: {plan.error[:300]}"}
        return
    from mes.tools.planner_tools import named_ids
    if not plan.in_scope and not plan.fallback and not any(named_ids(question).values()):
        yield {"type": "stage", "stage": "plan", "status": "done", "message": "Not a health question",
               "elapsed": timings["plan"]}
        yield from _notice(question, _localized(OUT_OF_SCOPE, plan.language), mode_cfg, sources, llm, t_start, plan)
        return
    concepts = ", ".join(e["name"] for e in plan.entities) or "keywords"
    facet_note = f"; facets: {', '.join(f['name'] for f in plan.facets)}" if plan.facets else ""
    yield {"type": "stage", "stage": "plan", "status": "done",
           "message": f"{plan.question_type.title()} question — concepts: {concepts}{facet_note}"
                      + (" (keyword fallback)" if plan.fallback else ""), "elapsed": timings["plan"]}

    query = plan.query
    query_vector = encode([query], is_query=True)[0]

    # ---- 2. parallel retrieval ---------------------------------------------------
    messages = queue.Queue()
    cancel = cancel or threading.Event()
    finished = False

    def reporter(stage):
        return lambda msg: messages.put((stage, msg))

    jobs, results, source_status, rwd_studies = {}, {}, {}, []
    pool = ThreadPoolExecutor(max_workers=6)
    try:
        if "literature" in sources:
            if "pubmed" in parts:
                jobs["literature"] = pool.submit(_literature, plan, mode_cfg, query, query_vector, reporter("literature"),
                                                 "fulltext" in parts)
            if uploads or ("pdfs" in parts and literature_tools.get_pdf_library().refresh()[1]):
                jobs["pdfs"] = pool.submit(_pdfs, uploads, query, query_vector, mode_cfg, reporter("pdfs"),
                                           "pdfs" in parts)
        if "trials" in sources:
            jobs["trials"] = pool.submit(trials_tools.search_trials, plan, mode_cfg, query, query_vector,
                                         reporter("trials"))
        if "kg" in sources:
            jobs["kg"] = pool.submit(_kg, plan, mode_cfg, query, query_vector, reporter("kg"))
        if "rwd" in sources:
            jobs["rwd"] = pool.submit(rwd_tools.search_rwd, plan, mode_cfg, query, query_vector, reporter("rwd"),
                                      llm, rwd_spec, rwd_engine, cancel)
        started = {name: time.time() for name in jobs}
        for name in jobs:
            yield {"type": "stage", "stage": name, "status": "running", "message": SOURCE_LABELS.get(name, "PDFs")}

        def collect(name):
            """Events for a finished job; stores its items and status."""
            nonlocal rwd_studies
            elapsed = round(time.time() - started[name], 2)
            timings[name] = elapsed
            try:
                items, stats = jobs[name].result()
                results[name] = items
                source_status[name] = {"ok": True, "n_items": len(items), "elapsed": elapsed,
                                       **{k: v for k, v in stats.items() if not isinstance(v, (dict, list))}}
                if name == "kg":
                    source_status[name]["links"] = stats.get("links", {})
                    source_status[name]["semantic_nodes"] = stats.get("semantic_nodes", [])
                if name == "literature":
                    source_status[name]["queries"] = stats.get("queries", {})
                if name == "rwd":
                    rwd_studies = stats.get("studies", [])
                status = "skipped" if stats.get("available") is False else "done"
                message = stats.get("error") if status == "skipped" else f"{len(items)} items"
                yield {"type": "stage", "stage": name, "status": status, "message": message, "elapsed": elapsed}
            except Exception as e:
                logger.warning("Source %s failed: %s", name, e)
                results[name] = []
                source_status[name] = {"ok": False, "error": str(e), "elapsed": elapsed}
                yield {"type": "stage", "stage": name, "status": "error", "message": str(e)[:300], "elapsed": elapsed}

        def drain():
            while not messages.empty():
                stage, msg = messages.get()
                yield {"type": "log", "stage": stage, "message": msg}

        pending, others_done = set(jobs), None
        early = False
        while pending or not messages.empty():
            try:
                stage, msg = messages.get(timeout=0.1)
                yield {"type": "log", "stage": stage, "message": msg}
            except queue.Empty:
                pass
            for name in [n for n in pending if jobs[n].done()]:
                pending.discard(name)
                yield from collect(name)
            # only the real-world data analysis left: answer from the other sources first
            if progressive and pending == {"rwd"} and len(jobs) > 1:
                others_done = others_done or time.time()
                if time.time() - others_done >= RWD_GRACE_SECONDS:
                    early = True
                    break

        literature = results.get("literature", []) + results.get("pdfs", [])
        if mode_cfg.completeness_round and "literature" in sources and "pubmed" in parts:
            yield {"type": "stage", "stage": "coverage", "status": "running", "message": "Checking facet coverage"}
            t = time.time()
            extra = []
            try:
                extra = _snow_coverage(plan, literature, mode_cfg, query,
                                       lambda msg: messages.put(("coverage", msg)))
            except Exception as e:
                logger.warning("Coverage round failed: %s", e)
            yield from drain()
            literature += extra
            timings["coverage"] = round(time.time() - t, 2)
            yield {"type": "stage", "stage": "coverage", "status": "done",
                   "message": f"{len(extra)} articles added for uncovered facets", "elapsed": timings["coverage"]}

        def write(with_rwd, phase):
            out = yield from _write(question, plan, query, llm, mode_cfg, sources if with_rwd else
                                    [x for x in sources if x != "rwd"], literature, results if with_rwd else
                                    {k: v for k, v in results.items() if k != "rwd"}, source_status, rwd_studies if with_rwd
                                    else [], history, verify, timings, t_start, calls_before, phase)
            if out is not None:
                out.literature = list(parts) if "literature" in sources else []
            return out

        if not early:
            result = yield from write(True, 0)
        else:
            # ---- progressive: the answer without MIMIC-IV now, the full answer when the analysis ends
            source_status["rwd"] = {"ok": None, "pending": True}
            partial = yield from write(False, 1)
            if partial is None:
                return
            partial.timings["first_answer"] = partial.timings.get("total")
            yield {"type": "result", "result": partial, "final": False, "pending": ["rwd"]}
            while not jobs["rwd"].done():
                try:
                    stage, msg = messages.get(timeout=0.2)
                    yield {"type": "log", "stage": stage, "message": msg}
                except queue.Empty:
                    pass
            yield from drain()
            yield from collect("rwd")
            useful = [it for it in results.get("rwd") or [] if it.meta.get("primary")]
            if not useful:      # no estimate (failed, not answerable, too few patients): the first answer stands
                partial.source_status["rwd"] = source_status.get("rwd", {})
                partial.rwd_studies = rwd_studies
                partial.sources = list(sources)
                result = partial
            else:
                yield {"type": "stage", "stage": "synthesis", "status": "running", "phase": 2,
                       "message": "Adding the real-world data results to the answer"}
                result = yield from write(True, 2)
                if result is None:
                    result = partial
                result.timings["first_answer"] = partial.timings.get("total")
        if result is None:
            return
        finished = True
    finally:
        if not finished:
            cancel.set()          # the consumer stopped (e.g. a new question): stop a running analysis
        pool.shutdown(wait=False, cancel_futures=True)

    if result.answer_markdown and result.evidence and rwd_spec is None \
            and all(st.get("ok") for st in source_status.values()):
        try:
            path = _cache_path(_cache_key(question, sources, mode_cfg.key, llm, history, uploads, verify,
                                          rwd_engine, parts))
            path.parent.mkdir(parents=True, exist_ok=True)
            save_result(result, path)
        except Exception as e:
            logger.warning("Could not cache result: %s", e)
    yield {"type": "stage", "stage": "done", "status": "done",
           "message": f"Done in {result.timings['total']:.0f}s with {result.n_llm_calls} LLM calls"}
    yield {"type": "result", "result": result, "final": True}


# Seconds to wait for a still-running real-world data analysis before answering from the other sources first
RWD_GRACE_SECONDS = 3.0


def _write(question, plan, query, llm, mode_cfg, sources, literature, results, source_status, rwd_studies, history,
           verify, timings, t_start, calls_before, phase):
    """Evidence IDs, extraction, streamed synthesis and checks; yields events and returns the MESResult (None when
    the LLM failed)."""
    timings = dict(timings)
    evidence = _assign_ids({"literature": literature, "trial": results.get("trials", []),
                            "kg": results.get("kg", []), "rwd": results.get("rwd", [])})
    for item in evidence:   # the study record knows which citation (R1, R2, ...) it became
        for study in rwd_studies:
            if item.source_type == "rwd" and item.meta.get("study_key") == study.get("key"):
                study["evidence_id"] = item.id
    by_id = {it.id: it for it in evidence}

    # ---- 3. Sackett / Snow: structured extraction ------------------------------------
    study_table = []
    if mode_cfg.extract_studies and any(it.source_type in ("literature", "trial") for it in evidence):
        yield {"type": "stage", "stage": "extract", "status": "running",
               "message": "Extracting study designs and effect estimates"}
        t = time.time()
        try:
            study_table = extract_study_table(query, evidence, llm)
            status, message = "done", f"{len(study_table)} studies tabulated"
        except Exception as e:
            logger.warning("Extraction failed: %s", e)
            status, message = "error", str(e)[:300]
        timings["extract"] = round(time.time() - t, 2)
        yield {"type": "stage", "stage": "extract", "status": status, "message": message, "elapsed": timings["extract"]}

    # ---- 4. streamed synthesis -------------------------------------------------------
    yield {"type": "evidence", "phase": phase,
           "items": [{"id": it.id, "type": it.source_type, "title": it.title, "design": it.design,
                      "year": it.year, "venue": it.venue} for it in evidence]}
    yield {"type": "stage", "stage": "synthesis", "status": "running", "phase": phase,
           "message": f"Synthesizing {len(evidence)} evidence items"}
    t = time.time()
    first_token = None
    raw = []
    budget = int(config.CONTEXT_BUDGET * mode_cfg.budget_scale)
    gate = PreambleGate()
    if evidence:
        stream = stream_synthesis(plan, evidence, llm, mode_cfg, sources, budget, history)
    else:  # nothing retrieved: say so instead of letting the model answer from memory
        failed = [SOURCE_LABELS.get(n, n) for n, s in source_status.items() if s.get("ok") is False]
        note = f" Sources that could not be searched: {', '.join(failed)}." if failed else ""
        stream = iter([("text", NO_EVIDENCE_ANSWER.format(note=note))])
    try:
        for kind, text in stream:
            if kind == "reasoning":
                yield {"type": "reasoning", "text": text, "phase": phase}
                continue
            visible = gate.feed(text)
            if visible:
                # the prefilled heading is not a model token: time the first generated text
                if first_token is None and not (text == ANSWER_PREFILL and not raw):
                    first_token = round(time.time() - t_start, 2)
                raw.append(text)
                yield {"type": "token", "text": visible, "phase": phase}
            if gate.repeated:   # stop generating; closing the stream cancels the request
                break
        raw = [gate.released or gate.buffer]
    except Exception as e:
        logger.warning("Synthesis failed: %s: %s", type(e).__name__, e)
        yield {"type": "stage", "stage": "synthesis", "status": "error", "message": f"LLM call failed: {e}"}
        yield {"type": "error", "message": f"The LLM backend ({llm.label}) failed: {e}"}
        return None
    finally:
        if hasattr(stream, "close"):
            stream.close()
    key = "synthesis" if phase < 2 else "synthesis_rwd"
    timings[key] = round(time.time() - t, 2)
    if phase < 2:
        timings["first_token"] = first_token or round(time.time() - t_start, 2)
    yield {"type": "stage", "stage": "synthesis", "status": "done", "message": "Answer written", "phase": phase,
           "elapsed": timings[key]}

    # ---- 5. checks (deterministic): citations, number grounding, citation repair -------------
    markdown, display, order, sections, takeaways, grounding = postprocess_answer("".join(raw), by_id)
    if not study_table and sections.get("Evidence table"):
        study_table = parse_evidence_table(sections["Evidence table"])
    if verify:
        yield {"type": "stage", "stage": "verify", "status": "running", "message": "Verifying claims against sources"}
        t = time.time()
        try:
            grounding = verify_claims(grounding, by_id, llm)
            status, message = "done", f"{grounding.get('llm_verified', 0)} claims checked"
        except Exception as e:
            logger.warning("Claim verification failed: %s", e)
            status, message = "error", str(e)[:300]
        timings["verify"] = round(time.time() - t, 2)
        yield {"type": "stage", "stage": "verify", "status": status, "message": message, "elapsed": timings["verify"]}

    timings["total"] = round(time.time() - t_start, 2)
    return MESResult(
        question=question, mode=mode_cfg.key, sources=list(sources), plan=asdict(plan), evidence=evidence,
        answer_markdown=markdown, answer_display=display, sections=sections, takeaways=takeaways,
        citation_order=order, verification=grounding, study_table=study_table,
        source_status={k: dict(v) for k, v in source_status.items()}, timings=timings, llm=llm.label,
        n_llm_calls=llm.n_calls - calls_before, history=history, rwd_studies=list(rwd_studies))


_BACKEND_ERRORS = ("APIConnectionError", "ConnectError", "Connection error", "Connection refused", "APITimeoutError",
                   "EndpointConnectionError", "NoCredentialsError", "ExpiredToken", "UnrecognizedClient",
                   "AccessDenied", "ProfileNotFound", "AuthenticationError", "NotFoundError",
                   "AWS credentials for Bedrock")          # llm_zoo.BedrockClaudeModel: how to log in again


def is_backend_error(message):
    """True for errors meaning the LLM backend cannot be used (down, not loaded, no credentials)."""
    return any(key in message for key in _BACKEND_ERRORS)


def answer_question(question, llm, sources=DEFAULT_SOURCES, mode="osler", uploads=None, history=None,
                    use_cache=True, verify=None, on_event=None, rwd_spec=None, rwd_engine="auto", literature=None):
    """Run ``iter_answer`` to completion and return the ``MESResult`` (raises on LLM failure)."""
    result = None
    for event in iter_answer(question, llm, sources, mode, uploads, history, use_cache, verify, rwd_spec,
                             rwd_engine, literature=literature):
        if on_event:
            on_event(event)
        if event["type"] == "result":
            result = event["result"]
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
    return result


def conversation_turn(result):
    """Memory of one answered turn for follow-up questions (with the design of its MIMIC-IV study, which a
    follow-up such as "and in women?" refines)."""
    turn = {"question": result.plan.get("standalone_question") or result.question,
            "summary": answer_summary(result.sections)}
    study = next((s for s in result.rwd_studies or [] if s.get("status") == "done" and s.get("kind") != "analyst"
                  and (s.get("spec") or {}).get("analysis") not in (None, "catalog", "explore")), None)
    if study:
        try:
            from mes.tools.rwd.designer import describe
            from mes.tools.rwd.spec import StudySpec
            turn["rwd_design"] = describe(StudySpec.from_dict(study["spec"]))
        except Exception:
            pass
    return turn


def prune_answer_cache():
    """Delete cached answers older than twice their lifetime (they are never used again)."""
    cutoff = time.time() - 2 * config.HTTP_CACHE_TTL_HOURS * 3600
    for path in (config.CACHE_DIR / "results").glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def warm_up():
    """Load the encoder and the iBKH store (and start PDF indexing) so the first question is fast."""
    t = time.time()
    prune_answer_cache()
    encode(["warm up"], is_query=True)
    get_kg()
    literature_tools.get_pdf_library().refresh()
    if rwd_tools.available():
        from mes.tools.rwd.concepts import get_index
        get_index().warm()
    return round(time.time() - t, 2)
