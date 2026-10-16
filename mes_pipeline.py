"""
MES orchestration: one call of ``run_med_synth`` answers a clinical question with the six agents.

Round r (r = 1..max_rounds):
  Supervisor     decomposes the (follow-up) query and selects the specialist agents;
  LiteratureMiner, RWD-Analyst and KG-Specialist collect source-specific evidence;
  Statistician   normalizes and synthesizes all evidence collected so far;
  SafeChecker    screens the synthesis claim by claim;
  Supervisor     writes the round report, checks it (PRISMA-lite, completeness) and either
                 stops or launches a targeted follow-up query for selected agents.
Finally the Supervisor combines all round reports into the final evidence report.
"""

import json
import logging
import os

from KGSpecialist import assess_triplet_consistency, assess_triplet_quality
from LiteratureMiner import (add_numerical_summary, aggregate_evidence, annotate_limitations, assess_study_quality,
                             extract_evidence, generate_synthesis_summary, search_literature)
from RWDAnalyst import build_dataframe_with_df_meta, generate_target_trial_protocol, run_rwd_analyst
from SafeChecker import run_safechecker_pipeline
from Statistician import normalize_evidence_list, run_statistical_synthesis_pipeline
from Supervisor import (check_final_answer_relevance, check_report_format, clarify_and_decompose_query,
                        combine_final_reports, get_primary_agents, synthesize_report)
from rwd_extended_functions import (determine_rwd_task, run_clustering_pipeline, run_descriptive_analysis,
                                    run_patient_similarity_pipeline, run_supervised_prediction_pipeline)
from rwd_preprocessor import process_and_complete_query

logger = logging.getLogger("MES")


def run_rwd_analysis(query, llm, preprocessor, max_patient_num=10_000):
    """RWD-Analyst: complete the query, route it to an analysis pipeline, build the cohort and run the analysis."""
    result = {"query": query, "data_source": "MIMIC-IV", "report": None, "structured_result": None, "error": None}
    try:
        completion = process_and_complete_query(query, llm)
        refined = completion["completed_query"]
        task = determine_rwd_task(refined, llm)
        result.update({"completed_query": refined, "task_type": task["task_type"], "routing_reasoning": task.get("reasoning")})
        logger.info("RWD task: %s", task["task_type"])
        if task["task_type"] == "unknown":
            result["error"] = "The query does not correspond to a supported real-world data analysis."
            return result

        df, metadata = preprocessor.process_query(refined, max_patient_num=max_patient_num)
        result["cohort"] = {k: v for k, v in metadata.items() if k != "col_info"}
        result["data_source"] = metadata.get("data_source", "MIMIC-IV")
        if df is None or df.empty:
            result["error"] = "No patients met the eligibility criteria."
            return result

        covariates, outcome = metadata["cols_covariates"], metadata["col_outcome"]
        if task["task_type"] == "trial_emulation":
            missing = [name for name, col in (("outcome", outcome), ("treatment", metadata["col_treatment"]),
                                              ("follow-up time", metadata["col_time_to_event"])) if not col]
            if missing:
                result["error"] = f"Trial emulation requires {', '.join(missing)}, which could not be derived from the query."
                return result
            state = build_dataframe_with_df_meta(df, metadata)
            state["trial_info"] = generate_target_trial_protocol(refined, llm)
            state["initial_question"] = refined
            state = run_rwd_analyst(state, llm)
            result["report"], result["structured_result"] = state["current_report"], state["primary_result"]
            result["analysis_details"] = {k: state[k] for k in ("trial_info", "trial_protocol", "balancing_information",
                                                                "analysis_results", "subgroup_results", "sample_size", "e_value")}
        elif task["task_type"] == "supervised_prediction":
            if not outcome:
                result["error"] = "Prediction requires an outcome, which could not be derived from the query."
                return result
            out = run_supervised_prediction_pipeline(refined, df, covariates, outcome, llm)
            result["report"], result["structured_result"] = out.get("summary"), out.get("metrics")
            result["error"] = out.get("error")
        elif task["task_type"] == "descriptive_analysis":
            out = run_descriptive_analysis(refined, df, metadata["col_info"], metadata["col_patient_id"], llm)
            result["report"], result["error"] = out.get("summary"), out.get("error")
        elif task["task_type"] == "unsupervised_clustering":
            out = run_clustering_pipeline(refined, df, covariates, metadata["col_patient_id"], metadata["col_info"], llm)
            result["report"], result["error"] = out.get("summary"), out.get("error")
        elif task["task_type"] == "patient_retrieval":
            out = run_patient_similarity_pipeline(refined, df, metadata["col_info"], metadata["col_patient_id"], llm)
            result["report"], result["error"] = out.get("summary"), out.get("error")
    except Exception as e:
        logger.warning("RWD analysis failed: %s", e)
        result["error"] = str(e)
    return result


def run_literature_mining(query, llm, max_docs_per_source=10, max_workers=8):
    """LiteratureMiner: search, extract, appraise, annotate and aggregate published evidence."""
    search = search_literature(query, llm)
    extracted = extract_evidence(search, llm, max_docs_per_source=max_docs_per_source, max_workers=max_workers)
    with_numbers, with_numbers_no_text = add_numerical_summary(extracted, llm, max_workers=max_workers)
    enriched = annotate_limitations(assess_study_quality(extracted, llm, max_workers), llm, max_workers)
    table = aggregate_evidence(enriched)
    return {
        "search_summary": search["search_summary"], "search_strings": search["search_strings"],
        "n_clinical_trials_found": len(search["clinical_trials"]), "n_pubmed_articles_found": len(search["pubmed_articles"]),
        "evidence": enriched, "numerical_summaries": with_numbers_no_text, "summary_table": table,
        "narrative_synthesis": generate_synthesis_summary(table, query, llm),
    }


def run_kg_specialist(query, llm, kg, top_k=100, top_k_used=10, max_workers=8):
    """
    KG-Specialist: retrieve triplets and multi-hop paths and assess the reliability of the ``top_k_used``
    best-ranked items passed downstream (up to a third of them are reserved for multi-hop paths).
    """
    retrieved, info = kg.retrieve(query, llm, top_k=top_k)
    paths = [r for r in retrieved if len(r["edges"]) > 1][:top_k_used // 3]
    triplets = [r for r in retrieved if len(r["edges"]) == 1][:top_k_used - len(paths)]
    selected = sorted(triplets + paths, key=lambda r: r["similarity"], reverse=True)
    used = assess_triplet_consistency(assess_triplet_quality(selected, llm, max_workers), retrieved)
    return {"retrieval_info": info, "retrieved": retrieved, "evidence": used}


def _unique_documents(documents):
    """Literature documents of all rounds without duplicates (same DOI / NCT ID, or same title)."""
    seen, unique = set(), []
    for doc in documents:
        key = doc.get("id") if doc.get("id") not in (None, "N/A") else doc.get("title")
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _distinct_rwd(rwd_outputs):
    """RWD analyses with a usable report, without repeats of an identical estimate in a later round."""
    distinct = []
    for rwd in rwd_outputs:
        result = rwd.get("structured_result")
        if result is None or all(prev.get("structured_result") != result for prev in distinct):
            distinct.append(rwd)
    return distinct


def run_med_synth(user_query, llm, preprocessor, kg, max_rounds=2, max_clarification_rounds=0, use_safechecker=True,
                  max_docs_per_source=10, kg_top_k=100, kg_top_k_used=10, max_patient_num=10_000, max_workers=8):
    """Answer ``user_query`` with the MES multi-agent workflow and return the full state of the run."""
    state = {"initial_question": user_query, "queries": [user_query], "rounds": [], "final_report": None}
    literature_rounds, rwd_outputs, kg_output = [], [], None
    normalized_cache = {}
    target_agents = None

    for round_idx in range(max_rounds):
        query = state["queries"][round_idx]
        logger.info("===== Round %d: %s", round_idx + 1, query)
        structured_query = clarify_and_decompose_query(
            llm, query, max_clarification_rounds=max_clarification_rounds if round_idx == 0 else 0)
        agents = target_agents if target_agents else get_primary_agents(structured_query)
        round_state = {"query": query, "structured_query": structured_query, "agents": agents}

        if "RWD-Analyst" in agents:
            rwd_outputs.append(run_rwd_analysis(query, llm, preprocessor, max_patient_num))
        if "KG-Specialist" in agents:
            kg_output = run_kg_specialist(query, llm, kg, kg_top_k, kg_top_k_used, max_workers)
        if "LiteratureMiner" in agents:
            literature_rounds.append(run_literature_mining(query, llm, max_docs_per_source, max_workers))

        # Evidence collected so far (literature and RWD from all rounds, KG from the latest retrieval).
        evidence = []
        documents = _unique_documents([doc for lit in literature_rounds for doc in lit["evidence"]])
        for i, item in enumerate(documents):
            evidence.append({**item, "study_id": f"LIT-{item.get('id') if item.get('id') not in (None, 'N/A') else i + 1}"})
        for i, rwd in enumerate(_distinct_rwd(rwd_outputs)):
            if rwd.get("report"):
                item = {"source_type": "RWD", "study_id": f"RWD-{i + 1}", "finding": rwd["report"]}
                if rwd.get("task_type") == "trial_emulation" and rwd.get("structured_result"):
                    item.update({k: rwd["structured_result"].get(k) for k in
                                 ("effect_measure", "effect_size", "ci_lower", "ci_upper", "p_value")})
                evidence.append(item)
        if kg_output:
            for i, t in enumerate(kg_output["evidence"]):
                evidence.append({"source_type": "KG", "study_id": f"KG-{i + 1}", "source": t["source"],
                                 "finding": t["text"], "quality_assessment": t["quality_assessment"],
                                 "reliability": t["reliability"], "consistency": t["consistency"]})

        new_items = [e for e in evidence if json.dumps(e, sort_keys=True, default=str) not in normalized_cache]
        for raw, norm in zip(new_items, normalize_evidence_list(new_items, query, llm, max_workers)):
            normalized_cache[json.dumps(raw, sort_keys=True, default=str)] = norm
        normalized = [normalized_cache[json.dumps(e, sort_keys=True, default=str)] for e in evidence]
        normalized = [e for e in normalized if e and e.get("relevance_to_query") != "NOT_RELEVANT"]

        statistician_report = run_statistical_synthesis_pipeline(normalized, llm, user_query)
        screened = statistician_report
        if use_safechecker:
            kg_constraints = [t["text"] for t in (kg_output or {}).get("evidence", [])]
            screened = run_safechecker_pipeline(statistician_report, normalized, llm, kg_constraints, max_workers)

        report = synthesize_report(llm, structured_query, screened)
        format_check = check_report_format(llm, report)
        completeness = check_final_answer_relevance(llm, structured_query, report)
        round_state.update({"n_evidence": len(evidence), "n_normalized_evidence": len(normalized),
                            "normalized_evidence": normalized, "statistician_report": statistician_report,
                            "safechecker_report": screened.get("safechecker") if use_safechecker else None,
                            "report": report, "format_check": format_check, "completeness_check": completeness})
        state["rounds"].append(round_state)

        if completeness.get("is_complete") or not completeness.get("follow_up_query") or round_idx + 1 >= max_rounds:
            break
        logger.info("Report incomplete (%s). Follow-up: %s", completeness.get("reasoning"), completeness.get("follow_up_query"))
        state["queries"].append(completeness["follow_up_query"])
        target_agents = completeness.get("target_agents") or None

    data_source = next((r["data_source"] for r in rwd_outputs if r.get("data_source")), "MIMIC-IV")
    rwd_evidence = [{"query": r["query"], "task_type": r.get("task_type"), "data_source": r.get("data_source"),
                     "result": r["structured_result"], "report": r["report"] or f"No usable real-world analysis: {r['error']}"}
                    for r in _distinct_rwd(rwd_outputs)]
    numerical_summaries = _unique_documents([doc for lit in literature_rounds for doc in lit["numerical_summaries"]])
    state["final_report"], state["final_prompt"] = combine_final_reports(
        llm, state["rounds"][0]["structured_query"], [r["report"] for r in state["rounds"]], numerical_summaries,
        rwd_evidence, (kg_output or {}).get("evidence", []), data_source)
    state.update({"literature": literature_rounds, "rwd": rwd_outputs, "kg": kg_output})
    return state


def save_outputs(state, output_dir):
    """Write the final evidence report and all agent outputs (JSON) to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "final_report.md"), "w") as f:
        f.write(f"# Question\n\n{state['initial_question']}\n\n# MES evidence synthesis\n\n{state['final_report']}\n")
    serializable = {k: v for k, v in state.items() if k != "final_prompt"}
    serializable["literature"] = [{k: v for k, v in lit.items() if k != "summary_table"} for lit in state["literature"]]
    with open(os.path.join(output_dir, "agent_outputs.json"), "w") as f:
        json.dump(serializable, f, indent=2, default=str)
