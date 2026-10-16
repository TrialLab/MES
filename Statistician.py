"""
Statistician: normalization, appraisal and synthesis of evidence from all agents.

* Normalization converts literature, RWD and KG outputs into a standard evidence
  record (finding, source, relevance) and extracts effect estimates (measure,
  estimate, 95% CI, p-value, sample size) when they are reported.
* Quantitative evidence is grouped by effect measure and by evidence class
  (randomized literature, other literature, each newly generated RWE analysis). Within a
  group, two or more directly relevant estimates judged compatible (population,
  intervention/comparator, outcome, estimand) are pooled with a DerSimonian-Laird
  random-effects model with I^2 and Cochran's Q; otherwise estimates are reported individually.
  Estimates from different evidence classes are not pooled; they are compared for
  cross-source divergence instead.
* Qualitative findings (including KG relations) are synthesized thematically,
  preserving disagreement and uncertainty.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats

from llm_zoo import llm_map
from utils import parse_json_response, to_float

logger = logging.getLogger(__name__)

RATIO_MEASURES = {"HR", "OR", "RR"}


def _normalize_measure(measure):
    m = str(measure or "").strip().upper()
    aliases = {"HAZARD RATIO": "HR", "ODDS RATIO": "OR", "RISK RATIO": "RR", "RELATIVE RISK": "RR",
               "MEAN DIFFERENCE": "MD", "RISK DIFFERENCE": "RD", "ABSOLUTE RISK DIFFERENCE": "RD"}
    for alias, short in aliases.items():
        if alias in m:
            return short
    for short in ("HR", "OR", "RR", "MD", "RD", "SMD"):
        if m.startswith(short):
            return short
    return m or "UNKNOWN"


def evidence_class(study: Dict) -> str:
    """Evidence class used to keep different study designs apart in quantitative synthesis."""
    source = str(study.get("source_type", "")).upper()
    if source == "RWD":
        # Each real-world analysis is kept separate: analyses of overlapping cohorts are not independent.
        return f"Generated real-world evidence ({study.get('study_id')})"
    if source == "KG":
        return "Knowledge graph"
    design = str((study.get("quality_assessment") or {}).get("study_design", "")).lower()
    if "randomized controlled" in design or "meta-analysis" in design:
        return "Literature: randomized trials and systematic reviews"
    return "Literature: other designs"


# ============================================================================ quantitative synthesis
def _prepare_meta_analysis_data(evidence_list: List[Dict]) -> List[Dict]:
    """Validate estimates and derive standard errors from 95% confidence intervals."""
    prepared = []
    for study in evidence_list:
        effect, lower, upper = (to_float(study.get(k)) for k in ("effect_size", "ci_lower", "ci_upper"))
        if None in (effect, lower, upper) or lower >= upper:
            continue
        study = dict(study)
        if _normalize_measure(study.get("effect_measure")) in RATIO_MEASURES:
            if min(effect, lower, upper) <= 0:
                continue
            study["log_effect"] = np.log(effect)
            study["se"] = (np.log(upper) - np.log(lower)) / 3.92
        else:
            study["effect"] = effect
            study["se"] = (upper - lower) / 3.92
        if study["se"] > 0:
            prepared.append(study)
    return prepared


def run_meta_analysis(evidence_list: List[Dict]) -> Optional[Dict]:
    """DerSimonian-Laird random-effects meta-analysis with I^2 and Cochran's Q."""
    data = _prepare_meta_analysis_data(evidence_list)
    if len(data) < 2:
        return None
    is_log = "log_effect" in data[0]
    effects = np.array([s["log_effect" if is_log else "effect"] for s in data])
    ses = np.array([s["se"] for s in data])
    w = 1 / ses ** 2
    q_stat = float(np.sum(w * (effects - np.average(effects, weights=w)) ** 2))
    df = len(data) - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q_stat - df) / c) if c > 0 else 0.0
    w_re = 1 / (ses ** 2 + tau2)
    pooled = np.sum(w_re * effects) / np.sum(w_re)
    pooled_se = np.sqrt(1 / np.sum(w_re))
    lo, hi = pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se
    tr = np.exp if is_log else (lambda x: x)
    return {
        "n_studies": len(data), "study_ids": [s.get("study_id") for s in data],
        "pooled_effect": float(tr(pooled)), "ci_lower": float(tr(lo)), "ci_upper": float(tr(hi)),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(pooled / pooled_se)))), "is_log_scale": is_log,
        "heterogeneity": {"i_squared": max(0.0, (q_stat - df) / q_stat) * 100 if q_stat > 0 else 0.0,
                          "q_statistic": q_stat, "q_p_value": float(1 - stats.chi2.cdf(q_stat, df)), "tau_squared": tau2},
    }


def assess_poolability(studies: List[Dict], query: str, llm_client: Any) -> List[Dict]:
    """Ask whether studies are compatible enough (population, intervention/comparator, outcome, estimand) to be pooled."""
    summary = [{k: s.get(k) for k in ("study_id", "finding", "effect_measure", "effect_size", "ci_lower", "ci_upper",
                                      "result_description")} for s in studies]
    prompt = f"""You are a biostatistician deciding whether studies can be combined in a meta-analysis for the question:
"{query}"
Pooling requires sufficient compatibility in study design, target population, intervention and comparator definitions,
outcome measurement and effect estimand. Studies:
{json.dumps(summary, indent=2, default=str)}
Return ONLY a JSON object: {{"poolable_study_ids": [<ids of the largest set of mutually compatible studies>], "reasoning": "..."}}.
Return an empty list if no two studies are compatible."""
    result = parse_json_response(llm_client.invoke(prompt), default={})
    ids = set(result.get("poolable_study_ids") or []) if isinstance(result, dict) else set()
    return [s for s in studies if s.get("study_id") in ids]


def analyze_heterogeneity(meta_results: Dict, studies: List[Dict], llm_client: Any) -> Dict:
    """Interpret heterogeneity statistics in light of the populations and designs of the pooled studies."""
    context = [{k: s.get(k) for k in ("study_id", "finding", "effect_measure", "effect_size", "ci_lower", "ci_upper")}
               for s in studies]
    prompt = f"""Interpret these heterogeneity statistics from a random-effects meta-analysis: {json.dumps(meta_results['heterogeneity'])}.
The pooled studies are: {json.dumps(context, default=str)}.
I^2 describes inconsistency but not its cause; discuss plausible clinical and methodological explanations
(population, intervention definitions, outcome ascertainment, design). Return a JSON object with keys "interpretation" and "confidence"."""
    return parse_json_response(llm_client.invoke(prompt), default={"interpretation": None})


def generate_visualization_data(evidence_list: List[Dict], meta_results: Optional[Dict]) -> Dict:
    """Structured data for forest and funnel plots."""
    data = _prepare_meta_analysis_data(evidence_list)
    forest = [{"type": "study", "label": s.get("study_id"), "effect": to_float(s["effect_size"]),
               "ci_lower": to_float(s["ci_lower"]), "ci_upper": to_float(s["ci_upper"])} for s in data]
    if meta_results:
        forest.append({"type": "summary", "label": "Pooled Effect",
                       **{k: meta_results[k] for k in ("pooled_effect", "ci_lower", "ci_upper")}})
    funnel = [{"effect_size": s.get("log_effect", s.get("effect")), "standard_error": s["se"]} for s in data]
    return {"forest_plot": forest, "funnel_plot": funnel}


def _comparison_units(group_reports: Dict[str, Dict], max_per_group: int = 5):
    """(label, group, estimate, lower, upper, details) for pooled estimates, single studies and unpooled studies."""
    units = []
    for group, report in group_reports.items():
        if report.get("status") == "pooled":
            m = report["meta_analysis"]
            units.append((f"{group} (pooled, {m['n_studies']} studies)", group, m["pooled_effect"], m["ci_lower"],
                          m["ci_upper"], m))
        elif report.get("status") == "single_study_summary":
            st = report["study_summary"]
            units.append((f"{group}: {st.get('study_id')}", group, to_float(st.get("effect_size")),
                          to_float(st.get("ci_lower")), to_float(st.get("ci_upper")), st))
        elif report.get("status") == "not_pooled":
            for st in report["studies"][:max_per_group]:
                units.append((f"{group}: {st.get('study_id')}", group, to_float(st.get("effect_size")),
                              to_float(st.get("ci_lower")), to_float(st.get("ci_upper")), st))
    return [u for u in units if None not in u[2:5]]


def assess_cross_source_divergence(measure: str, group_reports: Dict[str, Dict], query: str, llm,
                                   qualitative_context: Optional[List[Dict]] = None, max_appraisals: int = 5) -> List[Dict]:
    """
    Compare estimates of the same effect measure across evidence classes. Divergence is tested with a
    ratio-of-estimates z-test (difference on the log scale for ratio measures). Discordant sources are
    reported separately, with an LLM appraisal of plausible explanations; they are never pooled.
    """
    is_ratio = measure in RATIO_MEASURES
    tr = np.log if is_ratio else (lambda x: x)
    null = 1.0 if is_ratio else 0.0
    units = [u for u in _comparison_units(group_reports) if not is_ratio or min(u[2:5]) > 0]
    comparisons = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            (n1, g1, e1, l1, u1, d1), (n2, g2, e2, l2, u2, d2) = units[i], units[j]
            if g1 == g2 or (g1.startswith("Generated") and g2.startswith("Generated")):
                continue  # same evidence class, or real-world analyses of overlapping cohorts
            se1, se2 = (tr(u1) - tr(l1)) / 3.92, (tr(u2) - tr(l2)) / 3.92
            z = (tr(e1) - tr(e2)) / np.sqrt(se1 ** 2 + se2 ** 2)
            p = float(2 * (1 - stats.norm.cdf(abs(z))))
            opposite = (e1 - null) * (e2 - null) < 0 and (l1 > null or u1 < null) and (l2 > null or u2 < null)
            comparison = {"effect_measure": measure, "source_1": n1, "estimate_1": [e1, l1, u1], "source_2": n2,
                          "estimate_2": [e2, l2, u2], "z": float(z), "p_value": p, "divergent": bool(p < 0.05 or opposite)}
            if comparison["divergent"] and sum("appraisal" in c for c in comparisons) < max_appraisals:
                prompt = f"""
You are an expert epidemiologist. Two evidence sources addressing the clinical question below report different
estimates of the same effect measure ({measure}).

Clinical question: {query}
Source 1 ({n1}): estimate {e1:.3g} (95% CI {l1:.3g}-{u1:.3g}). Details: {json.dumps(d1, default=str)[:3000]}
Source 2 ({n2}): estimate {e2:.3g} (95% CI {l2:.3g}-{u2:.3g}). Details: {json.dumps(d2, default=str)[:3000]}
Test for difference: z = {z:.2f}, p = {p:.3g}.
Additional qualitative context: {json.dumps([q.get('finding') for q in (qualitative_context or [])][:15], default=str)}

Treat this as cross-source disagreement, not something to be resolved statistically. Compare study designs,
populations, eligibility, intervention/comparator definitions, outcome ascertainment, follow-up and estimands;
consider confounding and other biases for observational evidence. Do not rank the sources by statistical significance
and do not force a consensus.

Return a JSON object with keys: "summary", "plausible_explanations" (list), "remaining_uncertainty".
"""
                comparison["appraisal"] = parse_json_response(llm.invoke(prompt), default=None)
            comparisons.append(comparison)
    return comparisons


# ============================================================================ qualitative synthesis
def synthesize_qualitative_findings(qualitative_evidence: List[Dict], llm_client: Any) -> Dict:
    """Thematic synthesis of textual findings (literature, RWD and KG) that preserves nuance and disagreement."""
    if not qualitative_evidence:
        return {"status": "skipped", "reason": "No qualitative findings provided."}
    findings = [{"id": item.get("study_id"), "source": item.get("source_type"), "evidence_class": evidence_class(item),
                 "finding": item.get("finding"), "relevance_to_query": item.get("relevance_to_query")}
                for item in qualitative_evidence if item.get("finding")]
    prompt = f"""
You are an expert in qualitative evidence synthesis.

Your task is to synthesize multiple qualitative evidence findings from heterogeneous sources, such as randomized trials, observational real-world evidence, knowledge graphs, and other evidence sources.

Core principle:
Do not rely only on grouping.
Grouping is helpful, but the synthesis must preserve the important meaning, nuance, uncertainty, and disagreements across findings.
The most important output is a faithful, information-dense overall synthesis written in natural language.

What you should do:
1. Read all findings carefully, keeping track of both evidence ID and source type.
2. Group findings into a small number of meaningful evidence themes.
3. For each theme: provide a short label, summarize what the grouped findings collectively suggest, cite the supporting evidence IDs, and mention source types when they matter for interpretation.
4. Explicitly identify findings that support benefit, findings that raise concerns, harms or counterevidence, findings that provide mechanistic rationale, and findings that are only indirect or contextual.
5. Write an overall synthesis that integrates the balance of evidence, distinguishes direct evidence from indirect/mechanistic support, notes conflicts, inconsistency or uncertainty, and does not overstate conclusions.
6. If findings are sparse, weak, or partially relevant, say so clearly.

Important rules:
- Use only the provided findings. Do not use outside knowledge.
- Do not invent study IDs, source types, mechanisms, conflicts, or conclusions.
- Do not treat all sources as equally strong by default; if source type matters, reflect that cautiously.
- Do not force all findings into agreement and do not discard minority or conflicting findings.

Return exactly one JSON object with the following keys:
{{
  "synthesis_summary": string,
  "themed_groups": [{{"theme": string, "summary": string, "evidence_ids": [string], "source_types": [string]}}],
  "conflict_analysis": string
}}

- synthesis_summary: 1-2 paragraphs; a faithful overall synthesis discussing the balance of support, concerns, mechanistic rationale, indirect evidence, and uncertainty.
- conflict_analysis: whether the findings are broadly consistent, partially conflicting, or strongly conflicting, and the main tensions. If there is little or no conflict, say that explicitly.

Qualitative Findings:
{json.dumps(findings, ensure_ascii=False, indent=2, default=str)}

Return only valid JSON. Do not wrap in markdown fences.
"""
    response = llm_client.invoke(prompt)
    parsed = parse_json_response(response)
    return parsed if isinstance(parsed, dict) else {"error": "Failed to parse qualitative synthesis output.",
                                                    "raw_response": response}


def run_statistical_synthesis_pipeline(evidence_list: List[Dict], llm_client: Any, query: str = "") -> Dict:
    """Quantitative synthesis by effect measure and evidence class, cross-source comparison, qualitative synthesis."""
    quantitative = [s for s in evidence_list if s.get("effect_size") is not None and s.get("ci_lower") is not None
                    and s.get("ci_upper") is not None]
    qualitative = [s for s in evidence_list if s.get("finding")]
    logger.info("Statistician: %d quantitative and %d qualitative evidence items.", len(quantitative), len(qualitative))

    by_measure = {}
    for study in quantitative:
        by_measure.setdefault(_normalize_measure(study.get("effect_measure")), {}).setdefault(
            evidence_class(study), []).append(study)

    quantitative_reports, cross_source = {}, []
    for measure, groups in by_measure.items():
        group_reports = {}
        for group, studies in groups.items():
            valid = _prepare_meta_analysis_data(studies)
            if not valid:
                group_reports[group] = {"status": "skipped", "reason": "No study with a usable estimate and 95% CI."}
            elif len(valid) == 1:
                s = valid[0]
                group_reports[group] = {
                    "status": "single_study_summary",
                    "reason": "Only one valid study was found, so no meta-analysis was performed.",
                    "study_summary": {k: s.get(k) for k in ("study_id", "source_type", "effect_measure", "effect_size",
                                                            "ci_lower", "ci_upper", "p_value", "finding")},
                    "visualization_data": generate_visualization_data(valid, None)}
            else:
                poolable = assess_poolability([v for v in valid if v.get("relevance_to_query") != "INDIRECTLY_RELEVANT"],
                                              query, llm_client)
                individual = [{k: s.get(k) for k in ("study_id", "effect_size", "ci_lower", "ci_upper", "p_value")}
                              for s in valid if s not in poolable]
                if len(poolable) < 2:
                    group_reports[group] = {"status": "not_pooled", "reason": "Estimates were not sufficiently "
                                            "compatible for pooling; they are reported individually.", "studies": individual}
                    continue
                meta = run_meta_analysis(poolable)
                group_reports[group] = {"status": "pooled", "meta_analysis": meta,
                                        "heterogeneity_interpretation": analyze_heterogeneity(meta, poolable, llm_client),
                                        "visualization_data": generate_visualization_data(poolable, meta),
                                        "not_pooled_studies": individual}
        quantitative_reports[measure] = group_reports
        if len(group_reports) > 1:
            cross_source.extend(assess_cross_source_divergence(measure, group_reports, query, llm_client, qualitative))

    return {"quantitative_synthesis": quantitative_reports, "cross_source_comparison": cross_source,
            "qualitative_synthesis": synthesize_qualitative_findings(qualitative, llm_client)}


# ============================================================================ normalization
def classify_entry_relevance(entry: Dict, user_query: str, llm_client: Any) -> str:
    """Classify an evidence entry as DIRECTLY_RELEVANT, INDIRECTLY_RELEVANT or NOT_RELEVANT to the query."""
    compact = {k: entry.get(k) for k in ("study_id", "source_type", "finding", "population", "intervention",
                                         "comparator", "outcome", "metrics", "effect_size", "ci_lower", "ci_upper",
                                         "quality_assessment", "limitations", "source_text")}
    compact = {k: v for k, v in compact.items() if v not in (None, "")}
    prompt = f"""
You are evaluating whether a piece of evidence is relevant to a clinical or biomedical user query.

Your task is to classify the evidence into exactly one of the following three labels:

1. DIRECTLY_RELEVANT
Use this label when the evidence directly helps answer the user query (same or very similar condition,
intervention/exposure, comparator, outcome or target population).

2. INDIRECTLY_RELEVANT
Use this label when the evidence does not directly answer the query, but still provides useful support, context,
or partial evidence (related population, related intervention class, related outcome, mechanism, safety signal,
or background that helps interpret or bound the answer).

3. NOT_RELEVANT
Use this label when the evidence has no meaningful value for answering the query, i.e., it is unrelated across the
core dimensions such as condition, population, intervention/exposure, comparator, outcome, mechanism, or decision context.

Output only one label: DIRECTLY_RELEVANT, INDIRECTLY_RELEVANT, or NOT_RELEVANT.

User Query:
{user_query}

Evidence Entry:
{json.dumps(compact, ensure_ascii=False, indent=2, default=str)}

Return only one label.
""".strip()
    try:
        response = str(llm_client.invoke(prompt)).strip().upper()
    except Exception as e:
        logger.warning("Relevance assessment failed: %s", e)
        return "INDIRECTLY_RELEVANT"
    for label in ("DIRECTLY_RELEVANT", "INDIRECTLY_RELEVANT", "NOT_RELEVANT"):
        if label in response:
            return label
    return "INDIRECTLY_RELEVANT"


def extract_structured_info_from_document(document_text: str, user_query: str, llm_client: Any) -> Optional[Dict]:
    """Descriptive summary plus the main reported effect estimate (measure, estimate, CI, p-value, sample size)."""
    prompt = f"""
You are an expert clinical evidence analyst.

Your job is to read the provided document text and produce a light-structured evidence summary for downstream evidence synthesis.

Use only the provided document text. Do not use outside knowledge.
Do not invent values, results, p-values, confidence intervals, subgroup findings, or conclusions that are not supported by the text.

Return exactly one JSON object with the following fields:

{{
  "descriptive_summary": string,
  "main_quantitative_result": {{
    "effect_measure": string or null,
    "effect_size": number or null,
    "ci_lower": number or null,
    "ci_upper": number or null,
    "p_value": number or string or null,
    "sample_size": number or null,
    "description": string or null
  }},
  "subgroup_results": [
    {{"subgroup": string or null, "effect_measure": string or null, "effect_size": number or null,
      "ci_lower": number or null, "ci_upper": number or null, "p_value": number or string or null}}
  ]
}}

Field instructions:
1. descriptive_summary: 3-6 sentences that faithfully summarize the main evidence, preserving population,
   intervention/exposure, comparator, outcome, effect direction, important uncertainty and notable qualifications.
2. main_quantitative_result: the single most important numerical result for the user query, with effect_measure
   written as HR, OR, RR, RD, MD or another standard abbreviation. If no clear numerical result with a 95% confidence
   interval is reported, return an empty object {{}}. Do not guess or reconstruct missing numbers.
3. subgroup_results: only explicitly reported subgroup findings; [] if none.

Return only valid JSON. Do not wrap the JSON in markdown fences.

User Query:
{user_query}

Document Text:
---
{document_text}
---
""".strip()
    parsed = parse_json_response(llm_client.invoke(prompt))
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("descriptive_summary", None)
    if not isinstance(parsed.get("main_quantitative_result"), dict):
        parsed["main_quantitative_result"] = {}
    if not isinstance(parsed.get("subgroup_results"), list):
        parsed["subgroup_results"] = []
    return parsed


def _lift_quantitative(entry: Dict, result: Dict) -> Dict:
    """Copy a parsed main result to the top-level fields used by the quantitative synthesis."""
    effect, lower, upper = (to_float(result.get(k)) for k in ("effect_size", "ci_lower", "ci_upper"))
    if None not in (effect, lower, upper) and result.get("effect_measure"):
        entry.update({"effect_measure": _normalize_measure(result["effect_measure"]), "effect_size": effect,
                      "ci_lower": lower, "ci_upper": upper, "p_value": result.get("p_value"),
                      "sample_size": result.get("sample_size"), "result_description": result.get("description")})
    return entry


def synthesize_finding_from_structured(evidence_dict: Dict, user_query: str, llm_client: Any) -> str:
    """Summarize semi-structured literature evidence (PICO, metrics, quality, limitations) into one faithful finding."""
    structured_view = {k: evidence_dict.get(k) for k in ("population", "intervention", "comparator", "outcome", "metrics",
                                                         "quality_assessment", "limitations")}
    structured_view = {k: v for k, v in structured_view.items() if v not in (None, "", [], {})}
    prompt = f"""
You are an expert in clinical evidence synthesis.

Your task is to produce ONE concise, faithful evidence finding from the provided semi-structured evidence.

Important rules:
1. Use only the provided evidence. Do not use outside knowledge.
2. Do not invent missing facts, effect directions, numbers, statistical significance, or conclusions.
3. Do not directly answer the user query unless the evidence itself clearly supports that answer; the query is only provided to prioritize details.
4. If the evidence is only partially informative, summarize the informative part.
5. Avoid overstating causality unless clearly supported by the evidence.

The synthesized finding should be 3-6 sentences, evidence-focused, neutral and concrete, and preserve (when available)
population, intervention/exposure, comparator, outcome, main effect or direction, and important uncertainty or limitation.
Do not produce JSON or bullet points, and do not start with labels such as "Answer:" or "Finding:".

Only output exactly:
Insufficient information to synthesize a finding.
if the provided evidence contains essentially no meaningful clinical or analytical content to summarize.

USER QUERY:
{user_query}

STRUCTURED EVIDENCE:
{json.dumps(structured_view, ensure_ascii=False, indent=2, default=str)}

SOURCE TEXT:
{evidence_dict.get("source_text") or "N/A"}

Output only the synthesized finding text.
""".strip()
    response = str(llm_client.invoke(prompt) or "").strip()
    if not response or response.lower().startswith("insufficient information"):
        parts = [f"{k}: {evidence_dict[k]}" for k in ("population", "intervention", "comparator", "outcome", "metrics")
                 if evidence_dict.get(k)]
        return ("The evidence provides partial structured information (" + "; ".join(parts) + ") but not enough detail "
                "for a more specific finding.") if parts else "The evidence does not contain enough information for a finding."
    return response


def normalize_evidence(raw_evidence: Dict, user_query: str, llm_client: Any, min_doc_length: int = 500) -> Optional[Dict]:
    """
    Normalize one raw evidence entry into a standard record with a textual 'finding', its source, a relevance label
    and, when reported, a standardized effect estimate.
    """
    entry_id = str(raw_evidence.get("study_id", raw_evidence.get("id", "N/A")))
    source_type = raw_evidence.get("source_type", raw_evidence.get("source", "unknown_source"))
    finding = raw_evidence.get("finding")

    if all(raw_evidence.get(k) is not None for k in ("effect_size", "ci_lower", "ci_upper")):
        entry = dict(raw_evidence)  # already quantitative (e.g., structured RWD estimate)
    elif isinstance(finding, str) and len(finding) >= min_doc_length:
        extracted = extract_structured_info_from_document(finding, user_query, llm_client)
        if not extracted:
            logger.warning("Extraction failed for '%s'; entry discarded.", entry_id)
            return None
        entry = {"study_id": entry_id, "source_type": source_type,
                 "finding": extracted.get("descriptive_summary") or finding[:2000],
                 "subgroup_results": extracted.get("subgroup_results"),
                 "quality_assessment": raw_evidence.get("quality_assessment"),
                 "limitations": raw_evidence.get("limitations")}
        _lift_quantitative(entry, extracted["main_quantitative_result"])
    elif isinstance(finding, str):
        entry = dict(raw_evidence)
        entry.setdefault("study_id", entry_id)
        entry["source_type"] = source_type
    elif sum(k in raw_evidence for k in ("population", "intervention", "comparator", "outcome", "metrics", "source_text")) >= 3:
        entry = {"study_id": entry_id, "source_type": source_type, "title": raw_evidence.get("title"),
                 "finding": synthesize_finding_from_structured(raw_evidence, user_query, llm_client),
                 "quality_assessment": raw_evidence.get("quality_assessment"),
                 "limitations": raw_evidence.get("limitations")}
        if any(ch.isdigit() for ch in str(raw_evidence.get("metrics", ""))):
            extracted = extract_structured_info_from_document(raw_evidence.get("source_text", ""), user_query, llm_client)
            if extracted:
                _lift_quantitative(entry, extracted["main_quantitative_result"])
    else:
        logger.warning("Unrecognized evidence format for '%s'; entry discarded.", entry_id)
        return None

    entry["relevance_to_query"] = classify_entry_relevance(entry, user_query, llm_client)
    return entry


def normalize_evidence_list(evidence_list: List[Dict], user_query: str, llm_client: Any, max_workers: int = 8) -> List:
    """Normalize evidence entries in parallel; returns one result per input (None for discarded entries)."""
    return llm_map(lambda e: normalize_evidence(e, user_query, llm_client), evidence_list, max_workers)
