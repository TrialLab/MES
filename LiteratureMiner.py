"""
LiteratureMiner: retrieval and structuring of published evidence.

Pipeline: search strategy (PubMed and/or ClinicalTrials.gov with Boolean keyword
queries) -> evidence extraction (PICO and reported metrics) -> study quality
assessment (design and provisional risk-of-bias label) -> limitation annotation
(verbatim quotes) -> evidence aggregation (summary table and narrative synthesis)
-> numerically focused summaries used by the Supervisor.
"""

import copy
import json
import logging
import os
import re
import tempfile

import pandas as pd

import utils
from llm_zoo import llm_map
from utils import parse_json_response

logger = logging.getLogger(__name__)


def _keyword_field(response, name, next_name=None):
    """Value of NAME in a response of the form 'DISEASE: ...; TREATMENT: ...; OUTCOME: ...'."""
    end = rf";?\s*{next_name}:" if next_name else "$"
    m = re.search(rf"{name}:\s*(.*?)\s*(?:{end})", response, re.S)
    return m.group(1) if m else ""


def _to_essie(keywords):
    """Alternative keywords of a field (separated by commas or semicolons) combined with OR in Essie syntax."""
    return " OR ".join(k.replace(".", "").strip() for k in re.split(r"[,;]", keywords) if k.strip())


def _decide_source(response):
    text = str(response).lower()
    if "both" in text or ("pubmed" in text and "clinicaltrials" in text):
        return "Both"
    if "clinicaltrials" in text:
        return "ClinicalTrials.gov"
    if "pubmed" in text:
        return "PubMed"
    return "Both"


def search_literature(query: str, llm, max_pubmed_results: int = 100) -> dict:
    """Search ClinicalTrials.gov and/or PubMed for literature relevant to a clinical question."""
    source_decision = _decide_source(llm.invoke(f"""
    Analyze the following question and decide the best data source.
    Your answer MUST be one of: PubMed, ClinicalTrials.gov, Both. Do not provide explanations.

    Question: "{query}"
    """))
    logger.info("Literature source decision: %s", source_decision)
    results = {"search_summary": f"Query: '{query}'. Search performed on: {source_decision}.",
               "source_decision": source_decision, "clinical_trials": pd.DataFrame(), "pubmed_articles": [],
               "search_strings": {}}

    if source_decision in ("ClinicalTrials.gov", "Both"):
        cleaned = llm.invoke(f"""
            You are given a USER QUESTION.

            USER QUESTION: {query}

            Your task is to **extract only the core biological content** from the USER QUESTION.

            ### Keep:
            - Treatment (drug, intervention, biomarker, etc.)
            - Disease/condition
            - Outcome (if relevant)

            ### Remove:
            - Any mention of study design, methodology, instructions, protocols, trial setup
            - Generic phrases like "design a trial," "develop a study," or "assess impact"

            Return only the cleaned-up question in the format:

            CLEANED USER QUESTION: <cleaned-up question>
            """).strip()
        cleaned = cleaned.split("CLEANED USER QUESTION:")[-1].strip()
        keyword_response = llm.invoke(f"""
            You are given a USER QUESTION.

            USER QUESTION: {cleaned}

            Your task is to extract the keywords of disease and treatment from the USER QUESTION.
            If there is an outcome, extract the keywords of the outcome as well.
            If there is no outcome, should return OUTCOME: None
            The treatment keywords should not include "treatment" in them.
            Separate alternative keywords of the same field (synonyms, related conditions or outcomes) with commas.

            Return the keywords in the expected structure output:
            DISEASE: <disease keywords>; TREATMENT: <treatment keywords>; OUTCOME: <outcome keywords>

            Do not provide any additional information or explanation.
            """)
        try:
            disease = _to_essie(_keyword_field(keyword_response, "DISEASE", "TREATMENT"))
            treatment = _to_essie(_keyword_field(keyword_response, "TREATMENT", "OUTCOME"))
            outcome = _keyword_field(keyword_response, "OUTCOME")
            outcome = None if not outcome or outcome.lower().startswith("none") else _to_essie(outcome)
            results["search_strings"]["ClinicalTrials.gov"] = {"condition": disease, "intervention": treatment,
                                                              "outcome": outcome}
            results["clinical_trials"] = utils.get_clinical_trials_data(disease, treatment, outcome=outcome)
            logger.info("ClinicalTrials.gov: %d trials found.", len(results["clinical_trials"]))
        except Exception as e:
            logger.warning("Could not retrieve data from ClinicalTrials.gov: %s", e)

    if source_decision in ("PubMed", "Both"):
        from paperscraper.pubmed import get_and_dump_pubmed_papers
        from paperscraper.utils import load_jsonl
        response = llm.invoke(f"""
        You are a keyword extraction assistant. The output MUST only be a comma-separated list of keywords.
        The keywords should be related to biomedicine. Provide at most 4 keywords from the question below.
        Question: "{query}"
        """)
        # Punctuation becomes a space so that, e.g., "all-cause" is searched as "all cause" rather than "allcause".
        keywords = [" ".join(re.sub(r"[^\w\s]", " ", k).split()) for k in response.split(",")]
        keywords = [k for k in keywords if k]
        results["search_strings"]["PubMed"] = " AND ".join(f"({k})" for k in keywords)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "pubmed.jsonl")
                get_and_dump_pubmed_papers([[k] for k in keywords], output_filepath=path, max_results=max_pubmed_results)
                results["pubmed_articles"] = [{"title": p.get("title", "No Title"), "abstract": p["abstract"],
                                               "doi": p.get("doi"), "date": p.get("date")}
                                              for p in load_jsonl(path) if p.get("abstract")]
            logger.info("PubMed: %d articles with abstracts found.", len(results["pubmed_articles"]))
        except Exception as e:
            logger.warning("Could not retrieve data from PubMed: %s", e)
    return results


EXTRACTION_PROMPT = """
    You are a biomedical researcher analyzing a scientific document.
    From the provided text, extract the following information: Population, Intervention,
    Comparator, and Outcome (PICO). Also extract any key statistical metrics like
    Odds Ratio (OR), Hazard Ratio (HR), p-value, and confidence intervals.

    Return the information in a valid JSON format with the following keys:
    "population": "...",
    "intervention": "...",
    "comparator": "...",
    "outcome": "...",
    "metrics": "..."

    If a piece of information is not available, use "Not specified".

    DOCUMENT TEXT:
    ---
    {document_text}
    ---
    """


def extract_evidence(literature_results: dict, llm, max_docs_per_source: int = 10, max_workers: int = 8) -> list:
    """Extract PICO elements and reported metrics from the retrieved trials and abstracts."""
    documents = []
    trials = literature_results.get("clinical_trials", pd.DataFrame())
    for _, row in trials.head(max_docs_per_source).iterrows():
        text = (f"Title: {row.get('Brief Title', '')}. Official Title: {row.get('Official Title', '')}. "
                f"Conditions: {row.get('Conditions', '')}. Interventions: {row.get('Interventions', '')}. "
                f"Primary Outcomes: {row.get('Primary Outcomes', '')}. Eligibility Criteria: {row.get('Eligibility Criteria', '')}")
        documents.append({"source": "ClinicalTrials.gov", "id": row.get("NCT ID", "N/A"), "source_text": text})
    for article in literature_results.get("pubmed_articles", [])[:max_docs_per_source]:
        documents.append({"source": "PubMed", "title": article.get("title", "N/A"), "id": article.get("doi") or "N/A",
                          "source_text": f"Title: {article.get('title', '')}. Abstract: {article.get('abstract', '')}"})

    def _extract(doc):
        parsed = parse_json_response(llm.invoke(EXTRACTION_PROMPT.format(document_text=doc["source_text"])))
        if not isinstance(parsed, dict):
            logger.warning("Could not parse evidence extraction for %s.", doc.get("id"))
            return None
        return {**parsed, **doc}

    return [d for d in llm_map(_extract, documents, max_workers) if d is not None]


def assess_study_quality(structured_evidence: list, llm, max_workers: int = 8) -> list:
    """Provisional study design and risk-of-bias label (Low/Moderate/High) with a one-sentence justification."""
    template = """
    You are an expert in clinical trial methodology. Based on the provided study information,
    identify the study design and assess the potential risk of bias. Consider the study's design
    (e.g., RCT, observational), population, and reported outcomes.

    Return your assessment as a JSON object with three keys:
    "study_design": One of "Randomized controlled trial", "Non-randomized trial", "Cohort study",
                    "Case-control study", "Cross-sectional study", "Systematic review or meta-analysis",
                    "Case report or series", "Preclinical study", "Trial registration without results", "Other".
    "risk_of_bias": Classify as "Low", "Moderate", "High", or "Unclear".
    "reasoning": A brief, one-sentence explanation for your classification.

    Your response MUST be only the valid JSON object.

    STUDY INFORMATION:
    ---
    {study_info}
    ---
    """

    def _assess(study):
        result = parse_json_response(llm.invoke(template.format(study_info=json.dumps(study, indent=2))))
        if not isinstance(result, dict):
            result = {"study_design": "Other", "risk_of_bias": "Unclear", "reasoning": "Failed to parse LLM response."}
        return {**study, "quality_assessment": result}

    return llm_map(_assess, structured_evidence, max_workers)


def annotate_limitations(evidence_with_quality: list, llm, max_workers: int = 8) -> list:
    """Extract verbatim statements of study limitations from the source text."""
    template = """
    You are a critical reviewer of scientific literature. Your task is to identify
    and extract sentences that state the limitations of the study from the provided text.
    Look for phrases like "a limitation of our study was...", "small sample size",
    "short follow-up", "observational nature", or "results may not be generalizable".

    Return your findings as a JSON object with one key, "limitations",
    which should contain a list of strings. Each string should be a direct quote
    of a limitation found in the text. If no limitations are found, return an empty list.

    Your response MUST be only the valid JSON object.

    STUDY TEXT:
    ---
    {study_text}
    ---
    """

    def _annotate(study):
        parsed = parse_json_response(llm.invoke(template.format(study_text=study.get("source_text", ""))))
        limitations = parsed.get("limitations", []) if isinstance(parsed, dict) else []
        return {**study, "limitations": limitations}

    return llm_map(_annotate, evidence_with_quality, max_workers)


def aggregate_evidence(fully_enriched_evidence: list) -> pd.DataFrame:
    """Evidence Aggregator: one row per document with PICO, metrics, quality and limitations."""
    if not fully_enriched_evidence:
        return pd.DataFrame()
    df = pd.json_normalize(fully_enriched_evidence, max_level=1).rename(columns={
        "quality_assessment.study_design": "study_design",
        "quality_assessment.risk_of_bias": "risk_of_bias",
        "quality_assessment.reasoning": "quality_reasoning"})
    order = ["source", "id", "title", "study_design", "population", "intervention", "comparator", "outcome", "metrics",
             "risk_of_bias", "quality_reasoning", "limitations"]
    return df[[c for c in order if c in df.columns]].fillna("N/A")


def generate_synthesis_summary(aggregated_df: pd.DataFrame, query: str, llm, batch_size: int = 5) -> str:
    """Narrative synthesis of the aggregated literature evidence (summarized in batches, then combined)."""
    if aggregated_df.empty:
        return "No evidence was found to generate a summary."
    summaries = []
    for start in range(0, len(aggregated_df), batch_size):
        block = ""
        for _, study in aggregated_df.iloc[start:start + batch_size].iterrows():
            limitations = study.get("limitations")
            block += (f"### Study from: {study.get('source')}\n- ID: {study.get('id')}\n- Title: {study.get('title', 'N/A')}\n"
                      f"- Design: {study.get('study_design', 'N/A')}\n- Population: {study.get('population')}\n"
                      f"- Intervention: {study.get('intervention')}\n- Comparator: {study.get('comparator')}\n"
                      f"- Outcome: {study.get('outcome')}\n- Key Metrics: {study.get('metrics')}\n"
                      f"- Quality (Risk of Bias): {study.get('risk_of_bias')} ({study.get('quality_reasoning', '')})\n"
                      f"- Stated Limitations: {'; '.join(limitations) if isinstance(limitations, list) and limitations else 'N/A'}\n\n")
        summaries.append(llm.invoke(f"""
        You are a medical evidence synthesis expert. Based on the following batch of evidence,
        write a concise summary that addresses the user's query.
        Critically evaluate the evidence, mentioning the source, findings, quality, and limitations.

        USER QUERY: "{query}"

        EVIDENCE BATCH:
        ---
        {block}
        ---

        CONCISE SUMMARY OF THIS BATCH:
        """))
    if len(summaries) == 1:
        return summaries[0]
    joined = "\n\n---\n\n".join(summaries)
    return llm.invoke(f"""
        You are a medical evidence synthesis expert. Synthesize the following summaries into a single, comprehensive, and non-redundant final answer.

        USER QUERY: "{query}"

        BATCH SUMMARIES:
        ---
        {joined}
        ---

        COMPREHENSIVE FINAL ANSWER:
        """)


def add_numerical_summary(evidence_list: list, llm, max_workers: int = 8):
    """
    Add a compact, numerically focused summary ('extracted_numerical_results') to every evidence item.
    Returns the list with and without the full source text.
    """
    template = """
You are an expert medical evidence synthesis assistant.

Your task is to read the evidence text and write a HIGHLY CONDENSED, information-dense, COMPACT natural-language summary that preserves as many IMPORTANT NUMERICAL RESULTS as possible.

Important instructions:
- Focus on preserving numerical content, not on making the summary long.
- Extract the important quantitative findings from the evidence text in natural language.
- If several important numerical findings are reported, include them together rather than keeping only one.
- Keep clinically meaningful numbers such as sample sizes, event rates, percentages, subgroup differences, effect sizes, confidence intervals, p-values, baseline/follow-up values, time windows, thresholds, and other important numerical findings when they matter.
- Remove background, interpretation, and generic statements.
- Do not invent numbers.
- If no actual outcome results are reported, explicitly say so and summarize available numerical design details.
- Output only one paragraph of plain text. Do NOT output JSON. Do NOT use bullet points.

Evidence text:
\"\"\"{document_text}\"\"\"
"""

    def _summarize(evidence):
        evidence = copy.deepcopy(evidence)
        text = evidence.get("source_text")
        if not text or not str(text).strip():
            evidence["extracted_numerical_results"] = "No source text was available."
        else:
            evidence["extracted_numerical_results"] = str(llm.invoke(template.format(document_text=text))).strip()
        return evidence

    with_text = llm_map(_summarize, evidence_list, max_workers)
    without_text = [{k: v for k, v in e.items() if k != "source_text"} for e in with_text]
    return with_text, without_text
