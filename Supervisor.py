"""
Supervisor: query structuring, orchestration and report synthesis.

* Classifies the clinical question (therapy, diagnosis, prognosis, etiology/harm,
  cost-effectiveness, quality of life), decomposes it with the matching EBM schema
  (PICO, TTR, exposure-outcome), detects ambiguity and asks clarification questions.
* Selects the specialist agents required for the question.
* Synthesizes screened evidence into a structured report, checks it against
  PRISMA-lite criteria and against the decomposed question, and proposes targeted
  follow-up queries (and the agents that should address them) when the report is incomplete.
* Combines the reports of all rounds into the final evidence report.
"""

import json
import logging
from typing import Any, Callable, List, Optional

from utils import parse_json_response

logger = logging.getLogger(__name__)

SPECIALIST_AGENTS = ["LiteratureMiner", "RWD-Analyst", "KG-Specialist"]


# ============================================================================ query decomposition
def _create_query_decomposition_prompt(user_query: str) -> str:
    return f"""
You are an expert assistant in an Evidence-Based Medicine (EBM) synthesis system. Your task is to analyze a user's clinical query, classify it, and decompose it into a structured JSON format based on the provided taxonomy.
**Taxonomy of Clinical Questions:**
| Question Type      | Description                               | Suggested Decomposition Schema | Primary Agents                               |
|--------------------|-------------------------------------------|--------------------------------|----------------------------------------------|
| Therapy            | What is the best treatment?               | PICO                           | LiteratureMiner, RWD-Analyst, KG-Specialist  |
| Diagnosis          | How accurate is a test?                   | TTR                            | LiteratureMiner, RWD-Analyst, KG-Specialist  |
| Prognosis          | What is the natural course or outcome?    | Exposure-Outcome               | LiteratureMiner, RWD-Analyst, KG-Specialist  |
| Etiology / Harm    | Does an exposure cause harm?              | Exposure-Outcome               | LiteratureMiner, RWD-Analyst, KG-Specialist  |
| Cost-effectiveness | Is an intervention worth the cost?        | Exposure-Outcome + Cost nodes  | LiteratureMiner                              |
| Quality of life    | How does it affect patient well-being?    | Population-Outcome             | LiteratureMiner                              |

**Instructions:**
1.  **Analyze the User Query:** Carefully read the user's query provided below.
2.  **Classify the Question Type:** Identify the single most appropriate 'Question Type' from the taxonomy table.
3.  **Apply Decomposition Schema:** Based on the classified question type, use the corresponding 'Suggested Decomposition Schema' (e.g., PICO).
4.  **Extract Components:** Extract the relevant components from the query. If a component is not mentioned, use an empty string "" or null.
5.  **Identify Primary Agents:** List the 'Primary Agents' associated with the chosen question type under the key "primary_agents".
6.  **Check for Ambiguity:** Assess if the query is clear enough. If it lacks critical information (e.g., a missing 'Outcome' in a 'Therapy' question), set 'is_ambiguous' to true and formulate a concise question to ask the user for clarification under the key "clarification_question".
7.  **Output in JSON Format:** Your final output MUST be a single, valid JSON object with the keys "question_type", "decomposition_schema", "components", "primary_agents", "is_ambiguous", "clarification_question".
**User Query:**
"{user_query}"
**JSON Output:**
"""


def _decompose_query(llm_client, user_query: str) -> Optional[dict]:
    structured = parse_json_response(llm_client.invoke(_create_query_decomposition_prompt(user_query)))
    if not isinstance(structured, dict):
        logger.warning("Query decomposition could not be parsed.")
        return None
    structured["input_user_query"] = user_query
    return structured


def clarify_and_decompose_query(llm_client, initial_query: str, max_clarification_rounds: int = 2,
                                ask_user: Callable[[str], str] = input) -> Optional[dict]:
    """
    Decompose a query and, while it is ambiguous, ask the user for clarification
    (``max_clarification_rounds=0`` disables clarification, e.g., for batch evaluation).
    """
    current_query = initial_query
    structured = None
    for round_idx in range(max_clarification_rounds + 1):
        structured = _decompose_query(llm_client, current_query)
        if structured is None or not structured.get("is_ambiguous"):
            return structured
        question = structured.get("clarification_question")
        if round_idx == max_clarification_rounds or not question:
            logger.info("Query remains ambiguous; continuing with the partial decomposition.")
            return structured
        answer = ask_user(f"{question}\nYour answer: ")
        current_query = f"Original question was: '{current_query}'. The user provided this clarification: '{answer}'."
    return structured


def get_primary_agents(structured_query: Optional[dict]) -> List[str]:
    """Specialist agents selected by the decomposition (all specialists if none could be identified)."""
    raw = (structured_query or {}).get("primary_agents")
    text = " ".join(raw) if isinstance(raw, list) else str(raw or "")
    selected = [a for a, key in (("LiteratureMiner", "literature"), ("RWD-Analyst", "rwd"), ("KG-Specialist", "kg"))
                if key in text.lower().replace("-", "").replace(" ", "")]
    return selected or list(SPECIALIST_AGENTS)


# ============================================================================ report synthesis
def synthesize_report(llm_client, structured_query: dict, validated_evidence: dict) -> Any:
    """Synthesize the screened evidence into a structured report (JSON)."""
    screening = validated_evidence.get("safechecker", {}) if isinstance(validated_evidence, dict) else {}
    flagged = [c.get("claim") for c in screening.get("flagged_claims", [])]
    prompt = f"""
        You are a senior medical writer specializing in evidence synthesis for clinical decision support. Your task is to synthesize the provided structured evidence into a high-quality, clear, and well-structured clinical evidence summary report.
        The answer to the clinical question must be in the final report.

        **1. Original Clinical Question (Decomposed):**
        ```json
        {json.dumps(structured_query, indent=2)}
        ```

        **2. Evidence Synthesis and Claim Screening Results:**
        This evidence has been retrieved from multiple sources (literature, real-world data, knowledge graph), synthesized by the Statistician and screened claim by claim by the SafeChecker.
        ```json
        {json.dumps(validated_evidence, indent=2, default=str)}
        ```

        **3. Claims that failed evidence-support screening (do NOT include them in the report):**
        {json.dumps(flagged, indent=2)}

        **4. Instructions for Report Generation:**

        **Structure:** Generate the report as a single, valid JSON object.

        **Tone:** Be objective, neutral, and data-driven. Distinguish trial evidence, observational or real-world associations and knowledge-graph (mechanistic) context. Report disagreement between sources and missing evidence explicitly.

        **Fidelity:** Describe the population, setting and design of each study exactly as stated in its evidence item; do not attribute the setting of the question (e.g., ICU) to a study whose population differs.

        **Organization by question type:** therapy and comparative-effectiveness questions emphasize effectiveness, comparative outcomes and safety; diagnostic questions emphasize test performance, reference standards and population-specific accuracy; prognostic and health-services questions emphasize outcome definitions, applicability and decision boundaries.

        **Content:**

        - **title:** Create a concise, descriptive title based on the clinical question.
        - **executive_summary:** Write a brief, top-level summary (2-3 sentences) of the key findings from all provided evidence items.
        - **answer:** Provide a direct answer to the clinical question based on the synthesized evidence, including how certain it is.
        - **sections:** Synthesize the findings, e.g., one section per evidence source or theme. In each section, describe the findings and explicitly list the source citations for each piece of evidence you mention.
        - **summary_table:** Create a table that consolidates the main quantitative results from the evidence list. The table should have headers and rows.
        - **figure_placeholders:** Suggest relevant figures (e.g., forest plots) that could visualize the data, but do not generate the figures themselves.
        - **limitations:** State the main limitations of the evidence (e.g., study quality, heterogeneity, confounding in observational data, evidence gaps).
        - **citations:** Ensure every piece of evidence mentioned is attributed to its source using the citation keys from the input evidence list.

        **JSON Output:**
        """
    response = llm_client.invoke(prompt)
    report = parse_json_response(response)
    return report if isinstance(report, dict) else str(response).strip()


def check_report_format(llm_client, generated_report: Any, standard: str = "PRISMA-lite") -> Any:
    """Check the report against simplified PRISMA-style reporting principles."""
    report_str = generated_report if isinstance(generated_report, str) else json.dumps(generated_report, indent=2)
    checklist = """
    - **Title**: Does the title clearly identify the report's content?
    - **Objectives / Clinical Question**: Is the clinical question (e.g., PICO) clearly stated or summarized?
    - **Information Sources**: Does the report explicitly mention the types of sources used (e.g., Randomized Controlled Trials, Real-World Data)?
    - **Synthesis of Results**: Are the main results presented clearly, including quantitative data in the text and/or a summary table?
    - **Limitations**: Does the report mention limitations of the evidence (e.g., study quality, heterogeneity, potential for bias)?
    - **Conclusions**: Is there a general interpretation of the results that is balanced and consistent with the presented evidence?
    """
    prompt = f"""
    You are a quality assurance specialist for medical reports, with expertise in reporting guidelines like PRISMA. Your task is to check if the provided report draft adheres to the key principles of the specified reporting standard.

    **1. Reporting Standard:** {standard}
    **Simplified {standard} Checklist:**
    {checklist}

    **2. Generated Report Draft:**
    {report_str}

    **3. Instructions:**
    - Review the report draft against each point in the checklist.
    - Determine if the report passes the check (`passed`: true/false). The report only passes if all major principles, especially **Limitations**, are adequately addressed.
    - For any unmet principle, create an entry in the `issues` list with the `principle` and a concrete `suggestion`.
    - Your final output MUST be a single, valid JSON object with no other text.
    """
    response = llm_client.invoke(prompt)
    parsed = parse_json_response(response)
    return parsed if isinstance(parsed, dict) else str(response).strip()


def check_final_answer_relevance(llm_client, structured_query: dict, final_report: Any) -> dict:
    """Check whether the report answers the decomposed question; otherwise propose a targeted follow-up."""
    prompt = f"""
    You are a meticulous clinical research auditor. Your job is to determine if the 'Evidence Summary Report' completely and directly answers the 'Original Clinical Question'.

    **1. Original Clinical Question (Decomposed):**
    {json.dumps(structured_query, indent=2)}

    **2. Generated Evidence Summary Report:**
    {json.dumps(final_report, indent=2, default=str) if not isinstance(final_report, str) else final_report}

    **3. Your Task and Instructions:**
    1.  Compare the report against all components of the original question (e.g., Population, Intervention, Comparator, Outcome).
    2.  Does the report provide a clear answer for the specific outcome requested, in the requested population?
        - If YES: set `is_complete` to true and `follow_up_query` to null.
        - If NO: set `is_complete` to false, identify the specific missing information and formulate a concise
          `follow_up_query` that would fill this gap (e.g., a question restricted to the population subgroup that is not covered).
    3.  If the report is incomplete, list in `target_agents` which specialist agents should address the follow-up query
        (any of "LiteratureMiner", "RWD-Analyst", "KG-Specialist"): e.g., literature retrieval for a missing population,
        a stratified real-world analysis, or mechanistic context from the knowledge graph.
    4.  Provide a brief `reasoning` for your decision.
    5.  Output a single, valid JSON object with the keys "is_complete", "follow_up_query", "target_agents", "reasoning".
    """
    parsed = parse_json_response(llm_client.invoke(prompt))
    if not isinstance(parsed, dict):
        return {"is_complete": True, "follow_up_query": None, "target_agents": [],
                "reasoning": "The completeness check could not be parsed; no follow-up was launched."}
    agents = parsed.get("target_agents") or []
    parsed["target_agents"] = [a for a in SPECIALIST_AGENTS if any(a.lower().replace("-", "") in str(x).lower().replace("-", "") for x in agents)]
    return parsed


# ============================================================================ final report
def _create_final_report_prompt(structured_query: dict, final_reports: list, literature_evidence: list,
                                rwd_evidence: list, kg_evidence: list, data_source: str) -> str:
    return f"""
You are a senior medical evidence writer preparing a concise, numerically grounded clinical evidence report.

--------------------------------------------------
INPUTS
--------------------------------------------------
You will be given:
- the decomposed clinical question;
- evidence summary reports generated in each round (these already contain the Statistician's synthesis and the
  SafeChecker's claim screening);
- structured numerical evidence from the published literature (PubMed, ClinicalTrials.gov);
- real-world evidence (RWE) newly generated from {data_source} by the RWD-Analyst;
- knowledge-graph (iBKH) relations retrieved by the KG-Specialist.

--------------------------------------------------
PRINCIPLES (MANDATORY)
--------------------------------------------------
1. Answer the question directly in an opening "bottom line" paragraph that also states how strong and how consistent
   the evidence is.
2. Keep evidence classes distinct and say which class supports each statement:
   (a) randomized trials and systematic reviews, (b) other published studies, (c) newly generated RWE from
   {data_source} (observational associations: name the design, the adjustment for confounding and key limitations),
   (d) knowledge-graph relations (mechanistic or contextual information, not evidence of a clinical effect).
3. When sources agree, state the conclusion clearly. When sources conflict or are inconclusive, say so explicitly,
   report each estimate with its source and design, and describe plausible explanations (population, design, dose,
   outcome definition, follow-up, confounding). Do not force a consensus and do not choose between sources on the
   basis of statistical significance alone.
4. When evidence is missing for part of the question (e.g., no trials in the target population, no usable
   real-world analysis, no relevant knowledge-graph relation), state the gap explicitly instead of filling it.
5. RWE and knowledge-graph findings may qualify the published evidence and suggest hypotheses, but must not be
   presented as overriding randomized evidence. Where they diverge from the literature, report the divergence.
6. Use only numbers present in the inputs; prefer the most clinically meaningful measures (RR, HR, OR, absolute risk
   difference, NNT, sensitivity/specificity) with 95% CIs and p-values when available.
7. Do not include claims that the SafeChecker flagged as unsupported or contradicted.
8. General medical background may be used only to explain mechanisms or context, must be labelled as background,
   and must not be cited as evidence.
9. Describe the population, setting and design of each study exactly as stated in its evidence record; do not
   attribute the setting of the question (e.g., ICU) to a study whose population differs.

--------------------------------------------------
STRUCTURE
--------------------------------------------------
Bottom line. <answer to the question and certainty>

Evidence from the published literature. <key trials, pooled estimates, subgroups, safety, applicability; organize
by paragraphs or short named subsections as fits the question>

Real-world evidence ({data_source}). <design, cohort, effect estimate with 95% CI, adjustment, subgroup results and
limitations; one sentence if no usable real-world analysis was obtained>

Knowledge-graph context (iBKH). <relevant relations and their reliability; one sentence if none were relevant>

Agreement and divergence across sources. <only if more than one source reports on the same outcome>

Limitations. <main limitations of the evidence base>

References:
[1] ...

--------------------------------------------------
CITATION RULES (MANDATORY)
--------------------------------------------------
- Re-index all cited evidence as [1], [2], [3], ... in order of first appearance; do not use original IDs and do not skip numbers.
- Every index used in the text appears exactly once in References, and every reference is cited in the text.
- Reference formats: ClinicalTrials.gov -> full URL (https://clinicaltrials.gov/study/NCT...); journal articles -> title
  followed by "doi:" and the DOI when the evidence record has one;
  knowledge-graph findings -> "Knowledge graph relations from iBKH"; real-world analyses -> "Real-world data analysis on {data_source}".
- Do not output internal labels such as "Part A" or the words "Generated Evidence Summary Reports".

--------------------------------------------------
INPUT DATA (READ-ONLY)
--------------------------------------------------
1. Clinical Question:
{json.dumps(structured_query, indent=2)}

2. Evidence Summary Reports:
{json.dumps(final_reports, indent=2, default=str)}

3. Structured Numerical Evidence (literature):
{json.dumps(literature_evidence, indent=2, default=str)}

4. Real-world Evidence ({data_source}):
{json.dumps(rwd_evidence, indent=2, default=str)}

5. Knowledge-graph Relations (iBKH):
{json.dumps(kg_evidence, indent=2, default=str)}

--------------------------------------------------
YOUR OUTPUT (plain text report):
"""


def combine_final_reports(llm_client, structured_query: dict, final_reports: list, literature_evidence: list,
                          rwd_evidence: list, kg_evidence: list, data_source: str = "MIMIC-IV"):
    """Combine the reports of all rounds and the source-level evidence into the final report."""
    prompt = _create_final_report_prompt(structured_query, final_reports, literature_evidence, rwd_evidence,
                                         kg_evidence, data_source)
    return str(llm_client.invoke(prompt)).strip(), prompt
