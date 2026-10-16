"""
SafeChecker: claim-level evidence-support screening of the Statistician's synthesis.

1. Claim decomposition: the synthesis text is decomposed into atomic, verifiable claims.
2. Provenance tracing: each claim is linked to the most relevant evidence items.
3. Hallucination detection by negation probing: both the claim and its negation are
   verified against the evidence; a claim is accepted only if the evidence supports the
   claim and not its negation.
4. Consistency checking across accepted claims, using KG relations as biomedical constraints.
5. Each claim receives a screening status and a confidence score; unsupported or
   contradicted claims are flagged and passed to the Supervisor for exclusion.
The screening labels are not guarantees of factual or clinical correctness.
"""

import copy
import json
import logging
import re
from typing import Any, Dict, List

from llm_zoo import llm_map
from utils import parse_json_response

logger = logging.getLogger(__name__)

CLAIM_FIELDS = {"synthesis_summary", "summary", "conflict_analysis", "interpretation", "remaining_uncertainty"}
STOP_WORDS = {"a", "an", "the", "is", "in", "of", "for", "was", "and", "to", "it", "that", "with", "on", "or", "by"}


def _collect_text_blocks(data: Any, path: str, blocks: List[Dict]):
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and key in CLAIM_FIELDS and len(value.split()) > 3:
                blocks.append({"path": new_path, "text": value})
            else:
                _collect_text_blocks(value, new_path, blocks)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _collect_text_blocks(item, f"{path}[{i}]", blocks)


def decompose_claims(report: Dict, llm_client: Any, max_claims: int = 25) -> List[Dict]:
    """Decompose the synthesis into atomic claims, each with its logical negation for probing."""
    blocks = []
    _collect_text_blocks(report, "", blocks)
    if not blocks:
        return []
    prompt = f"""
You are decomposing an evidence synthesis into atomic claims for fact-checking.
A claim is a concise, verifiable statement expressing one specific conclusion or relationship derived from the evidence
(e.g., an effect estimate, a direction of effect, a population boundary, a safety signal, a statement about conflicting evidence).

For every claim also write its negation: a statement that asserts the opposite (e.g., "X reduces mortality" ->
"X does not reduce mortality"). Keep at most {max_claims} of the most important claims.

Text blocks (with their location in the report):
{json.dumps(blocks, indent=2)}

Return ONLY a JSON list of objects with keys "claim", "negation" and "path" (the location of the text block).
"""
    claims = parse_json_response(llm_client.invoke(prompt), default=[])
    claims = [c for c in claims if isinstance(c, dict) and c.get("claim")] if isinstance(claims, list) else []
    for i, c in enumerate(claims[:max_claims]):
        c["claim_id"] = f"C{i + 1}"
    return claims[:max_claims]


def _find_relevant_evidence(claim_text: str, all_evidence: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rank evidence items by keyword overlap (Jaccard) with the claim."""
    words = lambda t: set(re.findall(r"\b\w+\b", t.lower())) - STOP_WORDS
    claim_words = words(claim_text)
    scored = []
    for evidence in all_evidence:
        ev_words = words(json.dumps(evidence, default=str))
        union = len(claim_words | ev_words)
        score = len(claim_words & ev_words) / union if union else 0
        if score > 0.05:
            scored.append((score, evidence))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def normalize_status(status: Any) -> str:
    s = str(status or "").lower().replace("_", " ")
    if "contradict" in s or "refut" in s:
        return "Contradicted"
    if "unsupported" in s or "not supported" in s or "no support" in s:
        return "Unsupported"
    if "partial" in s or "mostly" in s:
        return "Partially Supported"
    if "support" in s or "verified" in s:
        return "Fully Supported"
    return "Unsupported" if s else "Error"


def verify_statement(statement: str, evidence: List[Dict], llm_client: Any) -> Dict:
    """Verify one statement against evidence; returns a normalized status and supporting evidence IDs."""
    prompt = f"""
    You are a meticulous fact-checker. Verify the following claim against the provided evidence only.
    Claim: "{statement}"
    Evidence: {json.dumps(evidence, indent=2, default=str)}

    Return a single JSON object with keys:
    "verification_status": one of "Fully Supported", "Partially Supported", "Unsupported", "Contradicted";
    "supporting_evidence_ids": list of evidence IDs (study_id) that support the claim;
    "uncertainty_level": one of "Low", "Moderate", "High";
    "reasoning": one or two sentences.
    """
    result = parse_json_response(llm_client.invoke(prompt))
    if not isinstance(result, dict):
        return {"verification_status": "Error", "reasoning": "The verification response could not be parsed."}
    result["verification_status"] = normalize_status(result.get("verification_status"))
    return result


def _confidence(status: str, uncertainty: str, evidence: List[Dict], supporting_ids: List) -> float:
    base = {"Verified": 0.9, "Partially Supported": 0.6, "Uncertain": 0.4, "Unsupported": 0.15,
            "Contradicted": 0.05}.get(status, 0.1)
    base += {"low": 0.05, "high": -0.1}.get(str(uncertainty).lower(), 0.0)
    supporting = [e for e in evidence if e.get("study_id") in (supporting_ids or [])]
    weak = [e for e in supporting if str((e.get("quality_assessment") or {}).get("risk_of_bias", "")).lower() == "high"
            or str(e.get("reliability", "")).lower() == "low"]
    if supporting and len(weak) == len(supporting):
        base -= 0.1
    return round(min(max(base, 0.0), 1.0), 2)


def verify_claim(claim: Dict, all_evidence: List[Dict], llm_client: Any) -> Dict:
    """Trace a claim to its evidence and screen it with negation probing."""
    evidence = _find_relevant_evidence(claim["claim"], all_evidence) or all_evidence
    original = verify_statement(claim["claim"], evidence, llm_client)
    negated = verify_statement(claim.get("negation") or f"It is not true that {claim['claim']}", evidence, llm_client)
    supported = original["verification_status"] in ("Fully Supported", "Partially Supported")
    negation_supported = negated["verification_status"] in ("Fully Supported", "Partially Supported")
    if supported and not negation_supported:
        status = "Verified" if original["verification_status"] == "Fully Supported" else "Partially Supported"
    elif supported and negation_supported:
        status = "Uncertain"
    elif original["verification_status"] == "Contradicted" or negation_supported:
        status = "Contradicted"
    else:
        status = "Unsupported"
    return {
        **claim, "status": status, "hallucination_flag": status in ("Unsupported", "Contradicted"),
        "claim_check": original, "negation_check": negated,
        "evidence_context_ids": [e.get("study_id") for e in evidence],
        "confidence_score": _confidence(status, original.get("uncertainty_level"), evidence,
                                        original.get("supporting_evidence_ids")),
    }


def check_internal_consistency(claims: List[Dict], llm_client: Any, kg_constraints: List[str] = None) -> Dict:
    """Check logical consistency among accepted claims and against KG-derived biomedical constraints."""
    if len(claims) < 2:
        return {"is_consistent": True, "checked_claims": len(claims), "summary": "Fewer than two accepted claims."}
    prompt = f"""
    Check for internal consistency among these claims from an evidence synthesis. Identify contradictions, for example
    when findings from the literature conflict with real-world data results for the same population and outcome
    without the conflict being acknowledged, or when a claim violates the biomedical relations listed below.

    Claims: {json.dumps([{"claim_id": c["claim_id"], "claim": c["claim"]} for c in claims], indent=2)}
    Biomedical relations from the knowledge graph: {json.dumps((kg_constraints or [])[:20], indent=2)}

    Return a JSON object with keys: "is_consistent" (true/false), "summary", "conflicts_found" (list of
    {{"claim_ids": [...], "description": "..."}}).
    """
    result = parse_json_response(llm_client.invoke(prompt))
    if not isinstance(result, dict):
        return {"is_consistent": None, "checked_claims": len(claims), "summary": "Consistency check could not be parsed."}
    result["checked_claims"] = len(claims)
    return result


def run_safechecker_pipeline(statistician_report: Dict, original_evidence_list: List[Dict], llm_client: Any,
                             kg_constraints: List[str] = None, max_workers: int = 8) -> Dict:
    """Screen the Statistician report; returns the report with a 'safechecker' section added."""
    report = copy.deepcopy(statistician_report)
    claims = decompose_claims(report, llm_client)
    logger.info("SafeChecker: %d claims extracted.", len(claims))
    verified = llm_map(lambda c: verify_claim(c, original_evidence_list, llm_client), claims, max_workers)
    accepted = [c for c in verified if c["status"] in ("Verified", "Partially Supported")]
    consistency = check_internal_consistency(accepted, llm_client, kg_constraints)
    flagged = [{"claim_id": c["claim_id"], "claim": c["claim"], "status": c["status"],
                "reasoning": c["claim_check"].get("reasoning")} for c in verified if c["status"] not in ("Verified", "Partially Supported")]
    report["safechecker"] = {
        "claims": verified,
        "flagged_claims": flagged,
        "overall_consistency_check": consistency,
        "summary": {"n_claims": len(verified), "n_accepted": len(accepted), "n_flagged": len(flagged)},
    }
    return report
