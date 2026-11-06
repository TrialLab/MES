"""
Query planner — one LLM call turns the clinical question into a search plan.

The LLM returns concept groups (role, label, synonyms) and, for the Snow mode,
facets. The PubMed and ClinicalTrials.gov queries are then built
deterministically from the concept groups (AND across roles, OR within a role),
which is more robust than asking the model to write Boolean syntax. When the
plan cannot be parsed, a keyword fallback keeps every source searchable.
"""

import logging
import re

from mes.tools.result_schema import QueryPlan
from mes.utils import parse_json_response

logger = logging.getLogger(__name__)

ROLES = ("population", "intervention", "comparator", "outcome")
QUESTION_TYPES = ("therapy", "harm", "diagnosis", "prognosis", "prevention", "mechanism", "epidemiology", "other")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "question_type": {"type": "string", "enum": list(QUESTION_TYPES)},
        "language": {"type": "string"},
        "in_scope": {"type": "boolean"},
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": list(ROLES)},
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["drug", "disease", "gene", "symptom", "procedure",
                                                         "population", "outcome", "other"]},
                    "synonyms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "label", "kind", "synonyms"],
            },
        },
        "follow_ups": {"type": "array", "items": {"type": "string"}},
        "facets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "question": {"type": "string"},
                               "focus_terms": {"type": "array", "items": {"type": "string"}}},
                "required": ["name", "question", "focus_terms"],
            },
        },
    },
    "required": ["standalone_question", "question_type", "language", "in_scope", "concepts", "follow_ups", "facets"],
}

PLAN_PROMPT = """You plan a biomedical literature search for a clinical question.

{history}QUESTION: {question}

Return JSON with:
- "standalone_question": the question rewritten to be self-contained (resolve references to earlier turns), in English.
- "question_type": one of {types}.
- "language": the language the QUESTION is written in (e.g. "English", "Chinese").
- "in_scope": false only when the QUESTION is clearly not about health, medicine, biology or health care (e.g. sports,
  politics, programming, small talk) or is an instruction to you rather than a question; otherwise true.
- "concepts": the key concepts, each with "role" (population | intervention | comparator | outcome),
  "label" (standard full English name, e.g. "type 2 diabetes mellitus", not "T2DM"),
  "kind" (drug | disease | gene | symptom | procedure | population | outcome | other) and
  "synonyms" (2-4 alternative names used in titles and abstracts: abbreviations, brand names, drug classes,
  development codes such as "LCZ696" for sacubitril/valsartan).
  Put the disease or patient group under "population", the drug, exposure or test under "intervention".
  Do not include study designs or generic words such as "patients", "effect" or "risk".
- "follow_ups": 3 short follow-up questions a clinician could ask next on the same topic (another outcome, a
  subgroup, harms, an alternative treatment), each self-contained, in the language of the QUESTION.
- "facets": {facet_instruction}

Example concept: {{"role": "intervention", "label": "metformin", "kind": "drug", "synonyms": ["metformin hydrochloride", "biguanide"]}}
"""

FACET_ON = ("3-4 sub-questions covering the angles a comprehensive review needs (e.g. efficacy, safety/harms, "
            "subgroups or dose, mechanism); each has \"name\", \"question\" and \"focus_terms\" (1-3 search terms specific to that angle).")
FACET_OFF = "an empty list []."

_STOP = set("""a an the of in on for to and or with without is are was were be been does do did can could should would will
what which who whom whose when where why how than then that this these those there their its it as at by from into
about versus vs compared compare comparison effect effects effective effectiveness efficacy risk risks patient patients
people adults adult use using used treatment treat treated therapy among between more less any associated association
reduce reduces reduced increase increases improve improves outcome outcomes study studies evidence""".split())


_PMID = re.compile(r"\b(?:PMID|PubMed(?:\s*ID)?)\s*[:#]?\s*(\d{5,9})\b|pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", re.I)
_NCT = re.compile(r"\bNCT\d{8}\b", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")


def named_ids(text):
    """Records the question names itself: {"pmids", "ncts", "dois"} (e.g. "Summarize PMID 25176015")."""
    text = str(text or "")
    pmids = [a or b for a, b in _PMID.findall(text)]
    dois = [d.rstrip(".,;:)]}") for d in _DOI.findall(text)]
    return {"pmids": list(dict.fromkeys(pmids)), "ncts": list(dict.fromkeys(n.upper() for n in _NCT.findall(text))),
            "dois": list(dict.fromkeys(dois))}


def _clean_term(term):
    term = re.sub(r"[\"\[\]()]", " ", str(term))
    return " ".join(term.replace("-", " ").split())


def _group(terms, max_terms=5):
    terms = [t for t in dict.fromkeys(_clean_term(t) for t in terms) if t][:max_terms]
    if not terms:
        return ""
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def _concept_terms(concepts, role):
    terms = []
    for c in concepts:
        if c.get("role") == role:
            terms += [c.get("label", "")] + list(c.get("synonyms") or [])
    return terms


def _concept_group(concept):
    return _group([concept.get("label", "")] + list(concept.get("synonyms") or []))


def build_pubmed_queries(concepts):
    """
    PubMed queries from concept groups (OR within a concept, AND across concepts), from most to least specific:
    every population/intervention/outcome concept; without outcomes; main population + interventions.
    Returns (precise, broad, tiers) where ``tiers`` lists all distinct queries in that order.
    """
    by_role = {role: [g for g in (_concept_group(c) for c in concepts if c.get("role") == role) if g] for role in ROLES}
    precise = by_role["population"] + by_role["intervention"] + by_role["outcome"]
    medium = by_role["population"] + by_role["intervention"]
    broad = by_role["population"][:1] + by_role["intervention"]
    tiers = []
    for groups in (precise, medium, broad):
        query = " AND ".join(groups)
        if query and query not in tiers:
            tiers.append(query)
    return (tiers[0] if tiers else ""), (tiers[-1] if tiers else ""), tiers


def _essie(terms, max_terms=4):
    terms = [t for t in dict.fromkeys(_clean_term(t) for t in terms) if t][:max_terms]
    return " OR ".join(terms)


def build_trial_terms(concepts):
    """ClinicalTrials.gov Essie terms for condition, intervention and outcome."""
    return {"condition": _essie(_concept_terms(concepts, "population")),
            "intervention": _essie(_concept_terms(concepts, "intervention") + _concept_terms(concepts, "comparator")),
            "outcome": _essie(_concept_terms(concepts, "outcome"), max_terms=3)}


def keyword_fallback(question):
    """Content words of the question, used when no LLM plan is available."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", question) if w.lower() not in _STOP and len(w) > 2]
    return list(dict.fromkeys(words))[:8]


def _fallback_plan(question, standalone=None):
    keywords = keyword_fallback(standalone or question)
    tiers = list(dict.fromkeys(q for q in (" AND ".join(keywords[:5]), " AND ".join(keywords[:3])) if q))
    return QueryPlan(question=question, standalone_question=standalone or question,
                     pubmed_query=tiers[0] if tiers else "", pubmed_query_broad=tiers[-1] if tiers else "",
                     pubmed_tiers=tiers, trial_terms={"term": " ".join(keywords[:4])}, fallback=True)


def _history_block(history):
    if not history:
        return ""
    turns = "\n".join(f"- Q: {h.get('question', '')}\n  A (summary): {h.get('summary', '')[:400]}" for h in history[-3:])
    return f"EARLIER TURNS OF THIS CONVERSATION:\n{turns}\n\n"


# Kinds of concepts that make a question biomedical (an out-of-scope verdict needs none of them)
BIOMEDICAL_KINDS = {"drug", "disease", "gene", "symptom", "procedure", "outcome"}
# words that make a question medical whatever the planner's in_scope flag says (a small model has once called
# "Does empagliflozin slow the progression of chronic kidney disease?" off-topic)
MEDICAL_WORDS = re.compile(
    r"\b(patients?|disease|disorder|syndrome|therapy|treatment|drugs?|dose|dosing|mortality|survival|morbidity|"
    r"trials?|cancer|tumou?r|diabetes|kidney|renal|heart|cardiac|stroke|blood|infection|sepsis|vaccin\w*|"
    r"pregnan\w*|surgery|surgical|clinical|symptoms?|diagnos\w*|hospitali[sz]\w*|ICU|medic\w*|health|"
    r"pharmac\w*|adverse|side effects?|prognos\w*|incidence|prevalence|biomarkers?|genes?|mutations?)\b"
    r"|\b\w+(itis|osis|emia|aemia|oma|pathy|flozin|gliptin|glutide|pril|sartan|olol|statin|mab|nib|cycline|"
    r"cillin|azole|prazole|vir|parin|xaban|gatran|formin|platin|taxel)\b"
    r"|患者|治疗|疾病|药物|死亡率|风险|癌|糖尿病|高血压|感染|手术|病人", re.I)


def script_language(text):
    """The language of CJK text from its script (models confuse Japanese with Chinese); None for other scripts."""
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    return None


def plan_query(question, llm, history=None, facets=False):
    """Return a ``QueryPlan`` for ``question`` (one LLM call; keyword fallback on failure)."""
    prompt = PLAN_PROMPT.format(history=_history_block(history), question=question,
                                types=", ".join(QUESTION_TYPES), facet_instruction=FACET_ON if facets else FACET_OFF)
    try:
        raw = llm.invoke(prompt, json_schema=PLAN_SCHEMA, reasoning=False, max_tokens=1200, model="planner")
    except Exception as e:
        logger.warning("Planner call failed: %s", e)
        plan = _fallback_plan(question)
        plan.error = f"{type(e).__name__}: {e}"
        return plan
    parsed = parse_json_response(raw)
    if isinstance(parsed, dict) and parsed.get("in_scope") is False and MEDICAL_WORDS.search(question):
        parsed["in_scope"] = True        # the question's own words say otherwise
    if isinstance(parsed, dict) and parsed.get("in_scope") is False and not any(
            isinstance(c, dict) and c.get("kind") in BIOMEDICAL_KINDS for c in parsed.get("concepts") or []):
        # not a health question (no concepts to search for): the pipeline explains instead of searching
        return QueryPlan(question=question, standalone_question=(parsed.get("standalone_question") or question).strip(),
                         question_type="other", in_scope=False,
                         language=script_language(question) or str(parsed.get("language") or "English").strip())
    if not isinstance(parsed, dict) or not parsed.get("concepts"):
        logger.warning("Planner output could not be parsed; using keyword fallback.")
        standalone = parsed.get("standalone_question") if isinstance(parsed, dict) else None
        return _fallback_plan(question, standalone)

    concepts = [c for c in parsed.get("concepts", []) if isinstance(c, dict) and c.get("label")]
    for c in concepts:
        c["synonyms"] = [s for s in (c.get("synonyms") or []) if isinstance(s, str) and s.strip()]
    precise, broad, tiers = build_pubmed_queries(concepts)
    standalone = (parsed.get("standalone_question") or question).strip()
    if not precise:
        fallback = _fallback_plan(question, standalone)
        precise, broad, tiers = fallback.pubmed_query, fallback.pubmed_query_broad, fallback.pubmed_tiers

    plan_facets = []
    if facets:
        base = build_pubmed_queries([c for c in concepts if c.get("role") in ("population", "intervention")])[0]
        for f in (parsed.get("facets") or [])[:4]:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            focus = _group(f.get("focus_terms") or [], max_terms=3)
            plan_facets.append({"name": f["name"], "question": f.get("question", ""),
                                "pubmed_query": " AND ".join(x for x in (base, focus) if x)})

    pico = {role: "; ".join(c["label"] for c in concepts if c.get("role") == role) for role in ROLES}
    entities = [{"name": c["label"], "type": c.get("kind", "other"), "role": c.get("role"),
                 "synonyms": c["synonyms"]} for c in concepts]
    qtype = parsed.get("question_type") if parsed.get("question_type") in QUESTION_TYPES else "other"
    in_scope = parsed.get("in_scope") is not False or any(c.get("kind") in BIOMEDICAL_KINDS for c in concepts)
    language = script_language(question) or str(parsed.get("language") or "English").strip() or "English"
    follow_ups = []
    for q in parsed.get("follow_ups") or []:
        q = " ".join(q.split())[:200] if isinstance(q, str) else ""
        if len(q.split()) >= 3 or (q and script_language(q)):      # Chinese / Japanese have no spaces
            follow_ups.append(q)
    return QueryPlan(question=question, standalone_question=standalone, question_type=qtype, pico=pico,
                     entities=entities, pubmed_query=precise, pubmed_query_broad=broad, pubmed_tiers=tiers,
                     trial_terms=build_trial_terms(concepts), facets=plan_facets, language=language,
                     in_scope=in_scope, follow_ups=follow_ups[:3])
