"""
ClinicalTrials.gov tools (API v2).

``search_trials`` queries condition / intervention / outcome terms (relaxing
the query when recall is low), ranks the registry records against the
question and, for the selected trials with posted results, fetches the outcome
measures with their arm-level values and statistical analyses (effect estimate,
CI, p-value). No LLM calls.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mes.tools.http_utils import get_json
from mes.tools.ranking_tools import expanded_query, hybrid_scores, mmr_select
from mes.tools.result_schema import EvidenceItem, Passage

logger = logging.getLogger(__name__)

CTGOV = "https://clinicaltrials.gov/api/v2/studies"


def _year(date_struct):
    date = (date_struct or {}).get("date", "")
    return int(date[:4]) if date[:4].isdigit() else None


def parse_study(study):
    """Flat record of the protocol section of one study."""
    ps = study.get("protocolSection", {})
    ident, status, design = ps.get("identificationModule", {}), ps.get("statusModule", {}), ps.get("designModule", {})
    arms = ps.get("armsInterventionsModule", {})
    outcomes = ps.get("outcomesModule", {})
    info = design.get("designInfo", {})
    return {
        "nct": ident.get("nctId", ""),
        "title": ident.get("briefTitle", "") or ident.get("officialTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "status": status.get("overallStatus", ""),
        "why_stopped": status.get("whyStopped", ""),
        "start_year": _year(status.get("startDateStruct")),
        "completion_year": _year(status.get("completionDateStruct") or status.get("primaryCompletionDateStruct")),
        "study_type": design.get("studyType", ""),
        "phases": [p.replace("PHASE", "Phase ").replace("EARLY_", "Early ").strip() for p in design.get("phases", [])
                   if p != "NA"],
        "allocation": info.get("allocation", ""),
        "masking": (info.get("maskingInfo") or {}).get("masking", ""),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "conditions": ps.get("conditionsModule", {}).get("conditions", []),
        "interventions": [f"{i.get('type', '').title()}: {i.get('name', '')}" for i in arms.get("interventions", [])],
        "primary_outcomes": [f"{o.get('measure', '')} ({o.get('timeFrame', '')})" for o in outcomes.get("primaryOutcomes", [])],
        "summary": ps.get("descriptionModule", {}).get("briefSummary", ""),
        "has_results": bool(study.get("hasResults")),
        "sponsor": ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
    }


def _design_label(rec):
    if rec["study_type"] == "OBSERVATIONAL":
        return "Observational study (registry)", 3 if rec["has_results"] else 4
    phase = "/".join(rec["phases"]) or "Interventional"
    randomized = rec["allocation"] == "RANDOMIZED"
    label = f"{phase} {'randomized ' if randomized else ''}trial".strip()
    if rec["has_results"]:
        return label + " (posted results)", 2 if randomized else 3
    return label + " (registration, no posted results)", 4


def _record_passage(rec):
    status = rec["status"].replace("_", " ").lower()
    parts = [f"Registry record ({status}{'; stopped: ' + rec['why_stopped'] if rec['why_stopped'] else ''})."]
    design = ", ".join(x for x in ("/".join(rec["phases"]), rec["allocation"].replace("_", " ").lower(),
                                   rec["masking"].lower() and f"masking: {rec['masking'].lower()}") if x)
    if design:
        parts.append(f"Design: {design}.")
    if rec["enrollment"]:
        parts.append(f"Enrollment: {rec['enrollment']}.")
    parts.append(f"Conditions: {', '.join(rec['conditions'][:5])}.")
    parts.append(f"Interventions: {'; '.join(rec['interventions'][:5])}.")
    if rec["primary_outcomes"]:
        parts.append(f"Primary outcome: {'; '.join(rec['primary_outcomes'][:3])}.")
    if rec["summary"]:
        parts.append(f"Summary: {rec['summary'][:700]}")
    return " ".join(parts)


def _format_measure(om):
    """One outcome measure as text: arm values and statistical analyses."""
    groups = {g["id"]: g.get("title", g["id"]) for g in om.get("groups", [])}
    denoms = {}
    for d in om.get("denoms", []):
        for c in d.get("counts", []):
            denoms[c.get("groupId")] = c.get("value")
    unit = om.get("unitOfMeasure", "")
    lines = [f"{om.get('type', '').title()} outcome: {om.get('title', '')} ({om.get('timeFrame', '')})."]
    values = []
    for cls in om.get("classes", [])[:3]:
        for cat in cls.get("categories", [])[:3]:
            for m in cat.get("measurements", []):
                arm = groups.get(m.get("groupId"), m.get("groupId"))
                n = f" (n={denoms[m.get('groupId')]})" if denoms.get(m.get("groupId")) else ""
                spread = ""
                if m.get("lowerLimit") or m.get("upperLimit"):
                    spread = f" [{m.get('lowerLimit', '')} to {m.get('upperLimit', '')}]"
                elif m.get("spread"):
                    spread = f" ± {m['spread']}"
                label = " ".join(x for x in (cls.get("title", ""), cat.get("title", "")) if x)
                values.append(f"{arm}{n}{' ' + label if label else ''}: {m.get('value', '')}{spread} {unit}".strip())
    if values:
        lines.append("Values: " + "; ".join(values[:8]) + ".")
    for a in om.get("analyses", [])[:2]:
        est = a.get("paramType", "")
        ci = (f", {a.get('ciPctValue', '95')}% CI {a['ciLowerLimit']} to {a['ciUpperLimit']}"
              if a.get("ciLowerLimit") and a.get("ciUpperLimit") else "")
        p = f", p {a['pValue']}" if a.get("pValue") and str(a["pValue"])[:1] in "<>=" else (f", p = {a['pValue']}" if a.get("pValue") else "")
        method = f" ({a['statisticalMethod']})" if a.get("statisticalMethod") else ""
        if a.get("paramValue"):
            lines.append(f"Analysis: {est} {a['paramValue']}{ci}{p}{method}.")
        elif p:
            lines.append(f"Analysis: {p.strip(', ')}{method}.")
    return " ".join(lines)


def fetch_results(nct, max_measures=3):
    """Text of the primary (then secondary) outcome measures with posted results."""
    data = get_json(f"{CTGOV}/{nct}", {"fields": "resultsSection.outcomeMeasuresModule"}, timeout=20)
    measures = ((data or {}).get("resultsSection", {}).get("outcomeMeasuresModule", {}).get("outcomeMeasures", []))
    measures = sorted(measures, key=lambda m: m.get("type") != "PRIMARY")[:max_measures]
    return [_format_measure(m) for m in measures if m.get("reportingStatus", "POSTED") == "POSTED"]


def _queries(plan):
    terms = plan.trial_terms or {}
    cond, intr, outc = terms.get("condition", ""), terms.get("intervention", ""), terms.get("outcome", "")
    queries = []
    if cond and intr and outc:
        queries.append({"query.cond": cond, "query.intr": intr, "query.outc": outc})
    if cond and intr:
        queries.append({"query.cond": cond, "query.intr": intr})
    if intr:
        queries.append({"query.intr": intr} if not cond else {"query.intr": intr, "query.term": cond.split(" OR ")[0]})
    if terms.get("term"):
        queries.append({"query.term": terms["term"]})
    return queries


def search_trials(plan, mode, query, query_vector, report=None):
    """Ranked ClinicalTrials.gov evidence items for ``plan``; returns (items, stats). Trials the question names
    (NCT numbers) are always among them."""
    from mes.tools.planner_tools import named_ids
    report = report or (lambda msg: None)
    studies, used = {}, []
    named = named_ids(f"{plan.question} {plan.standalone_question}")["ncts"][:5]
    for nct in named:
        try:
            data = get_json(f"{CTGOV}/{nct}", {"fields": "protocolSection,hasResults"}, timeout=20)
            rec = parse_study(data) if data else None
            if rec and rec["nct"]:
                studies[rec["nct"]] = rec
        except Exception as e:
            logger.info("Trial %s could not be fetched: %s", nct, e)
    for params in _queries(plan):
        data = get_json(CTGOV, {**params, "pageSize": 40, "fields": "protocolSection,hasResults", "countTotal": "true"},
                        timeout=20)
        used.append(params)
        for s in (data or {}).get("studies", []):
            rec = parse_study(s)
            if rec["nct"] and rec["nct"] not in studies:
                studies[rec["nct"]] = rec
        if len(studies) >= max(10, 2 * mode.trials_max):
            break
    if not studies:
        report("ClinicalTrials.gov: no matching trials")
        return [], {"n_found": 0, "queries": used}

    records = list(studies.values())
    texts = [f"{r['title']}. {_record_passage(r)}" for r in records]
    scores, vectors = hybrid_scores(query, texts, query_vector=query_vector, lexical_query=expanded_query(plan))
    dense = vectors @ query_vector
    priors = []
    for r in records:
        prior = 0.15 * r["has_results"] + 0.05 * (r["status"] == "COMPLETED")
        prior += 0.1 * any(p in ("Phase 3", "Phase 4") for p in r["phases"]) + 0.05 * (r["allocation"] == "RANDOMIZED")
        prior += 0.05 * min(1.0, np.log10((r["enrollment"] or 1)) / 4)
        priors.append(prior)
    final = 0.7 * scores + np.array(priors, dtype=np.float32)
    forced = [i for i, r in enumerate(records) if r["nct"] in named]
    keep = [i for i in range(len(records)) if dense[i] >= 0.25 and i not in forced] or \
        [i for i in range(len(records)) if i not in forced]
    room = max(0, mode.trials_max - len(forced))
    chosen = forced + ([keep[j] for j in mmr_select(vectors[keep], final[keep], room)] if keep and room else [])

    selected = [records[i] for i in chosen]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda r: fetch_results(r["nct"]) if r["has_results"] else [], selected))

    items = []
    for i, rec, res in zip(chosen, selected, results):
        design, level = _design_label(rec)
        passages = [Passage(text=_record_passage(rec), section="Registry record")]
        passages += [Passage(text=t, section="Posted results") for t in res]
        items.append(EvidenceItem(
            id=f"nct:{rec['nct']}", source_type="trial", title=rec["title"],
            url=f"https://clinicaltrials.gov/study/{rec['nct']}", year=rec["completion_year"] or rec["start_year"],
            venue="ClinicalTrials.gov", design=design, evidence_level=level, origin="ClinicalTrials.gov",
            identifiers={"nct": rec["nct"]}, passages=passages, score=float(final[i]),
            meta={k: rec[k] for k in ("status", "why_stopped", "phases", "allocation", "enrollment", "conditions",
                                      "interventions", "primary_outcomes", "has_results", "sponsor")}
                 | {"results": res, "relevance": round(float(dense[i]), 3)}))
    n_results = sum(1 for it in items if it.meta["results"])
    report(f"ClinicalTrials.gov: {len(records)} trials found, {len(items)} kept ({n_results} with posted results)")
    return items, {"n_found": len(records), "n_selected": len(items), "n_with_results": n_results, "queries": used}
