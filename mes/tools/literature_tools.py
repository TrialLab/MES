"""
LiteratureMiner tools — PubMed abstracts, open-access full text and PDFs.

* ``search_pubmed``: esearch with the precise query, a high-evidence query
  (systematic reviews, meta-analyses, RCTs, guidelines) and, when recall is
  low, the broad query; efetch returns abstracts, publication types, MeSH, DOI
  and PMCID. Study design and evidence level come from the publication types.
* ``add_fulltext``: for the best-ranked open-access articles, the Europe PMC
  JATS full text (sections and tables) or, as a fallback, the Unpaywall PDF.
* ``LocalPdfLibrary``: a folder of PDFs (``MES_PDF_DIR``) indexed once and
  searched for every question; ``search_uploaded_pdfs`` does the same for PDFs
  attached to a single question.

All functions are pure computation + HTTP (no LLM calls).
"""

import hashlib
import json
import logging
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date
from pathlib import Path

import numpy as np

from mes import config
from mes.tools.http_utils import get_json, http_get
from mes.tools.ranking_tools import chunk_text, encode, hybrid_scores, lexical_scores, mmr_select
from mes.tools.result_schema import EvidenceItem, Passage

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL = "https://api.unpaywall.org/v2"
# Recent trials are often not yet tagged as trials (ACORN, JAMA 2023, is only "Journal Article"): title words too
TRIAL_FILTER = ("(randomized controlled trial[pt] OR pragmatic clinical trial[pt] OR meta-analysis[pt] OR "
                "systematic review[pt] OR randomized[ti] OR randomised[ti] OR meta-analysis[ti])")
HIGH_EVIDENCE_FILTER = TRIAL_FILTER[:-1] + " OR practice guideline[pt] OR guideline[pt])"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126 Safari/537.36")


# ============================================================================ study design
# (publication types / MeSH that identify the design, label, evidence level 1 = strongest)
_DESIGNS = [
    ({"meta-analysis"}, "Meta-analysis", 1),
    ({"systematic review"}, "Systematic review", 1),
    ({"practice guideline", "guideline", "consensus development conference"}, "Guideline", 1),
    ({"randomized controlled trial", "equivalence trial", "pragmatic clinical trial"}, "Randomized controlled trial", 2),
    ({"clinical trial, phase iii", "clinical trial, phase iv", "controlled clinical trial", "clinical trial",
      "clinical trial, phase ii", "clinical trial, phase i"}, "Clinical trial", 2),
    ({"cohort studies", "prospective studies", "longitudinal studies"}, "Cohort study", 3),
    ({"case-control studies"}, "Case-control study", 3),
    ({"observational study", "retrospective studies", "cross-sectional studies", "multicenter study",
      "comparative study"}, "Observational study", 3),
    ({"review"}, "Narrative review", 4),
    ({"case reports"}, "Case report", 4),
    ({"editorial", "comment", "letter", "news", "published erratum"}, "Editorial / comment", 5),
]
DESIGN_PRIOR = {1: 1.0, 2: 0.85, 3: 0.6, 4: 0.35, 5: 0.15}


_TITLE_DESIGNS = [(re.compile(r"\bmeta-?analys[ie]s\b", re.I), "Meta-analysis", 1),
                  (re.compile(r"\bsystematic review\b", re.I), "Systematic review", 1),
                  (re.compile(r"\brandomi[sz]ed\b.*\btrial\b", re.I), "Randomized controlled trial", 2)]
_NOT_RESULTS = re.compile(r"\b(protocol|design and rationale|rationale and design|statistical analysis plan)\b", re.I)


def classify_design(publication_types, mesh_terms=(), title=""):
    """Study design label and evidence level from PubMed publication types and MeSH headings (and, for articles
    not yet indexed, from the title: "… The ACORN Randomized Clinical Trial")."""
    tags = {t.lower() for t in publication_types} | {m.lower() for m in mesh_terms}
    tagged = next(((label, level) for keys, label, level in _DESIGNS if tags & keys), None)
    # generic tags ("Multicenter Study", "Prospective Studies", "Review") also describe trials and meta-analyses:
    # the title wins over them, not over comments, letters or case reports
    if (tagged is None or tagged[0] in ("Cohort study", "Observational study", "Narrative review")) and title \
            and not _NOT_RESULTS.search(title):
        for pattern, label, level in _TITLE_DESIGNS:
            if pattern.search(title) and (tagged is None or level < tagged[1]):
                return label, level
    if tagged:
        return tagged
    mesh = {m.lower() for m in mesh_terms}
    if "animals" in mesh and "humans" not in mesh:
        return "Preclinical study", 5
    return "Primary study", 3


# ============================================================================ PubMed
def _eutils_params(**params):
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    if config.CONTACT_EMAIL:
        params["email"] = config.CONTACT_EMAIL
    params["tool"] = "mes"
    return params


def esearch(term, retmax=40):
    """PMIDs for ``term`` in PubMed relevance (Best Match) order."""
    if not term:
        return []
    data = get_json(f"{EUTILS}/esearch.fcgi", _eutils_params(db="pubmed", term=term, retmax=retmax,
                                                             sort="relevance", retmode="json"))
    return (data or {}).get("esearchresult", {}).get("idlist", [])


def _text(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _year(article):
    for path in (".//Article/Journal/JournalIssue/PubDate/Year", ".//Article/ArticleDate/Year",
                 ".//Article/Journal/JournalIssue/PubDate/MedlineDate"):
        m = re.search(r"(19|20)\d{2}", _text(article.find(path)))
        if m:
            return int(m.group(0))
    return None


def parse_pubmed_xml(xml_bytes):
    """Records (title, abstract sections, journal, year, publication types, MeSH, IDs) from efetch XML."""
    records = []
    root = ET.fromstring(xml_bytes)
    for article in root.findall("PubmedArticle"):
        citation = article.find("MedlineCitation")
        pmid = _text(citation.find("PMID"))
        abstract = [(el.get("Label") or "Abstract", _text(el)) for el in citation.findall(".//Abstract/AbstractText")]
        ids = {a.get("IdType"): _text(a) for a in article.findall(".//PubmedData/ArticleIdList/ArticleId")}
        first_author = citation.find(".//AuthorList/Author/LastName")
        records.append({
            "pmid": pmid,
            "title": _text(citation.find(".//Article/ArticleTitle")),
            "abstract": [(label.title(), text) for label, text in abstract if text],
            "journal": _text(citation.find(".//Article/Journal/ISOAbbreviation")) or _text(citation.find(".//Article/Journal/Title")),
            "year": _year(citation),
            "publication_types": [_text(p) for p in citation.findall(".//PublicationTypeList/PublicationType")],
            "mesh": [_text(d) for d in citation.findall(".//MeshHeadingList/MeshHeading/DescriptorName")],
            "doi": ids.get("doi", ""),
            "pmcid": ids.get("pmc", ""),
            "first_author": _text(first_author),
        })
    return records


def efetch(pmids):
    records = []
    for start in range(0, len(pmids), 100):
        batch = pmids[start:start + 100]
        result = http_get(f"{EUTILS}/efetch.fcgi", _eutils_params(db="pubmed", id=",".join(batch), retmode="xml"),
                          timeout=30)
        if result.ok:
            records += parse_pubmed_xml(result.content)
    return records


def citation_info(pmids):
    """{pmid: {"cited_by", "year", "open_access", "pmcid"}} from Europe PMC (parallel requests of 100 PMIDs)."""
    def lookup(batch):
        try:
            data = get_json(f"{EUROPE_PMC}/search", {"query": f"EXT_ID:({' OR '.join(batch)}) AND SRC:MED",
                                                      "resultType": "lite", "format": "json", "pageSize": 1000},
                            timeout=20)
        except Exception as e:
            logger.info("Europe PMC citation lookup failed: %s", e)
            return []
        return ((data or {}).get("resultList") or {}).get("result", [])

    batches = [pmids[start:start + 100] for start in range(0, len(pmids), 100)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = [a for found in pool.map(lookup, batches) for a in found]
    info = {}
    for a in results:
        if a.get("pmid"):
            year = a.get("pubYear") or ""
            info[a["pmid"]] = {"cited_by": int(a.get("citedByCount") or 0), "year": int(year) if year.isdigit() else None,
                               "open_access": a.get("isOpenAccess") == "Y", "pmcid": a.get("pmcid") or ""}
    return info


def _most_cited(pmids, citations, n):
    """The ``n`` PMIDs with the most citations per year since publication."""
    this_year = date.today().year
    per_year = {p: citations[p]["cited_by"] / max(1, this_year - (citations[p]["year"] or this_year) + 1)
                for p in pmids if p in citations and citations[p]["cited_by"]}
    return sorted(per_year, key=per_year.get, reverse=True)[:n]


def _record_to_item(rec, rank):
    design, level = classify_design(rec["publication_types"], rec["mesh"], rec.get("title", ""))
    passages = []
    for label, text in rec["abstract"]:
        # Large chunks keep an abstract section (e.g. all of "Findings") together with its numbers
        for chunk in chunk_text(text, size=1400, overlap=120):
            passages.append(Passage(text=chunk, section=f"Abstract — {label}" if label != "Abstract" else "Abstract"))
    return EvidenceItem(
        id=f"pmid:{rec['pmid']}", source_type="literature", title=rec["title"],
        url=f"https://pubmed.ncbi.nlm.nih.gov/{rec['pmid']}/", year=rec["year"], venue=rec["journal"],
        design=design, evidence_level=level, origin="PubMed",
        identifiers={k: v for k, v in (("pmid", rec["pmid"]), ("doi", rec["doi"]), ("pmcid", rec["pmcid"])) if v},
        passages=passages,
        meta={"publication_types": rec["publication_types"], "mesh": rec["mesh"][:12],
              "first_author": rec["first_author"], "pubmed_rank": rank,
              # the whole abstract stays available for number grounding after passages are trimmed
              "abstract": " ".join(text for _, text in rec["abstract"])},
    )


# PubMed's relevance order is not stable between requests (a pivotal trial can move from 1st to 60th), so the
# precise query (every concept) is screened deeper and its most cited studies always join the candidates
SCREEN_PMIDS = 200
TOP_CITED = 6


def _head_to_head(plan):
    """Trials and reviews of the intervention against the comparator without the population restriction: the
    main trial of a drug comparison is often in a broader population than the question's ("critically ill")."""
    from mes.tools.planner_tools import _group
    groups = {}
    for e in plan.entities:
        if e.get("role") in ("intervention", "comparator") and e["role"] not in groups:
            groups[e["role"]] = _group([e.get("name", "")] + list(e.get("synonyms") or []))
    if not groups.get("intervention") or not groups.get("comparator"):
        return ""
    return f"{groups['intervention']} AND {groups['comparator']} AND {TRIAL_FILTER}"


def _cited_query(plan):
    """Europe PMC query for the trials and meta-analyses of the question's population and treatment (names and
    synonyms, title or abstract), to be sorted by citations: PubMed's Best Match order is not stable and a query with
    many synonyms can push the pivotal trial out of its first hundreds of hits (EMPA-KIDNEY, 2026-10)."""
    groups = []
    for role in ("population", "intervention"):
        names = []
        for e in plan.entities:
            if e.get("role") == role and e.get("name"):
                names += [e["name"]] + list(e.get("synonyms") or [])[:3]
        names = [n.replace('"', "").strip() for n in dict.fromkeys(names) if n and len(n) > 2][:5]
        if names:
            groups.append("(" + " OR ".join(f'TITLE:"{n}" OR ABSTRACT:"{n}"' for n in names) + ")")
    if len(groups) < 2:
        return ""
    return " AND ".join(groups) + (' AND SRC:MED AND (PUB_TYPE:"Randomized Controlled Trial" OR '
                                   'PUB_TYPE:"Meta-Analysis" OR PUB_TYPE:"Systematic Review")')


def most_cited(query, n=10):
    """PMIDs of the most cited articles for a Europe PMC ``query``."""
    if not query:
        return []
    try:
        data = get_json(f"{EUROPE_PMC}/search", {"query": query, "sort": "CITED desc", "resultType": "lite",
                                                  "format": "json", "pageSize": n}, timeout=20)
    except Exception as e:
        logger.info("Europe PMC most-cited search failed: %s", e)
        return []
    return [a["pmid"] for a in ((data or {}).get("resultList") or {}).get("result", []) if a.get("pmid")]


def doi_to_pmid(dois):
    """PMIDs of DOIs (Europe PMC)."""
    pmids = []
    for doi in dois[:5]:
        try:
            data = get_json(f"{EUROPE_PMC}/search", {"query": f'DOI:"{doi}"', "resultType": "lite", "format": "json"},
                            timeout=15)
        except Exception as e:
            logger.info("DOI lookup failed for %s: %s", doi, e)
            continue
        found = [a.get("pmid") for a in ((data or {}).get("resultList") or {}).get("result", []) if a.get("pmid")]
        pmids += found[:1]
    return pmids


def search_pubmed(plan, mode, report=None):
    """PubMed search for ``plan``; returns (items with abstract passages, stats). Articles the question names
    (PMID, PubMed link, DOI) are always among them."""
    from mes.tools.planner_tools import named_ids
    report = report or (lambda msg: None)
    named = named_ids(f"{plan.question} {plan.standalone_question}")
    named_pmids = named["pmids"] + doi_to_pmid(named["dois"]) if named["dois"] else named["pmids"]
    queries = [("precise", plan.pubmed_query, max(SCREEN_PMIDS, mode.pubmed_max)),
               ("high-evidence", f"({plan.pubmed_query}) AND {HIGH_EVIDENCE_FILTER}" if plan.pubmed_query else "",
                max(10, mode.pubmed_max // 2))]
    for facet in plan.facets:
        queries.append((f"facet: {facet['name']}", facet.get("pubmed_query", ""), max(15, mode.pubmed_max // 2)))
    queries.append(("head-to-head trials", _head_to_head(plan), 10))
    queries = [q for q in queries if q[1]]

    cited_query = _cited_query(plan)
    with ThreadPoolExecutor(max_workers=5) as pool:
        cited_job = pool.submit(most_cited, cited_query)
        hits = dict(zip([q[0] for q in queries], pool.map(lambda q: esearch(q[1], q[2]), queries)))
        cited = cited_job.result()
    if cited:
        hits["most cited trials (Europe PMC)"] = cited
        queries.append(("most cited trials (Europe PMC)", cited_query, len(cited)))
    if named_pmids:
        hits["named in the question"] = named_pmids
    screened = hits.get("precise", [])
    if screened:
        hits["precise"] = screened[:mode.pubmed_max]
    # Low recall: relax the query tier by tier (drop outcomes, then secondary population concepts)
    found = set(hits.get("precise", []))
    for tier, query in enumerate((plan.pubmed_tiers or [plan.pubmed_query_broad])[1:], start=1):
        if len(found) >= 15 or not query or query == plan.pubmed_query:
            continue
        name = f"relaxed-{tier}"
        hits[name] = esearch(query, mode.pubmed_max)
        queries.append((name, query, mode.pubmed_max))
        found |= set(hits[name])

    rank = {}
    facet_of = {}
    for name, ids in hits.items():
        for r, pmid in enumerate(ids):
            rank[pmid] = min(rank.get(pmid, r), r)
            if name.startswith("facet: "):
                facet_of.setdefault(pmid, name[7:])
    pmids = sorted(rank, key=rank.get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records_job = pool.submit(efetch, pmids)
        citations_job = pool.submit(citation_info, list(dict.fromkeys(pmids + screened)))
        records, citations = records_job.result(), citations_job.result()
    top_cited = set(_most_cited(list(dict.fromkeys(screened + cited)), citations, TOP_CITED))
    extra = [p for p in screened if p in top_cited and p not in rank]
    if extra:
        records += efetch(extra)
        for p in extra:
            rank[p] = screened.index(p)
    report(f"PubMed: {len(pmids)} unique articles from {len(hits)} queries"
           + (f" + {len(extra)} highly cited" if extra else ""))
    items = _dedupe([_record_to_item(r, rank.get(r["pmid"], 99)) for r in records
                     if r["title"] and (r["abstract"] or r["pmid"] in named_pmids)])
    for item in items:
        pmid = item.identifiers["pmid"]
        if pmid in top_cited:
            item.meta["top_cited"] = True
        if pmid in named_pmids:
            item.meta["named"] = True
        if pmid in facet_of:
            item.meta["facet"] = facet_of[pmid]
        cite = citations.get(pmid)
        if cite:
            item.meta["cited_by"] = cite["cited_by"]
            item.meta["open_access"] = cite["open_access"]
            if cite["pmcid"] and not item.identifiers.get("pmcid"):
                item.identifiers["pmcid"] = cite["pmcid"]
    stats = {"queries": {name: q for name, q, _ in queries}, "n_hits": {k: len(v) for k, v in hits.items()},
             "n_found": len(pmids) + len(extra), "n_top_cited_added": len(extra), "n_with_abstract": len(items)}
    return items, stats


def _dedupe(items):
    """One record per article: guidelines and statements are often co-published in several journals."""
    best = {}
    for item in items:
        key = re.sub(r"[^a-z0-9]+", " ", item.title.lower()).strip()[:150]
        if key not in best or item.meta["pubmed_rank"] < best[key].meta["pubmed_rank"]:
            best[key] = item
    kept = {id(it) for it in best.values()}
    return [it for it in items if id(it) in kept]


# ============================================================================ ranking of documents
def rank_documents(items, query, query_vector, k, passages_per_doc=2, facets=None, lexical_query=None):
    """
    Score passages, keep the best ``passages_per_doc`` per item and select ``k`` diverse items.
    Item score = passage relevance + priors: PubMed Best Match rank (a learned ranker that knows synonyms
    and drug codes), study design, citations per year (landmark studies) and recency.
    """
    if not items:
        return []
    flat = [(i, p) for i, item in enumerate(items) for p in item.passages]
    scores, vectors = hybrid_scores(query, [f"{items[i].title}. {p.text}" for i, p in flat], query_vector=query_vector,
                                    lexical_query=lexical_query)
    dense = vectors @ query_vector
    best = np.full(len(items), -1.0, dtype=np.float32)
    best_dense = np.zeros(len(items), dtype=np.float32)
    item_vectors = np.zeros((len(items), vectors.shape[1]), dtype=np.float32)
    for j, (i, p) in enumerate(flat):
        p.score = float(scores[j]) + passage_boost(p)   # within-document choice favours results
        if scores[j] > best[i]:
            best[i], item_vectors[i] = scores[j], vectors[j]
        best_dense[i] = max(best_dense[i], dense[j])
    best = np.maximum(best, 0)

    this_year = date.today().year
    for i, item in enumerate(items):
        recency = max(0.0, 1 - (this_year - (item.year or this_year - 15)) / 15)
        rank_prior = 1 / (1 + item.meta.get("pubmed_rank", 50) / 10)
        # Landmark studies: citations per year since publication, saturating around 300/year
        age = max(1, this_year - (item.year or this_year) + 1)
        item.meta["cites_per_year"] = round(item.meta.get("cited_by", 0) / age, 1)
        impact = min(1.0, np.log10(1 + item.meta["cites_per_year"]) / 2.5)
        item.meta["impact"] = round(float(impact), 3)
        item.score = float(0.52 * best[i] + 0.13 * DESIGN_PRIOR.get(item.evidence_level, 0.3)
                           + 0.12 * impact + 0.15 * rank_prior + 0.08 * recency)
        item.meta["relevance"] = round(float(best_dense[i]), 3)
        item.passages = sorted(item.passages, key=lambda p: p.score, reverse=True)[:passages_per_doc]

    # Off-topic hits (low absolute similarity) are dropped before selection; articles the question names stay
    named = [i for i in range(len(items)) if items[i].meta.get("named")]
    keep = [i for i in range(len(items)) if best_dense[i] >= 0.25 and i not in named] or \
        [i for i in range(len(items)) if i not in named]
    # Landmark slots: the most cited trials / reviews / guidelines on the topic are always considered
    # (pivotal trials often name drugs by code, e.g. LCZ696, which similarity models miss)
    # (ordered by citations weighted by relevance: the pivotal trial of the question before a famous trial of
    # the same drug in another disease)
    landmarks = sorted((i for i in keep if items[i].evidence_level <= 2 and items[i].meta["impact"] >= 0.6
                        and best_dense[i] >= 0.45
                        and (items[i].meta.get("pubmed_rank", 99) < 30 or items[i].meta.get("top_cited"))),
                       key=lambda i: items[i].meta["cites_per_year"] * float(best_dense[i]) ** 2,
                       reverse=True)[:max(1, k // 4)]
    for i in landmarks:
        items[i].meta["landmark"] = True
    rest = [i for i in keep if i not in landmarks]
    chosen = named + landmarks + ([rest[j] for j in mmr_select(item_vectors[rest],
                                                               np.array([items[i].score for i in rest]),
                                                               k - len(landmarks) - len(named))]
                                  if rest and k > len(landmarks) + len(named) else [])
    if facets:  # Snow: make sure every facet keeps its best article
        for facet in facets:
            members = [i for i in keep if items[i].meta.get("facet") == facet["name"] and i not in chosen]
            if members and not any(items[i].meta.get("facet") == facet["name"] for i in chosen):
                chosen.append(max(members, key=lambda i: items[i].score))
    # articles the question names, then landmark studies: a concise answer is written from the first sources
    return sorted((items[i] for i in chosen),
                  key=lambda it: (not it.meta.get("named"), not it.meta.get("landmark"), -it.score))


# ============================================================================ full text
_SECTION_BOOST = [(("result", "outcome", "finding", "table"), 0.15), (("discussion", "conclusion"), 0.05),
                  (("introduction", "background"), -0.1)]
# Boilerplate sections that never carry evidence
_SKIP_SECTIONS = re.compile(r"fund|acknowledg|conflict|competing|disclos|author.{0,5}contribution|data availab|"
                            r"ethic|abbreviation|supplement|reference|copyright|consent|declaration|orcid", re.I)


_ABSTRACT_BOOST = [(("result", "finding", "outcome"), 0.2), (("conclusion", "interpretation"), 0.08),
                   (("background", "introduction", "objective", "aim", "purpose", "importance", "context",
                     "rationale"), -0.15)]
# Effect estimates and numeric results ("HR 0.79", "95% CI", "p<0.001", "12.4%")
_EFFECT = re.compile(r"\b(?:a?HR|a?OR|a?RR|IRR|SMD|WMD|NNT|hazard ratio|odds ratio|risk ratio|relative risk|"
                     r"95\s?%|CI\b|p\s?[<=>]\s?0?\.\d)|\d+(?:\.\d+)?\s?%", re.I)


def _section_boost(section):
    s = section.lower()
    return next((boost for keys, boost in _SECTION_BOOST if any(k in s for k in keys)), 0.0)


def passage_boost(passage):
    """Prior for choosing passages within a document: results and effect estimates over background."""
    s = passage.section.lower()
    boost = next((b for keys, b in _ABSTRACT_BOOST if any(k in s for k in keys)), 0.0)
    return boost + min(0.15, 0.04 * len(_EFFECT.findall(passage.text)))


def parse_jats(xml_bytes):
    """(section, text) blocks of a JATS article body, with tables rendered as rows."""
    root = ET.fromstring(xml_bytes)
    blocks = []
    body = root.find(".//body")
    if body is not None:
        def walk(sec, title):
            own = _text(sec.find("title")) or title
            if own and _SKIP_SECTIONS.search(own):
                return
            paragraphs = [_text(p) for p in sec.findall("p")]
            if any(paragraphs):
                blocks.append((own or "Body", " ".join(p for p in paragraphs if p)))
            for child in sec.findall("sec"):
                walk(child, own)
        for sec in body.findall("sec"):
            walk(sec, "")
        loose = [_text(p) for p in body.findall("p")]
        if any(loose):
            blocks.insert(0, ("Body", " ".join(p for p in loose if p)))
    for table in root.iter("table-wrap"):
        label = _text(table.find("label")) or "Table"
        caption = _text(table.find("caption"))
        rows = [" | ".join(_text(c) for c in tr if c.tag in ("td", "th")) for tr in table.iter("tr")]
        text = f"{caption}\n" + "\n".join(r for r in rows if r.strip(" |"))
        blocks.append((label, text[:3000]))
    return blocks


def pdf_text(data, max_pages=40):
    """Plain text of a PDF (bytes), page by page."""
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [doc[i].get_text("text") for i in range(min(len(doc), max_pages))]
        title = (doc.metadata or {}).get("title", "")
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate line breaks
    text = re.sub(r"(?<![.\n])\n(?!\n)", " ", text)     # join wrapped lines
    return text, title


_fulltext_lock = threading.Lock()


def _fulltext_cache(key, value=None):
    path = config.CACHE_DIR / "fulltext" / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if value is None:
        return json.loads(path.read_text()) if path.exists() else None
    with _fulltext_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return value


def fetch_fulltext_blocks(item):
    """(origin, blocks) of an article's open-access full text, or (None, [])."""
    pmcid, doi = item.identifiers.get("pmcid"), item.identifiers.get("doi")
    key = pmcid or doi
    if not key:
        return None, []
    cached = _fulltext_cache(key)
    if cached is not None:
        return cached["origin"], [tuple(b) for b in cached["blocks"]]

    origin, blocks, transient = None, [], False
    # Europe PMC serves the XML of open-access articles only (others answer HTTP 500 after the retries)
    if pmcid and item.meta.get("open_access") is not False:
        try:
            result = http_get(f"{EUROPE_PMC}/{pmcid}/fullTextXML", timeout=20)
            if result.ok:
                origin, blocks = "Europe PMC full text", parse_jats(result.content)
        except Exception as e:
            transient = True
            logger.info("Europe PMC full text failed for %s: %s", pmcid, e)
    if not blocks and doi and config.CONTACT_EMAIL:
        try:
            meta = get_json(f"{UNPAYWALL}/{doi}", {"email": config.CONTACT_EMAIL}, timeout=15)
            pdf_url = ((meta or {}).get("best_oa_location") or {}).get("url_for_pdf")
            if pdf_url:
                pdf = http_get(pdf_url, timeout=30, cache=False, headers={"User-Agent": BROWSER_UA})
                if pdf.ok and pdf.content[:5] == b"%PDF-":
                    text, _ = pdf_text(pdf.content)
                    origin, blocks = "Open-access PDF", [("Full text (PDF)", text)]
        except Exception as e:
            transient = True
            logger.info("Unpaywall PDF failed for %s: %s", doi, e)
    if blocks or not transient:  # network errors are retried next time instead of cached as "no full text"
        _fulltext_cache(key, {"origin": origin, "blocks": blocks})
    return origin, blocks


# The answer waits this long for full texts; slower ones finish in the background and are cached for next time
FULLTEXT_BUDGET_S = 8
# Passages of one full text scored with the dense model (conference supplements have thousands)
MAX_FULLTEXT_PASSAGES = 300


def _fetched(future):
    try:
        return future.result()
    except Exception as e:
        logger.info("Full text could not be read: %s", e)
        return None, []


def add_fulltext(items, query, query_vector, max_articles=3, passages_per_doc=2, report=None, lexical_query=None):
    """Replace abstract passages of the best open-access items by their most relevant full-text passages."""
    report = report or (lambda msg: None)
    # Europe PMC XML for open-access records; other articles only through an Unpaywall PDF (needs MES_CONTACT_EMAIL)
    candidates = [it for it in items if (it.identifiers.get("pmcid") and it.meta.get("open_access") is not False)
                  or (it.identifiers.get("doi") and config.CONTACT_EMAIL)]
    candidates = sorted(candidates, key=lambda it: not it.meta.get("open_access", bool(it.identifiers.get("pmcid"))))
    candidates = candidates[:max_articles * 2]
    pool = ThreadPoolExecutor(max_workers=6)
    futures = [pool.submit(fetch_fulltext_blocks, it) for it in candidates]
    done, late = wait(futures, timeout=FULLTEXT_BUDGET_S)
    pool.shutdown(wait=False, cancel_futures=True)
    fetched = [_fetched(f) if f in done else (None, []) for f in futures]
    n_ok = 0
    for item, (origin, blocks) in zip(candidates, fetched):
        if not blocks or n_ok >= max_articles:
            continue
        passages = [Passage(text=chunk, section=section) for section, text in blocks
                    for chunk in chunk_text(text, size=900, overlap=120)]
        if not passages:
            continue
        if len(passages) > MAX_FULLTEXT_PASSAGES:  # cheap lexical pre-filter before the dense model
            lexical = lexical_scores(lexical_query or query, [p.text for p in passages])
            passages = [passages[i] for i in sorted(np.argsort(-lexical)[:MAX_FULLTEXT_PASSAGES])]
        scores, _ = hybrid_scores(query, [f"{item.title}. {p.text}" for p in passages], query_vector=query_vector,
                                  lexical_query=lexical_query)
        for p, s in zip(passages, scores):
            p.score = float(s) + _section_boost(p.section) + min(0.15, 0.04 * len(_EFFECT.findall(p.text)))
        best_ft = sorted(passages, key=lambda p: p.score, reverse=True)[:passages_per_doc]
        # keep the best abstract passage (study summary) plus the best full-text passages
        item.passages = item.passages[:1] + best_ft
        item.full_text, item.origin = True, f"PubMed + {origin}"
        n_ok += 1
    report(f"Full text: {n_ok} open-access articles read"
           + (f" ({len(late)} too slow, skipped for this answer)" if late else ""))
    return n_ok


# ============================================================================ PDFs (local library and uploads)
def _pdf_items(docs, query, query_vector, k, passages_per_doc, origin):
    """Rank passages of parsed PDFs and return the best ``k`` documents as evidence items."""
    passages, owner = [], []
    for i, doc in enumerate(docs):
        for chunk in doc["chunks"]:
            passages.append(chunk)
            owner.append(i)
    if not passages:
        return []
    scores, vectors = hybrid_scores(query, passages, query_vector=query_vector)
    dense = vectors @ query_vector
    per_doc = {}
    for j, i in enumerate(owner):
        boost = min(0.15, 0.04 * len(_EFFECT.findall(passages[j])))
        per_doc.setdefault(i, []).append((float(scores[j]) + boost, float(dense[j]), passages[j]))
    items = []
    for i, rows in per_doc.items():
        rows.sort(reverse=True)
        if max(r[1] for r in rows) < 0.2:
            continue
        doc = docs[i]
        summary = doc.get("summary", "")
        passages = [Passage(text=summary, section="Summary", score=rows[0][0])] if summary else []
        passages += [Passage(text=t, section="PDF", score=s) for s, _, t in rows
                     if not (summary and summary[:200] in t)][:passages_per_doc - len(passages)]
        items.append(EvidenceItem(
            id=f"pdf:{doc['name']}", source_type="literature", title=doc["title"] or doc["name"],
            design="Document (PDF)", evidence_level=3, origin=origin, full_text=True,
            identifiers={"file": doc["name"]}, passages=passages,
            score=rows[0][0], meta={"relevance": round(max(r[1] for r in rows), 3)}))
    return sorted(items, key=lambda it: it.score, reverse=True)[:k]


# Bump when PDF parsing changes, so library PDFs are indexed again
PDF_PARSE_VERSION = 2
_REFERENCES = re.compile(r"(?:^|\n|\s)(?:References|REFERENCES|Bibliography|Literature Cited)\s*(?:\n|1\.\s|\[1\])")
# Affiliations, e-mail addresses and the publisher sidebar (citation, dates, copyright, funding)
_BOILERPLATE = re.compile(r"Department of|University|Hospital|Institute|School of|\S+@\S+\.\w+|Copyright|Received:|"
                          r"Accepted:|Published:|Funding:|Competing interests|Data Availability|Academic Editor|"
                          r"Creative Commons|licen[cs]e")


def _strip_references(text):
    """Drop the reference list: titles of cited papers match the question and crowd out the paper's own results."""
    starts = [m.start() for m in _REFERENCES.finditer(text)]
    return text[:starts[-1]] if starts and starts[-1] > 0.5 * len(text) else text


def _pdf_summary(text):
    """
    The paper's own conclusion — the abstract's Conclusions / Interpretation heading, else the discussion's
    closing "In conclusion / In summary" — shown first like the abstract of a PubMed record: passages ranked by
    similarity to the question tend to be methods text, which names the same concepts as the question.
    """
    match = re.search(r"\b(?:Conclusions?|Interpretation)\b", text[: int(0.3 * len(text))])
    if match is None:
        closing = list(re.finditer(r"\bIn (?:conclusion|summary)\b", text))
        match = closing[-1] if closing else None
    return chunk_text(text[match.start(): match.start() + 1400], size=900, overlap=0)[0] if match else ""


def _parse_pdf(name, data):
    text, title = pdf_text(data)
    first_line = next((ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20), "")
    text = _strip_references(text)
    chunks = [c for c in chunk_text(text, size=900, overlap=150) if len(_BOILERPLATE.findall(c)) < 3]
    return {"name": name, "title": title or first_line[:200], "chunks": chunks, "summary": _pdf_summary(text)}


def search_uploaded_pdfs(files, query, query_vector, k=4, passages_per_doc=2, report=None):
    """
    Evidence items from PDFs attached to one question; ``files`` is a list of (name, bytes).
    The user attached these on purpose, so each gets two more passages than a search hit.
    ``report`` hears about files that could not be read (not a PDF, damaged, scanned without text).
    """
    docs = []
    for name, data in files or []:
        try:
            doc = _parse_pdf(name, data)
        except Exception as e:
            logger.warning("Could not parse uploaded PDF %s: %s", name, e)
            if report:
                report(f"Could not read the uploaded file {name} (not a readable PDF)")
            continue
        if report and not doc["chunks"]:
            report(f"The uploaded file {name} has no readable text (a scanned PDF?)")
        docs.append(doc)
    return _pdf_items(docs, query, query_vector, k, passages_per_doc + 2, origin="Uploaded PDF")


class LocalPdfLibrary:
    """
    PDF folder (``MES_PDF_DIR``) searched for every question.

    Each file is parsed and embedded once; the result is cached per file in
    ``data/pdf_index/<sha1>.json|.npy`` (keyed by path, size and mtime), so
    adding a PDF only indexes that PDF. Indexing runs in a background thread:
    searches use the documents indexed so far and never wait for the folder.
    """

    def __init__(self, pdf_dir=None, index_dir=None):
        self.pdf_dir = Path(pdf_dir or config.PDF_DIR)
        self.index_dir = Path(index_dir or config.DATA_DIR / "pdf_index")
        self._lock = threading.Lock()
        self._worker = None
        self._signature = None
        self.n_files = 0
        # Search snapshot: documents, flattened chunks, owner document per chunk, chunk embeddings
        self._snapshot = ([], [], np.zeros(0, dtype=np.int32), None)

    def _files(self):
        return sorted(self.pdf_dir.rglob("*.pdf")) if self.pdf_dir.exists() else []

    @staticmethod
    def _file_key(f):
        st = f.stat()
        return hashlib.sha1(f"{f}:{st.st_size}:{st.st_mtime}:{config.ENCODER}:{PDF_PARSE_VERSION}".encode()).hexdigest()

    def _load_or_index(self, f):
        key = self._file_key(f)
        meta_path, vec_path = self.index_dir / f"{key}.json", self.index_dir / f"{key}.npy"
        if meta_path.exists() and vec_path.exists():
            return json.loads(meta_path.read_text()), np.load(vec_path)
        doc = _parse_pdf(f.name, f.read_bytes())
        doc["path"] = str(f)
        vectors = encode(doc["chunks"]) if doc["chunks"] else np.zeros((0, 1), dtype=np.float32)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, vectors)
        meta_path.write_text(json.dumps(doc))
        return doc, vectors

    def _index(self, files):
        docs, vectors = [], []
        for f in files:
            try:
                doc, vec = self._load_or_index(f)
            except Exception as e:
                logger.warning("Skipping unreadable PDF %s: %s", f, e)
                continue
            if doc["chunks"]:
                docs.append(doc)
                vectors.append(vec)
                self._publish(docs, vectors)
        self._publish(docs, vectors)

    def _publish(self, docs, vectors):
        chunks = [c for d in docs for c in d["chunks"]]
        owner = np.array([i for i, d in enumerate(docs) for _ in d["chunks"]], dtype=np.int32)
        matrix = np.vstack(vectors) if vectors else None
        self._snapshot = (list(docs), chunks, owner, matrix)

    def refresh(self):
        """Start (re)indexing in the background when the folder changed; returns (indexed, total) files."""
        files = self._files()
        signature = hashlib.sha1("|".join(self._file_key(f) for f in files).encode()).hexdigest() if files else ""
        with self._lock:
            self.n_files = len(files)
            if signature != self._signature and not (self._worker and self._worker.is_alive()):
                self._signature = signature
                if files:
                    self._worker = threading.Thread(target=self._index, args=(files,), daemon=True)
                    self._worker.start()
                else:
                    self._publish([], [])
        return len(self._snapshot[0]), self.n_files

    @property
    def indexing(self):
        return bool(self._worker and self._worker.is_alive())

    def search(self, query, query_vector, k=4, passages_per_doc=2):
        self.refresh()
        docs, chunks, owner, matrix = self._snapshot
        if not docs or matrix is None:
            return []
        # Pre-filter with the stored embeddings, then score the shortlisted chunks with the hybrid ranker
        top = np.argsort(-(matrix @ query_vector))[:200]
        shortlist = {}
        for j in top:
            shortlist.setdefault(int(owner[j]), []).append(chunks[j])
        candidates = [{**docs[i], "chunks": cs} for i, cs in shortlist.items()]
        return _pdf_items(candidates, query, query_vector, k, passages_per_doc, origin="Local PDF library")


_library = None
_library_lock = threading.Lock()


def get_pdf_library():
    """Process-wide PDF library (shared by all UI sessions)."""
    global _library
    with _library_lock:
        if _library is None:
            _library = LocalPdfLibrary()
    return _library
