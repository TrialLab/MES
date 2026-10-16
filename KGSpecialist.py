"""
KG-Specialist: retrieval of structured biomedical knowledge from a knowledge graph.

The knowledge graph is a table of triplets (head, relation, tail, source); the demo uses a
small subset of iBKH (Data/KG/example_kg.csv). Two retrieval modes are supported:
  * structured retrieval: biomedical entities in the question are mapped to KG nodes and
    their 1-hop relations and 2-hop paths between mapped entities (multi-hop reasoning)
    are retrieved from the graph, then ranked by semantic similarity to the question;
  * vector retrieval: all triplets are embedded with a biomedical sentence encoder
    (embeddinggemma-300m-medical by default) and the top-k triplets by cosine similarity
    to the question are returned.
Structured retrieval is used when entities can be mapped to KG nodes, otherwise vector
retrieval. Retrieved triplets are assessed for reliability by combining an LLM
assessment of source quality with the consistency of relations across retrieved triplets.
"""

import json
import logging
import re
from itertools import combinations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from llm_zoo import llm_map
from utils import parse_json_response

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "sentence-transformers/embeddinggemma-300m-medical"

# Relation pairs that contradict each other when asserted for the same pair of entities.
CONTRADICTORY_RELATIONS = [({"treats", "palliates"}, {"causes"}), ({"upregulates"}, {"downregulates"})]


def _normalize(name):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split())


class KGSpecialist:
    def __init__(self, triplet_csv="Data/KG/example_kg.csv", encoder_name=DEFAULT_ENCODER):
        self.triplets = pd.read_csv(triplet_csv)
        self.triplets["text"] = self.triplets["head"] + " " + self.triplets["relation"] + " " + self.triplets["tail"]
        self.node_names = {}
        for name in pd.concat([self.triplets["head"], self.triplets["tail"]]).unique():
            self.node_names.setdefault(_normalize(name), name)
        logger.info("Embedding %d KG triplets with %s.", len(self.triplets), encoder_name)
        self.encoder = SentenceTransformer(encoder_name)
        self.embeddings = self._encode(self.triplets["text"].tolist(), is_query=False)

    def _encode(self, texts, is_query):
        # Queries and triplets are encoded with the query and document prompts of the encoder.
        encode = self.encoder.encode_query if is_query else self.encoder.encode_document
        return encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def _record(self, rows, score, retrieval):
        """One retrieved item: a single triplet or a multi-hop path made of several triplets."""
        edges = self.triplets.iloc[rows]
        return {"source": "; ".join(edges["source"]), "text": "; ".join(edges["text"]),
                "edges": edges[["head", "relation", "tail"]].values.tolist(),
                "retrieval": retrieval, "similarity": round(float(score), 4)}

    def _neighbors(self, node, scores):
        """For each neighbor of ``node``: the row of the most question-relevant triplet linking them."""
        best = {}
        for i in np.flatnonzero((self.triplets["head"] == node) | (self.triplets["tail"] == node)):
            row = self.triplets.iloc[i]
            neighbor = row["tail"] if row["head"] == node else row["head"]
            if neighbor not in best or scores[i] > scores[best[neighbor]]:
                best[neighbor] = i
        return best

    def extract_entities(self, question, llm):
        prompt = f"""
    Extract the biomedical entities (drugs, diseases, conditions, genes, pathways, symptoms, side effects)
    mentioned in the question below. Use standard full names (e.g., "type 2 diabetes mellitus" rather than "T2DM").
    Do not include outcomes such as mortality, settings such as ICU, or study designs.

    Question: {question}

    Return ONLY a JSON list of strings.
    """
        entities = parse_json_response(llm.invoke(prompt), default=[])
        return [e for e in entities if isinstance(e, str) and e.strip()] if isinstance(entities, list) else []

    def map_entities(self, entities):
        """Map entity names to KG node names (exact match after normalization, then token containment)."""
        mapping = {}
        for entity in entities:
            key = _normalize(entity)
            if key in self.node_names:
                mapping[entity] = self.node_names[key]
                continue
            tokens = set(key.split())
            candidates = [name for norm, name in self.node_names.items()
                          if tokens and tokens <= set(norm.split()) and len(norm.split()) - len(tokens) <= 1]
            if candidates:
                mapping[entity] = min(candidates, key=len)
        return mapping

    def retrieve(self, question, llm, top_k=100, max_paths=20):
        """
        Retrieve up to ``top_k`` triplets (and up to ``max_paths`` 2-hop paths in structured mode) for a
        question; returns (list of records ranked by similarity to the question, retrieval info).
        """
        scores = self.embeddings @ self._encode([question], is_query=True)[0]
        entities = self.extract_entities(question, llm)
        mapping = self.map_entities(entities)
        info = {"entities": entities, "mapped_nodes": mapping}

        if mapping:
            info["mode"] = "structured"
            nodes = set(mapping.values())
            one_hop = np.flatnonzero(self.triplets["head"].isin(nodes) | self.triplets["tail"].isin(nodes))
            one_hop = one_hop[np.argsort(-scores[one_hop])][:top_k]
            results = [self._record([i], scores[i], "structured: 1-hop") for i in one_hop]
            paths = []
            for a, b in combinations(sorted(nodes), 2):
                via_a, via_b = self._neighbors(a, scores), self._neighbors(b, scores)
                for mid in (via_a.keys() & via_b.keys()) - {a, b}:
                    ia, ib = via_a[mid], via_b[mid]
                    paths.append(self._record([ia, ib], (scores[ia] + scores[ib]) / 2, f"structured: 2-hop path via {mid}"))
            paths = sorted(paths, key=lambda r: r["similarity"], reverse=True)[:max_paths]
            info["n_paths"] = len(paths)
            results += paths
        else:
            info["mode"] = "vector"
            results = [self._record([i], scores[i], "vector") for i in np.argsort(-scores)[:top_k]]
        results = sorted(results, key=lambda r: r["similarity"], reverse=True)
        logger.info("KG retrieval mode: %s; mapped nodes: %s", info["mode"], mapping)
        return results, info


def assess_triplet_quality(triplets: list, llm, max_workers=8) -> list:
    """LLM assessment of the quality of each triplet given its text and source database."""
    template = """
    You are an expert in the clinical domain. Based on the triplet from a clinical knowledge graph,
    assess the quality of the triplet. Consider the triplet's text and source.

    Return your assessment as a JSON object with two keys:
    "quality": Classify as "Low", "Moderate", "High", or "Unclear".
    "reasoning": A brief, one-sentence explanation for your classification.

    Your response MUST be only the valid JSON object.

    TRIPLET INFORMATION:
    ---
    {triplet_info}
    ---
    """

    def _assess(triplet):
        result = parse_json_response(llm.invoke(template.format(
            triplet_info=json.dumps({k: triplet[k] for k in ("source", "text")}, indent=2))))
        if not isinstance(result, dict) or "quality" not in result:
            result = {"quality": "Unclear", "reasoning": "The quality assessment could not be parsed."}
        return {**triplet, "quality_assessment": result}

    return llm_map(_assess, triplets, max_workers=max_workers)


def assess_triplet_consistency(triplets: list, retrieved: list = None) -> list:
    """
    Add a reliability label combining source quality with consistency across all retrieved
    triplets and paths (``retrieved``, default: ``triplets``): an item is downgraded when one of
    its entity pairs is also linked by a contradictory relation (e.g., 'treats' vs 'causes',
    'upregulates' vs 'downregulates').
    """
    edges = {tuple(e) for t in (retrieved or triplets) + triplets for e in t["edges"]}
    pair_relations = {}
    for head, rel, tail in edges:
        pair_relations.setdefault(frozenset((head.lower(), tail.lower())), set()).add(rel)
    quality_score = {"high": 3, "moderate": 2, "low": 1}
    for t in triplets:
        support, contradicted = [], False
        for head, rel, tail in t["edges"]:
            relations = pair_relations[frozenset((head.lower(), tail.lower()))]
            support.append(len(relations))
            contradicted |= any(rel in a and relations & b or rel in b and relations & a for a, b in CONTRADICTORY_RELATIONS)
        score = quality_score.get(str(t.get("quality_assessment", {}).get("quality", "")).lower(), 1)
        score = score - 1 if contradicted else score
        # For a path, support is that of its least supported edge.
        t["consistency"] = {"supporting_relations": min(support), "contradicted": contradicted}
        t["reliability"] = "High" if score >= 3 else "Moderate" if score == 2 else "Low"
    return triplets
