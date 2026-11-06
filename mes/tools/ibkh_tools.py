"""
KG-Specialist tools — structured and semantic retrieval from iBKH.

The store is built once by ``scripts/build_ibkh_kg.py`` (DuckDB + node-name
embeddings) and opened read-only; every query uses its own cursor, so the graph
is shared safely by parallel requests.

Retrieval for a question:
  1. Entity linking: exact name → token-subset match → nearest node embedding
     (planner entities and synonyms; n-gram matching of the question as fallback).
  2. Structured retrieval: direct relations between the question's entities,
     2-hop paths A–X–B, and the highest-weighted 1-hop relations of each entity.
  3. Semantic retrieval: nodes closest to the question embedding (e.g. an outcome
     the planner did not list) linked to the question's entities; all candidate
     relations are re-ranked by dense similarity to the question.
  4. Reliability: relation class (curated > literature-mined > inferred),
     downgraded when the same pair has contradictory relations.

KG relations are mechanistic / contextual information, not evidence of a
clinical effect; the synthesis prompt treats them that way.
"""

import json
import logging
import re
import threading
from collections import defaultdict
from itertools import combinations

import numpy as np

from mes import config
from mes.tools.ranking_tools import encode
from mes.tools.result_schema import EvidenceItem, Passage

logger = logging.getLogger(__name__)

CONTRADICTORY = [({"treats", "palliates"}, {"causes"}), ({"upregulates"}, {"downregulates"})]
CLASS_RELIABILITY = {"curated": "High", "literature-mined": "Moderate", "inferred": "Low"}
CLASS_LABEL = {"curated": "Curated relation", "literature-mined": "Literature-mined relation (SemMedDB)",
               "inferred": "Inferred relation (CTD)"}
_DOWNGRADE = {"High": "Moderate", "Moderate": "Low", "Low": "Low"}
_GENERIC = set("disease diseases syndrome disorder patients patient adult adults child children human humans "
               "drug drugs therapy treatment mortality death risk outcome outcomes effect cell cells protein "
               "gene genes male female".split())


def _normalize(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


class IBKHGraph:
    """Read-only iBKH store with entity linking, structured and semantic retrieval."""

    def __init__(self, kg_dir=None):
        import duckdb
        self.kg_dir = config.IBKH_DIR if kg_dir is None else kg_dir
        self.meta = json.loads((self.kg_dir / "meta.json").read_text())
        self._db = duckdb.connect(str(self.kg_dir / "ibkh.duckdb"), read_only=True)
        self._lock = threading.Lock()

        nodes = self._db.execute("SELECT id, name, norm, degree FROM nodes ORDER BY id").fetchall()
        self.names = [n[1] for n in nodes]
        self.degree = np.array([n[3] for n in nodes], dtype=np.int64)
        self.norm_index = {}
        self.token_index = defaultdict(list)
        for node_id, _, norm, degree in nodes:
            if norm not in self.norm_index or degree > self.degree[self.norm_index[norm]]:
                self.norm_index[norm] = node_id
            for token in set(norm.split()):
                self.token_index[token].append(node_id)
        self.norms = [n[2] for n in nodes]

        rels = self._db.execute("SELECT id, phrase, class, weight FROM rels ORDER BY id").fetchall()
        self.rel_phrase = {r[0]: r[1] for r in rels}
        self.rel_class = {r[0]: r[2] for r in rels}
        self.rel_weight = {r[0]: float(r[3]) for r in rels}

        self.node_vectors = None
        emb_path = self.kg_dir / "node_emb.npy"
        if emb_path.exists() and self.meta.get("encoder") == config.ENCODER:
            self.node_vectors = np.load(emb_path).astype(np.float32)
        elif emb_path.exists():
            logger.warning("Node embeddings were built with %s, not %s: semantic linking disabled "
                           "(rebuild with scripts/build_ibkh_kg.py).", self.meta.get("encoder"), config.ENCODER)
        logger.info("iBKH loaded: %d nodes, %d edges", len(self.names), self.meta.get("n_edges", 0))

    def _query(self, sql, params=()):
        with self._lock:
            cursor = self._db.cursor()
        try:
            return cursor.execute(sql, params).fetchall()
        finally:
            cursor.close()

    # ------------------------------------------------------------------ entity linking
    def _token_match(self, norm):
        tokens = norm.split()
        if not tokens:
            return None
        postings = sorted((self.token_index.get(t, []) for t in tokens), key=len)
        if not postings[0]:
            return None
        candidates = set(postings[0]).intersection(*map(set, postings[1:])) if len(postings) > 1 else set(postings[0])
        candidates = [c for c in candidates if len(self.norms[c].split()) - len(tokens) <= 2]
        if not candidates:
            return None
        return min(candidates, key=lambda c: (len(self.norms[c].split()), -self.degree[c]))

    def link_entities(self, entities, question=""):
        """Map planner entities ({"name", "synonyms"}) to node IDs; n-gram matching of the question as fallback."""
        links = {}
        pending = []
        for entity in entities:
            name = entity.get("name", "")
            variants = [name] + list(entity.get("synonyms") or [])
            hit = None
            for v in variants:
                norm = _normalize(v)
                if norm in self.norm_index and norm not in _GENERIC:
                    hit = (self.norm_index[norm], "exact", 1.0)
                    break
            if hit is None:
                for v in variants:
                    norm = _normalize(v)
                    if norm and norm not in _GENERIC:
                        node = self._token_match(norm)
                        if node is not None:
                            hit = (node, "token match", 0.9)
                            break
            if hit is None:
                pending.append((name, variants))
            else:
                links[name] = {"node": hit[0], "name": self.names[hit[0]], "method": hit[1], "score": hit[2]}

        if pending and self.node_vectors is not None:
            texts = [v for _, variants in pending for v in variants]
            vectors = encode(texts, is_query=True)
            offset = 0
            for name, variants in pending:
                block = vectors[offset:offset + len(variants)]
                offset += len(variants)
                sims = self.node_vectors @ block.T                 # nodes × variants
                best_node = int(np.argmax(sims.max(axis=1)))
                score = float(sims[best_node].max())
                if score >= 0.8 and self.norms[best_node] not in _GENERIC:
                    links[name] = {"node": best_node, "name": self.names[best_node], "method": "embedding",
                                   "score": round(score, 3)}

        if not links and question:  # no plan entities: longest n-grams of the question that are node names
            words = _normalize(question).split()
            used = set()
            for n in range(min(5, len(words)), 0, -1):
                for i in range(len(words) - n + 1):
                    span = set(range(i, i + n))
                    gram = " ".join(words[i:i + n])
                    if span & used or gram in _GENERIC or len(gram) < 4 or gram not in self.norm_index:
                        continue
                    node = self.norm_index[gram]
                    links[gram] = {"node": node, "name": self.names[node], "method": "question n-gram", "score": 0.85}
                    used |= span
        return links

    def semantic_nodes(self, query_vector, exclude=(), k=8, min_score=0.5):
        """Nodes closest to the question embedding (unstructured retrieval of concepts)."""
        if self.node_vectors is None:
            return []
        sims = self.node_vectors @ query_vector
        order = np.argpartition(-sims, k * 3)[:k * 3]
        order = order[np.argsort(-sims[order])]
        picked = [int(i) for i in order if sims[i] >= min_score and int(i) not in exclude
                  and self.norms[int(i)] not in _GENERIC][:k]
        return picked

    # ------------------------------------------------------------------ edges
    def _edge(self, node, nbr, r, direction, n):
        head, tail = (node, nbr) if direction == 0 else (nbr, node)
        return {"h": head, "t": tail, "r": int(r), "n": int(n),
                "text": f"{self.names[head]} {self.rel_phrase[int(r)]} {self.names[tail]}"}

    def direct_edges(self, nodes, targets):
        if not nodes or not targets:
            return []
        rows = self._query(
            f"SELECT node, nbr, r, dir, n FROM adj WHERE node IN ({','.join(map(str, nodes))}) "
            f"AND nbr IN ({','.join(map(str, targets))})")
        seen, edges = set(), []
        for row in rows:
            e = self._edge(*row)
            key = (e["h"], e["r"], e["t"])
            if key not in seen:
                seen.add(key)
                edges.append(e)
        return edges

    def neighborhood(self, node, limit=1500):
        """Highest-weighted relations of ``node`` (curated first, then by support)."""
        rows = self._query(
            "SELECT a.node, a.nbr, a.r, a.dir, a.n FROM adj a JOIN rels ON a.r = rels.id "
            "WHERE a.node = ? ORDER BY rels.weight DESC, a.n DESC LIMIT ?", (int(node), int(limit)))
        return [self._edge(*row) for row in rows]

    def two_hop_paths(self, a, b, limit=60):
        """Paths a–x–b through a shared neighbour, best relation weights first."""
        rows = self._query(
            "SELECT x.nbr, x.r, x.dir, x.n, y.r, y.dir, y.n FROM adj x JOIN adj y ON x.nbr = y.nbr "
            "JOIN rels rx ON x.r = rx.id JOIN rels ry ON y.r = ry.id "
            "WHERE x.node = ? AND y.node = ? AND x.nbr NOT IN (?, ?) "
            "ORDER BY rx.weight + ry.weight DESC, x.n + y.n DESC LIMIT ?",
            (int(a), int(b), int(a), int(b), int(limit)))
        paths = []
        for mid, r1, d1, n1, r2, d2, n2 in rows:
            paths.append({"mid": int(mid), "edges": [self._edge(a, mid, r1, d1, n1), self._edge(b, mid, r2, d2, n2)]})
        return paths

    def pair_relations(self, pairs):
        """All relation phrases per unordered node pair (for contradiction checks)."""
        pairs = list(pairs)
        if not pairs:
            return {}
        values = ",".join(f"({int(a)},{int(b)})" for a, b in pairs)
        rows = self._query(f"SELECT a.node, a.nbr, a.r FROM adj a JOIN (VALUES {values}) p(x, y) "
                           f"ON a.node = p.x AND a.nbr = p.y")
        relations = defaultdict(set)
        for node, nbr, r in rows:
            relations[frozenset((node, nbr))].add(self.rel_phrase[int(r)])
        return relations

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, plan, query, query_vector, k=12, report=None):
        """Ranked KG evidence items for the question; returns (items, info)."""
        report = report or (lambda msg: None)
        links = self.link_entities(plan.entities, question=query)
        linked = list(dict.fromkeys(v["node"] for v in links.values()))
        semantic = self.semantic_nodes(query_vector, exclude=set(linked))
        info = {"links": {k: {kk: vv for kk, vv in v.items()} for k, v in links.items()},
                "semantic_nodes": [self.names[i] for i in semantic],
                "mode": "structured + semantic" if linked else "semantic"}

        candidates = []   # (kind, edges, bonus, label)
        for e in self.direct_edges(linked, linked):
            candidates.append(("direct", [e], 1.0, "structured: relation between question entities"))
        for e in self.direct_edges(linked, semantic):
            candidates.append(("semantic-link", [e], 0.7, "semantic: relation to a question concept"))
        for a, b in combinations(linked, 2):
            for p in self.two_hop_paths(a, b):
                candidates.append(("path", p["edges"], 0.6, f"structured: 2-hop path via {self.names[p['mid']]}"))
        # 1-hop neighbourhoods of the treatments and conditions, not of outcomes: the neighbours of "hospitalization"
        # or "mortality" are unrelated drugs ("atorvastatin treats hospitalization")
        roles = {e.get("name"): e.get("role") for e in plan.entities or []}
        anchors = list(dict.fromkeys(v["node"] for name, v in links.items() if roles.get(name) != "outcome")) or linked
        for node in (anchors or semantic[:3]):
            for e in self.neighborhood(node):
                candidates.append(("neighborhood", [e], 0.0, "structured: 1-hop relation" if linked else "semantic: neighborhood"))
        if not candidates:
            report("iBKH: no entity of the question could be linked to the graph")
            return [], info

        # Dense re-ranking of all candidates against the question
        texts = ["; ".join(e["text"] for e in c[1]) for c in candidates]
        unique = list(dict.fromkeys(texts))
        vectors = encode(unique)
        sim = dict(zip(unique, (vectors @ query_vector).tolist()))
        scored = []
        for (kind, edges, bonus, label), text in zip(candidates, texts):
            weight = min(self.rel_weight[e["r"]] for e in edges)
            scored.append((0.55 * sim[text] + 0.3 * weight + 0.15 * bonus, kind, edges, label, text))
        scored.sort(key=lambda s: s[0], reverse=True)

        chosen, seen_pairs, n_paths = [], set(), 0
        for score, kind, edges, label, text in scored:
            key = tuple(sorted((e["h"], e["t"]) for e in edges)) + ((edges[0]["r"],) if len(edges) == 1 else ())
            if key in seen_pairs or (kind == "path" and n_paths >= max(1, k // 3)):
                continue
            seen_pairs.add(key)
            n_paths += kind == "path"
            chosen.append((score, kind, edges, label, text))
            if len(chosen) >= k:
                break

        relations = self.pair_relations({(e["h"], e["t"]) for _, _, edges, _, _ in chosen for e in edges})
        items = []
        for score, kind, edges, label, text in chosen:
            classes = [self.rel_class[e["r"]] for e in edges]
            weakest = min(classes, key=lambda c: ["inferred", "literature-mined", "curated"].index(c))
            reliability = CLASS_RELIABILITY[weakest]
            contradicted = False
            for e in edges:
                rels = relations.get(frozenset((e["h"], e["t"])), set())
                contradicted |= any(rels & a and rels & b for a, b in CONTRADICTORY)
            if contradicted:
                reliability = _DOWNGRADE[reliability]
            support = min(len(relations.get(frozenset((e["h"], e["t"])), {1})) for e in edges)
            items.append(EvidenceItem(
                id=f"kg:{text}", source_type="kg", title=text, venue="iBKH",
                design=CLASS_LABEL[weakest] + (" — 2-hop path" if len(edges) > 1 else ""),
                evidence_level=5, origin="iBKH", score=float(score),
                passages=[Passage(text=text, section=label)],
                meta={"edges": [[self.names[e["h"]], self.rel_phrase[e["r"]], self.names[e["t"]]] for e in edges],
                      "relation_class": weakest, "reliability": reliability, "contradicted": contradicted,
                      "supporting_relations": support, "retrieval": label,
                      "sources_support": min(e["n"] for e in edges)}))
        linked_text = ", ".join(f"{name}→{v['name']}" for name, v in links.items()) or "no entities"
        report(f"iBKH: linked {linked_text}; {len(candidates)} candidate relations, {len(items)} kept")
        info.update({"n_candidates": len(candidates), "n_selected": len(items)})
        return items, info


_graph = None
_graph_error = None
_graph_lock = threading.Lock()


def get_kg():
    """Process-wide iBKH graph, or ``None`` if the store has not been built (see ``kg_status``)."""
    global _graph, _graph_error
    with _graph_lock:
        if _graph is None and _graph_error is None:
            if not (config.IBKH_DIR / "ibkh.duckdb").exists():
                _graph_error = (f"iBKH store not found in {config.IBKH_DIR}. "
                                "Build it with: python -m mes.scripts.build_ibkh_kg")
            else:
                try:
                    _graph = IBKHGraph()
                except Exception as e:
                    _graph_error = f"iBKH store could not be opened: {e}"
    return _graph


def kg_status():
    get_kg()
    return {"available": _graph is not None, "error": _graph_error,
            "meta": _graph.meta if _graph is not None else None}
