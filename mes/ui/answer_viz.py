"""
Specs of the answer-level views (drawn by ui/web/viz_answer.js): the answer at a glance — what the cited evidence is
made of, how well its numbers are grounded, which designs it rests on — the map of the evidence by year and design,
and the knowledge-graph relations as a network. Pure functions of an ``MESResult``.
"""

from mes.tools.rwd import database_of
from mes.ui.components import TYPE_META, source_type

# The tiers of the evidence (the pyramid of the glance and the rows of the map); chips carry data-tier
TIERS = [(1, "Systematic reviews · guidelines"), (2, "Randomized trials"), (3, "Observational · real-world"),
         (4, "Reviews · case reports · other"), (6, "Mechanism (knowledge graph)")]
SHORT_TIERS = {1: "Reviews & guidelines", 2: "Randomized trials", 3: "Observational", 4: "Other designs", 6: "Mechanism (KG)"}


def tier(item):
    """The evidence tier of an item (1 strongest … 4), 6 for knowledge-graph relations."""
    if item.source_type == "kg":
        return 6
    if item.source_type == "rwd":
        return 3
    level = int(item.evidence_level or 5)
    if item.source_type == "trial":
        return 2 if "random" in (item.design or "").lower() else 3 if level <= 3 else 4
    return min(max(level, 1), 4)


def _anchor(turn_id):
    return f"mes-{turn_id}"


def glance(result, turn_id):
    """Cards: the cited sources by type, their grounding, the tiers of the cited evidence."""
    by_id = result.evidence_by_id()
    cited = [by_id[i] for i in result.citation_order if i in by_id]
    if not cited:
        return None
    parts = []
    for t, meta in TYPE_META.items():
        n = sum(1 for e in cited if e.source_type == t)
        parts.append({"cls": meta["cls"], "label": meta["label"] if t != "literature" else "Literature", "n": n})
    v = result.verification or {}
    counts = v.get("counts") or {}
    flagged = sum(counts.get(k, 0) for k in ("unsupported_number", "uncited_number", "unsupported"))
    cards = [{"kind": "sources", "label": "Evidence cited", "cited": len(cited), "reviewed": len(result.evidence),
              "parts": parts}]
    if v.get("n_numbers"):
        sub = f"{int(round(100 * v.get('citation_coverage', 0)))}% of statements cited"
        if flagged:
            sub += f" · {flagged} flagged"
        cards.append({"kind": "grounding", "label": "Grounding", "grounded": v.get("n_numbers_grounded", 0),
                      "numbers": v.get("n_numbers", 0), "coverage": v.get("citation_coverage", 0), "flagged": flagged,
                      "sub": sub})
    rows = []
    for rank, (t, label) in enumerate(TIERS):
        n = sum(1 for e in cited if tier(e) == t)
        rows.append({"tier": t, "label": SHORT_TIERS[t], "title": label, "n": n, "rank": rank})
    cards.append({"kind": "levels", "label": "Strength of the evidence", "rows": rows})
    landmarks = [e for e in cited if (e.meta or {}).get("landmark")]
    if landmarks:
        top = max(landmarks, key=lambda e: float((e.meta or {}).get("cites_per_year") or 0))
        cards.append({"kind": "stat", "label": "Landmark studies", "value": str(len(landmarks)), "unit": "cited",
                      "sub": f"most cited: {top.title[:90]}{'…' if len(top.title) > 90 else ''}"})
    elif result.rwd_studies:
        ran = [s for s in result.rwd_studies if isinstance(s, dict) and s.get("status") == "done"
               and (s.get("spec") or {}).get("analysis") != "catalog"]
        looked = [s for s in result.rwd_studies if isinstance(s, dict) and (s.get("spec") or {}).get("analysis") == "catalog"]
        if ran:
            cards.append({"kind": "stat", "label": "Real-world data", "value": str(len(ran)),     # the database they ran on
                          "unit": f"{database_of(ran[0])} {'study' if len(ran) == 1 else 'studies'}",
                          "sub": "designed and run for this question"})
        elif looked:
            cards.append({"kind": "stat", "label": "Real-world data", "value": database_of(looked[0]), "unit": "data catalog",
                          "sub": "what the data records about the question's terms"})
    return {"anchor": _anchor(turn_id), "cards": cards}


def landscape(result, turn_id):
    """The cited and reviewed studies by year and tier (literature, trials, real-world data)."""
    anchor = _anchor(turn_id)
    index = {i: n + 1 for n, i in enumerate(result.citation_order)}
    items = []
    for e in result.evidence:
        if e.source_type == "kg":
            continue
        year = e.year
        if e.source_type == "rwd" and not year:
            year = int(str(result.created_at or "2026")[:4]) if str(result.created_at or "")[:4].isdigit() else None
        if not year:
            continue
        meta = e.meta or {}
        if e.source_type == "trial":
            try:
                size, size_label = float(meta.get("enrollment") or 0), "enrolled"
            except (TypeError, ValueError):
                size, size_label = 0.0, ""
        elif e.source_type == "rwd":
            size, size_label = float(meta.get("n") or 0), "patients"
        else:
            try:
                size, size_label = float(meta.get("cites_per_year") or 0), "citations per year"
            except (TypeError, ValueError):
                size, size_label = 0.0, ""
        items.append({"cite_id": f"{anchor}-{e.id}" if e.id in index else "", "number": index.get(e.id),
                      "title": e.title[:200], "year": int(year), "tier": tier(e), "cls": TYPE_META[e.source_type]["cls"],
                      "cited": e.id in index, "size": size, "size_label": size_label, "venue": e.venue or "",
                      "design": e.design or "", "landmark": bool(meta.get("landmark"))})
    if len(items) < 3:
        return None
    return {"anchor": anchor, "items": items, "tiers": [{"tier": t, "label": label} for t, label in TIERS[:4]]}


def network(items):
    """The knowledge-graph relations of the cited KG items: entities and their relations, by reliability."""
    nodes, edges, seen = {}, [], set()
    for item in items:
        rel = (item.meta or {}).get("reliability") or "Low"
        for edge in (item.meta or {}).get("edges") or []:
            if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                continue
            head, relation, tail = (" ".join(str(x).split())[:80] for x in edge)
            if (head, relation, tail) in seen or len(edges) >= 36:
                continue
            seen.add((head, relation, tail))
            for name in (head, tail):
                nodes.setdefault(name.lower(), {"id": name.lower(), "label": name})
            edges.append({"source": head.lower(), "target": tail.lower(), "label": relation, "reliability": rel})
    if not edges:
        return None
    return {"nodes": list(nodes.values()), "edges": edges}


def chip_tier(evidence_id, item):
    """The tier of a citation chip (data-tier), from its item."""
    return tier(item) if item is not None else 4 if source_type(evidence_id) != "kg" else 6
