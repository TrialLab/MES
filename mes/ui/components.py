"""
HTML / markdown builders for the MES page (no Streamlit calls — easy to test).

The answer markdown cites evidence IDs ([L1], [T2], [K3]); ``answer_html`` escapes
the model text, inserts grounding badges after flagged sentences and replaces
each ID with a numbered, colored citation chip that links to the source card.
"""

import html
import re

from mes.tools.rwd.report import no_traceback

TYPE_META = {
    "literature": {"label": "Literature", "short": "Lit", "cls": "lit", "icon": "📄"},
    "trial": {"label": "ClinicalTrials.gov", "short": "Trials", "cls": "trial", "icon": "🧪"},
    "kg": {"label": "Knowledge graph", "short": "KG", "cls": "kg", "icon": "🕸️"},
    "rwd": {"label": "Real-world data", "short": "RWD", "cls": "rwd", "icon": "📊"},
}

BADGES = {
    "verified": ("ok", "✓ verified", "Checked against the cited source by the claim verifier."),
    "repaired": ("ok", "↻ source corrected", "The numbers were cited to another source or not cited; MES added the source "
                                             "that contains them."),
    "grounded": ("ok", "✓", "Every number in this sentence appears in the cited source."),
    "partially_supported": ("warn", "≈ partial", "The verifier found this claim only partly supported by the cited source."),
    "unsupported": ("bad", "⚠ unsupported", "The verifier did not find this claim in the cited source."),
    "unsupported_number": ("bad", "⚠ check number", "A number in this sentence does not appear in the cited source."),
    "uncited_number": ("warn", "⚠ uncited", "This sentence reports a number without citing a source."),
}

_CITE = re.compile(r"\[([LTKR]\d+)\]")


def esc(text):
    """Text for an HTML context of st.markdown; "$" as an entity so that it never starts LaTeX math, "~" so that a
    range written with tildes ("0.82~1.10", common in Chinese answers) never becomes strikethrough."""
    return html.escape(str(text or ""), quote=False).replace("$", "&#36;").replace("~", "&#126;")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$:])")


def md(text, limit=None):
    """User or model text for a Streamlit markdown context (button and expander labels, alerts): one line,
    shortened to ``limit`` characters, markdown syntax shown as typed."""
    text = " ".join(str(text or "").split())
    if limit and len(text) > limit:
        text = text[:limit - 1].rstrip() + "…"
    return _MD_SPECIAL.sub(r"\\\1", text)


def attr(text):
    return html.escape(str(text or ""), quote=True)


def keep_units(text):
    """``text`` with each number and its unit of time on one line ("first 24 h" never breaks before the "h")."""
    return re.sub(r"(\d) (h|d|min)\b", "\\1\u00a0\\2", str(text or ""))


def source_type(evidence_id):
    return {"L": "literature", "T": "trial", "K": "kg", "R": "rwd"}.get(evidence_id[:1], "literature")


def _card_passage(evidence, limit=300):
    """The passage a hover card quotes: the results for RWD, else the first (best-ranked) passage."""
    passages = evidence.passages or []
    if evidence.source_type == "rwd":
        passages = [x for x in passages if x.section == "RWD results"] or passages
    text = passages[0].text if passages else (evidence.meta.get("abstract") or "")
    text = " ".join(no_traceback(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def chip(number, evidence, anchor):
    """Citation chip with a hover card (type, title, venue, the quoted passage, link); click → the source card."""
    meta = TYPE_META[evidence.source_type]
    bits = [evidence.venue, str(evidence.year or "")]
    if evidence.identifiers.get("pmid"):
        bits.append(f"PMID {evidence.identifiers['pmid']}")
    if evidence.identifiers.get("nct"):
        bits.append(evidence.identifiers["nct"])
    if evidence.source_type == "rwd" and evidence.meta.get("n"):
        bits.append(f"{evidence.meta['n']:,} patients")
    detail = " · ".join(esc(b) for b in dict.fromkeys(bits) if b and b != "None")
    passage = _card_passage(evidence)
    link = (f'<a class="mes-cc-link" href="{attr(evidence.url)}" target="_blank" rel="noopener">Open ↗</a>'
            if evidence.url else "")
    card = (f'<span class="mes-cc" role="tooltip"><span class="mes-cc-type mes-cc-{meta["cls"]}">{meta["icon"]} '
            f'{esc(evidence.design or meta["label"])}</span><span class="mes-cc-title">{esc(evidence.title)}</span>'
            + (f'<span class="mes-cc-meta">{detail}</span>' if detail else "")
            + (f'<span class="mes-cc-text">“{esc(passage)}”</span>' if passage else "")
            + f'<span class="mes-cc-foot">{link}<span>Click the number for the source card</span></span></span>')
    from mes.ui.answer_viz import tier
    return (f'<span class="mes-cw"><a class="mes-cite mes-cite-{meta["cls"]}" href="#{anchor}" data-tier="{tier(evidence)}" '
            f'aria-label="{attr(meta["label"] + ": " + evidence.title)}">{number}</a>{card}</span>')


def _badge(status, reason=""):
    if status not in BADGES:
        return ""
    cls, text, tip = BADGES[status]
    if reason:
        tip = f"{tip} {reason}"
    return f'<span class="mes-badge mes-badge-{cls}" title="{attr(tip)}">{text}</span>'


def mark_sentences(markdown, sentences, show_grounded=True):
    """Insert badge markers after the sentences with a grounding / verification status."""
    for n, s in enumerate(sentences):
        status = s.get("status")
        if status == "grounded" and not show_grounded:
            continue
        if status not in BADGES:
            continue
        text = s["text"]
        pos = markdown.find(text)
        if pos < 0:
            continue
        end = pos + len(text)
        markdown = markdown[:end] + f"⟦B{n}⟧" + markdown[end:]
    return markdown


def answer_html(markdown, evidence_by_id, order, sentences, anchor_prefix, show_grounded=True):
    """Markdown with HTML citation chips and badges (safe: model text is escaped)."""
    index = {i: n + 1 for n, i in enumerate(order)}
    marked = mark_sentences(markdown, sentences, show_grounded)
    text = esc(marked)

    def to_chip(m):
        i = m.group(1)
        if i not in evidence_by_id or i not in index:
            return ""
        return chip(index[i], evidence_by_id[i], f"{anchor_prefix}-{i}")

    text = _CITE.sub(to_chip, text)
    statuses = {f"B{n}": s for n, s in enumerate(sentences)}

    def to_badge(m):
        s = statuses.get(m.group(1))
        return _badge(s["status"], (s.get("llm_check") or {}).get("issue", "")) if s else ""

    return re.sub(r"⟦(B\d+)⟧", to_badge, text)


def source_card(number, item, anchor, cited=True):
    meta = TYPE_META[item.source_type]
    bits = [item.design]
    if item.source_type == "literature":
        bits += [item.venue, str(item.year or "")]
        if item.identifiers.get("pmid"):
            bits.append(f"PMID {item.identifiers['pmid']}")
        if item.full_text:
            bits.append("full text")
    elif item.source_type == "trial":
        m = item.meta
        bits += [item.identifiers.get("nct", ""), str(m.get("status", "")).replace("_", " ").title()]
        if m.get("enrollment"):
            bits.append(f"n={m['enrollment']}")
    elif item.source_type == "kg":
        bits = [item.meta.get("retrieval", ""), f"reliability: {item.meta.get('reliability')}"]
    detail = " · ".join(esc(b) for b in bits if b)
    title = esc(item.title)
    if item.url:
        title = f'<a href="{attr(item.url)}" target="_blank" rel="noopener">{title}</a>'
    label = number if cited else "·"
    extra = ""
    if item.source_type == "kg":
        rel = item.meta.get("reliability", "")
        extra = f'<span class="mes-rel mes-rel-{attr(rel.lower())}">{esc(rel)}</span>'
        if item.meta.get("contradicted"):
            extra += '<span class="mes-rel mes-rel-low">contradictory relations</span>'
    elif item.evidence_level <= 2:
        extra = f'<span class="mes-level">{esc(item.design.split(" (")[0])}</span>'
    if item.source_type == "literature" and item.meta.get("landmark"):
        try:
            per_year = float(item.meta.get("cites_per_year") or 0)
        except (TypeError, ValueError):
            per_year = 0
        extra += ('<span class="mes-landmark">★ landmark' + (f' · {per_year:,.0f} citations/yr' if per_year >= 1 else "")
                  + "</span>")
    elif item.source_type == "trial" and item.meta.get("has_results"):
        extra += '<span class="mes-level">posted results</span>'
    return (f'<div class="mes-src {"" if cited else "mes-src-dim"}" id="{attr(anchor)}">'
            f'<span class="mes-src-num mes-num-{meta["cls"]}">{label}</span>'
            f'<div class="mes-src-body"><div class="mes-src-title">{title}</div>'
            f'<div class="mes-src-detail">{detail}</div>{extra}</div></div>')


def takeaways_html(takeaways, evidence_by_id, order, anchor_prefix, title="Key takeaways"):
    items = "".join(f"<li>{answer_html(t, evidence_by_id, order, [], anchor_prefix)}</li>" for t in takeaways)
    from mes.ui.icons import icon
    return f'<div class="mes-card mes-takeaways"><div class="mes-card-title">{icon("spark")} {esc(title)}</div><ul>{items}</ul></div>'


def pill(label, count, cls, icon_name=None):
    from mes.ui.icons import icon
    active = "" if count else " mes-pill-off"
    mark = icon(icon_name) if icon_name else ""
    return f'<span class="mes-pill mes-pill-{cls}{active}">{mark}{esc(label)} <b>{count if count else "—"}</b></span>'


def status_pill(text, ok):
    return f'<span class="mes-status mes-status-{"ok" if ok else "bad"}">{"●" if ok else "○"} {esc(text)}</span>'


def kg_dot(items, max_edges=24):
    """Graphviz DOT of the KG relations used in an answer."""
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fillcolor="#FFFBEB", color="#F59E0B", fontname="Helvetica", fontsize=10];',
             'edge [fontname="Helvetica", fontsize=9, color="#999999"];']
    seen = set()
    colors = {"High": "#15803D", "Moderate": "#B45309", "Low": "#9CA3AF"}
    for item in items:
        for head, rel, tail in item.meta.get("edges", []):
            key = (head, rel, tail)
            if key in seen or len(seen) >= max_edges:
                continue
            seen.add(key)
            color = colors.get(item.meta.get("reliability"), "#9CA3AF")
            q = lambda s: '"' + " ".join(str(s).replace('"', "'").replace("\\", "/").split())[:60] + '"'
            lines.append(f'{q(head)} -> {q(tail)} [label={q(rel)}, color="{color}", fontcolor="{color}"];')
    lines.append("}")
    return "\n".join(lines) if seen else ""


TABLE_HEADINGS = {"sample_size": "n", "n": "n", "effect": "Effect (95% CI)"}


def table_html(rows):
    """A table of the answer (rows: dicts with the same keys) whose cells wrap, so no column is hidden; a narrow
    window scrolls it sideways inside its own box."""
    if not rows:
        return ""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    label = lambda k: TABLE_HEADINGS.get(k.lower()) or (k.replace("_", " ").capitalize() if k == k.lower() else k)
    head = "".join(f"<th>{esc(label(str(k)))}</th>" for k in keys)
    body = "".join("<tr>" + "".join(f"<td>{esc(row.get(k, ''))}</td>" for k in keys) + "</tr>" for row in rows)
    return f'<div class="mes-table-wrap"><table class="mes-table"><thead><tr>{head}</tr></thead>' \
           f"<tbody>{body}</tbody></table></div>"


def export_markdown(result):
    """Standalone markdown report: question, answer with numbered citations, references."""
    by_id = result.evidence_by_id()
    lines = [f"# {result.question}", "", f"_MES · {result.mode.title()} mode · {result.llm} · {result.created_at}_", "",
             result.answer_display, ""]
    if result.protocol:             # an edited protocol (ui/protocol.py): version, changes, what ran again
        from mes.ui.exports import protocol_lines
        lines += ["## Protocol", ""] + [f"- {line}" for line in protocol_lines(result.protocol)] + [""]
        if result.protocol.get("diff"):
            lines += ["```diff", result.protocol["diff"], "```", ""]
    lines.append("## References")
    for n, i in enumerate(result.citation_order, start=1):
        e = by_id.get(i)
        if e is None:                       # a citation whose source is not in the record (never written by MES)
            continue
        where = e.url or e.venue
        lines.append(f"{n}. {e.title}. {e.design}{', ' + e.venue if e.venue and e.venue not in where else ''}"
                     f"{', ' + str(e.year) if e.year else ''}. {where}")
    lines += ["", "_Research prototype — not a medical device; verify with the cited sources before clinical use._"]
    return "\n".join(lines)
