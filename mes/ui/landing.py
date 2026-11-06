"""
The landing of the MES page as HTML (no Streamlit calls): the hero with its evidence constellation, the line of what
is connected, the four sources MES answers from and how an answer is made. Icons are inline SVG (Lucide-style
strokes), so they take the colour of their card.
"""

import math
import random

from mes.ui.components import esc
from mes.ui.icons import icon

def _constellation(n=34, seed=11):
    """A quiet field of linked points behind the headline (the same every time: seeded)."""
    rng = random.Random(seed)
    pts = [(rng.uniform(1, 99), rng.uniform(2, 38)) for _ in range(n)]
    lines = []
    for i, (x, y) in enumerate(pts):
        near = sorted(range(n), key=lambda j: (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2)[1:3]
        for j in near:
            if i < j and math.dist((x, y), pts[j]) < 16:
                lines.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}"/>')
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rng.choice((0.14, 0.2, 0.28)):.2f}" '
                   f'style="animation-delay:-{rng.uniform(0, 9):.1f}s"/>' for x, y in pts)
    return (f'<div class="mes-hero-bg" aria-hidden="true"><svg viewBox="0 0 100 40" preserveAspectRatio="xMidYMid slice">'
            f'{"".join(lines)}{dots}</svg></div>')


def hero():
    """The headline (what is connected on this computer is in the status menu and under the sources below)."""
    return (f'<div class="mes-hero">{_constellation()}'
            '<div class="mes-hero-tag"><span class="mes-mark">M</span>Multi-agent Evidence Synthesis · Weill Cornell '
            'Medicine</div>'
            '<h1>What would you like to <em>investigate</em>?</h1>'
            "<p>MES searches PubMed (abstracts and open-access full text), your PDFs, ClinicalTrials.gov, the iBKH "
            "knowledge graph and real-world data, then writes an answer in which every statement cites its "
            "source.</p></div>")


CAPS = [("literature", "Literature", "PubMed abstracts, open-access full text and your PDFs; landmark trials first.", "lit"),
        ("trials", "Clinical trials", "ClinicalTrials.gov registrations with their posted results — arm values, "
         "hazard ratios, p-values.", "trial"),
        ("kg", "Knowledge graph", "iBKH relations and two-hop paths that explain a mechanism, with their reliability.", "kg"),
        ("rwd", "Real-world data", "A new study on the connected clinical database: target trial emulation, patients "
         "like yours, survival, rates, prognostic models — designed, run and checked for your question.", "rwd")]


def _dot(ok):
    """Connected (green), not available (amber) or not connected yet (an empty grey ring)."""
    return f'<i class="mes-dot{"" if ok else " mes-dot-none" if ok is None else " mes-dot-off"}"></i>'


def capabilities(stats):
    """The four sources; ``stats``: {key: (text, ok) or [(text, ok), …]} — ok None: not connected yet."""
    cards = []
    for key, name, desc, cls in CAPS:
        entries = stats.get(key, ("", True))
        entries = [entries] if isinstance(entries, tuple) else list(entries)
        stat = "".join(f'<span class="{"mes-cap-none" if ok is None else ""}" title="'
                       f'{"connected" if ok else "not connected yet" if ok is None else "not available"}">'
                       f'{_dot(ok)}{esc(text)}</span>' for text, ok in entries if text)
        stat = f'<div class="mes-cap-stat">{stat}</div>' if stat else ""
        cards.append(f'<div class="mes-cap" style="--c:var(--mes-{cls});--cb:var(--mes-{cls}-bg)">'
                     f'<div class="mes-cap-ico">{icon(key)}</div>'
                     f'<div class="mes-cap-name">{name}</div><div class="mes-cap-desc">{desc}</div>{stat}</div>')
    return f'<div class="mes-caps">{"".join(cards)}</div>'


def databases(rows):
    """The real-world databases side by side: [(name, ok, about)] — ok True connected, None not connected yet."""
    chips = "".join(f'<span class="mes-db{" mes-db-none" if ok is None else ""}" title="{esc(about)} — '
                    f'{"connected" if ok else "not connected yet" if ok is None else "not available here"}">'
                    f'{_dot(ok)}{esc(name)}<small>{"connected" if ok else "not connected yet" if ok is None else "not built"}'
                    f'</small></span>' for name, ok, about in rows)
    return f'<div class="mes-dbs">{chips}</div>'


FLOW = [("plan", "Plan", "PICO, queries"), ("search", "Search", "4 sources at once"), ("rwd", "Analyze", "RWD study"),
        ("write", "Synthesize", "cited answer"), ("check", "Check", "citations · numbers"), ("answer", "Answer", "in seconds")]


def how_it_works():
    steps = []
    for i, (ico, name, sub) in enumerate(FLOW):
        if i:
            steps.append('<div class="mes-flow-link" aria-hidden="true"></div>')
        steps.append(f'<div class="mes-flow-step"><div class="mes-flow-ico">{icon(ico)}</div>'
                     f'<div class="mes-flow-name">{name}</div><div class="mes-flow-sub">{sub}</div></div>')
    return f'<div class="mes-flow" role="list" aria-label="How MES answers">{"".join(steps)}</div>'
