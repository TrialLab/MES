// MES charts — the answer at a glance (the evidence mix, its grounding, the designs it rests on), the map of the
// evidence (year × design) and the knowledge-graph relations as a network. Hovering a part of the glance lights up
// the citations of the answer it counts (the chips are in the page, outside this component).

function citeChips(anchor) {
  if (!anchor) return [];
  try {
    return [...document.querySelectorAll(`a.mes-cite[href^="#${CSS.escape(anchor)}-"]`)];
  } catch (e) {
    return [];
  }
}

function lightUp(anchor, test) {
  const chips = citeChips(anchor);
  for (const c of chips) {
    const on = test ? test(c) : false;
    c.classList.toggle("mes-hl", !!test && on);
    c.classList.toggle("mes-dim", !!test && !on);
  }
}

function ringSvg(share, color, track) {
  const r = 22, C = 2 * Math.PI * r;
  const svgEl = sv("svg", { width: 56, height: 56, viewBox: "0 0 56 56", "aria-hidden": "true" });
  sv("circle", { cx: 28, cy: 28, r, fill: "none", stroke: track, "stroke-width": 6 }, svgEl);
  const arc = sv("circle", { cx: 28, cy: 28, r, fill: "none", stroke: color, "stroke-width": 6, "stroke-linecap": "round",
                             "stroke-dasharray": `${C}`, "stroke-dashoffset": `${C}`, transform: "rotate(-90 28 28)" }, svgEl);
  const target = C * (1 - Math.max(0, Math.min(1, share)));
  arc.style.transition = motion() ? "stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1)" : "none";
  requestAnimationFrame(() => arc.setAttribute("stroke-dashoffset", `${target}`));
  const t = svgText(svgEl, 28, 32, `${Math.round(100 * share)}%`, { "text-anchor": "middle", "font-size": 12, "font-weight": 700 });
  t.setAttribute("fill", "currentColor");
  return svgEl;
}

// spec: {anchor, cards: [{kind: "sources" | "grounding" | "levels" | "stat", …}]}
function glanceView(root, spec, ctx) {
  const grid = h("div", "mv-glance");
  root.appendChild(grid);
  for (const card of spec.cards || []) {
    const box = h("div", "mv-gcard");
    box.appendChild(h("div", "mv-glabel", card.label));
    if (card.kind === "sources") {
      const v = h("div", "mv-gvalue", int(card.cited));
      v.appendChild(h("small", null, `cited of ${int(card.reviewed)} reviewed`));
      box.appendChild(v);
      const mix = h("div", "mv-mix");
      const leg = h("div", "mv-mixleg");
      const total = (card.parts || []).reduce((a, p) => a + (p.n || 0), 0) || 1;
      for (const p of card.parts || []) {
        if (!p.n) continue;
        const color = cssVar(ctx, `--${p.cls}`);
        const seg = h("i");
        seg.style.flex = `${p.n} 1 0`;
        seg.style.background = color;
        seg.title = `${p.label}: ${p.n}`;
        mix.appendChild(seg);
        const b = button("", "", () => {}, false);
        const sw = h("i"); sw.style.background = color;
        b.append(sw, document.createTextNode(`${p.label} ${p.n}`));
        const on = () => { lightUp(spec.anchor, (c) => c.classList.contains(`mes-cite-${p.cls}`)); b.setAttribute("aria-pressed", "true"); };
        const off = () => { lightUp(spec.anchor, null); b.setAttribute("aria-pressed", "false"); };
        for (const el of [seg, b]) {
          el.addEventListener("pointerenter", on); el.addEventListener("pointerleave", off);
          el.addEventListener("focus", on); el.addEventListener("blur", off);
        }
        leg.appendChild(b);
      }
      box.append(mix, leg);
      void total;
    } else if (card.kind === "grounding") {
      const ring = h("div", "mv-ring");
      const share = card.numbers ? card.grounded / card.numbers : card.coverage || 0;
      ring.appendChild(ringSvg(share, cssVar(ctx, card.flagged ? "--warn" : "--ok"), cssVar(ctx, "--line2")));
      const txt = h("div");
      txt.append(h("div", "mv-gvalue", `${int(card.grounded)}/${int(card.numbers)}`),
                 h("div", "mv-gsub", "numbers found in the cited sources"), h("div", "mv-gsub", card.sub));
      ring.appendChild(txt);
      box.appendChild(ring);
    } else if (card.kind === "levels") {
      const pyr = h("div", "mv-pyr");
      const max = Math.max(1, ...(card.rows || []).map((r) => r.n || 0));
      for (const r of (card.rows || []).filter((x) => x.n || x.tier <= 3)) {    // the three main tiers, the rest when cited
        const row = h("div", "mv-pyr-row");
        row.tabIndex = 0;
        row.setAttribute("role", "button");
        row.setAttribute("aria-pressed", "false");
        row.setAttribute("aria-label", `${r.label}: ${r.n || 0} cited — highlight them in the answer`);
        const track = h("div", "mv-pyr-track");
        const bar = h("i", r.n ? "" : "mv-zero");
        bar.style.width = `${r.n ? Math.max(6, (100 * r.n) / max) : 0}%`;
        track.appendChild(bar);
        const name = h("div", "mv-pyr-l", r.label);
        name.title = r.title || r.label;
        const cell = h("div", "mv-pyr-c");
        cell.append(name, track);
        row.append(cell, h("div", "mv-pyr-n", String(r.n || 0)));
        const on = () => { lightUp(spec.anchor, (c) => (c.dataset.tier || "") === String(r.tier)); row.setAttribute("aria-pressed", "true"); };
        const off = () => { lightUp(spec.anchor, null); row.setAttribute("aria-pressed", "false"); };
        row.addEventListener("pointerenter", on); row.addEventListener("pointerleave", off);
        row.addEventListener("focus", on); row.addEventListener("blur", off);
        pyr.appendChild(row);
      }
      box.appendChild(pyr);
    } else {
      const v = h("div", "mv-gvalue", card.value);
      if (card.unit) v.appendChild(h("small", null, card.unit));
      box.appendChild(v);
      if (card.sub) box.appendChild(h("div", "mv-gsub", card.sub));
    }
    grid.appendChild(box);
  }
}

// spec: {anchor, items: [{cite_id, title, year, tier, cls, cited, size, size_label, venue, design, landmark}], tiers}
function landscapeView(root, spec, ctx) {
  const items = (spec.items || []).filter((it) => isNum(it.year) && isNum(it.tier));
  if (items.length < 2) return;
  const tiers = spec.tiers || [];
  const width = Math.max(300, root.clientWidth || 600);
  const rowH = 38, m = { l: 0, r: 18, t: 10, b: 30 };
  const H = m.t + rowH * tiers.length + m.b;
  const wrap = h("div", "mv-chart");
  root.appendChild(wrap);
  const legend = h("div", "mv-legend");
  for (const [cls, label] of [["lit", "literature"], ["trial", "trial"], ["rwd", "real-world data"]]) {
    if (!items.some((it) => it.cls === cls)) continue;
    const it = h("span"); const k = h("i", "mv-key mv-key-dot"); k.style.background = cssVar(ctx, `--${cls}`);
    it.append(k, document.createTextNode(label)); legend.appendChild(it);
  }
  const hollow = h("span"); hollow.append(h("i", "mv-key-ring"), document.createTextNode("reviewed, not cited"));
  legend.appendChild(hollow);
  wrap.appendChild(legend);
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img", "aria-label": "The evidence by year and design" }, wrap);
  // the names of the designs whole — on one line, or two in a narrow column — and the plot after the longest
  const names = tiers.map((t) => fitLines(svgEl, t.label, "mv-tick", Math.min(220, Math.max(96, width * 0.34))));
  m.l = Math.ceil(Math.max(60, ...names.flat().map((line) => textWidth(svgEl, line, "mv-tick")))) + 16;
  const years = items.map((it) => it.year);
  const [y0, y1] = [Math.min(...years) - 1, Math.max(...years) + 1];
  const x = scale(y0, y1, m.l + 10, width - m.r);
  tiers.forEach((t, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    sv("line", { x1: m.l, x2: width - m.r, y1: cy, y2: cy, class: "mv-gridline" }, svgEl);
    svgLines(svgEl, m.l - 8, cy, names[i], { "text-anchor": "end", class: "mv-tick" }, t.label);
  });
  for (const t of ticks(y0, y1, Math.max(3, Math.floor((width - m.l) / 70)))) {
    if (t % 1) continue;
    svgTick(svgEl, x(t), H - 10, String(t), 0, width);
  }
  const maxSize = Math.max(1, ...items.map((it) => it.size || 0));
  const g = sv("g", {}, svgEl);
  const jitter = {};
  items.sort((a, b) => (b.size || 0) - (a.size || 0)).forEach((it) => {
    const row = tiers.findIndex((t) => t.tier === it.tier);
    if (row < 0) return;
    const k = `${row}:${it.year}`;
    jitter[k] = (jitter[k] || 0) + 1;
    const off = ((jitter[k] - 1) % 3 - 1) * 7;
    const cx = x(it.year), cy = m.t + row * rowH + rowH / 2 + off;
    const r = 4 + 10 * Math.sqrt((it.size || 0) / maxSize);
    const color = cssVar(ctx, `--${it.cls || "lit"}`);
    const node = sv("g", { class: "mv-bubble", tabindex: 0 }, g);
    sv("circle", { cx, cy, r: Math.max(r, 9), fill: "transparent" }, node);
    sv("circle", { cx, cy, r, fill: it.cited ? color : cssVar(ctx, "--surface"), "fill-opacity": it.cited ? 0.78 : 1, stroke: color,
                   "stroke-width": it.landmark ? 3 : 1.5 }, node);
    hoverable(node, ctx, () => ({ title: it.title, rows: [
      { label: "design", value: it.design || "—" }, { label: "year", value: String(it.year) },
      ...(it.venue ? [{ label: "venue", value: it.venue }] : []),
      ...(it.size_label ? [{ label: it.size_label, value: int(it.size) }] : [])],
      note: it.cited ? `Cited as [${it.number}] — click to find it in the answer` : "Reviewed, not cited" }));
    node.addEventListener("click", () => {
      const chip = it.cite_id ? document.querySelector(`a.mes-cite[href="#${CSS.escape(it.cite_id)}"]`) : null;
      if (!chip) return;
      chip.scrollIntoView({ block: "center", behavior: motion() ? "smooth" : "auto" });
      chip.classList.remove("mes-flash");
      void chip.offsetWidth;
      chip.classList.add("mes-flash");
    });
    node.addEventListener("pointerenter", () => {
      lightUp(spec.anchor, (c) => c.getAttribute("href") === `#${it.cite_id}`);
    });
    node.addEventListener("pointerleave", () => lightUp(spec.anchor, null));
  });
}

// spec: {nodes: [{id, label}], edges: [{source, target, label, reliability}]}
function networkView(root, spec, ctx) {
  const nodes = (spec.nodes || []).map((n) => ({ ...n }));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges = (spec.edges || []).filter((e) => byId.has(e.source) && byId.has(e.target) && e.source !== e.target);
  if (!nodes.length || !edges.length) return;
  const width = Math.max(260, root.clientWidth || 360);
  const H = Math.max(240, Math.min(420, 90 + nodes.length * 18));
  // a small force layout, run to rest before drawing (seeded: the same graph is drawn the same way)
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / nodes.length;
    n.x = width / 2 + Math.cos(a) * width * 0.3 + rand() * 4;
    n.y = H / 2 + Math.sin(a) * H * 0.3 + rand() * 4;
  });
  const degree = new Map();
  for (const e of edges) { degree.set(e.source, (degree.get(e.source) || 0) + 1); degree.set(e.target, (degree.get(e.target) || 0) + 1); }
  const pos = ctx.state.netpos || {};
  for (const n of nodes) if (pos[n.id]) { n.x = pos[n.id][0] * width; n.y = pos[n.id][1] * H; }
  if (!ctx.state.netpos) {
    for (let it = 0; it < 260; it++) {
      const k = 1 - it / 260;
      for (const a of nodes) {
        let fx = (width / 2 - a.x) * 0.01, fy = (H / 2 - a.y) * 0.01;
        for (const b of nodes) {
          if (a === b) continue;
          const dx = a.x - b.x, dy = a.y - b.y, d2 = Math.max(dx * dx + dy * dy, 25);
          fx += (dx / d2) * 1800; fy += (dy / d2) * 1800;
        }
        a.fx = fx; a.fy = fy;
      }
      for (const e of edges) {
        const a = byId.get(e.source), b = byId.get(e.target);
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(Math.hypot(dx, dy), 1), f = (d - 110) * 0.04;
        a.fx += (dx / d) * f; a.fy += (dy / d) * f; b.fx -= (dx / d) * f; b.fy -= (dy / d) * f;
      }
      for (const n of nodes) {
        n.x = Math.max(40, Math.min(width - 40, n.x + Math.max(-12, Math.min(12, n.fx)) * k));
        n.y = Math.max(24, Math.min(H - 24, n.y + Math.max(-12, Math.min(12, n.fy)) * k));
      }
    }
  }
  const wrap = h("div", "mv-chart");
  root.appendChild(wrap);
  const legend = h("div", "mv-legend");
  const REL = { High: "--ok", Moderate: "--s3", Low: "--before" };
  for (const [rel, v] of Object.entries(REL)) {
    if (!edges.some((e) => e.reliability === rel)) continue;
    const it = h("span"); const k = h("i", "mv-key"); k.style.background = cssVar(ctx, v);
    it.append(k, document.createTextNode(`${rel.toLowerCase()} reliability`)); legend.appendChild(it);
  }
  wrap.appendChild(legend);
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, class: "mv-net", role: "img",
                            "aria-label": `Knowledge-graph relations among ${nodes.length} entities` }, wrap);
  const defs = sv("defs", {}, svgEl);
  for (const [rel, v] of Object.entries(REL)) {
    const mk = sv("marker", { id: `${ctx.scope}-arrow-${rel}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
    sv("path", { d: "M0,0L10,5L0,10z", fill: cssVar(ctx, v) }, mk);
  }
  const lines = edges.map((e) => {
    const color = cssVar(ctx, REL[e.reliability] || "--before");
    const path = sv("path", { class: "mv-net-edge", stroke: color, "marker-end": `url(#${ctx.scope}-arrow-${REL[e.reliability] ? e.reliability : "Low"})` }, svgEl);
    const label = svgText(svgEl, 0, 0, e.label, { class: "mv-net-label", "text-anchor": "middle" });
    return { e, path, label };
  });
  const circles = nodes.map((n) => {
    const g = sv("g", { class: "mv-net-node", tabindex: 0 }, svgEl);
    const r = 6 + Math.min(8, 2 * (degree.get(n.id) || 1));
    sv("circle", { r, fill: cssVar(ctx, "--accent-tint"), stroke: cssVar(ctx, "--accent"), "stroke-width": 2 }, g);
    svgText(g, 0, -r - 5, n.label.length > 28 ? `${n.label.slice(0, 27)}…` : n.label, { "text-anchor": "middle" });
    hoverable(g, ctx, () => ({ title: n.label, rows: edges.filter((e) => e.source === n.id || e.target === n.id).slice(0, 8)
      .map((e) => ({ label: e.source === n.id ? `→ ${byId.get(e.target).label}` : `← ${byId.get(e.source).label}`, value: e.label })) }));
    return { n, g, r };
  });
  const place = () => {
    for (const { e, path, label } of lines) {
      const a = byId.get(e.source), b = byId.get(e.target);
      const rb = circles.find((c) => c.n === b).r + 3;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(Math.hypot(dx, dy), 1);
      const ex = b.x - (dx / d) * rb, ey = b.y - (dy / d) * rb;
      const mx = (a.x + ex) / 2 - (dy / d) * 14, my = (a.y + ey) / 2 + (dx / d) * 14;
      path.setAttribute("d", `M${a.x},${a.y}Q${mx},${my} ${ex},${ey}`);
      label.setAttribute("x", (a.x + 2 * mx + ex) / 4);
      label.setAttribute("y", (a.y + 2 * my + ey) / 4);
    }
    for (const c of circles) c.g.setAttribute("transform", `translate(${c.n.x},${c.n.y})`);
  };
  place();
  const focus = (id) => {
    svgEl.classList.toggle("mv-focus", !!id);
    for (const { e, path, label } of lines) {
      const on = !!id && (e.source === id || e.target === id);
      path.classList.toggle("mv-on", on); label.classList.toggle("mv-on", on);
    }
    for (const c of circles) {
      c.g.classList.toggle("mv-on", !!id && (c.n.id === id || edges.some((e) => (e.source === id && e.target === c.n.id) || (e.target === id && e.source === c.n.id))));
    }
  };
  for (const c of circles) {
    c.g.addEventListener("pointerenter", () => focus(c.n.id));
    c.g.addEventListener("pointerleave", () => { if (!drag) focus(null); });
    c.g.addEventListener("pointerdown", (ev) => { drag = c; c.g.setPointerCapture && c.g.setPointerCapture(ev.pointerId); ev.preventDefault(); });
  }
  let drag = null;
  svgEl.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const box = svgEl.getBoundingClientRect();
    drag.n.x = Math.max(20, Math.min(width - 20, ev.clientX - box.left));
    drag.n.y = Math.max(20, Math.min(H - 20, ev.clientY - box.top));
    place();
  });
  const release = () => {
    if (!drag) return;
    drag = null;
    ctx.state.netpos = Object.fromEntries(nodes.map((n) => [n.id, [n.x / width, n.y / H]]));
    focus(null);
  };
  svgEl.addEventListener("pointerup", release);
  svgEl.addEventListener("pointercancel", release);
  ctx.state.netpos = ctx.state.netpos || Object.fromEntries(nodes.map((n) => [n.id, [n.x / width, n.y / H]]));
}
