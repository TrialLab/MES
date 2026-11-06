// MES charts — descriptive studies: the trajectory of a measurement after time zero (median line and IQR band per
// group, a crosshair reading every group at an interval) and treatment pathways (first, second and third line as a
// flow; hover a link or a node for its patients).

// c: {groups: [{label, role, points: [{t0, t1, n, median, q1, q3}]}], unit, bin_h, horizon_h, variable}
function trajectory(body, c, ctx, s) {
  const groups = (c.groups || []).filter((g) => g.points && g.points.length);
  if (!groups.length) { body.appendChild(h("div", "mv-empty", "No interval has 10 patients.")); return; }
  const colors = groupColors(ctx, groups, false);
  if (groups.length > 1) {
    const legend = h("div", "mv-legend");
    groups.forEach((g, i) => { const k = h("span"); const dot = h("i", "mv-key mv-key-dot"); dot.style.background = colors[i]; k.append(dot, document.createTextNode(g.label)); legend.appendChild(k); });
    body.appendChild(legend);
  }
  const width = Math.max(300, body.clientWidth || 600), height = 260;
  const m = { l: 48, r: 16, t: 12, b: 34 };
  const xs = [0, c.horizon_h || Math.max(...groups.flatMap((g) => g.points.map((p) => p.t1)))];
  const ys = groups.flatMap((g) => g.points.flatMap((p) => [p.q1, p.q3, p.median])).filter(isNum);
  let [y0, y1] = extent(ys, 0.08);
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const x = scale(xs[0], xs[1], m.l, width - m.r), y = scale(y0, y1, height - m.b, m.t);
  const svgEl = sv("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                           "aria-label": `${c.variable || "Measurement"} over time` }, body);
  const ax = sv("g", { class: "mv-axis" }, svgEl);
  for (const t of ticks(y0, y1, 5)) {
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, ax);
    svgText(ax, m.l - 6, y(t) + 4, fmt(t), { "text-anchor": "end", class: "mv-tick" });
  }
  const step = c.bin_h || 6;
  const xt = ticks(xs[0], xs[1], Math.max(3, Math.floor(width / 90))).filter((t) => t % step === 0 || step > 24);
  for (const t of (xt.length >= 2 ? xt : ticks(xs[0], xs[1], 5))) {
    sv("line", { x1: x(t), x2: x(t), y1: height - m.b, y2: height - m.b + 4, class: "mv-baseline" }, ax);
    svgTick(ax, x(t), height - m.b + 16, `${fmt(t, 0)} h`, 0, width);
  }
  sv("line", { x1: m.l, x2: width - m.r, y1: height - m.b + 0.5, y2: height - m.b + 0.5, class: "mv-baseline" }, ax);
  if (c.unit) svgText(ax, 10, m.t + 4, c.unit, { class: "mv-axis-title", "dominant-baseline": "hanging" });
  const anim = fresh(ctx, s);
  groups.forEach((g, i) => {
    const pts = g.points.map((p) => ({ ...p, t: (p.t0 + p.t1) / 2 }));
    const band = pts.map((p) => `${x(p.t).toFixed(1)},${y(p.q3).toFixed(1)}`).join(" L") + " L" +
      pts.slice().reverse().map((p) => `${x(p.t).toFixed(1)},${y(p.q1).toFixed(1)}`).join(" L");
    sv("path", { d: `M${band}Z`, fill: colors[i], "fill-opacity": 0.12, stroke: "none" }, svgEl);
    const line = sv("path", { d: "M" + pts.map((p) => `${x(p.t).toFixed(1)},${y(p.median).toFixed(1)}`).join(" L"),
                             fill: "none", stroke: colors[i], "stroke-width": 2.2, "stroke-linejoin": "round",
                             class: anim ? "mv-draw" : "" }, svgEl);
    if (anim) line.style.setProperty("--len", `${line.getTotalLength ? Math.ceil(line.getTotalLength()) : 600}`);
    for (const p of pts) sv("circle", { cx: x(p.t), cy: y(p.median), r: 3, fill: colors[i], stroke: cssVar(ctx, "--surface"), "stroke-width": 1.5 }, svgEl);
    const last = pts[pts.length - 1];
    if (groups.length > 1) svgText(svgEl, Math.min(width - m.r, x(last.t) + 4), y(last.median) - 6, g.label.length > 26 ? `${g.label.slice(0, 25)}…` : g.label,
                                   { "text-anchor": x(last.t) > width - 140 ? "end" : "start", class: "mv-direct", fill: colors[i] });
  });
  // the crosshair: every group at the interval under the pointer
  const cross = sv("line", { y1: m.t, y2: height - m.b, class: "mv-cross", opacity: 0 }, svgEl);
  const hit = sv("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: "transparent", class: "mv-hit" }, svgEl);
  hit.setAttribute("tabindex", "0");
  const at = (px) => {
    const t = x.invert(px);
    const b = Math.floor(t / step);
    return { b, rows: groups.map((g, i) => ({ g, i, p: g.points.find((p) => Math.floor(p.t0 / step) === b) })) };
  };
  hit.addEventListener("pointermove", (e) => {
    const r = svgEl.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * width;
    const { b, rows } = at(px);
    cross.setAttribute("x1", x((b + 0.5) * step)); cross.setAttribute("x2", x((b + 0.5) * step)); cross.setAttribute("opacity", 1);
    ctx.tip.show(e, `${fmt(b * step, 0)}–${fmt((b + 1) * step, 0)} h`, rows.map(({ g, i, p }) => ({
      label: g.label, color: colors[i], value: p ? `${fmt(p.median)} (${fmt(p.q1)}–${fmt(p.q3)}) · n ${int(p.n)}` : "<10" })));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); ctx.tip.hide(); });
}

// c: {links: [{from: [level, name], to: [level, name], n}], names, n}
function sankey(body, c, ctx) {
  const links = (c.links || []).filter((l) => isNum(l.n) && l.n > 0);
  if (!links.length) { body.appendChild(h("div", "mv-empty", "No line has 10 patients.")); return; }
  const width = Math.max(320, body.clientWidth || 640);
  const levels = [1, 2, 3];
  const nodes = {};
  const key = (lv, name) => `${lv}|${name}`;
  for (const l of links) {
    const a = key(...l.from), b = key(...l.to);
    nodes[a] = nodes[a] || { lv: l.from[0], name: l.from[1], out: 0, in: 0 };
    nodes[b] = nodes[b] || { lv: l.to[0], name: l.to[1], out: 0, in: 0 };
    nodes[a].out += l.n; nodes[b].in += l.n;
  }
  const names = [...(c.names || []), "other", "none"];
  const color = (name) => name === "none" ? cssVar(ctx, "--line") : name === "other" ? cssVar(ctx, "--before")
    : cssVar(ctx, SERIES[Math.max(0, names.indexOf(name)) % SERIES.length]);
  const used = levels.filter((lv) => Object.values(nodes).some((n) => n.lv === lv));
  const total = Math.max(...used.map((lv) => Object.values(nodes).filter((n) => n.lv === lv).reduce((a, n) => a + Math.max(n.in, n.out), 0)));
  const height = Math.max(220, Math.min(420, 60 + 18 * Object.keys(nodes).length));
  const m = { t: 22, b: 10, l: 8, r: 8 }, gap = 8, nodeW = 14;
  const colX = (lv) => m.l + ((width - m.l - m.r - nodeW) * (used.indexOf(lv))) / Math.max(1, used.length - 1);
  const k = (height - m.t - m.b - gap * 6) / total;
  const svgEl = sv("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Treatment lines" }, body);
  for (const lv of used) {
    let yy = m.t;
    const col = Object.values(nodes).filter((n) => n.lv === lv).sort((a, b) => (a.name === "none") - (b.name === "none") || Math.max(b.in, b.out) - Math.max(a.in, a.out));
    for (const n of col) {
      n.h = Math.max(2, Math.max(n.in, n.out) * k); n.y = yy; n.x = colX(lv); n.oy = yy; n.iy = yy;
      yy += n.h + gap;
    }
    svgText(svgEl, colX(lv) + (lv === used[used.length - 1] ? nodeW : 0), 12, ["first", "second", "third"][lv - 1] + " treatment",
            { "text-anchor": lv === used[0] ? "start" : lv === used[used.length - 1] ? "end" : "middle", class: "mv-axis-title" });
  }
  const linkLayer = sv("g", {}, svgEl);
  for (const l of links.sort((a, b) => b.n - a.n)) {
    const a = nodes[key(...l.from)], b = nodes[key(...l.to)];
    const w = Math.max(1, l.n * k);
    const y1 = a.oy + w / 2, y2 = b.iy + w / 2;
    a.oy += w; b.iy += w;
    const x1 = a.x + nodeW, x2 = b.x, mx = (x1 + x2) / 2;
    const p = sv("path", { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, fill: "none", stroke: color(l.from[1]),
                           "stroke-opacity": b.name === "none" ? 0.18 : 0.35, "stroke-width": w, class: "mv-link" }, linkLayer);
    p.setAttribute("tabindex", "0");
    hoverable(p, ctx, () => ({ title: `${l.from[1]} → ${l.to[1] === "none" ? "no other treatment" : l.to[1]}`,
                               rows: [{ label: "patients", value: int(l.n) }, ...(isNum(c.n) ? [{ label: "share", value: pct((100 * l.n) / c.n) }] : [])] }));
  }
  // a narrow card: the labels of the last line would cross those of the middle one (they stay in the tooltip / table)
  const narrow = width < 560;
  for (const n of Object.values(nodes)) {
    const r = sv("rect", { x: n.x, y: n.y, width: nodeW, height: n.h, rx: 3, fill: color(n.name) }, svgEl);
    r.setAttribute("tabindex", "0");
    hoverable(r, ctx, () => ({ title: `${n.name} (${["first", "second", "third"][n.lv - 1]})`, rows: [{ label: "patients", value: int(Math.max(n.in, n.out)) }] }));
    if (n.h >= (narrow ? 16 : 12) && !(narrow && used.length > 2 && n.lv === used[used.length - 1])) {
      const right = n.lv !== used[used.length - 1];
      svgText(svgEl, right ? n.x + nodeW + 5 : n.x - 5, n.y + n.h / 2 + 4, `${n.name} · ${int(Math.max(n.in, n.out))}`,
              { "text-anchor": right ? "start" : "end", class: "mv-sk-label" });
    }
  }
}
