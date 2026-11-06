// MES charts — covariate balance (Love plot: each covariate's standardized difference before and after weighting, the
// weighting replayed as motion) and the overlap of the propensity scores of both groups.

// c: {rows: [{label, before, after, mean_t, mean_c}], threshold, groups: [treated, comparison], weighting}
function love(body, c, ctx, s) {
  let rows = (c.rows || []).filter((r) => isNum(r.before) || isNum(r.after));
  if (!rows.length) { body.appendChild(h("div", "mv-empty", "No covariates.")); return; }
  rows.sort((a, b) => Math.abs(b.before ?? 0) - Math.abs(a.before ?? 0));
  const allKey = `all:${s.id}`, showKey = `show:${s.id}`;
  const LIMIT = 15;
  const all = !!ctx.state[allKey];
  const show = ctx.state[showKey] || "both";
  s.addTool(segmented([{ key: "both", label: "Before → after" }, { key: "before", label: "Before" }, { key: "after", label: "After" }],
                      show, (k) => { ctx.state[showKey] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx)); }, "Weighting"));
  const play = button("▶ Replay weighting", "mv-chip-btn", () => replay(true));
  s.addTool(play);
  if (rows.length > LIMIT) {
    s.addTool(button(all ? `Top ${LIMIT}` : `All ${rows.length}`, "mv-chip-btn", () => {
      ctx.state[allKey] = !all; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }));
    if (!all) rows = rows.slice(0, LIMIT);
  }
  const legend = h("div", "mv-legend");
  const lb = h("span"); lb.append(h("i", "mv-key-ring"), document.createTextNode("before weighting"));
  const la = h("span"); const ka = h("i", "mv-key mv-key-dot"); ka.style.background = cssVar(ctx, "--accent");
  la.append(ka, document.createTextNode(`after ${c.weighting || "weighting"}`));
  legend.append(lb, la);
  body.appendChild(legend);

  const width = Math.max(300, body.clientWidth || 600);
  const labelW = Math.min(250, Math.max(120, width * 0.36));
  const rowH = 22, m = { t: 6, b: 38, r: 16 };
  const H = m.t + rows.length * rowH + m.b;
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img",
                            "aria-label": "Standardized mean difference of each covariate before and after weighting" }, body);
  const maxV = Math.max(c.threshold * 2 || 0.2, ...rows.map((r) => Math.max(Math.abs(r.before ?? 0), Math.abs(r.after ?? 0))));
  const x = scale(0, maxV * 1.08, labelW + 10, width - m.r);
  for (const t of ticks(0, maxV * 1.08, Math.max(3, Math.floor((width - labelW) / 70)))) {
    sv("line", { x1: x(t), x2: x(t), y1: m.t, y2: m.t + rows.length * rowH, class: "mv-gridline" }, svgEl);
    svgTick(svgEl, x(t), m.t + rows.length * rowH + 14, fmt(t, t < 1 ? 2 : 1), 0, width);
  }
  svgText(svgEl, (x.r[0] + x.r[1]) / 2, m.t + rows.length * rowH + 31, "|standardized mean difference|",
          { "text-anchor": "middle", class: "mv-axis-title" });
  if (isNum(c.threshold)) {
    sv("line", { x1: x(c.threshold), x2: x(c.threshold), y1: m.t - 2, y2: m.t + rows.length * rowH, class: "mv-thresh" }, svgEl);
    svgText(svgEl, x(c.threshold) + 4, m.t + 8, `${fmt(c.threshold, 2)} balance threshold`, { class: "mv-tick" });
  }
  const accent = cssVar(ctx, "--accent"), before = cssVar(ctx, "--before"), surface = cssVar(ctx, "--surface");
  const movers = [];
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    const g = sv("g", { tabindex: 0, "aria-label": `${r.label}: ${fmt(Math.abs(r.before), 3)} before, ${fmt(Math.abs(r.after), 3)} after` }, svgEl);
    sv("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" }, g);
    const name = fitText(svgEl, r.label, "mv-tick", labelW - 6);
    svgText(g, labelW, cy + 4, name, { "text-anchor": "end", class: "mv-tick", fill: Math.abs(r.after ?? 0) >= (c.threshold || 0.1) ? cssVar(ctx, "--bad") : undefined });
    const xb = isNum(r.before) ? x(Math.abs(r.before)) : null, xa = isNum(r.after) ? x(Math.abs(r.after)) : null;
    if (show === "both" && xb !== null && xa !== null) {
      sv("line", { x1: Math.min(xa, xb), x2: Math.max(xa, xb), y1: cy, y2: cy, stroke: before, "stroke-width": 1.5, opacity: 0.55 }, g);
    }
    if (show !== "after" && xb !== null) sv("circle", { cx: xb, cy, r: 4.5, fill: surface, stroke: before, "stroke-width": 2 }, g);
    if (show !== "before" && xa !== null) {
      const dot = sv("circle", { cx: 0, cy: 0, r: 5, fill: accent, stroke: surface, "stroke-width": 2, class: "mv-move" }, g);
      movers.push({ dot, xa, xb: xb ?? xa, cy });
    }
    hoverable(g, ctx, () => ({ title: r.label, rows: [
      { label: "before", value: fmt(r.before, 3) }, { label: "after", value: fmt(r.after, 3) },
      ...(r.mean_t !== undefined ? [{ label: `mean · ${(c.groups || [])[0] || "treated"}`, value: fmt(r.mean_t) },
                                    { label: `mean · ${(c.groups || [])[1] || "comparison"}`, value: fmt(r.mean_c) }] : [])],
      note: "Means before weighting; 0.1 is the usual threshold of balance" }));
  });
  // the weighting as motion: each covariate's dot starts where it was before weighting and moves to where it is after
  function place(atBefore) {
    for (const mv of movers) mv.dot.style.transform = `translate(${atBefore ? mv.xb : mv.xa}px, ${mv.cy}px)`;
  }
  function replay(force) {
    if (!motion() || (show === "after" && !force)) { place(false); return; }
    for (const mv of movers) mv.dot.style.transition = "none";
    place(true);
    void svgEl.getBoundingClientRect();
    requestAnimationFrame(() => {
      for (const [i, mv] of movers.entries()) { mv.dot.style.transition = ""; mv.dot.style.transitionDelay = `${Math.min(i, 20) * 25}ms`; }
      place(false);
    });
  }
  if (fresh(ctx, s)) replay(false); else place(false);
}

// The density of a distribution given as quantiles, on a regular grid: the slope of its (piecewise linear) distribution
// function between the 2.5th and 97.5th percentile, smoothed, scaled to a peak of 1 — the shape, not the counts
function quantileDensity(q, v, x0, x1, bins = 44) {
  const step = (x1 - x0) / bins;
  const raw = [];
  for (let j = 0; j < bins; j++) {
    const a = x0 + j * step, b = a + step;
    const inside = b > v[0] && a < v[v.length - 1];
    raw.push(inside ? (cdf(q, v, b) - cdf(q, v, a)) / step : 0);
  }
  const kernel = [1, 2, 3, 2, 1];
  const smooth = raw.map((_, j) => {
    if (!(x0 + (j + 1) * step > v[0] && x0 + j * step < v[v.length - 1])) return 0;   // nothing outside the percentiles
    let sum = 0, wsum = 0;
    kernel.forEach((w, k) => { const i = j + k - 2; if (i >= 0 && i < raw.length) { sum += w * raw[i]; wsum += w; } });
    return sum / wsum;
  });
  const top = Math.max(...smooth, 1e-9);
  return smooth.map((d, j) => [x0 + (j + 0.5) * step, d / top]);
}

function cdf(q, v, x0) {               // the share below x0, linear between the given percentiles
  if (x0 <= v[0]) return q[0];
  for (let i = 0; i + 1 < v.length; i++) {
    if (x0 <= v[i + 1]) return q[i] + (q[i + 1] - q[i]) * (x0 - v[i]) / Math.max(v[i + 1] - v[i], 1e-9);
  }
  return q[q.length - 1];
}

// c: {q, treated, control, names: [treated, comparison], overlap} — or {ranges: [{name, p5, p50, p95}]} (older studies)
function overlap(body, c, ctx, s) {
  const width = Math.max(300, body.clientWidth || 600);
  const colors = [cssVar(ctx, "--s1"), cssVar(ctx, "--s2")];
  const names = c.names || ["treated", "comparison"];
  const legend = h("div", "mv-legend");
  names.forEach((n, i) => { const it = h("span"); const k = h("i", "mv-key"); k.style.background = colors[i]; it.append(k, document.createTextNode(n)); legend.appendChild(it); });
  body.appendChild(legend);
  if (!c.q) {                          // three quantiles only: the 5th–95th percentile of each group, its median
    const rows = (c.ranges || []).map((r, i) => ({ label: r.name, value: r.p50, lo: r.p5, hi: r.p95, thick: true, median: true, color: colors[i],
                                                   text: fmt(r.p50, 3), textSub: `5–95%: ${fmt(r.p5, 2)}–${fmt(r.p95, 2)}` }));
    intervalRows(body, { rows, domain: [0, 1], axis: "Propensity score (5th, 50th and 95th percentile)", labelWidth: "30%" }, ctx, s);
    return;
  }
  const H = 210, m = { l: 14, r: 14, t: 12, b: 36 };
  const mid = m.t + (H - m.t - m.b) / 2;
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img",
                            "aria-label": "Propensity score distributions of both groups, mirrored" }, body);
  const maxV = Math.min(1, Math.max(...c.treated, ...c.control) * 1.05);
  const x = scale(0, Math.max(maxV, 0.05), m.l, width - m.r);
  const amp = (H - m.t - m.b) / 2 - 4;
  const support = [Math.max(c.treated[0], c.control[0]), Math.min(c.treated[c.treated.length - 1], c.control[c.control.length - 1])];
  if (support[1] > support[0]) {
    sv("rect", { x: x(support[0]), y: m.t, width: x(support[1]) - x(support[0]), height: H - m.t - m.b, fill: cssVar(ctx, "--surface2") }, svgEl);
  }
  const anim = fresh(ctx, s);
  [[c.treated, -1, 0], [c.control, 1, 1]].forEach(([v, sign, i]) => {
    const dens = quantileDensity(c.q, v, 0, x.d[1]);
    let d = `M${x(dens[0][0]).toFixed(1)},${mid}`;
    for (const [vx, dy] of dens) d += `L${x(vx).toFixed(1)},${(mid + sign * dy * amp).toFixed(1)}`;
    d += `L${x(dens[dens.length - 1][0]).toFixed(1)},${mid}Z`;
    sv("path", { d, fill: colors[i], opacity: 0.22, class: anim ? (sign < 0 ? "mv-grow-up" : "mv-fade") : "" }, svgEl);
    let line = "";
    dens.forEach(([vx, dy], k) => { line += `${k ? "L" : "M"}${x(vx).toFixed(1)},${(mid + sign * dy * amp).toFixed(1)}`; });
    sv("path", { d: line, fill: "none", stroke: colors[i], "stroke-width": 2, "stroke-linejoin": "round" }, svgEl);
  });
  sv("line", { x1: m.l, x2: width - m.r, y1: mid, y2: mid, class: "mv-baseline" }, svgEl);
  svgText(svgEl, width - m.r, m.t + 10, `▲ ${names[0]}`, { "text-anchor": "end", class: "mv-tick" });
  svgText(svgEl, width - m.r, H - m.b - 4, `▼ ${names[1]}`, { "text-anchor": "end", class: "mv-tick" });
  if (support[1] > support[0]) s.caption.textContent = [s.caption.textContent, "Grey: the scores both groups share (between their 2.5th and 97.5th percentiles)."].filter(Boolean).join(" ");
  for (const t of ticks(0, x.d[1], Math.max(3, Math.floor(width / 80)))) {
    svgTick(svgEl, x(t), H - m.b + 16, fmt(t, 2), 0, width);
  }
  svgText(svgEl, (m.l + width - m.r) / 2, H - 4, "Propensity score (probability of treatment)", { "text-anchor": "middle", class: "mv-axis-title" });
  const cross = sv("line", { y1: m.t, y2: H - m.b, class: "mv-cross", opacity: 0 }, svgEl);
  const hit = sv("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: H - m.t - m.b, class: "mv-hit" }, svgEl);
  hit.addEventListener("pointermove", (e) => {
    const box = svgEl.getBoundingClientRect();
    const v = Math.max(0, x.invert(e.clientX - box.left));
    cross.setAttribute("x1", x(v)); cross.setAttribute("x2", x(v)); cross.setAttribute("opacity", 1);
    ctx.tip.show(e, `Score ${fmt(v, 3)}`, [
      { label: `${names[0]} below`, value: pct(100 * cdf(c.q, c.treated, v), 0), color: colors[0] },
      { label: `${names[1]} below`, value: pct(100 * cdf(c.q, c.control, v), 0), color: colors[1] }]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); ctx.tip.hide(); });
}
