// MES charts — distributions (histogram with a range you can drag), trends over periods, the ROC curve and the
// calibration of a prognostic model, and bars for any aggregated table (the analyst's, an exploratory query's).

// c: {edges, counts, label, unit, summary: {median, q1, q3, p5, p95, mean, n}}
function hist(body, c, ctx, s) {
  const edges = c.edges || [], counts = c.counts || [];
  if (edges.length !== counts.length + 1 || !counts.length) { body.appendChild(h("div", "mv-empty", "No histogram.")); return; }
  const shown = counts.map((k) => (k > 0 && k < 10 ? null : k));          // bins of fewer than 10 patients: not shown
  const approx = counts.map((k) => (k > 0 && k < 10 ? 5 : k));
  const total = approx.reduce((a, b) => a + b, 0) || 1;
  const width = Math.max(300, body.clientWidth || 600);
  const H = 220, m = { l: 48, r: 14, t: 16, b: 40 };
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img", "aria-label": `Distribution of ${c.label || "the variable"}` }, body);
  const x = scale(edges[0], edges[edges.length - 1], m.l, width - m.r);
  const top = Math.max(...approx, 10);
  const y = scale(0, top * 1.08, H - m.b, m.t);
  for (const t of ticks(0, top * 1.08, 4)) {
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, svgEl);
    svgText(svgEl, m.l - 8, y(t) + 4, int(t), { "text-anchor": "end", class: "mv-tick" });
  }
  const sm = c.summary || {};
  if (isNum(sm.q1) && isNum(sm.q3)) {
    sv("rect", { x: x(sm.q1), y: m.t, width: Math.max(1, x(sm.q3) - x(sm.q1)), height: H - m.t - m.b, fill: cssVar(ctx, "--accent-tint") }, svgEl);
  }
  const anim = fresh(ctx, s);
  const accent = cssVar(ctx, "--accent");
  const bars = counts.map((k, i) => {
    const x0 = x(edges[i]) + 1, x1 = x(edges[i + 1]) - 1;
    const small = shown[i] === null;
    const r = sv("rect", { x: x0, y: y(approx[i]), width: Math.max(1, x1 - x0), height: Math.max(0, y(0) - y(approx[i])), rx: 2,
                           fill: small ? cssVar(ctx, "--line") : accent, opacity: small ? 1 : 0.85, class: anim ? "mv-grow-up" : "" }, svgEl);
    if (anim) r.style.animationDelay = `${i * 18}ms`;
    return r;
  });
  sv("line", { x1: m.l, x2: width - m.r, y1: y(0), y2: y(0), class: "mv-baseline" }, svgEl);
  for (const t of ticks(edges[0], edges[edges.length - 1], Math.max(3, Math.floor(width / 80)))) {
    svgTick(svgEl, x(t), H - m.b + 15, fmt(t), 0, width);
  }
  svgText(svgEl, (m.l + width - m.r) / 2, H - 6, `${c.label || ""}${c.unit ? ` (${c.unit})` : ""}`, { "text-anchor": "middle", class: "mv-axis-title" });
  if (isNum(sm.median)) {
    sv("line", { x1: x(sm.median), x2: x(sm.median), y1: m.t - 4, y2: y(0), stroke: cssVar(ctx, "--ink"), "stroke-width": 1.5 }, svgEl);
    svgText(svgEl, x(sm.median) + 4, m.t + 6, `median ${fmt(sm.median)}`, { class: "mv-tick", fill: cssVar(ctx, "--ink") });
  }
  // the dragged range: the share of patients inside it, from the bins (prorated; bins of fewer than 10 count as 5)
  const sel = sv("rect", { y: m.t, height: H - m.t - m.b, fill: cssVar(ctx, "--s2"), opacity: 0, rx: 3 }, svgEl);
  const hit = sv("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: H - m.t - m.b, class: "mv-hit", tabindex: 0,
                           "aria-label": "Drag across the histogram to measure the share of patients in a range" }, svgEl);
  const readout = h("div", "mv-caption");
  body.appendChild(readout);
  const share = (a, b) => {
    let k = 0;
    counts.forEach((_, i) => {
      const lo = Math.max(a, edges[i]), hi = Math.min(b, edges[i + 1]);
      if (hi > lo) k += approx[i] * (hi - lo) / Math.max(edges[i + 1] - edges[i], 1e-12);
    });
    return k;
  };
  const say = (a, b) => {
    const k = share(a, b);
    readout.textContent = `Selected ${fmt(a)}–${fmt(b)}${c.unit ? ` ${c.unit}` : ""}: about ${pct((100 * k) / total, 0)} of patients `
      + `(≈ ${int(Math.round(k / 10) * 10)}) · double-click to clear`;
  };
  const state = ctx.state[`brush:${s.id}`];
  const paint = (a, b) => {
    sel.setAttribute("x", x(Math.min(a, b)));
    sel.setAttribute("width", Math.max(1, Math.abs(x(b) - x(a))));
    sel.setAttribute("opacity", 0.16);
  };
  if (state) { paint(state[0], state[1]); say(Math.min(...state), Math.max(...state)); }
  else readout.textContent = isNum(sm.median) ? `Median ${fmt(sm.median)} (IQR ${fmt(sm.q1)}–${fmt(sm.q3)})${isNum(sm.n) ? ` · ${int(sm.n)} patients` : ""} · drag across the bars to measure a range` : "Drag across the bars to measure a range";
  let start = null;
  const valueAt = (e) => {
    const box = svgEl.getBoundingClientRect();
    return Math.max(edges[0], Math.min(edges[edges.length - 1], x.invert(e.clientX - box.left)));
  };
  hit.addEventListener("pointerdown", (e) => { start = valueAt(e); hit.setPointerCapture && hit.setPointerCapture(e.pointerId); ctx.tip.hide(); });
  hit.addEventListener("pointermove", (e) => {
    const v = valueAt(e);
    if (start !== null) { paint(start, v); say(Math.min(start, v), Math.max(start, v)); return; }
    const i = Math.max(0, Math.min(counts.length - 1, edges.findIndex((edge, j) => j + 1 < edges.length && v < edges[j + 1])));
    bars.forEach((b, j) => b.setAttribute("opacity", j === i ? 1 : shown[j] === null ? 1 : 0.85));
    ctx.tip.show(e, `${fmt(edges[i])}–${fmt(edges[i + 1])}${c.unit ? ` ${c.unit}` : ""}`,
                 [{ label: "patients", value: shown[i] === null ? "<10" : int(counts[i]) },
                  { label: "share", value: shown[i] === null ? "—" : pct((100 * counts[i]) / total, 1) }]);
  });
  const end = (e) => {
    if (start === null) return;
    const v = valueAt(e);
    if (Math.abs(x(v) - x(start)) > 3) ctx.state[`brush:${s.id}`] = [Math.min(start, v), Math.max(start, v)];
    start = null;
  };
  hit.addEventListener("pointerup", end);
  hit.addEventListener("pointerleave", () => { ctx.tip.hide(); bars.forEach((b, j) => b.setAttribute("opacity", shown[j] === null ? 1 : 0.85)); });
  hit.addEventListener("dblclick", () => {
    delete ctx.state[`brush:${s.id}`];
    sel.setAttribute("opacity", 0);
    readout.textContent = "Drag across the bars to measure a range";
  });
}

// c: {rows: [{period, value, lo, hi, n}], unit, measure, axis}
function trend(body, c, ctx, s) {
  const rows = (c.rows || []).filter((r) => isNum(r.value));
  if (rows.length < 2) { body.appendChild(h("div", "mv-empty", "Fewer than two periods.")); return; }
  const width = Math.max(300, body.clientWidth || 600);
  const H = 230, m = { l: 50, r: 18, t: 16, b: 42 };
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img", "aria-label": `${c.measure || "Value"} by period` }, body);
  const xs = rows.map((_, i) => m.l + 20 + (i * (width - m.l - m.r - 40)) / Math.max(1, rows.length - 1));
  const [lo, hi] = extent(rows.flatMap((r) => [r.value, r.lo, r.hi]), 0.15);
  const y = scale(c.zero === false ? lo : Math.min(0, lo), hi, H - m.b, m.t);
  for (const t of ticks(y.d[0], y.d[1], 4)) {
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, svgEl);
    svgText(svgEl, m.l - 8, y(t) + 4, `${fmt(t, Math.abs(t) < 10 && t % 1 ? 1 : 0)}${c.unit || ""}`, { "text-anchor": "end", class: "mv-tick" });
  }
  const accent = cssVar(ctx, "--accent");
  const withCi = rows.every((r) => isNum(r.lo) && isNum(r.hi));
  if (withCi) {
    let d = rows.map((r, i) => `${i ? "L" : "M"}${xs[i]},${y(r.hi)}`).join("");
    d += rows.slice().reverse().map((r, i) => `L${xs[rows.length - 1 - i]},${y(r.lo)}`).join("") + "Z";
    sv("path", { d, fill: accent, opacity: 0.12 }, svgEl);
  }
  const line = sv("path", { d: rows.map((r, i) => `${i ? "L" : "M"}${xs[i]},${y(r.value)}`).join(""), fill: "none", stroke: accent,
                            "stroke-width": 2, "stroke-linejoin": "round" }, svgEl);
  if (fresh(ctx, s) && line.getTotalLength) {
    line.style.setProperty("--len", Math.ceil(line.getTotalLength()));
    line.classList.add("mv-draw");
  }
  rows.forEach((r, i) => {
    svgText(svgEl, xs[i], H - m.b + 16, r.period, { "text-anchor": "middle", class: "mv-tick" });
    const g = sv("g", { tabindex: 0 }, svgEl);
    sv("circle", { cx: xs[i], cy: y(r.value), r: 12, fill: "transparent" }, g);
    sv("circle", { cx: xs[i], cy: y(r.value), r: 5, fill: accent, stroke: cssVar(ctx, "--surface"), "stroke-width": 2 }, g);
    if (i === rows.length - 1 || i === 0) {
      svgText(svgEl, xs[i], y(r.value) - 11, `${fmt(r.value, 1)}${c.unit || ""}`, { "text-anchor": "middle", class: "mv-tick", fill: cssVar(ctx, "--ink") });
    }
    hoverable(g, ctx, () => ({ title: r.period, rows: [
      { label: c.measure || "value", value: `${fmt(r.value, 1)}${c.unit || ""}` },
      ...(isNum(r.lo) ? [{ label: c.interval || "95% CI", value: `${fmt(r.lo, 1)}–${fmt(r.hi, 1)}${c.unit || ""}` }] : []),
      ...(isNum(r.n) ? [{ label: "patients", value: int(r.n) }] : [])] }));
  });
  if (c.axis) svgText(svgEl, (m.l + width - m.r) / 2, H - 6, c.axis, { "text-anchor": "middle", class: "mv-axis-title" });
}

// c: {points: [[fpr, tpr]], auc}
function roc(body, c, ctx, s) {
  const pts = (c.points || []).filter((p) => Array.isArray(p) && p.length === 2);
  if (pts.length < 2) { body.appendChild(h("div", "mv-empty", "No ROC curve.")); return; }
  const width = Math.max(240, Math.min(body.clientWidth || 360, 420));
  const H = width, m = { l: 44, r: 12, t: 12, b: 40 };
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img", "aria-label": `ROC curve, AUC ${fmt(c.auc, 2)}` }, body);
  const x = scale(0, 1, m.l, width - m.r), y = scale(0, 1, H - m.b, m.t);
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    sv("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "mv-gridline" }, svgEl);
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, svgEl);
    svgTick(svgEl, x(t), H - m.b + 15, `${t * 100}%`, 0, width);
    svgText(svgEl, m.l - 6, y(t) + 4, `${t * 100}%`, { "text-anchor": "end", class: "mv-tick" });
  }
  sv("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: "mv-null", "stroke-dasharray": "4 4" }, svgEl);
  const accent = cssVar(ctx, "--accent");
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  sv("path", { d: `${d}L${x(1)},${y(0)}L${x(0)},${y(0)}Z`, fill: accent, opacity: 0.1 }, svgEl);
  const line = sv("path", { d, fill: "none", stroke: accent, "stroke-width": 2.2, "stroke-linejoin": "round" }, svgEl);
  if (fresh(ctx, s) && line.getTotalLength) { line.style.setProperty("--len", Math.ceil(line.getTotalLength())); line.classList.add("mv-draw"); }
  svgText(svgEl, x(0.55), y(0.18), `AUC ${fmt(c.auc, 2)}`, { class: "mv-axis-title", fill: cssVar(ctx, "--ink") });
  svgText(svgEl, (m.l + width - m.r) / 2, H - 6, "False positive rate (1 − specificity)", { "text-anchor": "middle", class: "mv-axis-title" });
  const dot = sv("circle", { r: 5, fill: accent, stroke: cssVar(ctx, "--surface"), "stroke-width": 2, opacity: 0 }, svgEl);
  const hit = sv("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: H - m.t - m.b, class: "mv-hit" }, svgEl);
  hit.addEventListener("pointermove", (e) => {
    const box = svgEl.getBoundingClientRect();
    const fx = x.invert(e.clientX - box.left);
    const p = pts.reduce((best, q) => (Math.abs(q[0] - fx) < Math.abs(best[0] - fx) ? q : best), pts[0]);
    dot.setAttribute("cx", x(p[0])); dot.setAttribute("cy", y(p[1])); dot.setAttribute("opacity", 1);
    ctx.tip.show(e, "A threshold of the model", [{ label: "sensitivity", value: pct(100 * p[1], 0) },
                                                 { label: "specificity", value: pct(100 * (1 - p[0]), 0) }]);
  });
  hit.addEventListener("pointerleave", () => { dot.setAttribute("opacity", 0); ctx.tip.hide(); });
}

// c: {bins: [{pred, obs, n}], slope}
function calibration(body, c, ctx, s) {
  const bins = (c.bins || []).filter((b) => isNum(b.pred) && isNum(b.obs));
  if (bins.length < 2) { body.appendChild(h("div", "mv-empty", "No calibration bins.")); return; }
  const width = Math.max(240, Math.min(body.clientWidth || 360, 420));
  const H = width, m = { l: 46, r: 12, t: 12, b: 40 };
  const top = Math.min(100, Math.ceil(Math.max(...bins.map((b) => Math.max(b.pred, b.obs))) / 10 + 0.0001) * 10);
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img", "aria-label": "Calibration of the predicted risks" }, body);
  const x = scale(0, top, m.l, width - m.r), y = scale(0, top, H - m.b, m.t);
  for (const t of ticks(0, top, 4)) {
    sv("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "mv-gridline" }, svgEl);
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, svgEl);
    svgTick(svgEl, x(t), H - m.b + 15, `${fmt(t, 0)}%`, 0, width);
    svgText(svgEl, m.l - 6, y(t) + 4, `${fmt(t, 0)}%`, { "text-anchor": "end", class: "mv-tick" });
  }
  sv("line", { x1: x(0), y1: y(0), x2: x(top), y2: y(top), class: "mv-null", "stroke-dasharray": "4 4" }, svgEl);
  svgText(svgEl, x(top * 0.62), y(top * 0.7) - 6, "perfect calibration", { class: "mv-tick", transform: `rotate(-45 ${x(top * 0.62)} ${y(top * 0.7) - 6})` });
  const accent = cssVar(ctx, "--accent");
  sv("path", { d: bins.map((b, i) => `${i ? "L" : "M"}${x(b.pred)},${y(b.obs)}`).join(""), fill: "none", stroke: accent, "stroke-width": 1.5, opacity: 0.6 }, svgEl);
  const maxN = Math.max(...bins.map((b) => b.n || 1));
  bins.forEach((b, i) => {
    const g = sv("g", { tabindex: 0 }, svgEl);
    sv("circle", { cx: x(b.pred), cy: y(b.obs), r: 12, fill: "transparent" }, g);
    sv("circle", { cx: x(b.pred), cy: y(b.obs), r: 3.5 + 3 * Math.sqrt((b.n || 1) / maxN), fill: accent, stroke: cssVar(ctx, "--surface"), "stroke-width": 2 }, g);
    hoverable(g, ctx, () => ({ title: `Tenth ${i + 1} of the predicted risks`, rows: [
      { label: "predicted", value: pct(b.pred, 1) }, { label: "observed", value: pct(b.obs, 1) }, { label: "patients", value: int(b.n) }] }));
  });
  svgText(svgEl, (m.l + width - m.r) / 2, H - 6, "Predicted risk", { "text-anchor": "middle", class: "mv-axis-title" });
  svgText(svgEl, 12, (m.t + H - m.b) / 2, "Observed", { "text-anchor": "middle", class: "mv-axis-title", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` });
}

// c: {rows: [{label, values: {key: number | "<10"}}], columns: [{key, label, unit, lo, hi}], default}
function bars(body, c, ctx, s) {
  const cols = (c.columns || []).filter((col) => (c.rows || []).some((r) => isNum((r.values || {})[col.key])));
  if (!cols.length) { body.appendChild(h("div", "mv-empty", "No numbers to draw.")); return; }
  const key = `col:${s.id}`;
  let chosen = ctx.state[key];
  if (!cols.some((col) => col.key === chosen)) chosen = (cols.find((col) => col.key === c.default) || cols[0]).key;
  if (cols.length > 1) {
    s.addTool(segmented(cols.slice(0, 6).map((col) => ({ key: col.key, label: col.label })), chosen, (k) => {
      ctx.state[key] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }, "Value"));
  }
  const col = cols.find((x) => x.key === chosen);
  const unit = col.unit || "";
  const accent = cssVar(ctx, "--accent");
  const rows = (c.rows || []).slice(0, 40).map((r) => {
    const v = (r.values || {})[col.key];
    const lo = col.lo ? (r.values || {})[col.lo] : null, hi = col.hi ? (r.values || {})[col.hi] : null;
    return { label: r.label, value: isNum(v) ? v : null, lo: isNum(lo) ? lo : null, hi: isNum(hi) ? hi : null, color: accent,
             text: isNum(v) ? `${fmt(v)}${unit}` : String(v ?? "—"), textSub: isNum(lo) && isNum(hi) ? `${fmt(lo)}–${fmt(hi)}` : "",
             tip: Object.entries(r.values || {}).slice(0, 8).map(([k2, v2]) => ({ label: k2, value: isNum(v2) ? fmt(v2) : String(v2 ?? "—") })) };
  });
  intervalRows(body, { rows, zero: true, unit, axis: col.label, labelWidth: "34%" }, ctx, s);
}
