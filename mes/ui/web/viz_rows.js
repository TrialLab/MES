// MES charts — interval rows: one row per estimate / group / stratum, label · interval on a shared axis · value.
// Used by the forest plot of effect estimates, the risks of each arm, rates by subgroup and medians with IQR.

// cfg: {rows: [{label, sub, value, lo, hi, text, primary, head, color, marks: [{value, hollow, color}]}],
//       log, nul (null value, drawn as a line), ref: {value, label}, axis (title), dir: [left, right],
//       unit, domain: [lo, hi] (optional), labelWidth}
function intervalRows(body, cfg, ctx, s) {
  const rows = cfg.rows || [];
  const grid = h("div", "mv-rows");
  if (cfg.labelWidth) grid.style.setProperty("--label-w", cfg.labelWidth);
  body.appendChild(grid);
  const values = [];
  for (const r of rows) {
    if (r.head) continue;
    values.push(r.value, r.lo, r.hi, ...((r.marks || []).map((m) => m.value)));
  }
  let domain = cfg.domain;
  if (!domain) {
    if (isNum(cfg.nul)) domain = nullDomain(values.concat(cfg.ref ? [cfg.ref.value] : []), cfg.nul, cfg.log);
    else {
      const [lo, hi] = extent(values.concat(cfg.ref ? [cfg.ref.value] : []), 0.08);
      domain = cfg.zero && lo > 0 ? [0, hi] : [lo, hi];
    }
  }
  const cells = [];
  rows.forEach((r, i) => {
    const row = h("div", r.head ? "mv-row mv-row-head" : r.primary ? "mv-row mv-row-primary" : "mv-row");
    if (r.head) {
      row.appendChild(h("div", null, r.label));
      grid.appendChild(row);
      return;
    }
    const label = h("div", "mv-rl", r.label);
    if (r.sub) label.appendChild(h("small", null, r.sub));
    const plot = h("div", "mv-rp");
    const value = h("div", "mv-rv");
    if (r.text) value.appendChild(h("b", null, r.text));
    if (r.textSub) { value.appendChild(document.createElement("br")); value.appendChild(document.createTextNode(r.textSub)); }
    row.append(label, plot, value);
    label.tabIndex = 0;
    label.setAttribute("aria-label", `${r.label}: ${r.full || [r.text, r.textSub].filter(Boolean).join(", ")}`);
    hoverable(row, ctx, () => ({ title: r.label, rows: r.tip || [{ label: cfg.measure || "Estimate", value: r.text || "—" }],
                                 note: r.note }), label);
    grid.appendChild(row);
    cells.push({ r, plot, i });
  });
  // the axis under the plot column
  const axisL = h("div", "mv-rowaxis-l");
  const axisCell = h("div", "mv-rowaxis");
  const axisR = h("div", "mv-rowaxis-r");
  grid.append(axisL, axisCell, axisR);
  const width = Math.max(120, (cells[0] ? cells[0].plot.clientWidth : axisCell.clientWidth) || 300);
  const x = scale(domain[0], domain[1], 8, width - 8, cfg.log);
  const anim = fresh(ctx, s);
  for (const { r, plot, i } of cells) {
    const g = sv("svg", { width: "100%", height: "100%", viewBox: `0 0 ${width} 26`, preserveAspectRatio: "none",
                         "aria-hidden": "true" }, plot);
    if (isNum(cfg.nul)) sv("line", { x1: x(cfg.nul), x2: x(cfg.nul), y1: 0, y2: 26, class: "mv-null" }, g);
    if (cfg.ref && isNum(cfg.ref.value)) sv("line", { x1: x(cfg.ref.value), x2: x(cfg.ref.value), y1: 0, y2: 26, class: "mv-thresh" }, g);
    const color = r.color || (r.primary ? cssVar(ctx, "--accent") : cssVar(ctx, "--ink2"));
    const mark = sv("g", { class: anim ? "mv-fade" : "" }, g);
    if (anim) mark.style.animationDelay = `${i * 45}ms`;
    if (isNum(r.lo) && isNum(r.hi)) {
      const a = Math.max(x.r[0], Math.min(x.r[1], x(r.lo))), b = Math.max(x.r[0], Math.min(x.r[1], x(r.hi)));
      sv("line", { x1: a, x2: b, y1: 13, y2: 13, stroke: color, "stroke-width": r.thick ? 6 : 2, "stroke-linecap": "round",
                   opacity: r.thick ? 0.35 : 1 }, mark);
      if (!r.thick) {
        for (const [v, p] of [[r.lo, a], [r.hi, b]]) {
          const clipped = cfg.log ? (v < domain[0] || v > domain[1]) : (v < domain[0] || v > domain[1]);
          if (clipped) sv("path", { d: p <= x.r[0] + 1 ? `M${p + 6},9 L${p},13 L${p + 6},17` : `M${p - 6},9 L${p},13 L${p - 6},17`,
                                    fill: "none", stroke: color, "stroke-width": 2 }, mark);
          else sv("line", { x1: p, x2: p, y1: 9, y2: 17, stroke: color, "stroke-width": 1.5 }, mark);
        }
      }
    }
    for (const m of r.marks || []) {
      if (!isNum(m.value)) continue;
      const mc = m.color || color;
      sv("circle", { cx: x(m.value), cy: 13, r: 4.5, fill: m.hollow ? cssVar(ctx, "--surface") : mc, stroke: mc,
                     "stroke-width": 2 }, mark);
    }
    if (isNum(r.value) && (!cfg.log || r.value > 0)) {
      const cx = x(r.value);
      if (r.primary) sv("rect", { x: cx - 6, y: 7, width: 12, height: 12, rx: 2, fill: color, stroke: cssVar(ctx, "--surface"), "stroke-width": 2 }, mark);
      else if (r.median) sv("line", { x1: cx, x2: cx, y1: 5, y2: 21, stroke: color, "stroke-width": 3, "stroke-linecap": "round" }, mark);
      else sv("circle", { cx, cy: 13, r: 5, fill: r.hollow ? cssVar(ctx, "--surface") : color, stroke: r.hollow ? color : cssVar(ctx, "--surface"),
                          "stroke-width": 2 }, mark);
    }
  }
  // ticks
  const ax = sv("svg", { width: "100%", height: 34, viewBox: `0 0 ${width} 34`, class: "mv-axis", "aria-hidden": "true" }, axisCell);
  sv("line", { x1: x.r[0], x2: x.r[1], y1: 0.5, y2: 0.5, class: "mv-baseline" }, ax);
  const tks = cfg.log ? logTicks(domain[0], domain[1], Math.max(3, Math.floor(width / 60))) : ticks(domain[0], domain[1], Math.max(3, Math.floor(width / 70)));
  for (const t of tks) {
    sv("line", { x1: x(t), x2: x(t), y1: 0, y2: 4, class: "mv-baseline" }, ax);
    svgTick(ax, x(t), 16, cfg.tickFormat ? cfg.tickFormat(t) : (cfg.log ? fmt(t, t < 1 ? 2 : t < 10 ? (t % 1 ? 2 : 0) : 0) : fmt(t, Math.abs(t % 1) > 1e-9 ? 1 : 0)) + (cfg.unit && !cfg.log ? cfg.unit : ""),
            0, width);
  }
  if (cfg.axis) {                       // the axis title whole: on two lines when the column is narrow
    const lines = fitLines(ax, cfg.axis, "mv-axis-title", width - 8);
    const title = svgText(ax, (x.r[0] + x.r[1]) / 2, 31, "", { "text-anchor": "middle", class: "mv-axis-title" });
    lines.forEach((line, i) => { const sp = sv("tspan", { x: (x.r[0] + x.r[1]) / 2, dy: i ? 13 : 0 }, title); sp.textContent = line; });
    if (lines.length > 1) { ax.setAttribute("height", 47); ax.setAttribute("viewBox", `0 0 ${width} 47`); }
  }
  if (cfg.dir && (cfg.dir[0] || cfg.dir[1])) {
    const dir = h("div", "mv-dir");
    dir.append(h("span", null, cfg.dir[0] ? `← ${cfg.dir[0]}` : ""), h("span", null, cfg.dir[1] ? `${cfg.dir[1]} →` : ""));
    grid.append(h("div"), dir, h("div"));
  }
  if (cfg.ref && cfg.ref.label) s.caption.textContent = [s.caption.textContent, `Dashed line: ${cfg.ref.label}.`].filter(Boolean).join(" ");
}

// ------------------------------------------------------------------------------------------- forest plot
// c: {rows: [{label, value, lo, hi, text, primary, sensitivity, section}], log, nul, axis, dir, sensitivity: bool}
function forest(body, c, ctx, s) {
  let rows = c.rows || [];
  const hasSens = rows.some((r) => r.sensitivity);
  if (hasSens) {
    const key = `sens:${s.id}`;
    const show = ctx.state[key] !== false;
    s.addTool(button(show ? "Sensitivity analyses: shown" : "Sensitivity analyses: hidden", "mv-chip-btn", () => {
      ctx.state[key] = !show;
      ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }, show));
    if (!show) rows = rows.filter((r) => !r.sensitivity);
  }
  const out = [];
  let section = null;
  for (const r of rows) {
    const sec = r.section || (r.sensitivity ? "Sensitivity analyses" : "");
    if (sec && sec !== section && (hasSens || r.section)) { out.push({ head: true, label: sec }); section = sec; }
    out.push({ ...r, color: r.primary ? cssVar(ctx, "--accent") : r.sensitivity ? cssVar(ctx, "--ink3") : cssVar(ctx, "--ink2"),
               hollow: !!r.sensitivity, tip: [{ label: r.measure_label || "Estimate", value: r.full || r.text || "—" }] });
  }
  intervalRows(body, { rows: out, log: !!c.log, nul: c.nul, axis: c.axis, dir: c.dir, unit: c.unit || "", zero: !!c.zero,
                       measure: c.measure, labelWidth: c.labelWidth }, ctx, s);
}

// ------------------------------------------------------------------------------------------- risks of each arm
// c: {rows: [{label, sub, role, crude: {value, lo, hi}, weighted, text}], unit, axis, weighting}
function arms(body, c, ctx, s) {
  const colors = groupColors(ctx, c.rows || [], false);
  const legend = h("div", "mv-legend");
  const ring = h("span"); ring.append(h("i", "mv-key-ring"), document.createTextNode("crude risk (95% CI)"));
  const dot = h("span"); const k = h("i", "mv-key mv-key-dot"); k.style.background = cssVar(ctx, "--ink2");
  dot.append(k, document.createTextNode(c.weighting ? `weighted risk (${c.weighting})` : "weighted risk"));
  legend.append(ring, dot);
  body.appendChild(legend);
  const rows = (c.rows || []).map((r, i) => ({
    label: r.label, sub: r.sub, lo: r.crude && r.crude.lo, hi: r.crude && r.crude.hi, color: colors[i],
    marks: [{ value: r.crude && r.crude.value, hollow: true, color: colors[i] }],
    value: r.weighted, text: r.text, textSub: r.textSub,
    tip: [{ label: "crude", value: r.crude ? `${pct(r.crude.value)} (${pct(r.crude.lo)}–${pct(r.crude.hi)})` : "—", color: colors[i] },
          ...(isNum(r.weighted) ? [{ label: "weighted", value: pct(r.weighted), color: colors[i], dot: true }] : []),
          ...(r.n ? [{ label: "patients · events", value: `${int(r.n)} · ${int(r.events)}` }] : [])],
  }));
  intervalRows(body, { rows, zero: true, axis: c.axis || "Risk (%)", unit: "%", labelWidth: c.labelWidth || "34%" }, ctx, s);
}

// ------------------------------------------------------------------------------------------- rates by subgroup
// c: {groups: [{variable, rows: [{level, value, lo, hi, n, events}]}], overall: {value, lo, hi, label}, unit, axis}
function dots(body, c, ctx, s) {
  const groups = (c.groups || []).filter((g) => g.rows && g.rows.length);
  if (!groups.length) { body.appendChild(h("div", "mv-empty", "No subgroups.")); return; }
  const vkey = `var:${s.id}`, okey = `sort:${s.id}`;
  let chosen = ctx.state[vkey];
  if (!groups.some((g) => g.variable === chosen)) chosen = groups[0].variable;
  const order = ctx.state[okey] || "level";
  if (groups.length > 1) {
    s.addTool(segmented(groups.map((g) => ({ key: g.variable, label: g.variable })), chosen, (k) => {
      ctx.state[vkey] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }, "Breakdown"));
  }
  s.addTool(segmented([{ key: "level", label: "By level" }, { key: "value", label: "By rate" }], order, (k) => {
    ctx.state[okey] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
  }, "Sort"));
  const group = groups.find((g) => g.variable === chosen);
  let rows = group.rows.slice();
  if (order === "value") rows.sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
  const unit = c.unit ?? "%";
  const ink = cssVar(ctx, "--accent");
  const out = rows.map((r) => ({
    label: r.level, sub: isNum(r.n) ? `${int(r.n)} patients` : "", value: r.value, lo: r.lo, hi: r.hi, color: ink,
    text: isNum(r.value) ? `${fmt(r.value, 1)}${unit}` : "—", textSub: isNum(r.lo) ? `${fmt(r.lo, 1)}–${fmt(r.hi, 1)}` : "",
    tip: [{ label: c.measure || "Rate", value: isNum(r.value) ? `${fmt(r.value, 1)}${unit}` : "—" },
          ...(isNum(r.lo) ? [{ label: "95% CI", value: `${fmt(r.lo, 1)}–${fmt(r.hi, 1)}${unit}` }] : []),
          ...(r.n !== undefined ? [{ label: "events / patients", value: `${int(r.events)} / ${int(r.n)}` }] : [])],
  }));
  const ov = c.overall && isNum(c.overall.value) ? c.overall : null;
  intervalRows(body, { rows: out, zero: true, unit, axis: c.axis, measure: c.measure,
                       ref: ov ? { value: ov.value, label: ov.label || `overall ${fmt(ov.value, 1)}${unit}` } : null,
                       labelWidth: c.labelWidth || "30%" }, ctx, s);
}

// ------------------------------------------------------------------------------------------- medians with IQR
// c: {groups: [{variable, rows: [{level, median, q1, q3, n}]}], unit, axis}
function ranges(body, c, ctx, s) {
  const groups = (c.groups || []).filter((g) => g.rows && g.rows.length);
  if (!groups.length) { body.appendChild(h("div", "mv-empty", "No subgroups.")); return; }
  const vkey = `var:${s.id}`;
  let chosen = ctx.state[vkey];
  if (!groups.some((g) => g.variable === chosen)) chosen = groups[0].variable;
  if (groups.length > 1) {
    s.addTool(segmented(groups.map((g) => ({ key: g.variable, label: g.variable })), chosen, (k) => {
      ctx.state[vkey] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }, "Breakdown"));
  }
  const ink = cssVar(ctx, "--accent");
  const unit = c.unit ? ` ${c.unit}` : "";
  const rows = groups.find((g) => g.variable === chosen).rows.map((r) => ({
    label: r.level, sub: isNum(r.n) ? `${int(r.n)} patients` : "", value: r.median, lo: r.q1, hi: r.q3, thick: true, median: true,
    color: ink, text: `${fmt(r.median)}${unit}`, textSub: `IQR ${fmt(r.q1)}–${fmt(r.q3)}`,
    tip: [{ label: "median", value: `${fmt(r.median)}${unit}` }, { label: "IQR", value: `${fmt(r.q1)}–${fmt(r.q3)}${unit}` },
          { label: "patients", value: int(r.n) }],
  }));
  intervalRows(body, { rows, axis: c.axis, labelWidth: "30%" }, ctx, s);
}

// ------------------------------------------------------------------------------------------- effect by subgroup
// c: {rows: [{label, n, value, lo, hi}], overall: {value, lo, hi}, unit, axis, dir}
function subgroups(body, c, ctx, s) {
  const ink = cssVar(ctx, "--accent");
  const rows = [];
  if (c.overall && isNum(c.overall.value)) {
    rows.push({ label: "All patients", value: c.overall.value, lo: c.overall.lo, hi: c.overall.hi, primary: true, color: ink,
                text: `${signed(c.overall.value)} ${c.unit || ""}`.trim(), textSub: isNum(c.overall.lo) ? `${signed(c.overall.lo)} to ${signed(c.overall.hi)}` : "" });
  }
  let section = null;
  for (const r of c.rows || []) {
    const [sec, level] = String(r.label).includes(": ") ? String(r.label).split(/: (.+)/) : ["", r.label];
    if (sec && sec !== section) { rows.push({ head: true, label: sec }); section = sec; }
    rows.push({ label: level, sub: isNum(r.n) ? `${int(r.n)} patients` : "", value: r.value, lo: r.lo, hi: r.hi,
                color: cssVar(ctx, "--ink2"), text: `${signed(r.value)} ${c.unit || ""}`.trim(),
                textSub: isNum(r.lo) ? `${signed(r.lo)} to ${signed(r.hi)}` : "",
                tip: [{ label: c.measure || "Risk difference", value: `${signed(r.value)} ${c.unit || ""}` },
                      { label: "95% CI", value: `${signed(r.lo)} to ${signed(r.hi)}` }, { label: "patients", value: int(r.n) }] });
  }
  intervalRows(body, { rows, nul: 0, axis: c.axis, dir: c.dir, measure: c.measure, labelWidth: "32%" }, ctx, s);
}
