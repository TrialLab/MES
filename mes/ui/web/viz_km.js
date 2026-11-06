// MES charts — survival curves (Kaplan-Meier): survival or cumulative incidence, the 95% band, a crosshair that reads
// every curve at the same day (and the difference of two), the patients at risk under the axis.

// The value of a step function at t: the last point at or before t
function stepAt(ts, ys, t) {
  let lo = 0, hi = ts.length - 1, k = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) { k = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return k < 0 ? (ys.length ? ys[0] : null) : ys[k];
}

function stepPath(ts, ys, x, y, horizon) {
  if (!ts.length) return "";
  let d = `M${x(ts[0]).toFixed(1)},${y(ys[0]).toFixed(1)}`;
  for (let i = 1; i < ts.length; i++) d += `H${x(ts[i]).toFixed(1)}V${y(ys[i]).toFixed(1)}`;
  if (isNum(horizon) && horizon > ts[ts.length - 1]) d += `H${x(horizon).toFixed(1)}`;
  return d;
}

function bandPath(ts, lo, hi, x, y, horizon) {
  if (!ts.length || !lo || !hi || lo.length !== ts.length || hi.length !== ts.length) return "";
  const end = isNum(horizon) && horizon > ts[ts.length - 1] ? horizon : ts[ts.length - 1];
  let d = `M${x(ts[0]).toFixed(1)},${y(hi[0]).toFixed(1)}`;
  for (let i = 1; i < ts.length; i++) d += `H${x(ts[i]).toFixed(1)}V${y(hi[i]).toFixed(1)}`;
  d += `H${x(end).toFixed(1)}V${y(lo[lo.length - 1]).toFixed(1)}`;
  for (let i = ts.length - 1; i > 0; i--) d += `H${x(ts[i]).toFixed(1)}V${y(lo[i - 1]).toFixed(1)}`;
  return `${d}H${x(ts[0]).toFixed(1)}Z`;
}

// c: {groups: [{name, role, t, s, lo, hi, n, at_risk: [[day, n]]}], horizon, x_label, ordered, start, weighted}
function km(body, c, ctx, s) {
  const groups = (c.groups || []).filter((g) => g.t && g.t.length && g.s && g.s.length === g.t.length);
  if (!groups.length) { body.appendChild(h("div", "mv-empty", "No survival curves.")); return; }
  const mkey = `km:${s.id}`, bkey = `band:${s.id}`;
  const mode = ctx.state[mkey] || c.mode || "survival";
  const hasBand = groups.some((g) => g.lo && g.hi);
  const band = hasBand && ctx.state[bkey] !== false;
  s.addTool(segmented([{ key: "survival", label: "Survival" }, { key: "incidence", label: "Cumulative incidence" }], mode, (k) => {
    ctx.state[mkey] = k; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
  }, "Scale"));
  if (hasBand) {
    s.addTool(button("95% band", "mv-chip-btn", () => {
      ctx.state[bkey] = !band; ctx.redrawBlock(s.block, (into) => chartBlock(into, s.spec, ctx));
    }, band));
  }
  const colors = groupColors(ctx, groups, !!c.ordered);
  const inc = mode === "incidence";
  const val = (v) => (isNum(v) ? (inc ? 1 - v : v) : null);
  // legend: a line key per group, its patients and its value at the horizon
  const legend = h("div", "mv-legend");
  groups.forEach((g, i) => {
    const item = h("span");
    const key = h("i", "mv-key");
    key.style.background = colors[i];
    item.append(key, document.createTextNode(`${g.name}${isNum(g.n) ? ` (n = ${int(g.n)})` : ""}`));
    legend.appendChild(item);
  });
  body.appendChild(legend);

  const horizon = isNum(c.horizon) ? c.horizon : Math.max(...groups.map((g) => g.t[g.t.length - 1]));
  const atRisk = groups.some((g) => g.at_risk && g.at_risk.length);
  const width = Math.max(280, body.clientWidth || 600);
  const m = { l: 46, r: 14, t: 10, b: 40 + (atRisk ? 22 + 18 * groups.length : 0) };
  const plotH = Math.max(190, Math.min(300, width * 0.42));
  const H = m.t + plotH + m.b;
  const svgEl = sv("svg", { width, height: H, viewBox: `0 0 ${width} ${H}`, role: "img",
                            "aria-label": `${inc ? "Cumulative incidence" : "Survival"} curves of ${groups.map((g) => g.name).join(", ")}` }, body);
  const x = scale(0, horizon, m.l, width - m.r);
  const all = [];
  for (const g of groups) {
    all.push(...g.s);
    if (band && g.lo) all.push(...g.lo.filter(isNum));
    if (band && g.hi) all.push(...g.hi.filter(isNum));
  }
  let lo = Math.min(...all.filter(isNum)), hi = Math.max(...all.filter(isNum));
  let y;
  if (inc) {
    const top = Math.min(1, Math.ceil((1 - lo) * 10 + 0.0001) / 10);
    y = scale(0, Math.max(top, 0.1), m.t + plotH, m.t);
  } else {
    const bottom = Math.max(0, Math.floor(lo * 10 - 0.0001) / 10);
    y = scale(Math.min(bottom, 0.9), 1, m.t + plotH, m.t);
  }
  // grid and axes
  const yt = ticks(y.d[0], y.d[1], 5);
  for (const t of yt) {
    sv("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: "mv-gridline" }, svgEl);
    svgText(svgEl, m.l - 8, y(t) + 4, `${fmt(100 * t, 0)}%`, { "text-anchor": "end", class: "mv-tick" });
  }
  const riskTimes = atRisk ? (groups.find((g) => g.at_risk && g.at_risk.length).at_risk || []).map((p) => p[0]) : null;
  const xt = riskTimes && riskTimes.length >= 2 ? riskTimes : ticks(0, horizon, Math.max(3, Math.floor(width / 80)));
  sv("line", { x1: m.l, x2: width - m.r, y1: m.t + plotH, y2: m.t + plotH, class: "mv-baseline" }, svgEl);
  for (const t of xt) {
    sv("line", { x1: x(t), x2: x(t), y1: m.t + plotH, y2: m.t + plotH + 4, class: "mv-baseline" }, svgEl);
    svgTick(svgEl, x(t), m.t + plotH + 16, fmt(t, t % 1 ? 1 : 0), 0, width);
  }
  svgText(svgEl, (m.l + width - m.r) / 2, m.t + plotH + 32, c.x_label || "Days", { "text-anchor": "middle", class: "mv-axis-title" });
  // bands, then curves
  const anim = fresh(ctx, s);
  groups.forEach((g, i) => {
    if (!band || !g.lo || !g.hi) return;
    const vlo = g.lo.map((v) => val(v)), vhi = g.hi.map((v) => val(v));
    const d = bandPath(g.t, inc ? vhi : vlo, inc ? vlo : vhi, x, y, horizon);
    if (d) sv("path", { d, fill: colors[i], opacity: 0.11, class: anim ? "mv-fade" : "" }, svgEl);
  });
  const paths = groups.map((g, i) => {
    const p = sv("path", { d: stepPath(g.t, g.s.map(val), x, y, horizon), fill: "none", stroke: colors[i], "stroke-width": 2,
                           "stroke-linejoin": "round", "stroke-linecap": "round" }, svgEl);
    if (anim && p.getTotalLength) {
      const len = Math.ceil(p.getTotalLength());
      p.style.setProperty("--len", len);
      p.classList.add("mv-draw");
      p.style.animationDelay = `${i * 120}ms`;
    }
    return p;
  });
  // direct labels at the right end when they do not collide (the legend carries identity either way)
  if (groups.length <= 4) {
    const ends = groups.map((g, i) => ({ i, y: y(val(stepAt(g.t, g.s, horizon))) })).sort((a, b) => a.y - b.y);
    const apart = ends.every((e, k) => k === 0 || e.y - ends[k - 1].y >= 13);
    if (apart && width > 520) {
      for (const e of ends) {
        const g = groups[e.i];
        svgText(svgEl, width - m.r - 2, e.y - 6, `${pct(100 * val(stepAt(g.t, g.s, horizon)))}`, { "text-anchor": "end", class: "mv-tick" });
      }
    }
  }
  // numbers at risk
  if (atRisk) {
    const top = m.t + plotH + 50;
    svgText(svgEl, 4, top, "At risk", { class: "mv-axis-title" });
    groups.forEach((g, i) => {
      const yy = top + 18 * (i + 1);
      sv("rect", { x: 4, y: yy - 8, width: 10, height: 3, rx: 1.5, fill: colors[i] }, svgEl);
      for (const [t, n] of g.at_risk || []) {
        if (t > horizon + 1e-9) continue;
        svgText(svgEl, x(t), yy - 3, n === null || n === undefined ? "<10" : int(n), { "text-anchor": "middle", class: "mv-tick" });
      }
    });
  }
  // crosshair: hover or arrow keys
  const cross = sv("line", { y1: m.t, y2: m.t + plotH, class: "mv-cross", opacity: 0 }, svgEl);
  const dotsG = sv("g", { opacity: 0 }, svgEl);
  const marks = groups.map((g, i) => sv("circle", { r: 4.5, fill: colors[i], stroke: cssVar(ctx, "--surface"), "stroke-width": 2 }, dotsG));
  const hit = sv("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: plotH, class: "mv-hit", tabindex: 0,
                           "aria-label": "Read the curves: move the pointer, or use the arrow keys" }, svgEl);
  let at = ctx.state[`t:${s.id}`];
  const read = (t, ev) => {
    t = Math.max(0, Math.min(horizon, t));
    at = t;
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("opacity", 1);
    dotsG.setAttribute("opacity", 1);
    const rows = [];
    const values = groups.map((g, i) => {
      const v = val(stepAt(g.t, g.s, t));
      marks[i].setAttribute("cx", x(t));
      marks[i].setAttribute("cy", isNum(v) ? y(v) : -20);
      let ci = "";
      if (g.lo && g.hi) {
        const a = val(stepAt(g.t, g.lo, t)), b = val(stepAt(g.t, g.hi, t));
        if (isNum(a) && isNum(b)) ci = ` (${fmt(100 * Math.min(a, b), 1)}–${fmt(100 * Math.max(a, b), 1)})`;
      }
      rows.push({ label: g.name, value: `${pct(100 * v)}${ci}`, color: colors[i] });
      return v;
    });
    if (groups.length === 2 && values.every(isNum)) rows.push({ label: "Difference", value: `${signed(100 * (values[0] - values[1]))} pp` });
    ctx.tip.show(ev || hit, `Day ${fmt(t, t < 10 ? 1 : 0)}${c.start ? ` after ${c.start}` : ""}`, rows,
                 inc ? "Cumulative incidence = 1 − survival" : "");
  };
  hit.addEventListener("pointermove", (e) => {
    const box = svgEl.getBoundingClientRect();
    read(x.invert(e.clientX - box.left), e);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dotsG.setAttribute("opacity", 0); ctx.tip.hide(); });
  hit.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = horizon <= 60 ? 1 : horizon / 60;
    read((isNum(at) ? at : 0) + (e.key === "ArrowRight" ? step : -step));
    ctx.state[`t:${s.id}`] = at;
  });
  hit.addEventListener("blur", () => { cross.setAttribute("opacity", 0); dotsG.setAttribute("opacity", 0); ctx.tip.hide(); });
  void paths;
}
