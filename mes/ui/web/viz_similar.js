// MES charts — patients like this: what the similar patients received (beside everyone), the patient beside them
// characteristic by characteristic (a fingerprint), and where the patient sits among all stays (a map of aggregates).

// c: {rows: [{label, value, lo, hi, cohort}]} — % of patients like this (dot, 95% CI) and of everyone (ring)
function received(body, c, ctx, s) {
  const accent = cssVar(ctx, "--accent"), ink = cssVar(ctx, "--ink3");
  const legend = h("div", "mv-legend");
  const names = c.legend || ["patients like this (95% CI)", "everyone"];
  const a = h("span"); const ka = h("i", "mv-key mv-key-dot"); ka.style.background = accent;
  a.append(ka, document.createTextNode(names[0]));
  legend.append(a);
  if (names[1]) {           // (a single share — the commonest items of everyone — has no ring)
    const b = h("span"); b.append(h("i", "mv-key-ring"), document.createTextNode(names[1]));
    legend.append(b);
  }
  body.appendChild(legend);
  const rows = (c.rows || []).map((r) => ({
    label: r.label, value: r.value, lo: r.lo, hi: r.hi, color: accent,
    marks: isNum(r.cohort) ? [{ value: r.cohort, hollow: true, color: ink }] : [],
    text: isNum(r.value) ? pct(r.value) : "<10", textSub: isNum(r.cohort) ? `${names[1].split(" (")[0]} ${pct(r.cohort)}` : "",
    tip: [{ label: names[0].split(" (")[0], value: isNum(r.value) ? (isNum(r.lo) ? `${pct(r.value)} (${pct(r.lo)}–${pct(r.hi)})` : pct(r.value)) : "<10", color: accent, dot: true },
          ...(names[1] ? [{ label: names[1].split(" (")[0], value: isNum(r.cohort) ? pct(r.cohort) : "<10" }] : [])],
  }));
  intervalRows(body, { rows, domain: [0, 100], unit: "%", axis: "Patients (%)", labelWidth: "36%" }, ctx, s);
}

// c: {rows: [{label, kind, unit, patient, weight, must, range: [lo, hi], similar: [q1, median, q3], cohort: [q1, median, q3],
//      share_similar, share_cohort, similar_text, cohort_text, missing}]}
function profile(body, c, ctx, s) {
  const rows = c.rows || [];
  if (!rows.length) { body.appendChild(h("div", "mv-empty", "No characteristics.")); return; }
  const legend = h("div", "mv-legend");
  const k1 = h("span"); const i1 = h("i", "mv-key mv-pf-key-all"); k1.append(i1, document.createTextNode("everyone (IQR)"));
  const k2 = h("span"); const i2 = h("i", "mv-key mv-pf-key-near"); k2.append(i2, document.createTextNode("patients like this (IQR, median)"));
  const k3 = h("span"); k3.append(h("i", "mv-pf-key-pt"), document.createTextNode("this patient"));
  legend.append(k1, k2, k3);
  body.appendChild(legend);
  const list = h("div", "mv-pf");
  body.appendChild(list);
  const anim = fresh(ctx, s);
  rows.forEach((r, i) => {
    const row = h("div", "mv-pf-row");
    const name = h("div", "mv-pf-name");
    name.appendChild(h("span", null, r.label));
    if (r.must) name.appendChild(h("em", "mv-pf-tag", "must match"));
    else if (isNum(r.weight) && r.weight !== 1) name.appendChild(h("em", "mv-pf-tag", `weight ×${fmt(r.weight, r.weight % 1 ? 1 : 0)}`));
    const track = h("div", "mv-pf-track");
    const text = h("div", "mv-pf-text");
    const pv = r.kind === "binary" ? String(r.patient) : isNum(r.patient) ? fmt(r.patient) : String(r.patient ?? "—");
    text.appendChild(h("b", null, `${pv}${r.kind === "numeric" && r.unit ? ` ${r.unit}` : ""}`));
    text.appendChild(h("span", null, `like this ${r.similar_text || "—"} · everyone ${r.cohort_text || "—"}`));
    if (r.kind === "numeric" && Array.isArray(r.range) && isNum(r.range[0]) && isNum(r.range[1]) && r.range[1] > r.range[0]) {
      const lo = Math.min(r.range[0], isNum(r.patient) ? r.patient : r.range[0]);
      const hi = Math.max(r.range[1], isNum(r.patient) ? r.patient : r.range[1]);
      const pos = (v) => `${Math.max(0, Math.min(100, (100 * (v - lo)) / ((hi - lo) || 1)))}%`;
      const band = (q, cls) => {
        if (!Array.isArray(q) || !isNum(q[0]) || !isNum(q[2])) return;
        const el = h("i", cls);
        el.style.left = pos(q[0]);
        el.style.width = `calc(${pos(q[2])} - ${pos(q[0])})`;
        track.appendChild(el);
      };
      band(r.cohort, "mv-pf-all");
      band(r.similar, "mv-pf-near");
      if (Array.isArray(r.similar) && isNum(r.similar[1])) { const m = h("b", "mv-pf-med"); m.style.left = pos(r.similar[1]); track.appendChild(m); }
      if (isNum(r.patient)) { const p = h("span", "mv-pf-pt"); p.style.left = pos(r.patient); track.appendChild(p); }
      track.appendChild(h("small", "mv-pf-lo", fmt(lo)));
      track.appendChild(h("small", "mv-pf-hi", fmt(hi)));
    } else {
      // yes/no or a category: the share of each group that has the patient's value
      for (const [v, cls, who] of [[r.share_cohort, "mv-pf-bar-all", "everyone"], [r.share_similar, "mv-pf-bar-near", "like this"]]) {
        const line = h("div", "mv-pf-bar");
        const fill = h("i", cls);
        fill.style.width = isNum(v) ? `${Math.max(1, v)}%` : "0%";
        line.appendChild(fill);
        line.appendChild(h("small", null, `${who} ${isNum(v) ? pct(v, 0) : "—"}`));
        track.appendChild(line);
      }
      track.classList.add("mv-pf-share");
    }
    if (anim) row.style.animationDelay = `${i * 40}ms`;
    row.append(name, track, text);
    row.tabIndex = 0;
    hoverable(row, ctx, () => ({ title: r.label, rows: [
      { label: "this patient", value: pv }, { label: "patients like this", value: r.similar_text || "—" },
      { label: "everyone", value: r.cohort_text || "—" },
      ...(isNum(r.missing) && r.missing > 0 ? [{ label: "missing among them", value: pct(r.missing, 0) }] : [])] }));
    list.appendChild(row);
  });
}

// c: {hexes: [{x, y, n, k}], radius, ellipse: {cx, cy, rx, ry, angle}, patient: {x, y}, extent: [x0, x1, y0, y1], axes}
function simmap(body, c, ctx, s) {
  const hexes = c.hexes || [];
  // the frame: what is drawn (the hexagons, the ellipse, the patient), not the percentiles of every stay
  const rad = c.radius || 0.1;
  const xs = [], ys = [];
  for (const hx of hexes) { xs.push(hx.x - rad, hx.x + rad); ys.push(hx.y - rad, hx.y + rad); }
  if (c.patient) { xs.push(c.patient.x - rad, c.patient.x + rad); ys.push(c.patient.y - rad, c.patient.y + rad); }
  if (c.ellipse) {
    const e = c.ellipse, big = Math.max(e.rx, e.ry);
    xs.push(e.cx - big, e.cx + big); ys.push(e.cy - big, e.cy + big);
  }
  let [x0, x1, y0, y1] = xs.length ? [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)] : (c.extent || [0, 1, 0, 1]);
  if (!(x1 > x0)) { x0 -= 1; x1 += 1; }
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const width = Math.max(260, body.clientWidth || 560);
  const m = { l: 16, r: 16, t: 24, b: 26 };
  // one scale for both axes (hexagons stay regular), the map as wide as the card
  let k = (width - m.l - m.r) / ((x1 - x0) || 1);
  let height = Math.round(k * ((y1 - y0) || 1)) + m.t + m.b;
  if (height > 460) { k *= (460 - m.t - m.b) / (height - m.t - m.b); height = 460; }
  height = Math.max(220, height);
  const ox = m.l + (width - m.l - m.r - k * (x1 - x0)) / 2, oy = m.t + (height - m.t - m.b - k * (y1 - y0)) / 2;
  const sx = (v) => ox + k * (v - x0), sy = (v) => height - oy - k * (v - y0);
  const svgEl = sv("svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                           "aria-label": "Map of all stays with the patients like this and the patient" }, body);
  const maxN = Math.max(1, ...hexes.map((x) => x.n));
  const r = (c.radius || 0.1) * k;
  const hexPath = (cx, cy) => {
    let d = "";
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 180) * (60 * i - 30);
      d += `${i ? "L" : "M"}${(cx + r * 0.95 * Math.cos(a)).toFixed(1)},${(cy + r * 0.95 * Math.sin(a)).toFixed(1)}`;
    }
    return `${d}Z`;
  };
  const base = cssVar(ctx, "--ink3"), accent = cssVar(ctx, "--accent"), surface = cssVar(ctx, "--surface");
  const layer = sv("g", {}, svgEl);
  for (const hx of hexes) {
    const p = sv("path", { d: hexPath(sx(hx.x), sy(hx.y)), fill: hx.k ? accent : base,
                           "fill-opacity": hx.k ? 0.25 + 0.55 * Math.min(1, hx.k / Math.max(10, hx.n)) : 0.06 + 0.5 * Math.sqrt(hx.n / maxN),
                           class: "mv-hex" }, layer);
    p.setAttribute("tabindex", "0");
    hoverable(p, ctx, () => ({ title: "Stays here", rows: [{ label: "stays", value: int(hx.n) },
      { label: "of them like this patient", value: hx.k ? int(hx.k) : "<10" }] }));
  }
  if (c.ellipse) {
    const e = c.ellipse;
    sv("ellipse", { cx: sx(e.cx), cy: sy(e.cy), rx: Math.max(4, e.rx * k), ry: Math.max(4, e.ry * k),
                    transform: `rotate(${-e.angle} ${sx(e.cx)} ${sy(e.cy)})`, fill: "none", stroke: accent,
                    "stroke-width": 2, "stroke-dasharray": "6 4" }, svgEl);
  }
  if (c.patient) {
    const px = sx(c.patient.x), py = sy(c.patient.y);
    let d = "";
    for (let i = 0; i < 10; i++) {
      const rr = i % 2 ? 4.5 : 11, a = (Math.PI / 5) * i - Math.PI / 2;
      d += `${i ? "L" : "M"}${(px + rr * Math.cos(a)).toFixed(1)},${(py + rr * Math.sin(a)).toFixed(1)}`;
    }
    const star = sv("path", { d: `${d}Z`, fill: accent, stroke: surface, "stroke-width": 1.5, class: "mv-sim-star" }, svgEl);
    star.setAttribute("tabindex", "0");
    hoverable(star, ctx, () => ({ title: "This patient", rows: [{ label: "here", value: "the star" }] }));
  }
  const axes = c.axes || ["", ""];
  if (axes[0]) svgText(svgEl, width - m.r, height - 8, `${axes[0]} →`, { "text-anchor": "end", class: "mv-axis-title" });
  if (axes[1]) svgText(svgEl, m.l, 14, `↑ ${axes[1]}`, { class: "mv-axis-title" });
  void s;
}
