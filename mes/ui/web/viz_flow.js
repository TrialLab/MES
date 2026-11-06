// MES charts — the flow of patients through the criteria, the protocol timeline and the analyst's steps.

function countUp(el, value, ctx, s) {
  if (!isNum(value)) { el.textContent = int(value); return; }
  if (!fresh(ctx, s)) { el.textContent = int(value); return; }
  const t0 = performance.now(), dur = 700;
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    el.textContent = int(value * (1 - (1 - k) ** 3));
    if (k < 1 && el.isConnected) requestAnimationFrame(tick);
  };
  el.textContent = "0";
  requestAnimationFrame(tick);
}

// c: {steps: [[label, n]], split: [{label, n, role}]}
function flow(body, c, ctx, s) {
  const steps = (c.steps || []).filter((st) => Array.isArray(st) && st.length >= 2);
  if (!steps.length) { body.appendChild(h("div", "mv-empty", "No cohort steps were reported.")); return; }
  const wrap = h("div", "mv-flow");
  body.appendChild(wrap);
  const n0 = isNum(steps[0][1]) ? steps[0][1] : null;
  steps.forEach(([label, n], i) => {
    if (i) {
      const arrow = h("div", "mv-farrow");
      wrap.appendChild(arrow);
    }
    const row = h("div", "mv-fstep");
    const box = h("div", i === steps.length - 1 && !(c.split && c.split.length) ? "mv-fbox mv-fbox-final" : "mv-fbox");
    const num = h("div", "mv-fbox-n");
    const val = h("span");
    num.appendChild(val);
    if (n0 && isNum(n) && i) num.appendChild(h("small", null, `${fmt((100 * n) / n0, n / n0 < 0.1 ? 1 : 0)}% of the first step`));
    box.append(num, h("div", "mv-fbox-l", label));
    const bar = h("div", "mv-fbar");
    const fill = h("i");
    fill.style.width = n0 && isNum(n) ? `${Math.max(1.5, (100 * n) / n0)}%` : "100%";
    if (!fresh(ctx, s)) fill.style.animation = "none";
    bar.appendChild(fill);
    box.appendChild(bar);
    row.appendChild(box);
    const prev = i ? steps[i - 1][1] : null;
    if (i && isNum(prev) && isNum(n) && prev > n) {
      const x = h("div", "mv-fx");
      const share = (100 * (prev - n)) / prev;
      x.append(h("b", null, `− ${int(prev - n)}`), document.createTextNode(` excluded (${fmt(share, share < 1 ? 1 : 0)}% of ${int(prev)})`));
      row.appendChild(x);
    } else row.appendChild(h("div"));
    wrap.appendChild(row);
    countUp(val, n, ctx, s);
  });
  if (c.split && c.split.length) {
    wrap.appendChild(h("div", "mv-farrow"));
    const split = h("div", "mv-fsplit");
    const colors = groupColors(ctx, c.split, false);
    c.split.forEach((g, i) => {
      const box = h("div", "mv-fbox");
      box.style.setProperty("--sc", colors[i]);
      const num = h("div", "mv-fbox-n");
      const val = h("span");
      num.appendChild(val);
      const total = c.split.reduce((a, b) => a + (isNum(b.n) ? b.n : 0), 0);
      if (total && isNum(g.n)) num.appendChild(h("small", null, `${fmt((100 * g.n) / total, 0)}%`));
      box.append(num, h("div", "mv-fbox-l", g.label));
      if (g.sub) box.appendChild(h("div", "mv-fbox-l", g.sub));
      split.appendChild(box);
      countUp(val, g.n, ctx, s);
    });
    wrap.appendChild(split);
  }
}

// c: {segments: [{label, kind, w, sub}], marks: [{at, label, sub, align}]}
function timeline(body, c, ctx) {
  const tl = h("div", "mv-tl");
  const track = h("div", "mv-tl-track");
  const total = (c.segments || []).reduce((a, sgm) => a + (sgm.w || 1), 0) || 1;
  for (const sgm of c.segments || []) {
    const seg = h("div", `mv-tl-seg mv-tl-${sgm.kind || "window"}`);
    seg.appendChild(h("span", null, sgm.label));         // (a span: a label too long for its share ends in "…")
    // the exact share of the track (padding and divider inside it): the marks below sit on the segments' edges
    seg.style.flex = `0 0 ${(100 * (sgm.w || 1)) / total}%`;
    seg.title = [sgm.label, sgm.sub].filter(Boolean).join(" — ");
    track.appendChild(seg);
  }
  tl.appendChild(track);
  const marks = h("div", "mv-tl-marks");
  // marks whose labels would overlap at this width go one row lower (a longer tick): a narrow card keeps them legible
  const width = body.clientWidth || 600;
  const spans = [];          // [left, right] in px of the labels on each row
  let rows = 1;
  for (const mk of c.marks || []) {
    const el = h("div", `mv-tl-mark${mk.align === "left" ? " mv-tl-left" : mk.align === "right" ? " mv-tl-right" : ""}`);
    const at = Math.max(0, Math.min(100, 100 * (mk.at ?? 0)));
    // a right-aligned mark is anchored by its right edge (at "left: 100%" it would have no room and wrap word by word)
    if (mk.align === "right") el.style.right = `${100 - at}%`; else el.style.left = `${at}%`;
    const text = Math.min(0.34 * width, 6.6 * Math.max(String(mk.label || "").length, 0.92 * String(mk.sub || "").length));
    const x = (at / 100) * width;
    const span = mk.align === "left" ? [x, x + text] : mk.align === "right" ? [x - text, x] : [x - text / 2, x + text / 2];
    let row = 0;
    while ((spans[row] || []).some(([a, b]) => span[0] < b + 10 && span[1] > a - 10)) row += 1;
    (spans[row] = spans[row] || []).push(span);
    rows = Math.max(rows, row + 1);
    el.style.setProperty("--row", row);
    el.append(h("b", null, mk.label));
    if (mk.sub) el.append(h("span", null, mk.sub));
    marks.appendChild(el);
  }
  marks.style.height = `${54 + (rows - 1) * 46}px`;
  tl.appendChild(marks);
  body.appendChild(tl);
  // once laid out, room for the lowest label however many lines its text wraps to
  requestAnimationFrame(() => {
    const bottom = Math.max(0, ...[...marks.children].map((e) => e.offsetTop + e.offsetHeight));
    if (bottom > 0) marks.style.height = `${bottom + 6}px`;
  });
}

const STEP_COLORS = { search: "--s2", sql: "--s4", python: "--s1", submit: "--s5", finish: "--s5", error: "--bad" };

// c: {rows: [{action, seconds, plan, ok}]}
function steps(body, c, ctx, s) {
  const rows = c.rows || [];
  if (!rows.length) { body.appendChild(h("div", "mv-empty", "No steps were recorded.")); return; }
  const total = rows.reduce((a, r) => a + (isNum(r.seconds) ? r.seconds : 0), 0) || 1;
  const list = h("div", "mv-steps");
  let start = 0;
  rows.forEach((r, i) => {
    const row = h("div", "mv-step");
    const color = cssVar(ctx, STEP_COLORS[r.ok === false ? "error" : r.action] || "--ink3");
    row.style.setProperty("--sc", color);
    const n = h("div", "mv-step-n", String(i + 1));
    const t = h("div", "mv-step-t");
    t.append(h("b", null, r.action || "step"), h("span", null, r.plan || ""));
    const bar = h("div", "mv-step-bar");
    const fill = h("i");
    const sec = isNum(r.seconds) ? r.seconds : 0;
    fill.style.left = `${(100 * start) / total}%`;
    fill.style.width = `${Math.max(1, (100 * sec) / total)}%`;
    bar.appendChild(fill);
    start += sec;
    const right = h("div");
    right.append(bar, h("div", "mv-step-s", `${fmt(sec, sec < 10 ? 1 : 0)} s`));
    row.append(n, t, right);
    row.tabIndex = 0;
    hoverable(row, ctx, () => ({ title: `Step ${i + 1} · ${r.action || ""}`, rows: [{ label: "time", value: `${fmt(sec, 1)} s` }], note: r.plan || "" }));
    list.appendChild(row);
  });
  body.appendChild(list);
}
