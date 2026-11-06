// MES charts — core: DOM helpers, number formats, scales, the tooltip, chart shells and the blocks of a panel.
// ui/viz.py joins viz_core.js, viz_charts.js and viz_views.js into one st.components.v2 module ("mes_viz"); every
// chart is drawn as SVG / HTML from a spec built in Python, text always through textContent (labels come from
// the model, the designer or the data, never as markup).

const NS = "http://www.w3.org/2000/svg";
// UI state of each mounted chart (tab, toggles, brush) — the component is called again on every rerun
const MEMORY = (globalThis.__mesViz = globalThis.__mesViz || {});

function motion() {
  return !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

// ------------------------------------------------------------------------------------------- DOM
function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null && text !== "") e.textContent = String(text);
  return e;
}

function sv(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) {
    if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  }
  if (parent) parent.appendChild(e);
  return e;
}

function svgText(parent, x, y, text, attrs) {
  const t = sv("text", { x, y, ...(attrs || {}) }, parent);
  t.textContent = String(text ?? "");
  return t;
}

// The drawn width of ``text`` with class ``cls`` in ``svgEl`` (estimated where it cannot be measured, as in a hidden tab)
function textWidth(svgEl, text, cls) {
  const t = svgText(svgEl, 0, -100, text, { class: cls, visibility: "hidden" });
  let w = 0;
  try { w = t.getComputedTextLength(); } catch (e) { w = 0; }
  t.remove();
  return w > 0 ? w : String(text ?? "").length * 6.6;
}

// ``text`` cut to ``maxW`` px, with "…" when it is cut
function fitText(svgEl, text, cls, maxW) {
  let s = String(text ?? "");
  if (textWidth(svgEl, s, cls) <= maxW) return s;
  while (s.length > 1 && textWidth(svgEl, `${s.trimEnd()}…`, cls) > maxW) s = s.slice(0, -1);
  return `${s.trimEnd()}…`;
}

// ``text`` as one line, or two of at most ``maxW`` px — broken at a " · " first, the two lines as even as they can be
// (the "·" at the break is dropped); cut with "…" only when two lines are not enough
function fitLines(svgEl, text, cls, maxW) {
  const s = String(text ?? "");
  if (textWidth(svgEl, s, cls) <= maxW) return [s];
  const words = s.split(" ");
  const breaks = (dots) => {
    let best = null;
    for (let i = 1; i < words.length; i++) {
      if (dots && words[i] !== "·" && words[i - 1] !== "·") continue;
      const a = words.slice(0, i).join(" ").replace(/\s*·$/, ""), b = words.slice(i).join(" ").replace(/^·\s*/, "");
      if (!a || !b) continue;
      const w = Math.max(textWidth(svgEl, a, cls), textWidth(svgEl, b, cls));
      if (w <= maxW && (!best || w < best.w)) best = { a, b, w };
    }
    return best;
  };
  const best = breaks(true) || breaks(false);
  if (best) return [best.a, best.b];
  const half = Math.ceil(words.length / 2);
  return [fitText(svgEl, words.slice(0, half).join(" "), cls, maxW), fitText(svgEl, words.slice(half).join(" "), cls, maxW)];
}

// An axis tick label centred on ``x`` but kept whole inside [lo, hi] (the chart's edges): the first and the last tick
// of an axis would otherwise be cut in half at the edge of their SVG
function svgTick(parent, x, y, text, lo, hi, attrs) {
  const half = textWidth(parent, text, (attrs && attrs.class) || "mv-tick") / 2;
  const cx = hi - lo > 2 * half ? Math.min(Math.max(x, lo + half + 1), hi - half - 1) : x;
  return svgText(parent, cx, y, text, { "text-anchor": "middle", class: "mv-tick", ...(attrs || {}) });
}

// a right-aligned label of one or two lines (from fitLines) centred on ``y``; the whole text as its tooltip when cut
function svgLines(parent, x, y, lines, attrs, full) {
  const t = svgText(parent, x, y + 4 - (lines.length - 1) * 6.5, "", attrs);
  lines.forEach((line, i) => { const sp = sv("tspan", { x, dy: i ? 13 : 0 }, t); sp.textContent = line; });
  if (full && lines.join(" ").includes("…")) sv("title", {}, t).textContent = full;
  return t;
}

function button(label, cls, onClick, pressed) {
  const b = h("button", cls, label);
  b.type = "button";
  if (pressed !== undefined) b.setAttribute("aria-pressed", String(!!pressed));
  b.addEventListener("click", onClick);
  return b;
}

// A segmented control: [{key, label}], the chosen key, onChange(key)
function segmented(options, chosen, onChange, label) {
  const seg = h("div", "mv-seg");
  seg.setAttribute("role", "group");
  if (label) seg.setAttribute("aria-label", label);
  for (const o of options) {
    seg.appendChild(button(o.label, "", () => { if (o.key !== chosen) onChange(o.key); }, o.key === chosen));
  }
  return seg;
}

// ------------------------------------------------------------------------------------------- numbers
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function fmt(v, digits) {
  if (!isNum(v)) return "—";
  const a = Math.abs(v);
  const d = digits ?? (a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.1 ? 2 : 3);
  const s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  return (v < 0 && Number(s.replace(/,/g, "")) !== 0 ? "−" : "") + s;
}

function pct(v, d = 1) {
  return isNum(v) ? `${fmt(v, d)}%` : "—";
}

function signed(v, d = 1) {
  if (!isNum(v)) return "—";
  const s = fmt(Math.abs(v), d);
  return (v > 0 && Number(s) !== 0 ? "+" : v < 0 && Number(s) !== 0 ? "−" : "") + s;
}

function int(v) {
  if (isNum(v)) return Math.round(v).toLocaleString("en-US");
  return v === null || v === undefined || v === "" ? "—" : String(v);
}

// ------------------------------------------------------------------------------------------- scales
function scale(d0, d1, r0, r1, log) {
  const t = log ? (v) => Math.log(Math.max(v, 1e-12)) : (v) => v;
  const a = t(d0), b = t(d1);
  const k = (r1 - r0) / ((b - a) || 1);
  const f = (v) => r0 + (t(v) - a) * k;
  f.invert = (p) => {
    const x = a + (p - r0) / k;
    return log ? Math.exp(x) : x;
  };
  f.d = [d0, d1];
  f.r = [r0, r1];
  f.log = !!log;
  return f;
}

function niceStep(span, n) {
  const raw = span / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const r = raw / mag;
  return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mag;
}

function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, n);
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

// Ticks of a log scale: the first set, from fine to coarse, that fits the domain with at least three ticks
const LOG_SETS = [[0.8, 0.9, 1, 1.1, 1.25], [0.5, 0.67, 0.8, 1, 1.25, 1.5, 2], [0.25, 0.5, 1, 2, 4],
                  [0.1, 0.2, 0.5, 1, 2, 5, 10], [0.01, 0.1, 1, 10, 100]];

function logTicks(lo, hi, max = 7) {
  for (const set of LOG_SETS) {
    const inside = set.filter((v) => v >= lo * 0.999 && v <= hi * 1.001);
    const span = set[set.length - 1] / set[0];
    if (inside.length >= 3 && inside.length <= max && hi / lo <= span * 1.6) return inside;
  }
  const last = LOG_SETS[LOG_SETS.length - 1].filter((v) => v >= lo * 0.999 && v <= hi * 1.001);
  return last.length >= 2 ? last : [lo, 1, hi].filter((v) => v >= lo && v <= hi);
}

// A domain around a null value (0, or 1 on a log scale) holding every value, symmetric so the null sits in the middle
function nullDomain(values, nul, log, pad = 1.12) {
  const vs = values.filter((v) => isNum(v) && (!log || v > 0));
  if (!vs.length) return log ? [0.5, 2] : [-1, 1];
  if (log) {
    const m = Math.max(...vs.map((v) => Math.abs(Math.log(v / nul))), Math.log(1.25)) * pad;
    return [nul * Math.exp(-m), nul * Math.exp(m)];
  }
  const m = Math.max(...vs.map((v) => Math.abs(v - nul)), 1e-9) * pad;
  return [nul - m, nul + m];
}

function extent(values, pad = 0) {
  const vs = values.filter(isNum);
  if (!vs.length) return [0, 1];
  let lo = Math.min(...vs), hi = Math.max(...vs);
  if (lo === hi) { lo -= 1; hi += 1; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

// ------------------------------------------------------------------------------------------- tooltip
function makeTip(root) {
  let tip = root.querySelector(":scope > .mv-tip");
  if (!tip) {
    tip = h("div", "mv-tip");
    tip.setAttribute("role", "status");
    tip.setAttribute("aria-live", "polite");
    root.appendChild(tip);
  }
  return {
    // at: a pointer event, or an element (keyboard focus); rows: [{label, value, color, line}]
    show(at, title, rows, note) {
      tip.textContent = "";
      if (title) tip.appendChild(h("div", "mv-tip-title", title));
      for (const r of rows || []) {
        const row = h("div", "mv-tip-row");
        const lab = h("span");
        if (r.color) {
          const key = h("i", r.dot ? "mv-key mv-key-dot" : "mv-key");
          key.style.background = r.color;
          lab.appendChild(key);
        }
        lab.appendChild(document.createTextNode(r.label ?? ""));
        row.append(lab, h("b", null, r.value ?? ""));
        tip.appendChild(row);
      }
      if (note) tip.appendChild(h("div", "mv-tip-note", note));
      const box = root.getBoundingClientRect();
      let x, y;
      if (at && typeof at.clientX === "number") {
        x = at.clientX - box.left; y = at.clientY - box.top;
      } else if (at && at.getBoundingClientRect) {
        const r = at.getBoundingClientRect();
        x = r.left + r.width / 2 - box.left; y = r.top - box.top;
      } else {
        x = 0; y = 0;
      }
      tip.classList.add("mv-show");
      const w = tip.offsetWidth, hgt = tip.offsetHeight;
      let left = x + 14, top = y + 14;
      if (left + w > box.width - 4) left = Math.max(4, x - w - 14);
      if (top + hgt > box.height + 40) top = Math.max(4, y - hgt - 10);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    },
    hide() { tip.classList.remove("mv-show"); },
  };
}

// Hover and keyboard focus show the same tooltip; ``focusEl``: the element that takes the focus when ``el`` cannot
// (a row of display: contents has no box, so its first cell is focused instead)
function hoverable(el, ctx, content, focusEl) {
  const show = (e) => { const c = content(); if (c) ctx.tip.show(e, c.title, c.rows, c.note); };
  const target = focusEl || el;
  el.addEventListener("pointermove", show);
  el.addEventListener("pointerleave", () => ctx.tip.hide());
  target.addEventListener("focus", () => show(target));
  target.addEventListener("blur", () => ctx.tip.hide());
}

// ------------------------------------------------------------------------------------------- colours
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
const ORDINAL = ["--o1", "--o2", "--o3", "--o4"];

function cssVar(ctx, name) {
  return getComputedStyle(ctx.root).getPropertyValue(name).trim() || "#888888";
}

// The colour of each group: by role (treated → slot 1, comparison → slot 2), one ordered hue for quartiles,
// else the categorical slots in order (a seventh group and beyond wear the neutral of "before")
function groupColors(ctx, groups, ordered) {
  if (ordered) {
    const n = groups.length;
    return groups.map((_, i) => cssVar(ctx, ORDINAL[n <= 1 ? 2 : Math.round((i * (ORDINAL.length - 1)) / (n - 1))]));
  }
  let next = 0;
  const used = new Set();
  return groups.map((g) => {
    if (g.role === "treated") { used.add(0); return cssVar(ctx, SERIES[0]); }
    if (g.role === "control") { used.add(1); return cssVar(ctx, SERIES[1]); }
    while (used.has(next)) next += 1;
    used.add(next);
    return next < SERIES.length ? cssVar(ctx, SERIES[next]) : cssVar(ctx, "--before");
  });
}

// ------------------------------------------------------------------------------------------- shells and blocks
// A titled block with its tools on the right; a chart that has a table gets a Chart / Table switch (every value is
// reachable without hovering)
function shell(parent, b, ctx, id) {
  const block = h("div", "mv-block");
  parent.appendChild(block);
  const head = h("div", "mv-bhead");
  const titles = h("div");
  if (b.title) titles.appendChild(h("div", "mv-btitle", b.title));
  if (b.subtitle) titles.appendChild(h("div", "mv-bsub", b.subtitle));
  const tools = h("div", "mv-tools");
  head.append(titles, tools);
  if (b.title || b.subtitle) block.appendChild(head);
  const body = h("div", "mv-chart");
  block.appendChild(body);
  const caption = h("div", "mv-caption");
  if (b.caption) caption.textContent = b.caption;
  block.appendChild(caption);
  return { block, head, tools, body, caption, id, addTool(el) { tools.appendChild(el); if (!head.parentNode) block.prepend(head); } };
}

function tableEl(columns, rows, opts = {}) {
  const wrap = h("div", "mv-table-wrap");
  if (opts.maxHeight) { wrap.style.maxHeight = `${opts.maxHeight}px`; wrap.style.overflowY = "auto"; }
  const table = h("table", "mv-table");
  const thead = h("thead");
  const tr = h("tr");
  for (const c of columns) {
    const th = h("th", c.numeric ? "mv-n" : "", c.label);
    th.setAttribute("scope", "col");
    tr.appendChild(th);
  }
  thead.appendChild(tr);
  const tbody = h("tbody");
  for (const r of rows) {
    const row = h("tr", r.__heading ? "mv-th" : r.__level ? "mv-lvl" : "");
    columns.forEach((c, i) => {
      const v = r.__heading && i ? "" : r[c.key];
      const td = h("td", c.numeric ? "mv-n" : "");
      if (v && typeof v === "object" && v.node) td.appendChild(v.node);
      else td.textContent = v === null || v === undefined ? (i === 0 ? "" : "—") : String(v);
      if (r.__cls && r.__cls[c.key]) td.classList.add(r.__cls[c.key]);
      row.appendChild(td);
    });
    tbody.appendChild(row);
  }
  table.append(thead, tbody);
  wrap.appendChild(table);
  return wrap;
}

function chartBlock(parent, b, ctx) {
  const c = b.chart || {};
  const id = b.id || c.type || "chart";
  const s = shell(parent, b, ctx, id);
  s.spec = b;
  const draw = CHARTS[c.type];
  const asTable = c.table && ctx.state[`table:${id}`];
  if (c.table && c.table.rows && c.table.rows.length) {
    s.addTool(button(asTable ? "Chart" : "Table", "mv-chip-btn", () => {
      ctx.state[`table:${id}`] = !asTable;
      ctx.redrawBlock(s.block, (into) => chartBlock(into, b, ctx));
    }, !!asTable));
  }
  if (asTable || !draw) {
    if (c.table && c.table.rows) s.body.appendChild(tableEl(c.table.columns, c.table.rows));
    else s.body.appendChild(h("div", "mv-empty", "Nothing to draw."));
    return s.block;
  }
  try {
    draw(s.body, c, ctx, s);
  } catch (e) {
    s.body.textContent = "";
    s.body.appendChild(h("div", "mv-error", "This chart could not be drawn; its numbers are in the table."));
    console.warn("MES chart", c.type, e);
  }
  ctx.state.seen[`${ctx.scope}:${id}`] = true;
  return s.block;
}

// The first time a chart is drawn it animates; redrawn (a resize, a tab switched back) it does not
function fresh(ctx, s) {
  return motion() && !ctx.state.seen[`${ctx.scope}:${s.id}`];
}

const BLOCKS = {
  chart: chartBlock,
  table(parent, b, ctx) {
    const s = shell(parent, b, ctx, b.id || "table");
    let columns = b.columns || [];
    if (b.toggle) {          // columns of two views, e.g. Table 1 before / after weighting
      const key = `toggle:${b.id}`;
      const chosen = ctx.state[key] || b.toggle.default || b.toggle.options[0].key;
      s.addTool(segmented(b.toggle.options, chosen, (k) => {
        ctx.state[key] = k;
        ctx.redrawBlock(s.block, (into) => BLOCKS.table(into, b, ctx));
      }, b.toggle.label));
      columns = columns.filter((c) => !c.view || c.view === chosen);
    }
    const rows = (b.rows || []).map((r) => ({ ...r }));
    if (b.smd) {             // a bar beside each standardized mean difference, red from 0.1
      for (const r of rows) {
        for (const c of columns) {
          if (!c.smd || !isNum(r[c.key])) continue;
          const v = r[c.key];
          const span = h("span", Math.abs(v) >= 0.1 ? "mv-smd mv-smd-hi" : "mv-smd");
          const bar = h("i");
          bar.style.width = `${Math.min(40, Math.abs(v) * 100)}px`;
          span.append(bar, document.createTextNode(fmt(v, 2)));
          r[c.key] = { node: span };
        }
      }
    }
    s.body.appendChild(tableEl(columns, rows, { maxHeight: b.max_height }));
    return s.block;
  },
  kv(parent, b, ctx) {
    const s = shell(parent, b, ctx, b.id || "kv");
    const kv = h("div", "mv-kv");
    for (const [k, v] of b.rows || []) kv.append(h("div", "mv-k", k), h("div", null, v));
    s.body.appendChild(kv);
    return s.block;
  },
  notes(parent, b, ctx) {
    const block = h("div", "mv-block");
    const box = h("div", `mv-notes mv-notes-${b.tone || "info"}`);
    if (b.title) box.appendChild(h("div", "mv-notes-title", b.title));
    const ul = h("ul");
    for (const item of b.items || []) ul.appendChild(h("li", null, item));
    box.appendChild(ul);
    block.appendChild(box);
    parent.appendChild(block);
    return block;
  },
  chips(parent, b, ctx) {
    const s = shell(parent, b, ctx, b.id || "chips");
    const wrap = h("div", "mv-chips");
    for (const c of b.items || []) {
      const chip = h("div", "mv-chip");
      chip.appendChild(h("b", null, c.text));
      if (c.detail) chip.appendChild(h("span", null, c.detail));
      wrap.appendChild(chip);
    }
    s.body.appendChild(wrap);
    return s.block;
  },
  text(parent, b) {
    const block = h("div", "mv-block");
    if (b.title) block.appendChild(h("div", "mv-btitle", b.title));
    block.appendChild(h("div", "mv-text", b.text));
    parent.appendChild(block);
    return block;
  },
  scorecard(parent, b, ctx) {
    const s = shell(parent, b, ctx, b.id || "score");
    const items = b.items || [];
    const counts = { ok: 0, warn: 0, bad: 0, info: 0 };
    for (const it of items) counts[it.status in counts ? it.status : "info"] += 1;
    const sum = h("div", "mv-score-sum");
    const checks = items.filter((i) => i.status !== "info").length;
    sum.appendChild(h("b", null, `${counts.ok} of ${checks}`));
    sum.appendChild(document.createTextNode(" checks met"));
    const bar = h("div", "mv-score-bar");
    for (const it of items) bar.appendChild(h("i", it.status in counts ? it.status : "info"));
    sum.appendChild(bar);
    s.body.appendChild(sum);
    const list = h("div", "mv-score");
    const ICON = { ok: "✓", warn: "!", bad: "✕", info: "i" };
    for (const it of items) {
      const st = it.status in ICON ? it.status : "info";
      const row = h("div", `mv-score-row mv-score-${st}`);
      const ico = h("div", "mv-score-ico", ICON[st]);
      ico.setAttribute("aria-label", { ok: "met", warn: "caution", bad: "not met", info: "note" }[st]);
      const text = h("div");
      text.appendChild(h("div", "mv-score-label", it.label));
      if (it.detail) text.appendChild(h("div", "mv-score-detail", it.detail));
      row.append(ico, text);
      list.appendChild(row);
    }
    s.body.appendChild(list);
    return s.block;
  },
  defs(parent, b, ctx) {
    const s = shell(parent, b, ctx, b.id || "defs");
    const grid = h("div", "mv-defs");
    for (const d of b.items || []) {
      const card = h("div", "mv-def");
      card.appendChild(h("div", "mv-def-role", d.role));
      card.appendChild(h("div", "mv-def-label", d.label));
      const where = b.database ? ` in ${b.database}` : "";          // the study's database, by name
      const meta = [d.kind, d.rule, isNum(d.patients) ? `${int(d.patients)} patients${where}` : ""].filter(Boolean);
      if (meta.length) card.appendChild(h("div", "mv-def-meta", meta.join(" · ")));
      if (d.detail) card.appendChild(h("div", "mv-def-detail", d.detail));
      for (const [k, list] of Object.entries(d.codes || {})) {
        if (!list || !list.length) continue;
        const codes = h("div", "mv-codes");
        codes.appendChild(h("div", "mv-code-k", k));
        for (const code of list.slice(0, 40)) codes.appendChild(h("span", "mv-code", code));
        if (list.length > 40) codes.appendChild(h("span", "mv-code-k", `+ ${list.length - 40} more`));
        card.appendChild(codes);
      }
      if (d.medications && d.medications.length) {
        const meds = h("div", "mv-codes");
        meds.appendChild(h("div", "mv-code-k", `Medication names matched (${int(d.n_medications || d.medications.length)})`));
        for (const m of d.medications) meds.appendChild(h("span", "mv-code", m));
        if ((d.n_medications || 0) > d.medications.length) {
          meds.appendChild(h("span", "mv-code-k", `+ ${int(d.n_medications - d.medications.length)} more`));
        }
        card.appendChild(meds);
      }
      grid.appendChild(card);
    }
    s.body.appendChild(grid);
    return s.block;
  },
  grid(parent, b, ctx) {
    const g = h("div", "mv-grid");
    parent.appendChild(g);
    renderBlocks(g, b.blocks, ctx);
    return g;
  },
};

function renderBlocks(parent, blocks, ctx) {
  for (const b of blocks || []) {
    if (!b) continue;
    try {
      (BLOCKS[b.type] || BLOCKS.text)(parent, b, ctx);
    } catch (e) {
      parent.appendChild(h("div", "mv-error", "This part could not be drawn."));
      console.warn("MES block", b.type, e);
    }
  }
}
