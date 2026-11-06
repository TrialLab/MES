// MES charts — the study card of a real-world data analysis: header (citation, design, patients, checks), the numbers a
// reader looks for first, and tabs (results, protocol, cohort, balance, definitions …) of blocks built in viz.py.

function kpiTiles(parent, kpis, ctx) {
  if (!kpis || !kpis.length) return;
  const grid = h("div", "mv-kpis");
  for (const k of kpis) {
    const tile = h("div", k.tone === "primary" ? "mv-kpi mv-kpi-primary" : "mv-kpi");
    tile.appendChild(h("div", "mv-kpi-label", k.label));
    const value = h("div", /[A-Za-z]{3}/.test(String(k.value)) ? "mv-kpi-value mv-kpi-word"
      : String(k.value ?? "").length > 9 ? "mv-kpi-value mv-kpi-long" : "mv-kpi-value");
    value.appendChild(h("span", null, k.value));
    if (k.unit) value.appendChild(h("span", "mv-kpi-unit", k.unit));
    tile.appendChild(value);
    if (k.ci) tile.appendChild(h("div", "mv-kpi-ci", k.ci));
    if (k.note) { const n = h("div", "mv-kpi-note", k.note); n.title = k.note; tile.appendChild(n); }
    // a small interval bar: the estimate and its CI against the null (or the 0–100 % scale of a rate)
    if (k.bar && isNum(k.bar.value)) {
      const lo = k.bar.min ?? 0, hi = k.bar.max ?? 100;
      const pos = (v) => `${Math.max(0, Math.min(100, (100 * (v - lo)) / ((hi - lo) || 1)))}%`;
      const bar = h("div", "mv-kpi-bar");
      const fill = h("i");
      const a = isNum(k.bar.lo) ? k.bar.lo : k.bar.value, b = isNum(k.bar.hi) ? k.bar.hi : k.bar.value;
      fill.style.left = pos(Math.min(a, b));
      fill.style.width = `calc(${pos(Math.max(a, b))} - ${pos(Math.min(a, b))} + 3px)`;
      bar.appendChild(fill);
      if (isNum(k.bar.nul)) { const tick = h("b"); tick.style.left = pos(k.bar.nul); bar.appendChild(tick); }
      tile.appendChild(bar);
    }
    if (k.tip) hoverable(tile, ctx, () => ({ title: k.label, rows: [{ label: k.tip_label || "estimate", value: k.tip }] }));
    grid.appendChild(tile);
  }
  parent.appendChild(grid);
}

function badges(parent, list) {
  if (!list || !list.length) return;
  const wrap = h("div", "mv-badges");
  for (const b of list) wrap.appendChild(h("span", `mv-badge mv-badge-${b.kind || "neutral"}`, b.text));
  parent.appendChild(wrap);
}

// spec: {cite, dataset, design, n, unit, seconds, title, subtitle, badges, kpis, tabs: [{id, label, count, blocks}]}
function studyView(root, spec, ctx) {
  const card = h("div", "mv-study");
  root.appendChild(card);
  const head = h("div", "mv-head");
  const kick = h("div", "mv-kicker");
  if (spec.cite) kick.appendChild(h("span", "mv-cite", spec.cite));
  const bits = [spec.dataset, spec.design, isNum(spec.n) ? `${int(spec.n)} ${spec.unit || "patients"}` : "",
                isNum(spec.seconds) ? `${fmt(spec.seconds, spec.seconds < 10 ? 1 : 0)} s` : ""].filter(Boolean);
  bits.forEach((b, i) => { if (i) kick.appendChild(h("span", "mv-sep", "·")); kick.appendChild(h("span", null, b)); });
  head.appendChild(kick);
  if (spec.title) head.appendChild(h("div", "mv-title", spec.title));
  if (spec.subtitle) head.appendChild(h("div", "mv-sub", spec.subtitle));
  badges(head, spec.badges);
  card.appendChild(head);
  kpiTiles(card, spec.kpis, ctx);
  const tabs = (spec.tabs || []).filter((t) => t.blocks && t.blocks.length);
  if (!tabs.length) return;
  let active = tabs.some((t) => t.id === ctx.state.tab) ? ctx.state.tab : tabs[0].id;
  const list = h("div", "mv-tabs");
  list.setAttribute("role", "tablist");
  card.appendChild(list);
  let panel = null;
  const buttons = tabs.map((t) => {
    const b = h("button", "mv-tab", t.label);
    b.type = "button";
    b.setAttribute("role", "tab");
    b.id = `${ctx.scope}-tab-${t.id}`;
    if (isNum(t.count)) b.appendChild(h("span", "mv-tab-count", String(t.count)));
    b.addEventListener("click", () => show(t.id, true));
    b.addEventListener("keydown", (e) => {
      const i = tabs.findIndex((x) => x.id === active);
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const j = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
        show(tabs[j].id, true);
        buttons[j].focus();
      }
    });
    list.appendChild(b);
    return b;
  });
  function show(id, user) {
    active = id;
    ctx.state.tab = id;
    buttons.forEach((b, i) => {
      const on = tabs[i].id === id;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    const next = h("div", "mv-panel");
    next.setAttribute("role", "tabpanel");
    next.setAttribute("aria-labelledby", `${ctx.scope}-tab-${id}`);
    if (!user || !motion()) next.style.animation = "none";
    if (panel) panel.replaceWith(next); else card.appendChild(next);
    panel = next;
    ctx.scope = `${ctx.base}:${id}`;
    ctx.tip.hide();
    renderBlocks(panel, tabs.find((t) => t.id === id).blocks, ctx);
  }
  show(active, false);
}

// spec: {blocks}: blocks without a card (e.g. a chart in the answer)
function blocksView(root, spec, ctx) {
  renderBlocks(root, spec.blocks || [], ctx);
}

const CHARTS = { forest, arms, dots, ranges, subgroups, km, love, overlap, flow, timeline, steps, hist, trend, roc,
                 calibration, bars, cohortflow, received, profile, simmap, trajectory, sankey };
