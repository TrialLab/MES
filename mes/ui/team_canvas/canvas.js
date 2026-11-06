// The agent team builder: palette, canvas (pan / zoom, drag agents, typed ports, edges), inspector and run bar.
// Nothing here knows what the agents do: the use case supplies `Canvas` ({graph, view}, e.g. mes/ui/team_graph.js,
// loaded before this file with graph.js), the rules, the catalogue and the templates.
// Actions go to Python with
// setTriggerValue("action", {type: "back" | "apply" | "save" | "delete", graph, id, at}).
const memory = (globalThis.__teamCanvas = globalThis.__teamCanvas || {});
const ZMIN = 0.4, ZMAX = 1.6;
const COLORS = ["1", "2", "3", "4", "5", "slate", "red", "green"];

const SKELETON = (L) => `
<aside class="tb-palette">
  <div class="tb-pal-head">
    <div class="tb-tabs" role="tablist">
      <button type="button" class="tb-tab" data-tab="agents" role="tab">Agents</button>
      <button type="button" class="tb-tab" data-tab="templates" role="tab">Templates</button>
    </div>
    <input class="tb-search" type="search" placeholder="Search…" aria-label="Search agents and templates" maxlength="80">
  </div>
  <div class="tb-pal-body"></div>
</aside>
<section class="tb-main">
  <div class="tb-toolbar">
    <button type="button" class="tb-tool" data-act="undo" title="Undo (⌘Z)" aria-label="Undo"><svg class="tb-svg" viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 4.5 4 8l3.5 3.5"/><path d="M4.5 8H12a4 4 0 0 1 0 8H8.5"/></svg></button>
    <button type="button" class="tb-tool" data-act="redo" title="Redo (⇧⌘Z)" aria-label="Redo"><svg class="tb-svg" viewBox="0 0 20 20" aria-hidden="true"><path d="M12.5 4.5 16 8l-3.5 3.5"/><path d="M15.5 8H8a4 4 0 0 0 0 8h3.5"/></svg></button>
    <span class="tb-sep"></span>
    <button type="button" class="tb-tool tb-tool-text" data-act="layout" title="Arrange the agents">Auto-layout</button>
    <button type="button" class="tb-tool tb-tool-text" data-act="fit" title="Fit the team in view">Fit</button>
    <span class="tb-sep"></span>
    <button type="button" class="tb-tool" data-act="zoom-out" title="Zoom out" aria-label="Zoom out">−</button>
    <span class="tb-zoom">100%</span>
    <button type="button" class="tb-tool" data-act="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
    <span class="tb-grow"></span>
    <button type="button" class="tb-badge" data-act="team" title="Team settings and checks"></button>
  </div>
  <div class="tb-canvas" tabindex="0" aria-label="Agent team canvas">
    <div class="tb-world">
      <svg class="tb-edges" viewBox="-5000 -5000 10000 10000" aria-hidden="true">
        <defs></defs><g class="tb-edge-layer"></g><path class="tb-wire" d=""></path>
      </svg>
      <div class="tb-nodes"></div>
      <button type="button" class="tb-edge-del" title="Remove this connection (Delete)" aria-label="Remove connection">×</button>
    </div>
    <div class="tb-hint">Drag agents in from the left · drag from a port to connect · ⌘ + scroll to zoom</div>
    <div class="tb-drop-hint">Drop to add the agent to the team</div>
    <div class="tb-toast" role="status"></div>
  </div>
  <div class="tb-runbar">
    <button type="button" class="tb-btn" data-act="back">← Back</button>
    <div class="tb-runbar-fill"></div>
    <button type="button" class="tb-btn" data-act="save">Save as template</button>
    <button type="button" class="tb-btn tb-primary" data-act="apply">${L.apply}</button>
  </div>
</section>
<aside class="tb-inspector"></aside>
<div class="tb-ghost" aria-hidden="true"></div>
<div class="tb-modal" hidden></div>
`;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}

function slug(name) {
  return String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48).replace(/^-+|-+$/g, "") || "team";
}

export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const root = parentElement.querySelector(".tb");
  if (!root || !data || !data.rules || !data.catalog) return;
  const G = Canvas.graph, V = Canvas.view || {};
  G.configure(data.rules);
  const L = { apply: "Use this team", consumer: "the question box", ...(data.catalog.labels || {}) };
  const fresh = !root.dataset.built;          // the builder was (re)opened: a new canvas, fitted again
  if (fresh) {
    root.innerHTML = SKELETON(L);
    root.dataset.built = "1";
  }
  if (root.__tbAbort) root.__tbAbort.abort();      // Streamlit calls this again on every rerun
  const abort = new AbortController();
  root.__tbAbort = abort;
  const on = (target, type, fn, opts) => target.addEventListener(type, fn, { signal: abort.signal, ...(opts || {}) });

  const CAT = data.catalog, AG = CAT.agents, RULES = data.rules;
  const avail = data.availability || {};
  const templates = Array.isArray(data.templates) ? data.templates : [];
  const mem = (memory[data.key] = memory[data.key] || {});
  if (mem.nonce !== data.nonce || !mem.graph) {
    mem.nonce = data.nonce;
    mem.graph = G.normalize(data.graph);
    mem.undo = []; mem.redo = [];
    mem.selected = null;
    mem.view = { x: 0, y: 0, z: 1 };
    mem.needFit = true;
    mem.tab = mem.tab || "agents";
    mem.search = "";
    mem.question = mem.question || "";
  }
  mem.confirm = null;
  if (fresh) mem.needFit = true;
  root.classList.toggle("tb-dark", !!data.dark);

  const $ = (sel) => root.querySelector(sel);
  const canvas = $(".tb-canvas"), world = $(".tb-world"), nodesLayer = $(".tb-nodes");
  const svg = $(".tb-edges"), edgeLayer = $(".tb-edge-layer"), wire = $(".tb-wire"), defs = svg.querySelector("defs");
  const edgeDel = $(".tb-edge-del"), ghost = $(".tb-ghost"), toastEl = $(".tb-toast"), modal = $(".tb-modal");
  const palBody = $(".tb-pal-body"), search = $(".tb-search"), inspector = $(".tb-inspector");
  // a use case without a question of its own (labels.input === false) gets a detached input: the code below is the same
  const question = $(".tb-question") || el("input"), badge = $(".tb-badge"), zoomLabel = $(".tb-zoom");
  const sizes = {};
  let compiled = G.compile(mem.graph, avail);
  // the team the app is set to use, to show whether the canvas still is that team
  const active = data.active && data.active.graph
    ? { name: data.active.name, cfg: G.compile(data.active.graph, avail) } : null;
  // the run settings of a team: everything the compiler says except what is only about the canvas
  const runOf = (c) => JSON.stringify(Object.keys(c).filter((k) => !["ok", "agents", "messages"].includes(k)).sort()
    .map((k) => [k, c[k]]));
  const sameRun = (a, b) => !!a && !!b && runOf(a) === runOf(b);
  const useState = () => !active ? "none"
    : sameRun(compiled, active.cfg) && G.normalize(mem.graph).name === active.name ? "in-use" : "changed";
  let drag = null;
  // a dialog or drag ghost left from before this rerun has lost its listeners
  modal.hidden = true; modal.textContent = "";
  ghost.className = "tb-ghost";
  canvas.classList.remove("tb-drop", "tb-panning");

  // ---------------------------------------------------------------- markers (one arrow per colour)
  if (!defs.childElementCount) {
    for (const c of [...COLORS, "idle", "sel"]) {
      const m = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      m.setAttribute("id", `tb-arrow-${c}`);
      m.setAttribute("viewBox", "0 0 10 10");
      m.setAttribute("refX", "8"); m.setAttribute("refY", "5");
      m.setAttribute("markerWidth", "7"); m.setAttribute("markerHeight", "7");
      m.setAttribute("orient", "auto-start-reverse");
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", "M 1 1 L 9 5 L 1 9 z");
      p.setAttribute("class", `tb-arrow tb-arrow-${c}`);
      m.appendChild(p);
      defs.appendChild(m);
    }
  }

  // ---------------------------------------------------------------- state changes
  const snapshot = () => JSON.stringify(mem.graph);

  function commit(next, opts) {
    const before = snapshot();
    const n = G.normalize(next);
    if (JSON.stringify(n) === before) return false;
    mem.undo.push(before);
    if (mem.undo.length > 80) mem.undo.shift();
    mem.redo = [];
    mem.graph = n;
    if (opts && opts.fit) mem.needFit = true;
    render();
    return true;
  }

  function restore(fromStack, toStack) {
    if (!fromStack.length) return;
    toStack.push(snapshot());
    mem.graph = G.normalize(JSON.parse(fromStack.pop()));
    if (mem.selected && mem.selected.kind === "node" && !nodeOf(mem.selected.id)) mem.selected = null;
    if (mem.selected && mem.selected.kind === "edge" && !hasEdge(mem.selected.from, mem.selected.to)) mem.selected = null;
    render();
  }
  const undo = () => restore(mem.undo, mem.redo);
  const redo = () => restore(mem.redo, mem.undo);

  const nodeOf = (id) => mem.graph.nodes.find((n) => n.id === id);
  const hasEdge = (a, b) => mem.graph.edges.some((e) => e.from === a && e.to === b);
  const title = (t) => (AG[t] ? AG[t].title : t);
  const modeLabel = (m) => (CAT.modes[m] ? CAT.modes[m].label : m);
  const level = (m) => (RULES.mode_levels || {})[m] || "";
  const depths = RULES.modes.length > 1;   // more than one depth: the team can choose one
  const inserted = RULES.insert || {};
  // an agent of the middle row, or an optional one, that the compiler does not count in: on the canvas but not running
  const isIdle = (t) => (RULES.parallel.includes(t) || inserted[t]) && !compiled.agents.includes(t);
  // where an agent lands when it is dropped in: the rules already say it
  const wireHint = (t) => {
    if (RULES.parallel.includes(t)) {
      const [up, down] = RULES.wire.parallel;
      return ` — wired between the ${title(up)} and the ${title(down)}`;
    }
    return inserted[t] ? ` — inserted before the ${title(inserted[t].before)}` : "";
  };

  function toast(text, kind) {
    toastEl.textContent = text;
    toastEl.className = `tb-toast show${kind ? " tb-toast-" + kind : ""}`;
    clearTimeout(root.__tbToast);
    root.__tbToast = setTimeout(() => { toastEl.className = "tb-toast"; }, 2600);
  }

  function select(sel) {
    mem.selected = sel;
    renderNodes(); renderEdges(); renderInspector();
  }

  function addAgent(t, x, y) {
    if (nodeOf(t)) { select({ kind: "node", id: t }); toast(`${title(t)} is already in the team`); return; }
    let px = x, py = y;
    if (px === undefined) [px, py] = G.layout([...mem.graph.nodes.map((n) => n.type), t])[t];
    [px, py] = free(px, py);          // never on top of another agent
    if (commit(G.addNode(mem.graph, t, px, py))) {
      mem.selected = { kind: "node", id: t };
      renderNodes(); renderEdges(); renderInspector();
      ensureVisible(t);
      toast(`${title(t)} added${wireHint(t)}`, "ok");
    }
  }

  function free(x, y) {           // the nearest spot to the right or left that does not cover another agent
    const W = RULES.layout.node_w, step = W + RULES.layout.gap_x;
    const clear = (cx) => !mem.graph.nodes.some((n) => Math.abs(n.x - cx) < W && Math.abs(n.y - y) < ((sizes[n.id] || {}).h || 84));
    for (let i = 0; i < 12; i++) {
      for (const cx of i ? [x + i * step, x - i * step] : [x]) if (clear(cx)) return [cx, y];
    }
    return [x, y];
  }

  function removeAgent(t) {
    if (RULES.core.includes(t)) { toast(`${title(t)} is a core agent: it is always in the team`); return; }
    if (commit(G.removeNode(mem.graph, t))) {
      mem.selected = null;
      render();
      toast(`${title(t)} removed — ⌘Z to undo`);
    }
  }

  function connect(a, b) {
    const [next, err] = G.connect(mem.graph, a, b);
    if (err === "incompatible") {
      const out = RULES.ports[a][1], inp = RULES.ports[b][0];
      const what = inp === null ? `${title(b)} has no input` : out === null ? `${title(a)} has no output`
        : a === b ? "An agent cannot feed itself"
        : `${title(a)} gives ${CAT.port_labels[out].toLowerCase()}; ${title(b)} needs ${CAT.port_labels[inp].toLowerCase()}`;
      flash(b);
      toast(`${what}.`, "err");
      return;
    }
    if (commit(next)) toast(`Connected ${title(a)} → ${title(b)}`, "ok");
  }

  function flash(id) {
    const n = nodesLayer.querySelector(`[data-id="${id}"]`);
    if (!n) return;
    n.classList.remove("tb-flash"); void n.offsetWidth; n.classList.add("tb-flash");
  }

  // ---------------------------------------------------------------- view
  function applyView() {
    const v = mem.view;
    world.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.z})`;
    const g = 22 * v.z;
    canvas.style.backgroundSize = `${g}px ${g}px`;
    canvas.style.backgroundPosition = `${v.x % g}px ${v.y % g}px`;
    zoomLabel.textContent = `${Math.round(v.z * 100)}%`;
    positionEdgeDelete();
  }

  function zoomAt(factor, cx, cy) {
    const v = mem.view;
    const z = Math.min(ZMAX, Math.max(ZMIN, v.z * factor));
    v.x = cx - ((cx - v.x) * z) / v.z;
    v.y = cy - ((cy - v.y) * z) / v.z;
    v.z = z;
    applyView();
  }

  function bounds() {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of mem.graph.nodes) {
      const s = sizes[n.id] || { w: RULES.layout.node_w, h: 80 };
      minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + s.w); maxY = Math.max(maxY, n.y + s.h);
    }
    return { minX, minY, maxX, maxY };
  }

  function fit() {
    const r = canvas.getBoundingClientRect();
    if (!r.width || !r.height || !mem.graph.nodes.length) { mem.needFit = true; return; }
    const b = bounds();
    const bw = Math.max(1, b.maxX - b.minX), bh = Math.max(1, b.maxY - b.minY);
    const z = Math.min(1, Math.max(ZMIN, Math.min((r.width - 56) / bw, (r.height - 48) / bh)));
    mem.view = { z, x: r.width / 2 - (b.minX + bw / 2) * z, y: r.height / 2 - (b.minY + bh / 2) * z };
    mem.needFit = false;
    applyView();
  }

  function ensureVisible(id) {     // pan (no zoom change) so that the agent is inside the canvas
    const n = nodeOf(id), s = sizes[id], r = canvas.getBoundingClientRect();
    if (!n || !s || !r.width || !r.height) return;
    const v = mem.view, pad = 24;
    const left = n.x * v.z + v.x, top = n.y * v.z + v.y, right = left + s.w * v.z, bottom = top + s.h * v.z;
    let dx = 0, dy = 0;
    if (right > r.width - pad) dx = r.width - pad - right;
    if (left + dx < pad) dx = pad - left;
    if (bottom > r.height - pad) dy = r.height - pad - bottom;
    if (top + dy < pad) dy = pad - top;
    if (dx || dy) { v.x += dx; v.y += dy; applyView(); }
  }

  const toCanvas = (clientX, clientY) => {
    const r = canvas.getBoundingClientRect(), v = mem.view;
    return { x: (clientX - r.left - v.x) / v.z, y: (clientY - r.top - v.y) / v.z };
  };

  // ---------------------------------------------------------------- nodes
  // the two words under an agent on the canvas: what it will actually do at this depth. Only the use case knows.
  function footChips(t) {
    if (!V.chips) return [];
    const a = AG[t], depth = mem.graph.depth;
    const val = (k) => { const b = (a.budgets || []).find((x) => x.key === k); return b ? b.values[depth] : null; };
    return V.chips({ type: t, agent: a, node: nodeOf(t), depth, val, avail, modeLabel, level }) || [];
  }

  function nodeEl(n) {
    const t = n.type, a = AG[t];
    const io = RULES.ports[t][0] === null || RULES.ports[t][1] === null;      // the two ends of the team
    const box = el("div", `tb-node tb-c-${a.color}${io ? " tb-io" : ""}`);
    box.dataset.id = n.id;
    box.style.transform = `translate(${n.x}px, ${n.y}px)`;
    box.style.width = `${RULES.layout.node_w}px`;
    box.setAttribute("role", "button");
    box.setAttribute("aria-label", `${a.title} — ${a.role}`);
    const msgs = compiled.messages.filter((m) => m.node === t && m.level !== "info");
    if (isIdle(t)) box.classList.add("tb-idle");
    if (msgs.some((m) => m.level === "error")) box.classList.add("tb-err");
    else if (msgs.length) box.classList.add("tb-warn");
    if (mem.selected && mem.selected.kind === "node" && mem.selected.id === n.id) box.classList.add("tb-sel");
    const [inK, outK] = RULES.ports[t];
    if (inK) {
      const p = el("span", "tb-port tb-in");
      p.title = `Input · ${CAT.port_labels[inK]}`;
      box.appendChild(p);
    }
    const head = el("div", "tb-node-head");
    head.appendChild(el("span", "tb-ico", a.icon));
    const txt = el("div", "tb-node-text");
    const nm = el("div", "tb-node-title", a.title);
    nm.title = a.title;
    txt.appendChild(nm);
    let sub = (a.canvas_sub || a.subtitle || "").split("{depth}").join(modeLabel(mem.graph.depth));
    if (t === RULES.input_node) sub = (mem.question || "").trim() || data.default_question || sub;
    const s = el("div", "tb-node-sub", sub);
    s.title = sub;
    txt.appendChild(s);
    head.appendChild(txt);
    if (a.core && !io) {
      const lock = el("span", "tb-node-lock", "🔒");
      lock.title = "Core agent: always in the team";
      head.appendChild(lock);
    }
    box.appendChild(head);
    const chips = footChips(t);
    if (chips.length || msgs.length) {
      const foot = el("div", "tb-node-foot");
      for (const c of chips) foot.appendChild(el("span", "tb-chip", c));
      if (msgs.length) {
        const f = el("span", `tb-chip tb-chip-${msgs.some((m) => m.level === "error") ? "err" : "warn"}`, msgs.some((m) => m.level === "error") ? "⚠ fix" : "⚠");
        f.title = msgs.map((m) => m.text).join("\n");
        foot.appendChild(f);
      }
      box.appendChild(foot);
    }
    if (outK) {
      const p = el("span", "tb-port tb-out");
      p.title = `Output · ${CAT.port_labels[outK]} — drag to connect`;
      box.appendChild(p);
    }
    return box;
  }

  function renderNodes() {
    nodesLayer.textContent = "";
    for (const n of mem.graph.nodes) nodesLayer.appendChild(nodeEl(n));
    for (const box of nodesLayer.children) sizes[box.dataset.id] = { w: box.offsetWidth || RULES.layout.node_w, h: box.offsetHeight || 80 };
  }

  function moveNodeEl(id) {
    const n = nodeOf(id), box = nodesLayer.querySelector(`[data-id="${id}"]`);
    if (n && box) box.style.transform = `translate(${n.x}px, ${n.y}px)`;
  }

  // ---------------------------------------------------------------- edges
  function ports(id) {
    const n = nodeOf(id), s = sizes[id] || { w: RULES.layout.node_w, h: 80 };
    return { inX: n.x + s.w / 2, inY: n.y - 8, outX: n.x + s.w / 2, outY: n.y + s.h + 7 };   // just outside the port circles
  }

  function curve(x1, y1, x2, y2) {
    const dy = Math.max(36, Math.abs(y2 - y1) / 2);
    return [x1, y1, x1, y1 + dy, x2, y2 - dy, x2, y2];
  }
  const pathOf = (c) => `M ${c[0]} ${c[1]} C ${c[2]} ${c[3]}, ${c[4]} ${c[5]}, ${c[6]} ${c[7]}`;
  const mid = (c) => [(c[0] + 3 * c[2] + 3 * c[4] + c[6]) / 8, (c[1] + 3 * c[3] + 3 * c[5] + c[7]) / 8];

  // the edges a run actually walks down: the backbone, the agents the compiler counted in, the optional ones spliced in
  function chainEdges() {
    const set = new Set((RULES.chain || []).map(([a, b]) => `${a}>${b}`));
    const [up, down] = RULES.wire.parallel;
    for (const t of RULES.parallel) {
      if (compiled.agents.includes(t)) { set.add(`${up}>${t}`); set.add(`${t}>${down}`); }
    }
    for (const [t, ins] of Object.entries(inserted)) {
      if (!compiled.agents.includes(t)) continue;
      set.delete(`${ins.after}>${ins.before}`);
      set.add(`${ins.after}>${t}`); set.add(`${t}>${ins.before}`);
    }
    return set;
  }

  // an edge takes the colour of the agent of the middle row it touches; the stretch past the optional agents is green
  function edgeColor(from, to) {
    for (const t of [from, to]) if (RULES.parallel.includes(t)) return AG[t].color;
    for (const [t, ins] of Object.entries(inserted)) {
      if (from === t || to === t || from === ins.after) return "green";
    }
    return "slate";
  }

  function renderEdges() {
    edgeLayer.textContent = "";
    const chain = chainEdges();
    const NS = "http://www.w3.org/2000/svg";
    for (const e of mem.graph.edges) {
      if (!nodeOf(e.from) || !nodeOf(e.to)) continue;
      const a = ports(e.from), b = ports(e.to);
      const c = curve(a.outX, a.outY, b.inX, b.inY);
      const color = edgeColor(e.from, e.to);
      const inChain = chain.has(`${e.from}>${e.to}`);
      const selected = mem.selected && mem.selected.kind === "edge" && mem.selected.from === e.from && mem.selected.to === e.to;
      const cls = selected ? "sel" : inChain ? color : "idle";
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", pathOf(c));
      p.setAttribute("class", `tb-edge tb-edge-${cls}${inChain && compiled.ok && !selected ? " tb-live" : ""}`);
      p.setAttribute("marker-end", `url(#tb-arrow-${cls})`);
      const hit = document.createElementNS(NS, "path");
      hit.setAttribute("d", pathOf(c));
      hit.setAttribute("class", "tb-edge-hit");
      hit.dataset.from = e.from; hit.dataset.to = e.to;
      const tip = document.createElementNS(NS, "title");
      tip.textContent = `${title(e.from)} → ${title(e.to)} · ${CAT.port_labels[RULES.ports[e.from][1]]}${inChain ? "" : " (not used)"}`;
      hit.appendChild(tip);
      edgeLayer.appendChild(p);
      edgeLayer.appendChild(hit);
    }
    positionEdgeDelete();
  }

  function positionEdgeDelete() {
    const s = mem.selected;
    if (!s || s.kind !== "edge" || !hasEdge(s.from, s.to)) { edgeDel.style.display = "none"; return; }
    const a = ports(s.from), b = ports(s.to);
    const [mx, my] = mid(curve(a.outX, a.outY, b.inX, b.inY));
    edgeDel.style.display = "grid";
    edgeDel.style.transform = `translate(${mx - 11}px, ${my - 11}px) scale(${1 / mem.view.z})`;
  }

  // ---------------------------------------------------------------- palette
  function paletteItem(key, a, opts) {
    const item = el("div", `tb-pal-item tb-c-${a.color}${opts.locked ? " tb-pal-locked" : ""}${opts.inTeam ? " tb-pal-in" : ""}`);
    item.dataset.type = key;
    if (opts.draggable) item.dataset.drag = "1";
    if (opts.locked) item.dataset.locked = "1";
    item.title = opts.locked ? `${a.title} — ${CAT.locked_text}` : opts.draggable ? `Drag ${a.title} onto the canvas` : a.title;
    item.appendChild(el("span", "tb-ico", a.icon));
    const txt = el("div", "tb-pal-text");
    txt.appendChild(el("div", "tb-pal-name", a.title));
    txt.appendChild(el("div", "tb-pal-sub", a.subtitle));
    item.appendChild(txt);
    if (opts.locked) item.appendChild(el("span", "tb-tag tb-tag-lock", "later"));
    else if (opts.inTeam) {
      const c = el("span", "tb-check", "✓");
      c.title = "In the team — click to select it on the canvas";
      item.appendChild(c);
    }
    else if (opts.draggable) {
      const add = el("button", "tb-add", "+");
      add.type = "button";
      add.dataset.add = key;
      add.title = `Add ${a.title} to the team`;
      add.setAttribute("aria-label", `Add ${a.title}`);
      item.appendChild(add);
    } else {
      const c = el("span", "tb-check tb-check-core", "🔒");
      c.title = "Core agent: always in the team";
      item.appendChild(c);
    }
    return item;
  }

  function matchesSearch(...texts) {
    const q = (mem.search || "").trim().toLowerCase();
    return !q || texts.some((t) => String(t || "").toLowerCase().includes(q));
  }

  function renderPalette() {
    root.querySelectorAll(".tb-tab").forEach((b) => {
      b.classList.toggle("on", b.dataset.tab === mem.tab);
      b.setAttribute("aria-selected", String(b.dataset.tab === mem.tab));
    });
    if (search.value !== (mem.search || "")) search.value = mem.search || "";
    palBody.textContent = "";
    if (mem.tab === "templates") return renderTemplates();
    const inTeam = new Set(mem.graph.nodes.map((n) => n.type));
    let shown = 0;
    for (const { name, hint, types } of CAT.groups) {
      const items = types.filter((t) => AG[t] && matchesSearch(AG[t].title, AG[t].subtitle, AG[t].desc));
      if (!items.length) continue;
      const sec = el("div", "tb-pal-sec");
      sec.appendChild(el("div", "tb-pal-title", name));
      if (hint) sec.appendChild(el("div", "tb-pal-hint", hint));
      for (const t of items) {
        const core = RULES.core.includes(t);
        sec.appendChild(paletteItem(t, AG[t], { draggable: !core && !inTeam.has(t), inTeam: !core && inTeam.has(t) }));
      }
      palBody.appendChild(sec);
      shown += items.length;
    }
    const locked = (CAT.locked_agents || []).filter((a) => matchesSearch(a.title, a.subtitle, a.desc));
    if (locked.length) {
      const sec = el("div", "tb-pal-sec");
      sec.appendChild(el("div", "tb-pal-title", "Coming in a later version"));
      for (const a of locked) sec.appendChild(paletteItem(a.key, a, { locked: true }));
      palBody.appendChild(sec);
      shown += locked.length;
    }
    if (!shown) palBody.appendChild(el("div", "tb-empty", "No agent matches."));
  }

  function renderTemplates() {
    const list = templates.filter((t) => matchesSearch(t.name, t.description));
    const sections = [["Built-in templates", list.filter((t) => t.builtin)], ["Saved teams", list.filter((t) => !t.builtin)]];
    for (const [name, items] of sections) {
      const sec = el("div", "tb-pal-sec");
      sec.appendChild(el("div", "tb-pal-title", name));
      if (!items.length) {
        sec.appendChild(el("div", "tb-pal-hint", name === "Saved teams"
          ? (mem.search ? "No saved team matches." : "Teams you save appear here.") : "No template matches."));
      }
      for (const t of items) {
        const g = G.normalize(t.graph);
        const c = G.compile(g, avail);
        const card = el("div", "tb-tpl");
        card.dataset.tpl = t.id;
        card.setAttribute("role", "button");
        card.tabIndex = 0;
        card.title = `Load “${t.name}” onto the canvas`;
        const top = el("div", "tb-tpl-top");
        top.appendChild(el("div", "tb-tpl-name", t.name));
        if (active && t.name === active.name && sameRun(c, active.cfg)) {
          card.classList.add("tb-tpl-active");
          top.appendChild(el("span", "tb-tag tb-tag-in", "✓ in use"));
        } else top.appendChild(el("span", "tb-tag", `${modeLabel(g.depth)}`));
        card.appendChild(top);
        if (t.description) card.appendChild(el("div", "tb-tpl-desc", t.description));
        const icons = el("div", "tb-tpl-icons");
        for (const a of c.agents) {
          const i = el("span", `tb-mini tb-c-${AG[a].color}`, AG[a].icon);
          i.title = AG[a].title;
          icons.appendChild(i);
        }
        card.appendChild(icons);
        if (!t.builtin) {
          if (mem.confirm === t.id) {
            const row = el("div", "tb-confirm");
            row.appendChild(el("span", "", "Delete this saved team?"));
            const yes = el("button", "tb-btn tb-btn-sm tb-danger", "Delete");
            yes.type = "button"; yes.dataset.del = t.id;
            const no = el("button", "tb-btn tb-btn-sm", "Cancel");
            no.type = "button"; no.dataset.cancel = "1";
            row.append(yes, no);
            card.appendChild(row);
          } else {
            const x = el("button", "tb-tpl-x", "✕");
            x.type = "button"; x.dataset.ask = t.id;
            x.title = "Delete this saved team";
            x.setAttribute("aria-label", `Delete ${t.name}`);
            card.appendChild(x);
          }
        }
        sec.appendChild(card);
      }
      palBody.appendChild(sec);
    }
  }

  function loadTemplate(id) {
    const t = templates.find((x) => x.id === id);
    if (!t) return;
    mem.selected = null;
    if (commit(t.graph, { fit: true })) toast(`Loaded “${t.name}” — ⌘Z to go back`, "ok");
    else toast(`“${t.name}” is already on the canvas`);
  }

  // ---------------------------------------------------------------- inspector
  function section(parent, name, extra) {
    const s = el("div", "tb-sec");
    const h = el("div", "tb-sec-title", name);
    if (extra) h.appendChild(extra);
    s.appendChild(h);
    parent.appendChild(s);
    return s;
  }

  function lockTag(kind) {
    const t = el("span", `tb-lock tb-lock-${kind}`, kind === "depth" ? "🔒 by depth" : "🔒 locked");
    t.title = kind === "depth" ? CAT.depth_text : CAT.locked_text;
    return t;
  }

  function switchEl(checked, disabled, label) {
    const w = el("label", `tb-switch${disabled ? " tb-switch-off" : ""}`);
    const i = el("input");
    i.type = "checkbox"; i.checked = !!checked; i.disabled = !!disabled;
    i.setAttribute("aria-label", label);
    w.append(i, el("span", "tb-slider"));
    return [w, i];
  }

  function renderInspector() {
    inspector.textContent = "";
    const s = mem.selected;
    if (s && s.kind === "node" && nodeOf(s.id)) renderNodePanel(nodeOf(s.id));
    else if (s && s.kind === "locked") renderLockedPanel(s.key);
    else renderTeamPanel();
    const legend = el("div", "tb-legend");
    legend.append(lockTag("locked"), el("span", "", ` fixed in this version of the pipeline${depths ? " · " : ""}`));
    if (depths) legend.append(lockTag("depth"), el("span", "", " follows the depth"));
    inspector.appendChild(legend);
  }

  function head(icon, color, name, role, closable) {
    const h = el("div", "tb-insp-head");
    h.appendChild(el("span", `tb-ico tb-ico-lg tb-c-${color}`, icon));
    const t = el("div", "tb-insp-text");
    t.appendChild(el("div", "tb-insp-title", name));
    t.appendChild(el("div", "tb-insp-role", role));
    h.appendChild(t);
    if (closable) {
      const x = el("button", "tb-x", "✕");
      x.type = "button"; x.dataset.act = "team"; x.title = "Back to the team settings";
      x.setAttribute("aria-label", "Close");
      h.appendChild(x);
    }
    inspector.appendChild(h);
  }

  function renderPrompts(t, a, depth) {     // prompts, or the search strategy / rules of the agent (read only)
    if (!a.prompts || !a.prompts.length) return;
    const sec = section(inspector, a.panel || "Prompts");
    for (const p of a.prompts) {
      const key = `${t}:${p.label}`;
      const det = el("details", "tb-prompt");
      det.open = mem.openPrompt === key;
      det.dataset.key = key;
      const sum = el("summary", "", p.label);
      sum.appendChild(lockTag("locked"));
      det.appendChild(sum);
      const extra = p.by_depth ? p.by_depth[depth] || "" : "";
      const body = (p.text || "") + (extra ? `${p.text ? "\n\n" : ""}— ${modeLabel(depth)} (${level(depth) || depth}) —\n${extra}` : "");
      const ta = el("textarea", "tb-prompt-text");
      ta.readOnly = true;
      ta.value = body;
      ta.rows = Math.min(14, Math.max(4, body.split("\n").length));
      ta.setAttribute("aria-label", `${p.label} (read only)`);
      det.appendChild(ta);
      const bar = el("div", "tb-prompt-bar");
      bar.appendChild(el("span", "tb-tool-desc", p.hint || "Editable in a later version."));
      const copy = el("button", "tb-btn tb-btn-sm", "Copy");
      copy.type = "button"; copy.dataset.copy = key;
      bar.appendChild(copy);
      det.appendChild(bar);
      sec.appendChild(det);
    }
  }

  function renderNodePanel(n) {
    const t = n.type, a = AG[t], depth = mem.graph.depth;
    head(a.icon, a.color, a.title, a.role, true);
    const msgs = compiled.messages.filter((m) => m.node === t);
    const status = el("div", "tb-status");
    const idle = isIdle(t);
    status.appendChild(el("span", `tb-pill ${idle ? "tb-pill-idle" : "tb-pill-ok"}`, idle ? "Not connected — will not run"
      : a.core ? "Core · always runs" : "In the team · runs"));
    inspector.appendChild(status);
    for (const m of msgs) inspector.appendChild(el("div", `tb-msg tb-msg-${m.level}`, m.text));
    inspector.appendChild(el("p", "tb-desc", a.desc));

    if (a.tools && a.tools.length) {
      const sec = section(inspector, "Tools");
      for (const tool of a.tools) {
        const row = el("div", "tb-tool-row");
        const txt = el("div", "tb-tool-text");
        const name = el("div", "tb-tool-name", tool.label);
        txt.appendChild(name);
        txt.appendChild(el("div", "tb-tool-desc", tool.desc));
        row.appendChild(txt);
        const right = el("div", "tb-tool-ctl");
        if (tool.kind === "toggle") {
          const value = n.tools[tool.key] !== false;
          const [w, input] = switchEl(value, false, tool.label);
          input.dataset.tool = tool.key;
          right.appendChild(w);
          // what this computer cannot do: a grey aside (info) or an amber warning (note), each on an availability flag
          if (tool.info && avail[tool.info.flag] === false) {
            const note = el("div", "tb-tool-desc tb-tool-info", tool.info.text);
            const why = avail[`${tool.info.flag}_reason`];
            if (why) note.title = why;
            txt.appendChild(note);
          }
          if (tool.note && avail[tool.note.flag] === false) {
            txt.appendChild(el("div", "tb-tool-note", avail[`${tool.note.flag}_reason`] || tool.note.text));
          }
        } else {
          const [w] = switchEl(tool.on[depth], true, tool.label);
          right.append(w, lockTag(tool.kind === "depth" ? "depth" : "locked"));
          row.title = tool.kind === "depth" ? CAT.depth_text : CAT.locked_text;
        }
        row.appendChild(right);
        sec.appendChild(row);
      }
    }

    if (a.budgets && a.budgets.length) {
      const sec = section(inspector, depths ? `Settings · ${modeLabel(depth)}` : "Settings",
        lockTag(depths ? "depth" : "locked"));
      for (const b of a.budgets) {
        const row = el("label", "tb-field");
        row.appendChild(el("span", "tb-field-label", b.label));
        const box = el("span", "tb-field-box");
        const i = el("input", "tb-input");
        i.disabled = true; i.readOnly = true;
        i.value = `${b.values[depth]}${b.unit ? " " + b.unit : ""}`;
        i.setAttribute("aria-label", b.label);
        box.appendChild(i);
        box.appendChild(el("span", "tb-field-lock", "🔒"));
        row.appendChild(box);
        row.title = depths ? `${CAT.locked_text} (now set by the depth)` : CAT.locked_text;
        sec.appendChild(row);
      }
    }

    if (a.model) {
      const msec = section(inspector, "Model");
      const row = el("label", "tb-field tb-field-col");
      row.appendChild(el("span", "tb-field-label", a.model_role || "Language model"));
      const box = el("span", "tb-field-box");
      const sel = el("select", "tb-input");
      sel.disabled = true;
      sel.appendChild(el("option", "", a.model));
      sel.setAttribute("aria-label", a.model_role || "Language model");
      box.append(sel, el("span", "tb-field-lock", "🔒"));
      row.appendChild(box);
      row.title = CAT.locked_text;
      msec.appendChild(row);
      if (a.llm) msec.appendChild(el("div", "tb-tool-desc", a.llm));
    }

    renderPrompts(t, a, depth);

    const ps = section(inspector, "Ports");
    const [inK, outK] = RULES.ports[t];
    ps.appendChild(el("div", "tb-tool-desc", `In: ${inK ? CAT.port_labels[inK] : "—"} · Out: ${outK ? CAT.port_labels[outK] : "—"}`));

    if (!a.core) {
      const rm = el("button", "tb-btn tb-danger tb-wide", `Remove ${a.title} from the team`);
      rm.type = "button"; rm.dataset.remove = t;
      inspector.appendChild(rm);
    }
  }

  function renderLockedPanel(key) {
    const a = (CAT.locked_agents || []).find((x) => x.key === key);
    if (!a) return renderTeamPanel();
    head(a.icon, a.color, a.title, a.role, true);
    const st = el("div", "tb-status");
    st.appendChild(lockTag("locked"));
    st.appendChild(el("span", "tb-tool-desc", " Coming in a later version"));
    inspector.appendChild(st);
    inspector.appendChild(el("p", "tb-desc", a.desc));
    renderPrompts(a.key, a, mem.graph.depth);
    inspector.appendChild(el("p", "tb-tool-desc tb-pad", CAT.locked_text));
  }

  function renderTeamPanel() {
    head("🧩", "red", "Agent team", "Team settings", false);
    const g = mem.graph;
    const use = useState();
    const st = el("div", "tb-status");
    st.appendChild(el("span", `tb-pill ${use === "in-use" ? "tb-pill-ok" : use === "changed" ? "tb-pill-warn" : "tb-pill-idle"}`,
      use === "in-use" ? `✓ In use in ${L.consumer}` : use === "changed" ? "Changed — not in use yet" : "Not in use yet"));
    inspector.appendChild(st);
    if (use !== "in-use") {
      inspector.appendChild(el("div", "tb-tool-desc tb-pad", use === "changed"
        ? `“${active.name}” as it was is still the team in ${L.consumer}. Press ${L.apply} to use these changes.`
        : `${L.apply} puts it into ${L.consumer}.`));
    }
    const ts = section(inspector, "Team");
    const nameRow = el("label", "tb-field tb-field-col");
    nameRow.appendChild(el("span", "tb-field-label", "Name"));
    const name = el("input", "tb-input tb-edit");
    name.value = g.name; name.maxLength = RULES.limits.name; name.dataset.meta = "name";
    nameRow.appendChild(name);
    ts.appendChild(nameRow);
    const descRow = el("label", "tb-field tb-field-col");
    descRow.appendChild(el("span", "tb-field-label", "Description"));
    const desc = el("textarea", "tb-input tb-edit");
    desc.value = g.description; desc.maxLength = RULES.limits.description; desc.rows = 3; desc.dataset.meta = "description";
    descRow.appendChild(desc);
    ts.appendChild(descRow);

    if (depths) {                        // a use case with one depth has nothing to choose
      const ds = section(inspector, "Depth");
      const seg = el("div", "tb-seg");
      for (const m of RULES.modes) {
        const b = el("button", `tb-seg-btn${g.depth === m ? " on" : ""}`);
        b.type = "button"; b.dataset.depth = m;
        b.setAttribute("aria-pressed", String(g.depth === m));
        b.appendChild(el("span", "tb-seg-level", level(m) || m));
        b.appendChild(el("span", "tb-seg-name", modeLabel(m)));
        seg.appendChild(b);
      }
      ds.appendChild(seg);
      ds.appendChild(el("div", "tb-tool-desc", CAT.modes[g.depth] ? CAT.modes[g.depth].tagline : ""));
    }

    if (CAT.cache) {
      const os = section(inspector, "Options");
      const row = el("div", "tb-tool-row");
      const txt = el("div", "tb-tool-text");
      txt.appendChild(el("div", "tb-tool-name", CAT.cache.name));
      txt.appendChild(el("div", "tb-tool-desc", CAT.cache.desc));
      row.appendChild(txt);
      const [w, input] = switchEl(g.use_cache, false, CAT.cache.name);
      input.dataset.cache = "1";
      const ctl = el("div", "tb-tool-ctl");
      ctl.appendChild(w);
      row.appendChild(ctl);
      os.appendChild(row);
    }

    // what the run will do, in the use case's own words: [label, text | {chips: [type]} | {text, lock}]
    const rs = section(inspector, "Run settings");
    const list = el("div", "tb-kv");
    for (const [k, v] of (V.summary ? V.summary(compiled, g, data, { avail, modeLabel, level }) : [])) {
      list.appendChild(el("span", "tb-k", k));
      const vv = el("span", "tb-v");
      if (v && v.chips) {
        const chips = el("span", "tb-chips");
        for (const t of v.chips) chips.appendChild(el("span", `tb-mini tb-mini-wide tb-c-${AG[t].color}`, `${AG[t].icon} ${AG[t].title}`));
        if (!v.chips.length) chips.textContent = v.empty || "none";
        vv.appendChild(chips);
      } else if (v && v.lock) {
        const w = el("span", "tb-v-lock");
        w.append(el("span", "", v.text || ""), lockTag(v.lock));
        vv.appendChild(w);
      } else vv.textContent = v;
      list.appendChild(vv);
    }
    rs.appendChild(list);

    const cs = section(inspector, "Checks");
    if (!compiled.messages.some((m) => m.level !== "info")) cs.appendChild(el("div", "tb-msg tb-msg-ok", "✓ The team is ready to run."));
    for (const m of compiled.messages) {
      const b = el("button", `tb-msg tb-msg-${m.level} tb-msg-link`, m.text);
      b.type = "button";
      if (m.node) b.dataset.goto = m.node;
      cs.appendChild(b);
    }
  }

  // ---------------------------------------------------------------- toolbar, run bar
  function renderBars() {
    const errors = compiled.messages.filter((m) => m.level === "error").length;
    const warnings = compiled.messages.filter((m) => m.level === "warning").length;
    badge.className = `tb-badge ${errors ? "tb-badge-err" : warnings ? "tb-badge-warn" : "tb-badge-ok"}`;
    badge.textContent = errors ? `✕ ${errors} problem${errors > 1 ? "s" : ""} to fix`
      : `✓ Ready · ${compiled.agents.length} agents · ${modeLabel(compiled.mode)}${warnings ? ` · ⚠ ${warnings}` : ""}`;
    root.querySelector('[data-act="undo"]').disabled = !mem.undo.length;
    root.querySelector('[data-act="redo"]').disabled = !mem.redo.length;
    const firstError = (compiled.messages.find((m) => m.level === "error") || {}).text;
    const applyBtn = root.querySelector('[data-act="apply"]');
    applyBtn.disabled = !compiled.ok;
    applyBtn.title = compiled.ok ? `${L.apply} in ${L.consumer}` : firstError;
  }

  function render() {
    compiled = G.compile(mem.graph, avail);
    renderPalette();
    renderNodes();
    renderEdges();
    renderInspector();
    renderBars();
    if (mem.needFit) fit(); else applyView();
  }

  // ---------------------------------------------------------------- actions to Python
  function send(type, extra) {
    setTriggerValue("action", { type, graph: mem.graph, at: Date.now(), ...(extra || {}) });
  }

  function openSave() {
    const g = mem.graph;
    modal.hidden = false;
    modal.textContent = "";
    const card = el("div", "tb-modal-card");
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-label", "Save as template");
    card.appendChild(el("div", "tb-insp-title", "Save as template"));
    const nameRow = el("label", "tb-field tb-field-col");
    nameRow.appendChild(el("span", "tb-field-label", "Name"));
    const name = el("input", "tb-input tb-edit");
    name.value = g.name === "Untitled team" ? "" : g.name;
    name.placeholder = "e.g. Sepsis RWE team";
    name.maxLength = RULES.limits.name;
    nameRow.appendChild(name);
    const descRow = el("label", "tb-field tb-field-col");
    descRow.appendChild(el("span", "tb-field-label", "Description"));
    const desc = el("textarea", "tb-input tb-edit");
    desc.value = g.description; desc.rows = 2; desc.maxLength = RULES.limits.description;
    descRow.appendChild(desc);
    const note = el("div", "tb-tool-desc tb-modal-note");
    const update = () => {
      const n = G.text(name.value, RULES.limits.name);
      const builtin = templates.some((t) => t.builtin && t.name.toLowerCase() === n.toLowerCase());
      const existing = templates.find((t) => !t.builtin && slug(t.name) === slug(n));
      note.textContent = !n ? "Give the team a name." : builtin ? "A built-in template has this name: it is saved as “… (custom)”."
        : existing ? `Replaces the saved team “${existing.name}”.` : "Saved on this computer; it appears under Templates.";
      saveBtn.disabled = !n;
    };
    const bar = el("div", "tb-modal-bar");
    const cancel = el("button", "tb-btn", "Cancel");
    cancel.type = "button";
    const saveBtn = el("button", "tb-btn tb-primary", "Save");
    saveBtn.type = "button";
    bar.append(cancel, saveBtn);
    card.append(nameRow, descRow, note, bar);
    modal.appendChild(card);
    update();
    const close = () => { modal.hidden = true; modal.textContent = ""; canvas.focus({ preventScroll: true }); };
    const doSave = () => {
      const n = G.text(name.value, RULES.limits.name);
      if (!n) return;
      commit({ ...mem.graph, name: n, description: desc.value });
      modal.hidden = true; modal.textContent = "";
      send("save");
    };
    name.addEventListener("input", update, { signal: abort.signal });
    cancel.addEventListener("click", close, { signal: abort.signal });
    saveBtn.addEventListener("click", doSave, { signal: abort.signal });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
      else if (e.key === "Enter" && e.target === name && !e.isComposing) { e.preventDefault(); doSave(); }
    }, { signal: abort.signal });
    modal.addEventListener("pointerdown", (e) => { if (e.target === modal) close(); }, { signal: abort.signal });
    setTimeout(() => name.focus(), 30);
  }

  function copyText(text) {
    const done = () => toast("Prompt copied", "ok");
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
    } else fallbackCopy(text, done);
  }
  function fallbackCopy(text, done) {
    const ta = el("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    root.appendChild(ta); ta.select();
    try { document.execCommand("copy"); done(); } catch (e) { toast("Select the text and copy it", "err"); }
    ta.remove();
  }

  // ---------------------------------------------------------------- events: clicks
  on(root, "click", (e) => {
    const t = e.target;
    const act = t.closest("[data-act]");
    if (act && root.contains(act) && !act.disabled) {
      const a = act.dataset.act;
      const r = canvas.getBoundingClientRect();
      if (a === "undo") undo();
      else if (a === "redo") redo();
      else if (a === "fit") fit();
      else if (a === "layout") {
        const pos = G.layout(mem.graph.nodes.map((n) => n.type));
        if (commit({ ...mem.graph, nodes: mem.graph.nodes.map((n) => ({ ...n, x: pos[n.type][0], y: pos[n.type][1] })) }, { fit: true })) toast("Arranged", "ok");
        else fit();
      }
      else if (a === "zoom-in") zoomAt(1.2, r.width / 2, r.height / 2);
      else if (a === "zoom-out") zoomAt(1 / 1.2, r.width / 2, r.height / 2);
      else if (a === "team") select(null);
      else if (a === "back") send("back");
      else if (a === "save") openSave();
      else if (a === "apply") { if (compiled.ok) send("apply"); }
      return;
    }
    const tab = t.closest(".tb-tab");
    if (tab) { mem.tab = tab.dataset.tab; mem.confirm = null; renderPalette(); return; }
    const add = t.closest("[data-add]");
    if (add) { addAgent(add.dataset.add); return; }
    const ask = t.closest("[data-ask]");
    if (ask) { mem.confirm = ask.dataset.ask; renderPalette(); return; }
    const del = t.closest("[data-del]");
    if (del) { mem.confirm = null; send("delete", { id: del.dataset.del }); return; }
    if (t.closest("[data-cancel]")) { mem.confirm = null; renderPalette(); return; }
    const tpl = t.closest("[data-tpl]");
    if (tpl) { loadTemplate(tpl.dataset.tpl); return; }
    const item = t.closest(".tb-pal-item");
    if (item && !root.__tbDragged) {
      const type = item.dataset.type;
      if (item.dataset.locked) select({ kind: "locked", key: type });
      else if (nodeOf(type)) select({ kind: "node", id: type });
      else if (item.dataset.drag) addAgent(type);
      return;
    }
    const rm = t.closest("[data-remove]");
    if (rm) { removeAgent(rm.dataset.remove); return; }
    const depth = t.closest("[data-depth]");
    if (depth) { if (commit({ ...mem.graph, depth: depth.dataset.depth })) toast(`Depth: ${modeLabel(depth.dataset.depth)} (${level(depth.dataset.depth)})`, "ok"); return; }
    const go = t.closest("[data-goto]");
    if (go) { if (nodeOf(go.dataset.goto)) select({ kind: "node", id: go.dataset.goto }); return; }
    const copy = t.closest("[data-copy]");
    if (copy) { const ta = copy.closest("details").querySelector("textarea"); copyText(ta.value); return; }
    if (t === edgeDel || t.closest(".tb-edge-del")) {
      const s = mem.selected;
      if (s && s.kind === "edge") { mem.selected = null; if (commit(G.disconnect(mem.graph, s.from, s.to))) toast("Connection removed — ⌘Z to undo"); }
    }
  });

  on(root, "keydown", (e) => {
    const tpl = e.target.closest && e.target.closest("[data-tpl]");
    if (tpl && (e.key === "Enter" || e.key === " ") && e.target === tpl) { e.preventDefault(); loadTemplate(tpl.dataset.tpl); }
  });

  on(root, "change", (e) => {
    const t = e.target;
    if (t.dataset.tool && mem.selected && mem.selected.kind === "node") {
      commit(G.setTool(mem.graph, mem.selected.id, t.dataset.tool, t.checked));
    } else if (t.dataset.cache) {
      commit({ ...mem.graph, use_cache: t.checked });
    } else if (t.dataset.meta) {         // name / description: one undo step for the whole edit
      const before = mem.metaBefore;
      mem.metaBefore = null;
      const n = G.normalize({ ...mem.graph, [t.dataset.meta]: t.value });
      if (before && JSON.stringify(n) !== before) {
        mem.undo.push(before);
        if (mem.undo.length > 80) mem.undo.shift();
        mem.redo = [];
      }
      mem.graph = n;
      renderBars();
    }
  });

  on(root, "input", (e) => {
    const t = e.target;
    if (t === search) { mem.search = search.value; mem.confirm = null; renderPalette(); }
    else if (t === question) {
      mem.question = question.value;
      const q = nodesLayer.querySelector(`[data-id="${RULES.input_node}"] .tb-node-sub`);
      if (q) { q.textContent = question.value.trim() || data.default_question || AG[RULES.input_node].subtitle; q.title = q.textContent; }
    } else if (t.dataset.meta) {          // name / description: kept while typing, one undo step on change
      if (!mem.metaBefore) mem.metaBefore = snapshot();
      mem.graph = { ...mem.graph, [t.dataset.meta]: t.value };
    }
  });

  on(root, "toggle", (e) => {
    const d = e.target;
    if (d.classList && d.classList.contains("tb-prompt")) {
      if (d.open) mem.openPrompt = d.dataset.key; else if (mem.openPrompt === d.dataset.key) mem.openPrompt = null;
    }
  }, { capture: true });

  // keyboard on the canvas and panels: Delete, ⌘Z / ⇧⌘Z, Escape
  on(root, "keydown", (e) => {
    const target = e.composedPath()[0];
    const typing = target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName || ""));
    if (!modal.hidden) return;
    const mod = e.metaKey || e.ctrlKey;
    if (mod && !typing && (e.key === "z" || e.key === "Z")) { e.preventDefault(); e.stopPropagation(); if (e.shiftKey) redo(); else undo(); return; }
    if (mod && !typing && (e.key === "y" || e.key === "Y")) { e.preventDefault(); e.stopPropagation(); redo(); return; }
    if (typing) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      const s = mem.selected;
      if (!s) return;
      e.preventDefault(); e.stopPropagation();
      if (s.kind === "node") removeAgent(s.id);
      else if (s.kind === "edge") { mem.selected = null; if (commit(G.disconnect(mem.graph, s.from, s.to))) toast("Connection removed — ⌘Z to undo"); }
    } else if (e.key === "Escape" && mem.selected) { e.preventDefault(); select(null); }
  });

  // ---------------------------------------------------------------- events: pointer (canvas)
  function nodeAt(p, margin) {
    for (let i = mem.graph.nodes.length - 1; i >= 0; i--) {
      const n = mem.graph.nodes[i], s = sizes[n.id] || { w: RULES.layout.node_w, h: 80 };
      if (p.x >= n.x - margin && p.x <= n.x + s.w + margin && p.y >= n.y - margin && p.y <= n.y + s.h + margin) return n;
    }
    return null;
  }

  on(canvas, "pointerdown", (e) => {
    if (e.button !== 0 || drag) return;
    const t = e.target;
    if (t.closest(".tb-edge-del")) return;
    canvas.focus({ preventScroll: true });
    const out = t.closest(".tb-port.tb-out");
    const node = t.closest(".tb-node");
    const hit = t.closest(".tb-edge-hit");
    if (out && node) {
      drag = { kind: "wire", from: node.dataset.id, id: e.pointerId };
      wire.setAttribute("class", "tb-wire on");
    } else if (node) {
      const n = nodeOf(node.dataset.id);
      if (!n) return;
      drag = { kind: "node", id: n.id, sx: e.clientX, sy: e.clientY, x: n.x, y: n.y, moved: false, before: snapshot(), pid: e.pointerId };
      if (!(mem.selected && mem.selected.kind === "node" && mem.selected.id === n.id)) select({ kind: "node", id: n.id });
    } else if (hit) {
      select({ kind: "edge", from: hit.dataset.from, to: hit.dataset.to });
      return;
    } else {
      drag = { kind: "pan", sx: e.clientX, sy: e.clientY, x: mem.view.x, y: mem.view.y, moved: false };
      canvas.classList.add("tb-panning");
    }
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* pointer already released */ }
    e.preventDefault();
  });

  on(canvas, "pointermove", (e) => {
    if (!drag) return;
    if (drag.kind === "pan") {
      const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      mem.view.x = drag.x + dx; mem.view.y = drag.y + dy;
      applyView();
    } else if (drag.kind === "node") {
      const dx = (e.clientX - drag.sx) / mem.view.z, dy = (e.clientY - drag.sy) / mem.view.z;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) < 3) return;
      drag.moved = true;
      const lim = RULES.limits.coord;
      const n = nodeOf(drag.id);
      if (!n) return;
      n.x = Math.round(Math.min(lim, Math.max(-lim, drag.x + dx)));
      n.y = Math.round(Math.min(lim, Math.max(-lim, drag.y + dy)));
      moveNodeEl(n.id);
      renderEdges();
    } else if (drag.kind === "wire") {
      const a = ports(drag.from), p = toCanvas(e.clientX, e.clientY);
      wire.setAttribute("d", pathOf(curve(a.outX, a.outY, p.x, p.y)));
      const target = nodeAt(p, 10);
      nodesLayer.querySelectorAll(".tb-target, .tb-target-bad").forEach((x) => x.classList.remove("tb-target", "tb-target-bad"));
      if (target && target.id !== drag.from) {
        const box = nodesLayer.querySelector(`[data-id="${target.id}"]`);
        if (box) box.classList.add(G.compatible(drag.from, target.id) ? "tb-target" : "tb-target-bad");
      }
    }
  });

  function endDrag(e, cancelled) {
    if (!drag) return;
    const d = drag;
    drag = null;
    canvas.classList.remove("tb-panning");
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* not captured */ }
    if (d.kind === "pan") {
      if (!d.moved && !cancelled && mem.selected) select(null);
    } else if (d.kind === "node") {
      if (d.moved) {
        mem.undo.push(d.before);
        if (mem.undo.length > 80) mem.undo.shift();
        mem.redo = [];
        mem.graph = G.normalize(mem.graph);
        renderBars();
      }
    } else if (d.kind === "wire") {
      wire.setAttribute("class", "tb-wire");
      wire.setAttribute("d", "");
      nodesLayer.querySelectorAll(".tb-target, .tb-target-bad").forEach((x) => x.classList.remove("tb-target", "tb-target-bad"));
      if (!cancelled) {
        const target = nodeAt(toCanvas(e.clientX, e.clientY), 10);
        if (target && target.id !== d.from) connect(d.from, target.id);
      }
    }
  }
  on(canvas, "pointerup", (e) => endDrag(e, false));
  on(canvas, "pointercancel", (e) => endDrag(e, true));
  on(canvas, "lostpointercapture", (e) => { if (drag) endDrag(e, true); });

  on(canvas, "wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return;          // a plain scroll keeps scrolling the page
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0022), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  // ---------------------------------------------------------------- events: drag from the palette
  on(palBody, "pointerdown", (e) => {
    if (e.button !== 0) return;
    if (e.target.closest("button")) return;
    const item = e.target.closest(".tb-pal-item[data-drag]");
    if (!item) return;
    root.__tbDragged = false;
    const p = { type: item.dataset.type, sx: e.clientX, sy: e.clientY, moved: false, item };
    try { item.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
    const move = (ev) => {
      if (!p.moved && Math.abs(ev.clientX - p.sx) + Math.abs(ev.clientY - p.sy) < 5) return;
      if (!p.moved) {
        p.moved = true;
        root.__tbDragged = true;
        const a = AG[p.type];
        ghost.textContent = "";
        ghost.className = `tb-ghost on tb-c-${a.color}`;
        ghost.append(el("span", "tb-ico", a.icon), el("span", "tb-ghost-name", a.title));
      }
      const rr = root.getBoundingClientRect();
      ghost.style.transform = `translate(${ev.clientX - rr.left + 10}px, ${ev.clientY - rr.top + 8}px)`;
      const c = canvas.getBoundingClientRect();
      canvas.classList.toggle("tb-drop", ev.clientX >= c.left && ev.clientX <= c.right && ev.clientY >= c.top && ev.clientY <= c.bottom);
    };
    const up = (ev, cancelled) => {
      item.removeEventListener("pointermove", move);
      item.removeEventListener("pointerup", upOk);
      item.removeEventListener("pointercancel", upCancel);
      ghost.className = "tb-ghost";
      const over = canvas.classList.contains("tb-drop");
      canvas.classList.remove("tb-drop");
      if (cancelled || !p.moved) { setTimeout(() => { root.__tbDragged = false; }, 0); return; }
      if (over) {
        const q = toCanvas(ev.clientX, ev.clientY);
        addAgent(p.type, Math.round(q.x - RULES.layout.node_w / 2), Math.round(q.y - 30));
      }
      setTimeout(() => { root.__tbDragged = false; }, 0);
    };
    const upOk = (ev) => up(ev, false), upCancel = (ev) => up(ev, true);
    item.addEventListener("pointermove", move, { signal: abort.signal });
    item.addEventListener("pointerup", upOk, { signal: abort.signal });
    item.addEventListener("pointercancel", upCancel, { signal: abort.signal });
    e.preventDefault();
  });

  // ---------------------------------------------------------------- first paint
  mem.metaBefore = null;
  render();
  if (mem.needFit) requestAnimationFrame(() => { if (mem.needFit) fit(); });
  if (root.__tbResize) root.__tbResize.disconnect();
  if (typeof ResizeObserver !== "undefined") {      // the canvas gets its size later (e.g. a hidden tab)
    const ro = new ResizeObserver(() => {      // fitted again when the window makes the canvas another size
      const r = canvas.getBoundingClientRect(), last = root.__tbSize;
      const changed = last && (Math.abs(r.width - last[0]) > 24 || Math.abs(r.height - last[1]) > 24);
      if (r.width && r.height) root.__tbSize = [r.width, r.height];
      if (mem.needFit || changed) fit();
    });
    ro.observe(canvas);
    root.__tbResize = ro;
    abort.signal.addEventListener("abort", () => ro.disconnect());
  }
  return () => abort.abort();
}
