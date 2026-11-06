// The graph of an agent team in the browser: the twin of graph.py (same rules, same results; the tests of
// each domain run both and compare them case by case).
//
// A domain makes its own graph object with the compiler it needs:
//
//     const TeamGraph = AgentGraph.create((ctx) => ({ ok: ctx.ok(), ... }));
//     TeamGraph.configure(rules);
//
// No DOM here: this file is loaded before canvas.js and, for the tests, in Node.
const AgentGraph = (() => {
  const SPACE = /[ \t\n\r\f\v]+/g;
  const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function create(compileFn) {
    let R = null;

    function configure(rules) { R = rules; }

    function text(value, limit) {
      if (typeof value !== "string") return "";
      const one = value.replace(SPACE, " ").replace(/^ +| +$/g, "");
      return Array.from(one).slice(0, limit).join("").replace(/^ +| +$/g, "");
    }

    function coord(value) {
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      const lim = R.limits.coord;
      return Math.floor(Math.min(lim, Math.max(-lim, value)) + 0.5);
    }

    function ends(edge) {
      if (Array.isArray(edge)) return edge.length === 2 ? [edge[0], edge[1]] : [null, null];
      if (edge && typeof edge === "object") return [edge.from, edge.to];
      return [null, null];
    }

    const isType = (t) => typeof t === "string" && has(R.ports, t);

    function compatible(a, b) {
      if (!isType(a) || !isType(b)) return false;
      const out = R.ports[a][1], inp = R.ports[b][0];
      return a !== b && out !== null && inp !== null && out === inp;
    }

    function layout(types) {
      const set = new Set(types);
      const present = R.types.filter((t) => set.has(t));
      const agents = R.parallel.filter((t) => present.includes(t));
      const W = R.layout.node_w, G = R.layout.gap_x, rows = R.layout.rows;
      const collapse = R.layout.collapse || {};
      const width = agents.length * W + Math.max(0, agents.length - 1) * G;
      const pos = {};
      for (const t of present) {
        if (R.parallel.includes(t)) {
          pos[t] = [Math.trunc(-width / 2 + agents.indexOf(t) * (W + G)), rows.parallel];
        } else {
          const row = has(collapse, t) && !present.includes(collapse[t]) ? collapse[t] : t;
          pos[t] = [Math.floor(-W / 2), rows[row]];
        }
      }
      return pos;
    }

    function normalize(raw) {
      const g = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
      const depth = typeof g.depth === "string" && R.modes.includes(g.depth) ? g.depth : R.default_mode;
      const useCache = typeof g.use_cache === "boolean" ? g.use_cache : true;
      const found = new Map();
      const rawNodes = Array.isArray(g.nodes) ? g.nodes.slice(0, R.limits.nodes) : [];
      for (const n of rawNodes) {
        if (!n || typeof n !== "object" || Array.isArray(n)) continue;
        const t = n.type;
        if (!isType(t) || found.has(t)) continue;
        const toolsIn = n.tools && typeof n.tools === "object" && !Array.isArray(n.tools) ? n.tools : {};
        const tools = {};
        for (const [k, def] of Object.entries(R.toggles[t] || {})) {
          tools[k] = typeof toolsIn[k] === "boolean" ? toolsIn[k] : def;
        }
        found.set(t, { x: coord(n.x), y: coord(n.y), tools });
      }
      // a core agent is always there, with its switches at their defaults
      for (const t of R.core) if (!found.has(t)) found.set(t, { x: null, y: null, tools: { ...(R.toggles[t] || {}) } });
      const pos = layout([...found.keys()]);
      const nodes = [];
      for (const t of R.types) {
        if (!found.has(t)) continue;
        let { x, y, tools } = found.get(t);
        if (x === null || y === null) [x, y] = pos[t];
        nodes.push({ id: t, type: t, x, y, tools });
      }
      const edges = [], seen = new Set(), hasInput = new Set();
      const rawEdges = Array.isArray(g.edges) ? g.edges.slice(0, R.limits.edges) : [];
      for (const e of rawEdges) {
        const [a, b] = ends(e);
        if (typeof a !== "string" || typeof b !== "string" || !found.has(a) || !found.has(b)) continue;
        const key = a + "\u0000" + b;
        if (!compatible(a, b) || seen.has(key) || (hasInput.has(b) && !R.multi_input.includes(b))) continue;
        seen.add(key);
        hasInput.add(b);
        edges.push({ from: a, to: b });
      }
      return { version: R.version, name: text(g.name, R.limits.name) || "Untitled team",
               description: text(g.description, R.limits.description), depth, use_cache: useCache, nodes, edges };
    }

    const types = (graph) => new Set(graph.nodes.map((n) => n.type));

    function withEdge(graph, a, b) {
      const edges = graph.edges.filter((e) => !(e.to === b && !R.multi_input.includes(b)) && !(e.from === a && e.to === b));
      edges.push({ from: a, to: b });
      return { ...graph, edges };
    }

    function withoutEdge(graph, a, b) {
      return { ...graph, edges: graph.edges.filter((e) => !(e.from === a && e.to === b)) };
    }

    function connect(graph, a, b) {
      graph = normalize(graph);
      const ts = types(graph);
      if (!ts.has(a) || !ts.has(b)) return [graph, "missing"];
      if (!compatible(a, b)) return [graph, "incompatible"];
      if (graph.edges.some((e) => e.from === a && e.to === b)) return [graph, null];
      return [normalize(withEdge(graph, a, b)), null];
    }

    function disconnect(graph, a, b) {
      graph = normalize(graph);
      return normalize(withoutEdge(graph, a, b));
    }

    function addNode(graph, t, x, y) {
      graph = normalize(graph);
      if (!isType(t) || types(graph).has(t)) return graph;
      if (x === undefined || y === undefined || x === null || y === null) {
        [x, y] = layout([...types(graph), t])[t];
      }
      graph = normalize({ ...graph, nodes: [...graph.nodes, { id: t, type: t, x, y, tools: { ...(R.toggles[t] || {}) } }] });
      if (R.parallel.includes(t)) {
        const [up, down] = R.wire.parallel;
        graph = withEdge(withEdge(graph, up, t), t, down);
      } else if (has(R.insert || {}, t)) {
        const { after, before } = R.insert[t];
        graph = withEdge(withEdge(withoutEdge(graph, after, before), after, t), t, before);
      }
      return normalize(graph);
    }

    function removeNode(graph, t) {
      graph = normalize(graph);
      if (R.core.includes(t) || !types(graph).has(t)) return graph;
      graph = { ...graph, nodes: graph.nodes.filter((n) => n.type !== t),
                edges: graph.edges.filter((e) => e.from !== t && e.to !== t) };
      const spliced = (R.insert || {})[t];
      if (spliced && !graph.edges.some((e) => e.to === spliced.before)) {
        graph = withEdge(graph, spliced.after, spliced.before);
      }
      return normalize(graph);
    }

    function setTool(graph, t, tool, value) {
      graph = normalize(graph);
      const nodes = graph.nodes.map((n) => (n.type === t && has(n.tools, tool) ? { ...n, tools: { ...n.tools, [tool]: !!value } } : n));
      return normalize({ ...graph, nodes });
    }

    function compile(graph, availability) {
      graph = normalize(graph);
      const av = availability && typeof availability === "object" ? availability : {};
      const nodes = {};
      for (const n of graph.nodes) nodes[n.type] = n;
      const edges = new Set(graph.edges.map((e) => e.from + "\u0000" + e.to));
      const edge = (a, b) => edges.has(a + "\u0000" + b);
      const messages = [];
      const say = (level, code, node) => messages.push({
        level, code, node: node === undefined ? null : node,
        text: R.messages[code].split("{agent}").join(node ? R.titles[node] || "" : ""),
      });
      const [up, down] = R.wire.parallel;
      const connected = R.parallel.filter((t) => nodes[t] && edge(up, t) && edge(t, down));
      return compileFn({ R, graph, av, nodes, edges, messages, say, edge, connected,
                         ok: () => !messages.some((m) => m.level === "error") });
    }

    function empty(name, description, depth) {
      let graph = normalize({ name, description, depth });
      for (const [a, b] of R.chain || []) graph = withEdge(graph, a, b);
      return normalize(graph);
    }

    return { configure, normalize, layout, compatible, connect, disconnect, addNode, removeNode, setTool, compile,
             text, empty, rules: () => R };
  }

  return { create };
})();
