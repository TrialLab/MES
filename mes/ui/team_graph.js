// What the MES team means, in the browser: the compiler (the twin of teams.compile_team) and the few texts the canvas
// cannot work out from the rules. The graph itself is team_canvas/graph.js, which is loaded before this file.
// tests/test_teams.py runs both this and Python and compares them.
const TeamGraph = AgentGraph.create((ctx) => {
  const { nodes, edge, say, connected, R } = ctx;
  const rwdOk = ctx.av.rwd !== false, analystOk = ctx.av.analyst !== false;
  if (!edge("question", "planner")) say("error", "no_question", "planner");
  if (!connected.length) say("error", "no_evidence", "synthesizer");
  for (const t of R.parallel) if (nodes[t] && !connected.includes(t)) say("warning", "not_connected", t);
  if (!edge("synthesizer", "grounding")) say("error", "no_grounding", "grounding");
  const verify = !!nodes.safechecker && edge("grounding", "safechecker") && edge("safechecker", "answer");
  if (!verify && !edge("grounding", "answer")) say("error", "no_answer", "answer");
  if (nodes.safechecker && !verify) say("warning", "safechecker_off", "safechecker");
  const tools = nodes.rwd ? nodes.rwd.tools : { ...R.toggles.rwd };
  const templates = tools.templates !== undefined ? tools.templates : true;
  const analyst = tools.analyst !== undefined ? tools.analyst : true;
  const engine = templates && analyst ? "auto" : templates ? "templates" : analyst ? "analyst" : null;
  if (connected.includes("rwd")) {
    if (engine === null) say("error", "rwd_no_tool", "rwd");
    else if (!rwdOk) say("warning", "rwd_unavailable", "rwd");
    else if (engine !== "templates" && !analystOk) say("info", "analyst_unavailable", "rwd");
  }
  const agents = ["planner", ...connected, "synthesizer", "grounding", ...(verify ? ["safechecker"] : [])];
  return { ok: ctx.ok(), sources: connected, mode: ctx.graph.depth, verify, rwd_engine: engine || "auto",
           use_cache: ctx.graph.use_cache, agents, messages: ctx.messages };
});

// The two texts the canvas cannot work out from the rules: what an agent does at this depth, and the run settings.
const TeamView = {
  chips({ type, node, depth, val, avail, modeLabel, level }) {
    switch (type) {
      case "planner": return [depth === "snow" ? "PICO + facets" : "PICO", "search queries"];
      case "literature": return [`${val("literature_max")} articles`, `${val("fulltext_max")} full texts`];
      case "trials": return [`${val("trials_max")} trials`, "posted results"];
      case "kg": return [`${val("kg_max")} relations`, "2-hop paths"];
      case "rwd": {
        const tl = node ? node.tools : {};
        const tools = [tl.templates ? "templates" : null, tl.analyst ? "analyst" : null].filter(Boolean);
        return [tools.length ? tools.join(" + ") : "no tool", avail.rwd === false ? "store not built" : avail.database || "real-world data"];
      }
      case "synthesizer": return [`${modeLabel(depth)} · ${level(depth)}`, depth === "osler" ? "concise" : "table + certainty"];
      case "grounding": return ["numbers", "citations"];
      case "safechecker": return [`${val("max_claims")} claims`, "LLM check"];
      default: return [];                   // the question and the answer say it in their subtitle
    }
  },

  // [[label, text | {chips: [agent]} | {text, lock}]]
  summary(compiled, graph, data, { avail, modeLabel, level }) {
    const rows = [["Evidence", { chips: compiled.sources }],
                  ["Depth", `${modeLabel(compiled.mode)} · ${level(compiled.mode)}`],
                  ["Claim check", compiled.verify ? "SafeChecker on" : "off"]];
    if (compiled.sources.includes("rwd")) {
      const engine = { auto: "templates, analyst when they do not fit", templates: "study templates only",
                       analyst: "LLM analyst" }[compiled.rwd_engine] || compiled.rwd_engine;
      rows.push(["Real-world data", compiled.rwd_engine !== "templates" && avail.analyst === false
        ? `${engine} (here: study templates)` : engine]);
    }
    rows.push(["Cache", compiled.use_cache ? "reuse recent answers" : "always search again"],
              ["Model", { text: data.model || "", lock: "locked" }]);
    return rows;
  },
};

const Canvas = { graph: TeamGraph, view: TeamView };
