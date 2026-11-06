"""
The graph of an agent team, without any knowledge of what the agents do.

A team graph is ``{"version", "name", "description", "depth", "use_cache", "nodes", "edges"}``. Every agent appears at
most once and its node id is its type. An edge joins an output port to an input port of the same kind. A ``Graph`` is
built from the rules of one domain (e.g. MES's agent teams) and offers the operations the canvas needs: ``normalize``,
``layout``, ``connect``, ``disconnect``, ``add_node``, ``remove_node``, ``set_tool``.

The rules describe a pipeline that runs top to bottom with one row of interchangeable agents in the middle:

    types          the node types, in the order they appear in a normalized graph
    core           the types that are always in a team and cannot be removed
    ports          type → [input kind, output kind]; None = no port on that side
    multi_input    the types whose input port takes more than one edge
    toggles        type → {tool: default} — the switches the user really has
    parallel       the types of the middle row (at least one of them is needed, the domain's compiler says so)
    wire           {"parallel": [upstream, downstream]} — what a new agent of the middle row is wired between
    insert         type → {"after", "before"} — an optional agent that is spliced into an existing edge
    chain          the edges between the core agents of an empty team
    modes          the depth levels; default_mode is the one a new graph gets
    limits         {"nodes", "edges", "name", "description", "coord"}
    layout         {"node_w", "gap_x", "rows": {row: y}, "collapse": {type: row of another type}}

The browser runs the same logic (``graph.js``), so both files must stay in step; the tests of each domain compare
them case by case.
"""

import math
import re
from types import SimpleNamespace

_SPACE = re.compile(r"[ \t\n\r\f\v]+")


class Graph:
    """The graph operations of one domain's rules (the rules are also sent to the browser as JSON)."""

    def __init__(self, rules):
        self.rules = rules
        self.version = rules["version"]
        self.types = tuple(rules["types"])
        self.core = tuple(rules["core"])
        self.ports = rules["ports"]
        self.multi_input = tuple(rules["multi_input"])
        self.toggles = rules["toggles"]
        self.parallel = tuple(rules.get("parallel", ()))
        self.wire = rules.get("wire", {})
        self.insert = rules.get("insert", {})
        self.modes = rules["modes"]
        self.default_mode = rules["default_mode"]
        limits = rules["limits"]
        self.max_nodes, self.max_edges = limits["nodes"], limits["edges"]
        self.max_name, self.max_description = limits["name"], limits["description"]
        self.coord_limit = limits["coord"]
        layout = rules["layout"]
        self.node_w, self.gap_x = layout["node_w"], layout["gap_x"]
        self.rows = layout["rows"]
        self.collapse = layout.get("collapse", {})

    # ------------------------------------------------------------------ values
    def text(self, value, limit):
        """One line of text: ASCII whitespace collapsed, at most ``limit`` characters (code points)."""
        if not isinstance(value, str):
            return ""
        return _SPACE.sub(" ", value).strip(" ")[:limit].strip(" ")

    def coord(self, value):
        """A finite number, clamped and rounded like JavaScript's Math.round; None otherwise."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            value = float(value)        # a huge JSON integer: infinite in JavaScript
        except OverflowError:
            return None
        if not math.isfinite(value):
            return None
        return int(math.floor(min(self.coord_limit, max(-self.coord_limit, value)) + 0.5))

    @staticmethod
    def edge_ends(edge):
        if isinstance(edge, dict):
            return edge.get("from"), edge.get("to")
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            return edge[0], edge[1]
        return None, None

    def compatible(self, source, target):
        """An edge from ``source`` to ``target`` joins an output to an input of the same kind."""
        if source not in self.ports or target not in self.ports:
            return False
        out, inp = self.ports[source][1], self.ports[target][0]
        return source != target and out is not None and inp is not None and out == inp

    # ------------------------------------------------------------------ layout
    def layout(self, types):
        """Positions {type: (x, y)} of the default layout for the node types ``types``."""
        present = [t for t in self.types if t in set(types)]
        agents = [t for t in self.parallel if t in present]
        width = len(agents) * self.node_w + max(0, len(agents) - 1) * self.gap_x
        pos = {}
        for t in present:
            if t in self.parallel:
                i = agents.index(t)
                pos[t] = (int(-width / 2 + i * (self.node_w + self.gap_x)), self.rows["parallel"])
            else:
                row = self.collapse.get(t)
                row = row if row is not None and row not in present else t
                pos[t] = (-self.node_w // 2, self.rows[row])
        return pos

    # ------------------------------------------------------------------ graph
    def normalize(self, raw):
        """A well-formed team graph from anything (data from the browser or a file is never trusted)."""
        g = raw if isinstance(raw, dict) else {}
        depth = g.get("depth") if isinstance(g.get("depth"), str) and g.get("depth") in self.modes \
            else self.default_mode
        use_cache = g.get("use_cache") if isinstance(g.get("use_cache"), bool) else True
        found = {}
        raw_nodes = g.get("nodes") if isinstance(g.get("nodes"), list) else []
        for n in raw_nodes[:self.max_nodes]:
            if not isinstance(n, dict):
                continue
            t = n.get("type")
            if not isinstance(t, str) or t not in self.ports or t in found:
                continue
            tools_in = n.get("tools") if isinstance(n.get("tools"), dict) else {}
            tools = {k: (tools_in[k] if isinstance(tools_in.get(k), bool) else default)
                     for k, default in self.toggles.get(t, {}).items()}
            found[t] = {"x": self.coord(n.get("x")), "y": self.coord(n.get("y")), "tools": tools}
        for t in self.core:                     # a core agent is always there, with its switches at their defaults
            found.setdefault(t, {"x": None, "y": None, "tools": dict(self.toggles.get(t, {}))})
        default_pos = self.layout(found)
        nodes = []
        for t in self.types:
            if t in found:
                x, y = found[t]["x"], found[t]["y"]
                if x is None or y is None:
                    x, y = default_pos[t]
                nodes.append({"id": t, "type": t, "x": x, "y": y, "tools": found[t]["tools"]})
        edges, seen, has_input = [], set(), set()
        raw_edges = g.get("edges") if isinstance(g.get("edges"), list) else []
        for e in raw_edges[:self.max_edges]:
            a, b = self.edge_ends(e)
            if not isinstance(a, str) or not isinstance(b, str) or a not in found or b not in found:
                continue
            if not self.compatible(a, b) or (a, b) in seen or (b in has_input and b not in self.multi_input):
                continue
            seen.add((a, b))
            has_input.add(b)
            edges.append({"from": a, "to": b})
        return {"version": self.version, "name": self.text(g.get("name"), self.max_name) or "Untitled team",
                "description": self.text(g.get("description"), self.max_description), "depth": depth,
                "use_cache": use_cache, "nodes": nodes, "edges": edges}

    @staticmethod
    def node_types(graph):
        return {n["type"] for n in graph["nodes"]}

    def with_edge(self, graph, a, b):
        """``graph`` with the edge a → b (an existing input of a single-input port is replaced)."""
        edges = [e for e in graph["edges"] if not (e["to"] == b and b not in self.multi_input)
                 and not (e["from"] == a and e["to"] == b)]
        edges.append({"from": a, "to": b})
        return {**graph, "edges": edges}

    def without_edge(self, graph, a, b):
        return {**graph, "edges": [e for e in graph["edges"] if not (e["from"] == a and e["to"] == b)]}

    def connect(self, graph, a, b):
        """(graph, error code or None): the edge a → b added; an incompatible edge leaves the graph unchanged."""
        graph = self.normalize(graph)
        types = self.node_types(graph)
        if a not in types or b not in types:
            return graph, "missing"
        if not self.compatible(a, b):
            return graph, "incompatible"
        if any(e["from"] == a and e["to"] == b for e in graph["edges"]):
            return graph, None
        return self.normalize(self.with_edge(graph, a, b)), None

    def disconnect(self, graph, a, b):
        graph = self.normalize(graph)
        return self.normalize(self.without_edge(graph, a, b))

    def add_node(self, graph, t, x=None, y=None):
        """``graph`` with agent ``t`` added and wired: an agent of the middle row between the two agents of
        ``rules["wire"]["parallel"]``, an optional agent into the edge of ``rules["insert"]``. A present agent is not
        added twice."""
        graph = self.normalize(graph)
        if t not in self.ports or t in self.node_types(graph):
            return graph
        if x is None or y is None:
            x, y = self.layout(self.node_types(graph) | {t})[t]
        graph = {**graph, "nodes": graph["nodes"] + [{"id": t, "type": t, "x": x, "y": y,
                                                      "tools": dict(self.toggles.get(t, {}))}]}
        graph = self.normalize(graph)
        if t in self.parallel:
            up, down = self.wire["parallel"]
            graph = self.with_edge(self.with_edge(graph, up, t), t, down)
        elif t in self.insert:
            after, before = self.insert[t]["after"], self.insert[t]["before"]
            graph = self.without_edge(graph, after, before)
            graph = self.with_edge(self.with_edge(graph, after, t), t, before)
        return self.normalize(graph)

    def remove_node(self, graph, t):
        """``graph`` without agent ``t`` (core agents stay); an optional agent's edge is joined up again."""
        graph = self.normalize(graph)
        if t in self.core or t not in self.node_types(graph):
            return graph
        graph = {**graph, "nodes": [n for n in graph["nodes"] if n["type"] != t],
                 "edges": [e for e in graph["edges"] if t not in (e["from"], e["to"])]}
        spliced = self.insert.get(t)
        if spliced and not any(e["to"] == spliced["before"] for e in graph["edges"]):
            graph = self.with_edge(graph, spliced["after"], spliced["before"])
        return self.normalize(graph)

    def set_tool(self, graph, t, tool, value):
        graph = self.normalize(graph)
        nodes = [{**n, "tools": {**n["tools"], tool: bool(value)}} if n["type"] == t and tool in n["tools"] else n
                 for n in graph["nodes"]]
        return self.normalize({**graph, "nodes": nodes})

    # ------------------------------------------------------------------ for the domain's compiler
    def context(self, graph, availability=None):
        """What every compiler starts from: the normalized graph, its nodes and edges, and a message list.

        ``ctx.edge(a, b)`` tells whether an edge is there, ``ctx.say(level, code, node)`` adds a message (its text
        comes from ``rules["messages"]``, ``{agent}`` filled from ``rules["titles"]``) and ``ctx.connected`` lists the
        agents of the middle row that are wired between the two agents of ``rules["wire"]["parallel"]``.
        """
        graph = self.normalize(graph)
        av = availability if isinstance(availability, dict) else {}
        nodes = {n["type"]: n for n in graph["nodes"]}
        edges = {(e["from"], e["to"]) for e in graph["edges"]}
        messages = []
        titles, texts = self.rules["titles"], self.rules["messages"]

        def say(level, code, node=None):
            messages.append({"level": level, "code": code, "node": node,
                             "text": texts[code].replace("{agent}", titles.get(node, ""))})

        up, down = self.wire["parallel"]
        connected = [t for t in self.parallel if t in nodes and (up, t) in edges and (t, down) in edges]
        return SimpleNamespace(graph=graph, av=av, nodes=nodes, edges=edges, messages=messages, say=say,
                               connected=connected, edge=lambda a, b: (a, b) in edges,
                               ok=lambda: not any(m["level"] == "error" for m in messages))

    # ------------------------------------------------------------------ graphs from nothing
    def empty_graph(self, name="Untitled team", description="", depth=None):
        """The core agents, wired along ``rules["chain"]``, without any agent of the middle row."""
        graph = self.normalize({"name": name, "description": description, "depth": depth})
        for a, b in self.rules.get("chain", ()):
            graph = self.with_edge(graph, a, b)
        return self.normalize(graph)

    def build_graph(self, name, description, agents, depth, extras=()):
        """A graph of the core agents plus ``agents`` and ``extras``, each wired and placed the default way."""
        graph = self.empty_graph(name, description, depth)
        for t in list(agents) + list(extras):
            graph = self.add_node(graph, t)
        return self.normalize({**graph, "nodes": [{**n, "x": None, "y": None} for n in graph["nodes"]]})
