"""
The agent team canvas: the drag-and-drop builder of agent teams (used by MES's team builder).

``graph.py`` / ``graph.js`` hold the graph itself (normalize, wire, lay out, operate), ``canvas.js`` / ``canvas.css``
the builder's interface and ``component.py`` registers it with Streamlit. Nothing here knows what the agents do: each
use case supplies its rules, its catalogue of agents and its compiler (``mes/ui/teams.py``).
"""

from mes.ui.team_canvas.graph import Graph

__all__ = ["Graph"]
