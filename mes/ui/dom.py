"""Small page helpers as script-only Streamlit components (st.components.v2: no iframe, runs in the page)."""

import time

import streamlit as st

_SCROLL_JS = """
export default function ({ data }) {
  const el = document.getElementById(data.anchor);
  if (el) el.scrollIntoView({ block: "start", behavior: data.smooth ? "smooth" : "auto" });
}
"""
_scroll = st.components.v2.component("mes_scroll", js=_SCROLL_JS)


def scroll_to(anchor, smooth=False, key=None):
    """Bring the element with id ``anchor`` to the top of the page (Streamlit keeps the old scroll position); ``anchor``
    "" scrolls nowhere (an element kept in its place, see page._render_result)."""
    _scroll(key=key or f"mes_scroll_{anchor}", data={"anchor": anchor, "smooth": smooth, "t": time.time()}, height=0)


_KICK_JS = """
const fired = (globalThis.__mesKick = globalThis.__mesKick || {});
export default function ({ data, setTriggerValue }) {
  if (fired[data.id]) return;
  fired[data.id] = true;
  setTimeout(() => setTriggerValue("go", data.id), 30);
}
"""
_kick = st.components.v2.component("mes_kick", js=_KICK_JS)


def _noop():
    pass


def rerun_when_shown(run_id):
    """Rerun the script once, right after this run has been drawn in the browser (once per ``run_id``)."""
    _kick(key=f"mes_kick_{run_id}", data={"id": run_id}, on_go_change=_noop, height=0)


_THEME_JS = """
// Marks the page html.mes-dark when Streamlit drew it dark (the dark styles of MES apply only then): the background
// that is actually on screen decides, not the operating system's preference. Checked again every second (the theme
// can be switched in Streamlit's menu without a rerun of this component).
export default function () {
  const check = () => {
    const app = document.querySelector(".stApp") || document.body;
    const m = (getComputedStyle(app).backgroundColor || "").match(/[\\d.]+/g);
    if (!m || m.length < 3) return;
    const [r, g, b] = m.map(Number);
    const dark = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.45;
    document.documentElement.classList.toggle("mes-dark", dark);
  };
  check();
  if (!globalThis.__mesThemeTimer) globalThis.__mesThemeTimer = setInterval(check, 1000);
}
"""
_theme = st.components.v2.component("mes_theme", js=_THEME_JS)


def theme_probe():
    """Mark the page dark or light as drawn (see ui/theme.py)."""
    _theme(key="mes_theme_probe", height=0)
