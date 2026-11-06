"""
Light and dark on the MES pages.

The MES theme (``.streamlit/config.toml``) draws its own white background whatever the operating system prefers,
while ``st.context.theme.type`` reports the browser's preference: "dark" on a dark Mac although the page is light. So
the page decides in the browser, from the background Streamlit actually drew: ``dom.theme_probe()`` marks the page
``html.mes-dark`` when it is dark, the dark styles only apply under that mark (``dark_scoped``), and MES's components
read the same background themselves. ``is_dark()`` is the best guess on the Python side, for components that take
the theme from Python (the agent canvas).
"""

import re

import streamlit as st

SCOPE = "html.mes-dark"


def _luminance(color):
    m = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", str(color or "").strip())
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_dark():
    """The page is drawn dark: a configured background decides (a custom theme draws it whatever the system prefers),
    else the theme Streamlit reports."""
    try:
        background = st.get_option("theme.backgroundColor")
    except Exception:
        background = None
    lum = _luminance(background)
    if lum is not None:
        return lum < 0.45
    try:
        return getattr(st.context.theme, "type", None) == "dark"
    except Exception:
        return False


def _split_selectors(text):
    """Selectors of a rule, split at the commas that are not inside parentheses (:has(a, b))."""
    out, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return [s.strip() for s in out if s.strip()]


def _scope_selector(selector, scope):
    """The page's own variables (:root) under the mark itself — so they beat the light :root — and every other rule
    under :where(mark), which adds nothing to its specificity: a dark rule overrides exactly the light rules it did
    before it was scoped (by coming later), never a more specific one."""
    if selector.startswith(":root"):
        return scope + selector[len(":root"):]
    if selector.startswith("html"):
        return scope + selector[len("html"):]
    return f":where({scope}) {selector}"


def dark_scoped(css, scope=SCOPE):
    """``css`` with every rule limited to a page marked dark (``html.mes-dark``); @media blocks are scoped inside,
    @keyframes are left as they are."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out, i, n = [], 0, len(css)
    while i < n:
        brace = css.find("{", i)
        if brace < 0:
            break
        head = css[i:brace].strip()
        depth, j = 1, brace + 1
        while j < n and depth:
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        body = css[brace + 1:j - 1]
        if head.startswith("@media") or head.startswith("@supports"):
            out.append(f"{head} {{{dark_scoped(body, scope)}}}")
        elif head.startswith("@"):
            out.append(f"{head} {{{body}}}")
        elif head:
            out.append(", ".join(_scope_selector(s, scope) for s in _split_selectors(head)) + f" {{{body.strip()}}}")
        i = j
    return "\n".join(out)
