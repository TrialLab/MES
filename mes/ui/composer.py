"""
MES question box: an auto-growing input, one or two labelled dropdowns saying what will answer the question, and a
send button, as one Streamlit custom component (st.components.v2), in the WCM palette (Dark Orange accent, neutral
ramp).

    res = composer("mes_composer", mode="osler", modes=[...], placeholder="Ask a medical question…")
    res["submit"]  # {"question", "mode", "depth", "at"} in the run triggered by sending, else None
    res["edited"]  # {"question", "at"} the first keystroke that leaves `routed` behind, else None
    res["action"]  # {"key", "at"} when a menu item of kind "action" was clicked, else None
    res["depth"]   # {"key", "at"} when the depth dropdown was changed, else None

The first dropdown (``mode`` / ``modes``) says *what* answers the question — the MES depths on the MES page,
or e.g. a task picker in an app that embeds the box. The second (``depth`` / ``depths``) is optional and says *how much effort*
it is worth; it is drawn only when ``depths`` is given. Both are labelled with a prefix when ``label`` /
``depth_label`` is set ("Task: Auto", "Depth: Fast"), and the prefixes drop together when the box is narrow.

With ``default_question`` the empty box shows that question in grey and can be sent as it is (sending an empty box
asks it). Enter sends (Shift+Enter: new line; ignored while an input method is composing, e.g. Chinese), Esc closes
an open menu, arrow keys move in it. The draft and both choices live in the browser (choosing a ``mode`` does not
rerun the page) and are sent with the question; a ``mode`` changed from Python (e.g. "Go deeper") replaces the
chosen one.

A menu item is ``{"key", "label", "desc", "time"}`` plus, optionally, ``kind``: "mode" (the default — it can be
chosen), "group" (a heading, which needs only ``label``) or "action" (clicking it reports ``res["action"]`` and
chooses nothing, e.g. "+ New custom team").

A dropdown whose items all carry a ``level`` (0, 1, 2 … — the depths) is an effort scale and is animated: its button
shows that many bars, lit up to the chosen level, which grow when the level changes; its menu ends in a meter whose
peak moves to the level under the pointer (FAST · BALANCED · DEEP, from the items' labels) and rises when it opens.

``routed`` is the question the page's current answer plan was made from: the first keystroke that leaves it
behind reports ``res["edited"]`` once — that is the whole write-back, so typing never reruns the page. A change
of depth reports ``res["depth"]`` instead: the effort is a knob on the plan, not a different plan.
"""

import uuid

import streamlit as st

from mes.ui.theme import is_dark               # noqa: F401  (re-exported: the MES pages read composer.is_dark)

HTML = """
<div class="mesc">
  <div class="mesc-box">
    <textarea rows="1" aria-label="Question" maxlength="4000"></textarea>
    <div class="mesc-actions">
      <div class="mesc-pickers"></div>
      <button type="button" class="mesc-send" aria-label="Send">
        <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 4.5 13 10l-5.5 5.5"/></svg>
      </button>
    </div>
  </div>
</div>
"""

CSS = """
/* WCM: Dark Orange is the interactive accent (Cornell Red is for brand marks only), the rest is the neutral ramp */
:host { --accent: #CF4520; --accent-dark: #A8381A; --accent-soft: #F3D5C9; --ink: #1A1A1A; --muted: #555555;
  --faint: #999999; --line: #DDDDDD; --hover: #F2F2F2; }
.mesc { position: relative; font-family: inherit; color: var(--ink); }
.mesc-box { display: flex; align-items: center; gap: 0.6rem; background: #fff; border: 1px solid var(--line);
  border-radius: 18px; padding: 12px 12px 12px 22px; box-shadow: 0 6px 24px rgba(26, 26, 26, 0.07), 0 1px 2px rgba(26, 26, 26, 0.04);
  transition: border-color .15s ease, box-shadow .15s ease; }
.mesc-box:focus-within { border-color: #CCCCCC; box-shadow: 0 8px 28px rgba(26, 26, 26, 0.10), 0 0 0 3px rgba(207, 69, 32, 0.10); }
textarea { flex: 1; border: 0; outline: none; resize: none; background: transparent; font: inherit; font-size: 1.12rem;
  line-height: 1.5; color: var(--ink); padding: 6px 0; min-height: 30px; max-height: 220px; overflow-y: auto; }
textarea::placeholder { color: var(--faint); }
.mesc-actions { display: flex; align-items: center; gap: 0.45rem; flex-shrink: 0; }
/* two controls, not one: the gap is what tells them apart */
.mesc-pickers { display: flex; align-items: center; gap: 0.5rem; }
.mesc-picker { position: relative; }
.mesc-mode { display: inline-flex; align-items: center; gap: 0.3rem; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 1.02rem; font-weight: 500; color: var(--ink); padding: 8px 10px; border-radius: 10px;
  white-space: nowrap; }
.mesc-mode:hover, .mesc-mode[aria-expanded="true"] { background: var(--hover); }
.mesc-prefix { color: var(--muted); font-weight: 500; }
.mesc-mode:focus-visible, .mesc-send:focus-visible, .mesc-item:focus-visible { outline: 2px solid rgba(207, 69, 32, .45); outline-offset: 2px; }
/* the level bars of an effort scale, on its button: lit up to the chosen level, grown again when it changes */
.mesc-lvl { display: inline-flex; align-items: flex-end; gap: 2px; height: 14px; margin: 0 3px 0 1px; }
.mesc-lvl i { width: 3px; border-radius: 1.5px; background: var(--line); transform-origin: bottom;
  transition: background-color .25s ease calc(var(--i) * 70ms); }
.mesc-lvl i.on { background: var(--accent); }
.mesc-lvl.bump i.on { animation: mesc-grow .45s cubic-bezier(.2, .8, .2, 1) both;
  animation-delay: calc(var(--i) * 70ms + var(--t, 0ms)); }
@keyframes mesc-grow { from { transform: scaleY(.25); opacity: .35; } }
.mesc-chevron { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round;
  stroke-linejoin: round; transform: rotate(180deg); transition: transform .2s ease; }
.mesc-picker.open .mesc-chevron { transform: rotate(0deg); }
/* narrow box: both prefixes go at once, and stay in the accessibility tree */
.mesc-tight .mesc-prefix { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%);
  white-space: nowrap; }
.mesc-send { width: 46px; height: 46px; border-radius: 12px; border: 0; cursor: pointer; display: grid; place-items: center;
  background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dark) 100%); color: #fff;
  box-shadow: 0 2px 6px rgba(168, 56, 26, .28); transition: transform .12s ease, opacity .15s ease, box-shadow .15s ease; }
.mesc-send svg { width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
.mesc-send:hover:not(:disabled) { transform: translateX(1px); box-shadow: 0 4px 12px rgba(168, 56, 26, .35); }
.mesc-send:active:not(:disabled) { transform: scale(.96); }
.mesc-send:disabled { opacity: .38; cursor: default; box-shadow: none; }

.mesc-menu { position: absolute; right: 0; top: calc(100% + 10px); width: var(--menu-w, 460px);
  max-width: min(88vw, 520px); background: #fff; border: 1px solid var(--line); border-radius: 14px;
  box-shadow: 0 18px 40px rgba(26, 26, 26, .14), 0 2px 6px rgba(26, 26, 26, .06);
  padding: 8px; z-index: 1000; opacity: 0; visibility: hidden; transform: translateY(-6px) scale(.985);
  transform-origin: top right; transition: opacity .16s ease, transform .18s cubic-bezier(.2, .8, .2, 1), visibility 0s linear .18s; }
.mesc-picker.up .mesc-menu { top: auto; bottom: calc(100% + 10px); transform-origin: bottom right; transform: translateY(6px) scale(.985); }
.mesc-picker.open .mesc-menu { opacity: 1; visibility: visible; transform: none; transition-delay: 0s; }
.mesc-picker[data-picker="depth"] .mesc-menu { --menu-w: 300px; }
.mesc-item { display: flex; width: 100%; align-items: flex-start; justify-content: space-between; gap: 12px; text-align: left;
  border: 0; background: transparent; cursor: pointer; font: inherit; color: var(--ink); padding: 11px 14px; border-radius: 10px;
  transition: background .12s ease; }
.mesc-item.active { background: #F7F7F7; }
.mesc-item-text { display: flex; flex-direction: column; gap: 3px; }
.mesc-item-name { font-size: 1.05rem; font-weight: 500; }
.mesc-item.selected .mesc-item-name { font-weight: 650; }
.mesc-item-desc { font-size: .9rem; color: var(--muted); }
.mesc-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 9px; background: transparent; flex-shrink: 0;
  transition: background .2s ease, transform .2s ease; transform: scale(.4); }
.mesc-item.selected .mesc-dot { background: var(--accent); transform: scale(1); }

/* A heading over a run of items, and an item that does something instead of being chosen */
.mesc-group { padding: 12px 14px 4px; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--faint); font-weight: 600; }
.mesc-items > .mesc-group:first-child { padding-top: 4px; }
.mesc-action { border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; margin-top: 4px; }
.mesc-action .mesc-item-name { color: var(--accent); }

/* the meter at the foot of an effort scale's menu: a peak at the level under the pointer, FAST at the left */
.mesc-meter { display: flex; align-items: flex-end; gap: 3px; height: 24px; margin: 12px 14px 6px; }
.mesc-meter i { flex: 1; height: 3px; border-radius: 2px; background: var(--accent-soft);
  transition: height .38s cubic-bezier(.2, .8, .2, 1) calc(var(--i) * 14ms), opacity .38s ease calc(var(--i) * 14ms),
              background-color .3s ease calc(var(--i) * 14ms); }
.mesc-meter i.on { background: var(--accent); }
.mesc-meter.intro i.on { animation: mesc-rise .55s cubic-bezier(.2, .8, .2, 1) both; animation-delay: calc(var(--i) * 22ms); }
@keyframes mesc-rise { from { height: 3px; opacity: .25; } }
.mesc-scale { display: flex; align-items: center; gap: 10px; margin: 6px 14px 6px; font-size: .74rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); font-weight: 500; }
.mesc-scale i { flex: 1; height: 1px; background: var(--line); }
.mesc-scale span { transition: color .25s ease; }
.mesc-scale span.on { color: var(--accent); font-weight: 650; }

.mesc-dark { --accent: #E8784A; --accent-dark: #CF4520; --accent-soft: #3D2A22; --ink: #F3F4F6; --muted: #9CA3AF;
  --faint: #6B7280; --line: #30343E; --hover: #232733; }
.mesc-dark .mesc-box { background: #161922; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); }
.mesc-dark .mesc-box:focus-within { border-color: #4B5563; box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), 0 0 0 3px rgba(232, 120, 74, 0.14); }
.mesc-dark .mesc-menu { background: #161922; box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55); }
.mesc-dark .mesc-item.active { background: #232733; }
.mesc-dark .mesc-scale span.on { color: #F09A73; }
/* the MES page's own boxes ("studio"): a warm gradient rim that brightens while the box has the focus */
.mesc-studio .mesc-box { border: 1px solid transparent;
  background: linear-gradient(#fff, #fff) padding-box,
              linear-gradient(115deg, var(--accent-soft) 0%, var(--line) 35%, var(--line) 65%, var(--accent-soft) 100%) border-box; }
.mesc-studio .mesc-box:focus-within { border-color: transparent;
  background: linear-gradient(#fff, #fff) padding-box,
              linear-gradient(115deg, var(--accent) 0%, var(--accent-soft) 45%, var(--line) 70%, var(--accent) 100%) border-box;
  box-shadow: 0 10px 34px rgba(207, 69, 32, 0.12), 0 0 0 4px rgba(207, 69, 32, 0.07); }
.mesc-dark.mesc-studio .mesc-box { background: linear-gradient(#161922, #161922) padding-box,
  linear-gradient(115deg, var(--accent-soft) 0%, var(--line) 35%, var(--line) 65%, var(--accent-soft) 100%) border-box; }
.mesc-dark.mesc-studio .mesc-box:focus-within { background: linear-gradient(#161922, #161922) padding-box,
  linear-gradient(115deg, var(--accent) 0%, var(--accent-soft) 45%, var(--line) 70%, var(--accent) 100%) border-box; }
.mesc-compact .mesc-box { border-radius: 14px; padding: 8px 8px 8px 16px; box-shadow: 0 2px 10px rgba(26, 26, 26, .05); }
.mesc-compact textarea { font-size: 1rem; min-height: 26px; }
.mesc-compact .mesc-send { width: 38px; height: 38px; border-radius: 10px; }
.mesc-compact .mesc-mode { font-size: .95rem; padding: 6px 8px; }
/* the menu hangs from its own button, so on a narrow screen it is capped rather than stretched */
@media (max-width: 640px) { .mesc-menu { --menu-w: 86vw; max-width: 86vw; } textarea { font-size: 1rem; } }
@media (prefers-reduced-motion: reduce) {
  .mesc-menu, .mesc-chevron, .mesc-meter i, .mesc-lvl i { transition: none !important; animation: none !important; } }
"""

JS = """
const CHEVRON = '<svg class="mesc-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 7.5 6 4.5 9 7.5"/></svg>';
const TIGHT = 560;                       // box width below which both label prefixes go
// The meter of an effort scale's menu: a peak that moves with the level, the lowest at the left
const BARS = 24;
const EDGE = 1.5;                        // bar index of the peak at either end of the scale
const SPREAD = 3.4;                      // width of the peak (bars)
const BUMP_MS = 700;                     // how long the level bars of a button take to grow (the last one included)
const memory = (globalThis.__mesComposer = globalThis.__mesComposer || {});

export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const root = parentElement.querySelector(".mesc");
  if (!root) return;
  const box = root.querySelector(".mesc-box");
  const input = root.querySelector("textarea");
  const rack = root.querySelector(".mesc-pickers");
  const send = root.querySelector(".mesc-send");
  const mem = (memory[data.key] = memory[data.key] || { draft: "", nonce: null, focused: false });
  // Streamlit calls this function again on every rerun without running the previous cleanup: remove the
  // listeners of the previous call here, otherwise one Enter sends twice (the second time from an empty box)
  if (root.__mescAbort) root.__mescAbort.abort();
  const abort = new AbortController();
  root.__mescAbort = abort;
  const on = (el, type, fn, opts) => el.addEventListener(type, fn, { signal: abort.signal, ...(opts || {}) });

  root.classList.toggle("mesc-compact", !!data.compact);
  root.classList.toggle("mesc-studio", data.variant === "studio");
  // dark when the background Streamlit drew is dark (a custom light theme stays light on a dark system)
  const bg = getComputedStyle(root).getPropertyValue("--st-background-color").trim();
  const hex = /^#?([0-9a-f]{6})$/i.exec(bg);
  const rgb = hex ? [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16)) : (bg.match(/[\\d.]+/g) || []).map(Number);
  root.classList.toggle("mesc-dark", rgb.length >= 3 ? (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255 < 0.45
                                                     : !!data.dark);
  input.placeholder = data.default_question || data.placeholder || "";
  if (data.nonce && data.nonce !== mem.nonce) {        // a value set from Python (e.g. an example question)
    mem.nonce = data.nonce;
    input.value = mem.draft = data.value || "";
    setTimeout(() => { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }, 30);
  } else if (!input.value && mem.draft) {
    input.value = mem.draft;                           // the draft survives a page change: it lives here
  }

  // ------------------------------------------------------------------------------- the dropdowns
  // One picker per labelled menu. Each owns its button, its list and its own chosen key in `mem`; the
  // only thing they share is that opening one closes the other.
  const pickers = [];

  function makePicker(spec) {
    const items = spec.items;
    const byKey = new Map(items.filter((it) => it.key).map((it) => [it.key, it]));
    const kindOf = (key) => (byKey.get(key) || {}).kind || "mode";
    const choosable = items.filter((it) => it.key && (it.kind || "mode") === "mode");
    const walkable = items.filter((it) => it.key && it.kind !== "group");   // what the arrows move over
    const known = (key) => choosable.some((m) => m.key === key);
    const sent = `${spec.name}FromPython`;
    const before = mem[spec.name];
    if (data[spec.name] !== mem[sent] || !known(mem[spec.name])) {   // first render, or changed from Python
      mem[sent] = data[spec.name];
      mem[spec.name] = known(data[spec.name]) ? data[spec.name] : (choosable[0] || {}).key;
    }
    let active = mem[spec.name];
    // An effort scale: every item says its level. Its button gets the bars and its menu the meter.
    const levels = choosable.length > 1 && choosable.every((m) => Number.isInteger(m.level))
      ? Math.max(...choosable.map((m) => m.level)) + 1 : 0;
    const levelOf = (key) => (byKey.get(key) || {}).level ?? 0;

    const wrap = document.createElement("div");
    wrap.className = "mesc-picker";
    wrap.dataset.picker = spec.name;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mesc-mode";
    button.setAttribute("aria-haspopup", "listbox");
    button.setAttribute("aria-expanded", "false");
    if (spec.label) {
      const prefix = document.createElement("span");
      prefix.className = "mesc-prefix";
      prefix.textContent = `${spec.label}:`;
      button.append(prefix);
    }
    let lvl = null;
    if (levels) {
      lvl = document.createElement("span");
      lvl.className = "mesc-lvl";
      lvl.setAttribute("aria-hidden", "true");
      for (let i = 0; i < levels; i++) {
        const bar = document.createElement("i");
        bar.style.setProperty("--i", i);
        bar.style.height = `${Math.round(5 + 9 * (i + 1) / levels)}px`;
        lvl.append(bar);
      }
      button.append(lvl);
    }
    const valueLabel = document.createElement("span");
    valueLabel.className = "mesc-mode-label";
    button.append(valueLabel);
    button.insertAdjacentHTML("beforeend", CHEVRON);
    const menu = document.createElement("div");
    menu.className = "mesc-menu";
    menu.setAttribute("role", "listbox");
    if (spec.label) menu.setAttribute("aria-label", spec.label);
    const list = document.createElement("div");
    list.className = "mesc-items";
    menu.append(list);
    let meter = null;
    const scale = [];
    if (levels) {
      meter = document.createElement("div");
      meter.className = "mesc-meter";
      meter.setAttribute("aria-hidden", "true");
      for (let i = 0; i < BARS; i++) {
        const bar = document.createElement("i");
        bar.style.setProperty("--i", i);
        meter.append(bar);
      }
      const ruler = document.createElement("div");
      ruler.className = "mesc-scale";
      ruler.setAttribute("aria-hidden", "true");
      [...choosable].sort((a, b) => a.level - b.level).forEach((m, j) => {
        if (j) ruler.append(document.createElement("i"));
        const mark = document.createElement("span");
        mark.dataset.level = m.level;
        mark.textContent = m.label;
        ruler.append(mark);
        scale.push(mark);
      });
      menu.append(meter, ruler);
    }
    wrap.append(button, menu);
    rack.append(wrap);

    for (const m of items) {
      if (m.kind === "group") {
        const heading = document.createElement("div");
        heading.className = "mesc-group";
        heading.setAttribute("role", "presentation");
        heading.textContent = m.label;
        list.appendChild(heading);
        continue;
      }
      const action = m.kind === "action";
      const item = document.createElement("button");
      item.type = "button";
      item.className = action ? "mesc-item mesc-action" : "mesc-item";
      item.dataset.mode = m.key;
      item.setAttribute("role", action ? "button" : "option");
      const text = document.createElement("span");
      text.className = "mesc-item-text";
      const name = document.createElement("span");
      name.className = "mesc-item-name";
      name.textContent = m.label;
      const desc = document.createElement("span");
      desc.className = "mesc-item-desc";
      desc.textContent = [m.desc, m.time].filter(Boolean).join(" · ");
      text.append(name, desc);
      item.append(text);
      if (!action) {
        const dot = document.createElement("span");
        dot.className = "mesc-dot";
        item.append(dot);
      }
      list.appendChild(item);
    }

    function paint(level) {
      const peak = EDGE + (levels > 1 ? level * (BARS - 1 - 2 * EDGE) / (levels - 1) : 0);
      [...meter.children].forEach((bar, i) => {
        const g = Math.exp(-((i - peak) ** 2) / (2 * SPREAD * SPREAD));    // 1 at the peak, → 0 away from it
        const isOn = g > 0.08;
        bar.classList.toggle("on", isOn);
        bar.style.height = isOn ? `${Math.round(3 + 19 * g)}px` : "3px";
        bar.style.opacity = isOn ? (0.3 + 0.7 * g).toFixed(2) : "1";
      });
      scale.forEach((mark) => mark.classList.toggle("on", Number(mark.dataset.level) === level));
    }

    // The chosen level changed: the lit bars grow again. The pickers are rebuilt on every rerun, and a depth
    // change reruns the page at once, so the start is kept in `mem` and a rebuilt button carries on from there.
    const bumpKey = `${spec.name}BumpAt`;
    function bump(startedAt) {
      const elapsed = Math.round(performance.now() - startedAt);
      if (elapsed > BUMP_MS) return;
      lvl.style.setProperty("--t", `${-elapsed}ms`);
      lvl.classList.remove("bump");
      void lvl.offsetWidth;                  // restart the animation
      lvl.classList.add("bump");
    }

    function render() {
      const chosen = mem[spec.name];
      valueLabel.textContent = (choosable.find((m) => m.key === chosen) || choosable[0] || { label: "" }).label;
      list.querySelectorAll(".mesc-item").forEach((item) => {
        item.classList.toggle("selected", item.dataset.mode === chosen);
        item.classList.toggle("active", item.dataset.mode === active);
        item.setAttribute("aria-selected", String(item.dataset.mode === chosen));
      });
      if (levels) {
        [...lvl.children].forEach((bar, i) => bar.classList.toggle("on", i <= levelOf(chosen)));
        paint(levelOf(known(active) ? active : chosen));
      }
    }

    function setOpen(show) {
      if (show) {
        pickers.forEach((p) => p !== picker && p.setOpen(false));
        const r = box.getBoundingClientRect();
        const below = window.innerHeight - r.bottom;
        wrap.classList.toggle("up", below < 330 && r.top > below);
        active = mem[spec.name];
        render();
        if (meter) {
          meter.classList.remove("intro");
          void meter.offsetWidth;            // restart the rise animation
          meter.classList.add("intro");
        }
      }
      wrap.classList.toggle("open", show);
      button.setAttribute("aria-expanded", String(show));
    }

    function choose(key) {
      if (kindOf(key) === "action") {       // an action does something; it is never the chosen item
        setOpen(false);
        active = mem[spec.name];
        render();
        setTriggerValue("action", { key, at: Date.now() });
        return;
      }
      const changed = key !== mem[spec.name];
      mem[spec.name] = active = key;
      setOpen(false);
      render();
      renderSend();
      if (changed && levels) bump((mem[bumpKey] = performance.now()));
      if (changed) spec.onChange(key);
    }

    on(button, "click", (e) => { e.stopPropagation(); setOpen(!wrap.classList.contains("open")); });
    on(list, "click", (e) => {
      const item = e.target.closest(".mesc-item");
      if (item) { choose(item.dataset.mode); input.focus(); }
    });
    on(list, "pointerover", (e) => {
      const item = e.target.closest(".mesc-item");
      if (item && item.dataset.mode !== active) { active = item.dataset.mode; render(); }
    });
    on(menu, "pointerleave", () => { if (active !== mem[spec.name]) { active = mem[spec.name]; render(); } });
    on(wrap, "keydown", (e) => {
      if (!wrap.classList.contains("open")) return;
      const keys = walkable.map((m) => m.key);
      const i = Math.max(0, keys.indexOf(active));
      if (e.key === "ArrowDown") { active = keys[(i + 1) % keys.length]; render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active = keys[(i - 1 + keys.length) % keys.length]; render(); e.preventDefault(); }
      else if (e.key === "Enter" || e.key === " ") { choose(active); e.preventDefault(); }
      else if (e.key === "Escape") { setOpen(false); button.focus(); e.preventDefault(); }
    });

    const picker = { name: spec.name, wrap, setOpen, render, choose,
                     chosen: () => mem[spec.name],
                     isOpen: () => wrap.classList.contains("open"),
                     active: () => active };
    pickers.push(picker);
    if (levels) {
      if (before !== undefined && before !== mem[spec.name]) mem[bumpKey] = performance.now();   // set from Python
      if (mem[bumpKey]) bump(mem[bumpKey]);
    }
    return picker;
  }

  // The write-back: the first keystroke that leaves the routed text behind says so, once. Python clears
  // the route and disables its button; typing on after that reruns nothing.
  function reportEdit(extra) {
    if (typeof data.routed !== "string") return;
    if (!extra && (mem.editedFor === data.routed || input.value === data.routed)) return;
    mem.editedFor = data.routed;
    setTriggerValue("edited", { question: input.value, mode: mem.mode, at: Date.now(), ...(extra || {}) });
  }

  rack.textContent = "";
  makePicker({
    name: "mode", items: data.modes || [], label: data.label,
    // Choosing from this menu does not rerun the page — except when a route is on screen, which the
    // choice has just made wrong: then it goes back the same way an edit does.
    onChange: (key) => { if (typeof data.routed === "string") reportEdit({ mode: key }); },
  });
  const depthPicker = (data.depths || []).length
    // A depth is a knob on the plan, not a different plan: Python wants it (the time estimate changes
    // with it) but must not treat it as an edit, so it comes back on its own channel.
    ? makePicker({ name: "depth", items: data.depths, label: data.depth_label,
                   onChange: (key) => setTriggerValue("depth", { key, at: Date.now() }) })
    : null;

  // ------------------------------------------------------------------------------- the box itself
  function renderSend() {
    send.disabled = (!input.value.trim() && !data.default_question) || !!data.disabled;
    send.title = !input.value.trim() && data.default_question ? "Ask this example question (Enter)" : (data.send_title || "Send (Enter)");
  }

  function grow() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, data.compact ? 140 : 220)}px`;
  }

  function submit() {
    // a second Enter right after sending (double press) would send the grey default question of the emptied box
    if (Date.now() - (mem.sentAt || 0) < 2500) return;
    const question = input.value.trim() || (data.default_question || "").trim();
    if (!question || data.disabled) return;
    mem.sentAt = Date.now();
    setTriggerValue("submit", { question, mode: mem.mode, depth: depthPicker ? depthPicker.chosen() : null,
                                at: Date.now() });
    if (!data.keep) {                 // a box with `keep` keeps the question (e.g. to show a route next to it)
      mem.draft = "";
      input.value = "";
    } else {
      input.value = mem.draft = question;
    }
    grow();
    renderSend();
  }

  on(input, "input", () => { mem.draft = input.value; reportEdit(null); grow(); renderSend(); });
  on(input, "keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      const open = pickers.find((p) => p.isOpen());
      if (open) { open.choose(open.active()); return; }
      submit();
    }
  });
  on(send, "click", submit);
  on(window, "pointerdown", (e) => {
    const path = e.composedPath();
    pickers.forEach((p) => { if (p.isOpen() && !path.includes(p.wrap)) p.setOpen(false); });
  });
  // "/" anywhere on the page (outside a text field) puts the cursor in the question box
  on(window, "keydown", (e) => {
    const el = e.composedPath()[0];
    const typing = el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName || ""));
    if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey && !e.altKey && input.isConnected) {
      e.preventDefault();
      input.focus();
    }
  });

  // Both prefixes go together when the box runs out of room, never just one
  const fit = () => root.classList.toggle("mesc-tight", box.clientWidth > 0 && box.clientWidth < TIGHT);
  const observer = new ResizeObserver(fit);
  observer.observe(box);
  fit();

  pickers.forEach((p) => p.render());
  grow();
  renderSend();
  if (data.autofocus && !mem.focused) {
    mem.focused = true;
    setTimeout(() => input.focus(), 60);
  }
  return () => { observer.disconnect(); abort.abort(); };
}
"""

_component = st.components.v2.component("mes_composer", html=HTML, css=CSS, js=JS)


def _noop():
    pass


def composer(key, mode, modes, placeholder, value=None, nonce=None, disabled=False, autofocus=False,
             compact=False, send_title="Send (Enter)", default_question=None, routed=None, keep=False,
             label=None, depth=None, depths=(), depth_label=None, variant=None):
    """
    Render the question box.

    ``modes``   the first dropdown: [{"key", "label", "desc", "time"}], plus optionally ``kind`` ("mode", the
                default, is chosen and stays; "group" is a heading; "action" reports and chooses nothing).
    ``label``   a prefix on that dropdown, e.g. "Task" → "Task: Auto". None draws the value alone.
    ``depths``  the second dropdown, same item shape; empty (the default) draws no second dropdown.
    ``value``   with a new ``nonce``, replaces the text in the box (e.g. an example question).
    ``routed``  the text a route was computed for; the first keystroke that leaves it behind reports
                ``edited`` once, as does choosing another item from the first dropdown while it is set. That
                is the whole write-back: without ``routed``, neither typing nor a menu choice reruns the page.
    ``keep``    keep the question in the box after sending (e.g. to show a route beside it).
    ``variant`` "studio": the MES page's own look (a warm gradient rim); None: the plain look.

    Returns {"submit": {"question", "mode", "depth", "at"}, "edited": {"question", "at"},
    "action": {"key", "at"}, "depth": {"key", "at"}}, each set only in the run its trigger fired in.
    """
    result = _component(key=key, data={"key": key, "mode": mode, "modes": modes, "placeholder": placeholder,
                                       "value": value, "nonce": nonce, "disabled": disabled, "autofocus": autofocus,
                                       "compact": compact, "send_title": send_title,
                                       "default_question": default_question, "routed": routed,
                                       "keep": keep, "label": label, "depth": depth, "depths": list(depths),
                                       "depth_label": depth_label, "dark": is_dark(), "variant": variant},
                        on_submit_change=_noop, on_edited_change=_noop, on_action_change=_noop,
                        on_depth_change=_noop, height="content")
    return {"submit": getattr(result, "submit", None),
            "edited": getattr(result, "edited", None),
            "action": getattr(result, "action", None),
            "depth": getattr(result, "depth", None)}


def new_nonce():
    return uuid.uuid4().hex
