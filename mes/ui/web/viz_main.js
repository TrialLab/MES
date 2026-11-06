// MES charts — the component's entry point. data: {key, dark, view, spec}. Streamlit calls this again on every rerun
// without the previous cleanup, so the previous call's observer is disconnected here; the state of each mounted chart
// (tab, toggles, brush, node positions) lives in MEMORY under its key and survives the reruns.

const VIEWS = { study: studyView, blocks: blocksView, glance: glanceView, landscape: landscapeView, network: networkView };

// Dark when the background Streamlit drew is dark (a custom light theme stays light on a dark system)
function pageDark(root, fallback) {
  const bg = getComputedStyle(root).getPropertyValue("--st-background-color").trim();
  const hex = /^#?([0-9a-f]{6})$/i.exec(bg);
  const rgb = hex ? [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16)) : (bg.match(/[\d.]+/g) || []).map(Number);
  if (rgb.length < 3) return !!fallback;
  return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255 < 0.45;
}

function drawView(root, data, state) {
  root.textContent = "";
  root.classList.toggle("mv-dark", pageDark(root, data.dark));
  const ctx = {
    root, state, dark: !!data.dark, base: data.key || "mv", scope: data.key || "mv", tip: null,
    // replace one block in place (its charts are measured in the live page) without redrawing the others
    redrawBlock(old, build) {
      if (!old || !old.parentNode) return;
      const holder = h("div");
      holder.style.display = "contents";
      old.parentNode.insertBefore(holder, old);
      old.remove();
      ctx.tip.hide();
      build(holder);
      while (holder.firstChild) holder.parentNode.insertBefore(holder.firstChild, holder);
      holder.remove();
    },
  };
  ctx.tip = makeTip(root);
  const view = VIEWS[data.view] || blocksView;
  try {
    view(root, data.spec || {}, ctx);
  } catch (e) {
    root.textContent = "";
    root.appendChild(h("div", "mv-error", "This view could not be drawn."));
    console.warn("MES view", data.view, e);
  }
  if (!root.querySelector(":scope > .mv-tip")) ctx.tip = makeTip(root);
}

export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement && parentElement.querySelector(".mv");
  if (!root || !data) return undefined;
  if (root.__mvCleanup) root.__mvCleanup();
  const state = (MEMORY[data.key] = MEMORY[data.key] || { seen: {} });
  let width = -1, frame = 0;
  const render = () => {
    frame = 0;
    const w = root.clientWidth;
    if (!w) return;                    // hidden (a closed expander or tab): drawn when it is shown
    width = w;
    drawView(root, data, state);
  };
  const observer = new ResizeObserver(() => {
    if (Math.abs(root.clientWidth - width) > 6 && !frame) frame = requestAnimationFrame(render);
  });
  observer.observe(root);
  render();
  const cleanup = () => { observer.disconnect(); if (frame) cancelAnimationFrame(frame); };
  root.__mvCleanup = cleanup;
  return cleanup;
}
