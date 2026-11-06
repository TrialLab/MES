// MES charts — the cohort builder's flow of patients: every step of the edited design beside the study's own (the
// change since the study; new, changed and switched-off criteria), the numbers moving from the values drawn last.

function cfCount(n, small) {
  if (small) return "<10";
  return isNum(n) ? int(n) : "—";
}

// A number that moves from the value shown last to the new one (and remembers it)
function cfTween(el, mem, key, to, small) {
  const from = mem[key];
  mem[key] = small ? null : to;
  if (small || !isNum(to)) { el.textContent = cfCount(to, small); return; }
  if (!isNum(from) || from === to || !motion()) { el.textContent = int(to); return; }
  const t0 = performance.now(), dur = 480;
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    el.textContent = int(from + (to - from) * (1 - (1 - k) ** 3));
    if (k < 1 && el.isConnected) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// The change against the study: ▲ +1,203 / ▼ −58 / = (text in ink; the arrow and the tint carry the direction)
function cfDelta(n, was, small, wasSmall) {
  if (small || wasSmall || !isNum(n) || !isNum(was)) return null;
  const d = n - was;
  if (d === 0) return h("span", "mv-cf-delta mv-cf-same", "= study");
  const chip = h("span", `mv-cf-delta ${d > 0 ? "mv-cf-up" : "mv-cf-down"}`);
  // a change of fewer than 10 patients is a small cell too
  chip.textContent = `${d > 0 ? "▲ +" : "▼ −"}${Math.abs(d) < 10 ? "<10" : int(Math.abs(d))}`;
  chip.title = `${int(was)} in the study's design`;
  return chip;
}

const CF_STATE = { new: "new", changed: "changed", off: "switched off" };

function cfBar(mem, key, n, n0) {
  const bar = h("div", "mv-fbar mv-cf-bar");
  const fill = h("i");
  const w = isNum(n) && isNum(n0) && n0 > 0 ? Math.max(1.5, (100 * n) / n0) : 0;
  const before = mem[`w:${key}`];
  mem[`w:${key}`] = w;
  fill.style.animation = "none";
  fill.style.width = `${isNum(before) && motion() ? before : w}%`;
  if (isNum(before) && before !== w && motion()) requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${w}%`; }));
  bar.appendChild(fill);
  return bar;
}

// c: {steps: [{key, label, n, was, small, was_small, state, role}], groups: {treated, control: {label, n, was, small}},
//     final: {n, was, small, was_small, label}}
function cohortflow(body, c, ctx) {
  const mem = (ctx.state.cf = ctx.state.cf || {});
  const steps = c.steps || [];
  const fin = c.final || {};
  // the headline: the cohort the analysis would use, and how it changed
  const head = h("div", "mv-cf-head");
  const big = h("div", "mv-cf-big");
  const num = h("span", "mv-cf-n");
  big.appendChild(num);
  big.appendChild(h("span", "mv-cf-unit", fin.label || "patients"));
  head.appendChild(big);
  const d = cfDelta(fin.n, fin.was, fin.small, fin.was_small);
  if (d) head.appendChild(d);
  cfTween(num, mem, "final", fin.n, fin.small);
  if (c.groups) {
    const g = h("div", "mv-cf-groups");
    const colors = groupColors(ctx, [{ role: "treated" }, { role: "control" }], false);
    ["treated", "control"].forEach((k, i) => {
      const x = c.groups[k];
      if (!x) return;
      const chip = h("div", "mv-cf-group");
      chip.style.setProperty("--sc", colors[i]);
      const n = h("b");
      chip.append(h("i"), n, h("span", null, x.label || k));
      const dd = cfDelta(x.n, x.was, x.small, x.was_small);
      if (dd) chip.appendChild(dd);
      cfTween(n, mem, `g:${k}`, x.n, x.small);
      g.appendChild(chip);
    });
    head.appendChild(g);
  }
  body.appendChild(head);
  if (!steps.length) { body.appendChild(h("div", "mv-empty", "No steps to count.")); return; }
  const wrap = h("div", "mv-cf");
  body.appendChild(wrap);
  const n0 = (steps.find((s) => isNum(s.n)) || {}).n;
  let prev = null;
  steps.forEach((s) => {
    const off = s.state === "off";
    if (!off && prev !== null && isNum(prev) && isNum(s.n) && prev > s.n && !s.small) {
      const share = (100 * (prev - s.n)) / prev;
      const x = h("div", "mv-cf-x");
      x.append(h("b", null, prev - s.n < 10 ? "− <10" : `− ${int(prev - s.n)}`),
               document.createTextNode(` excluded (${fmt(share, share < 1 ? 1 : 0)}%)`));
      wrap.appendChild(x);
    }
    const row = h("div", `mv-cf-step mv-cf-${s.state || "same"} mv-cf-role-${s.role || "fixed"}`);
    const count = h("div", "mv-cf-count");
    const val = h("span");
    count.appendChild(val);
    if (off) {
      val.textContent = "off";
      if (isNum(s.was) || s.was_small) count.appendChild(h("small", null, `was ${cfCount(s.was, s.was_small)}`));
    } else {
      cfTween(val, mem, s.key, s.n, s.small);
      const dd = cfDelta(s.n, s.was, s.small, s.was_small);
      if (dd) count.appendChild(dd);
    }
    const text = h("div", "mv-cf-text");
    const label = h("div", "mv-cf-label", s.label);
    text.appendChild(label);
    if (CF_STATE[s.state]) text.appendChild(h("span", `mv-cf-badge mv-cf-badge-${s.state}`, CF_STATE[s.state]));
    text.appendChild(cfBar(mem, s.key, off ? 0 : s.n, n0));
    row.append(count, text);
    row.tabIndex = 0;
    hoverable(row, ctx, () => ({ title: s.label, rows: [
      { label: "now", value: off ? "switched off" : cfCount(s.n, s.small) },
      { label: "in the study", value: isNum(s.was) || s.was_small ? cfCount(s.was, s.was_small) : "—" }] }));
    wrap.appendChild(row);
    if (!off) prev = s.small ? null : s.n;
  });
}
