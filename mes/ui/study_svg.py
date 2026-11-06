"""
Static SVG of the study card's charts, for reports that leave the page (the study report and the answer's HTML
report, printable and readable in Word): interval rows (forest plots, risks, rates, medians, bars), survival curves
with the patients at risk, the Love plot, the overlap of the propensity scores, histograms, trends, the ROC curve and
calibration. The same chart specs as the interactive card (ui/study_card.py), drawn without JavaScript.
"""

import html
import math

INK, INK2, INK3, GRID, AXIS = "#1A1A1A", "#555555", "#6B6B6B", "#EEEEEE", "#C8C8C8"
ACCENT, S2, BEFORE = "#CF4520", "#3B6EA8", "#9A9A9A"
SERIES = ["#CF4520", "#3B6EA8", "#B07F00", "#1A9C84", "#8A5CC9", "#C43E7A"]
ORDINAL = ["#F0A587", "#E2734A", "#CF4520", "#8E2E12"]
FONT = 'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"'


def _e(text):
    return html.escape(str(text if text is not None else ""), quote=True)


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(v, d=None):
    v = _num(v)
    if v is None:
        return "—"
    if d is None:
        d = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
    return f"{v:,.{d}f}"


def _ticks(lo, hi, n=5):
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return out


def _log_ticks(lo, hi):
    for ticks in ([0.8, 0.9, 1, 1.1, 1.25], [0.5, 0.67, 0.8, 1, 1.25, 1.5, 2], [0.25, 0.5, 1, 2, 4], [0.1, 0.2, 0.5, 1, 2, 5, 10]):
        inside = [t for t in ticks if lo <= t <= hi]
        if len(inside) >= 3 and hi / lo <= ticks[-1] / ticks[0] * 1.6:
            return inside
    return [t for t in (0.01, 0.1, 1, 10, 100) if lo <= t <= hi] or [lo, hi]


class _Scale:
    def __init__(self, d0, d1, r0, r1, log=False):
        self.log, self.d0, self.d1, self.r0, self.r1 = log, d0, d1, r0, r1
        t = (lambda v: math.log(max(v, 1e-12))) if log else (lambda v: v)
        self.t, self.a, self.b = t, t(d0), t(d1)

    def __call__(self, v):
        return self.r0 + (self.t(v) - self.a) * (self.r1 - self.r0) / ((self.b - self.a) or 1)


def _svg(width, height, body, label=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px" role="img" aria-label="{_e(label)}" {FONT}>{body}</svg>')


def _text(x, y, text, size=11, color=INK3, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{_e(text)}</text>')


# ============================================================================ interval rows
def intervals(rows, log=False, nul=None, axis="", ref=None, zero=False, width=760):
    """rows: [{label, value, lo, hi, text, primary, head, hollow, marks: [{value, hollow, color}], color}]"""
    rows = [r for r in rows if r]
    values = [v for r in rows if not r.get("head") for v in [r.get("value"), r.get("lo"), r.get("hi")]
              + [m.get("value") for m in r.get("marks") or []] if _num(v) is not None and (not log or _num(v) > 0)]
    if not values:
        return ""
    if nul is not None:
        if log:
            m = max(max(abs(math.log(v / nul)) for v in values), math.log(1.25)) * 1.12
            d0, d1 = nul * math.exp(-m), nul * math.exp(m)
        else:
            m = max(max(abs(v - nul) for v in values), 1e-9) * 1.12
            d0, d1 = nul - m, nul + m
    else:
        lo, hi = min(values + ([ref] if ref is not None else [])), max(values + ([ref] if ref is not None else []))
        pad = (hi - lo) * 0.08 or 1
        d0, d1 = (0.0 if zero and lo >= 0 else lo - pad), hi + pad
    label_w, value_w, row_h = 250, 150, 26
    x = _Scale(d0, d1, label_w + 8, width - value_w - 8, log)
    height = row_h * len(rows) + 40
    out = []
    if nul is not None:
        out.append(f'<line x1="{x(nul):.1f}" x2="{x(nul):.1f}" y1="0" y2="{row_h * len(rows)}" stroke="{INK3}"/>')
    if ref is not None:
        out.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="0" y2="{row_h * len(rows)}" stroke="{ACCENT}" '
                   f'stroke-dasharray="4 4" opacity=".7"/>')
    for i, r in enumerate(rows):
        cy = i * row_h + row_h / 2
        if r.get("head"):
            out.append(_text(4, cy + 4, str(r.get("label", "")).upper(), 10, INK3, weight=700))
            continue
        label = str(r.get("label", ""))
        out.append(_text(4, cy + 4, label if len(label) <= 44 else label[:43] + "…", 11.5, INK, weight=650 if r.get("primary") else 400))
        color = r.get("color") or (ACCENT if r.get("primary") else INK2)
        lo, hi, v = _num(r.get("lo")), _num(r.get("hi")), _num(r.get("value"))
        if lo is not None and hi is not None and (not log or (lo > 0 and hi > 0)):
            a, b = max(x.r0, min(x.r1, x(lo))), max(x.r0, min(x.r1, x(hi)))
            out.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{cy}" y2="{cy}" stroke="{color}" stroke-width="{6 if r.get("thick") else 2}" '
                       f'opacity="{0.35 if r.get("thick") else 1}" stroke-linecap="round"/>')
        for m in r.get("marks") or []:
            mv = _num(m.get("value"))
            if mv is not None:
                c = m.get("color") or color
                out.append(f'<circle cx="{x(mv):.1f}" cy="{cy}" r="4.5" fill="{"#FFFFFF" if m.get("hollow") else c}" stroke="{c}" stroke-width="2"/>')
        if v is not None and (not log or v > 0):
            if r.get("primary"):
                out.append(f'<rect x="{x(v) - 6:.1f}" y="{cy - 6}" width="12" height="12" rx="2" fill="{color}"/>')
            elif r.get("median"):
                out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{cy - 8}" y2="{cy + 8}" stroke="{color}" stroke-width="3"/>')
            else:
                out.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="5" fill="{"#FFFFFF" if r.get("hollow") else color}" '
                           f'stroke="{color}" stroke-width="2"/>')
        text = " ".join(str(t) for t in (r.get("text"), r.get("textSub")) if t)
        out.append(_text(width - 4, cy + 4, text, 11, INK, "end", 650 if r.get("primary") else 400))
    base = row_h * len(rows)
    out.append(f'<line x1="{x.r0}" x2="{x.r1}" y1="{base + 0.5}" y2="{base + 0.5}" stroke="{AXIS}"/>')
    ticks = _log_ticks(d0, d1) if log else _ticks(d0, d1, 6)
    for t in ticks:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{base}" y2="{base + 4}" stroke="{AXIS}"/>')
        out.append(_text(x(t), base + 16, _fmt(t, 2 if log and t < 1 else 0 if (log or abs(t) >= 10 or t == int(t)) else 1), 10.5,
                         INK3, "middle"))
    if axis:
        out.append(_text((x.r0 + x.r1) / 2, base + 32, axis, 11, INK2, "middle", 600))
    return _svg(width, height, "".join(out), axis)


# ============================================================================ survival curves
def km(groups, horizon=None, ordered=False, x_label="Days", width=760):
    groups = [g for g in groups if g.get("t") and g.get("s") and len(g["t"]) == len(g["s"])]
    if not groups:
        return ""
    colors = (ORDINAL if ordered and len(groups) <= 4 else SERIES)
    if not ordered:
        colors = [SERIES[0] if g.get("role") == "treated" else SERIES[1] if g.get("role") == "control" else SERIES[min(i, 5)]
                  for i, g in enumerate(groups)]
    else:
        colors = [ORDINAL[round(i * 3 / max(1, len(groups) - 1))] for i in range(len(groups))]
    horizon = _num(horizon) or max(g["t"][-1] for g in groups)
    at_risk = any(g.get("at_risk") for g in groups)
    m_l, m_r, m_t, plot_h = 46, 14, 26, 240
    height = m_t + plot_h + 44 + (22 + 18 * len(groups) if at_risk else 0)
    lows = [v for g in groups for v in (g.get("lo") or g["s"]) if _num(v) is not None]
    bottom = max(0.0, math.floor(min(lows + [min(g["s"]) for g in groups]) * 10 - 1e-9) / 10)
    x, y = _Scale(0, horizon, m_l, width - m_r), _Scale(min(bottom, 0.9), 1, m_t + plot_h, m_t)
    out = []
    for i, g in enumerate(groups):
        out.append(f'<rect x="{m_l + i * 220}" y="6" width="14" height="3" fill="{colors[i]}"/>')
        out.append(_text(m_l + 18 + i * 220, 11, f"{g.get('name', '')}" + (f" (n = {int(g['n']):,})" if _num(g.get('n')) else ""), 11, INK2))
    for t in _ticks(y.d0, 1, 5):
        out.append(f'<line x1="{m_l}" x2="{width - m_r}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
        out.append(_text(m_l - 8, y(t) + 4, f"{100 * t:.0f}%", 10.5, INK3, "end"))
    times = [d for d, _n in next((g["at_risk"] for g in groups if g.get("at_risk")), [])] or _ticks(0, horizon, 6)
    for t in times:
        out.append(_text(x(t), m_t + plot_h + 16, _fmt(t, 0), 10.5, INK3, "middle"))
    out.append(f'<line x1="{m_l}" x2="{width - m_r}" y1="{m_t + plot_h}" y2="{m_t + plot_h}" stroke="{AXIS}"/>')
    out.append(_text((m_l + width - m_r) / 2, m_t + plot_h + 32, x_label, 11, INK2, "middle", 600))
    for i, g in enumerate(groups):
        ts, ss = g["t"], g["s"]
        if g.get("lo") and g.get("hi") and len(g["lo"]) == len(ts):
            up = "".join(f"{'M' if k == 0 else 'H'}{x(ts[k]):.1f},{y(g['hi'][k]):.1f}" if k == 0 else
                         f"H{x(ts[k]):.1f}V{y(g['hi'][k]):.1f}" for k in range(len(ts)))
            down = "".join(f"H{x(ts[k]):.1f}V{y(g['lo'][k - 1]):.1f}" for k in range(len(ts) - 1, 0, -1))
            out.append(f'<path d="{up}H{x(horizon):.1f}V{y(g["lo"][-1]):.1f}{down}H{x(ts[0]):.1f}Z" fill="{colors[i]}" opacity=".11"/>')
        d = f"M{x(ts[0]):.1f},{y(ss[0]):.1f}" + "".join(f"H{x(ts[k]):.1f}V{y(ss[k]):.1f}" for k in range(1, len(ts)))
        out.append(f'<path d="{d}H{x(horizon):.1f}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
    if at_risk:
        top = m_t + plot_h + 52
        out.append(_text(4, top, "At risk", 11, INK2, weight=600))
        for i, g in enumerate(groups):
            yy = top + 18 * (i + 1)
            out.append(f'<rect x="4" y="{yy - 8}" width="10" height="3" fill="{colors[i]}"/>')
            for d, n in g.get("at_risk") or []:
                if d <= horizon + 1e-9:
                    out.append(_text(x(d), yy - 3, "<10" if n is None else f"{int(n):,}", 10.5, INK3, "middle"))
    return _svg(width, height, "".join(out), "Survival curves")


# ============================================================================ balance, distributions, models
def love(rows, threshold=0.1, width=760, limit=20):
    rows = sorted([r for r in rows if _num(r.get("before")) is not None or _num(r.get("after")) is not None],
                  key=lambda r: -abs(_num(r.get("before")) or 0))[:limit]
    if not rows:
        return ""
    label_w, row_h = 250, 20
    top = max([abs(_num(r.get("before")) or 0) for r in rows] + [abs(_num(r.get("after")) or 0) for r in rows] + [threshold * 2])
    x = _Scale(0, top * 1.08, label_w + 10, width - 16)
    height = row_h * len(rows) + 40
    out = [f'<line x1="{x(threshold):.1f}" x2="{x(threshold):.1f}" y1="0" y2="{row_h * len(rows)}" stroke="{ACCENT}" '
           f'stroke-dasharray="4 4" opacity=".7"/>']
    for i, r in enumerate(rows):
        cy = i * row_h + row_h / 2
        label = str(r.get("label", ""))
        out.append(_text(label_w, cy + 4, label if len(label) <= 40 else label[:39] + "…", 10.5, INK2, "end"))
        b, a = _num(r.get("before")), _num(r.get("after"))
        if a is not None and b is not None:
            out.append(f'<line x1="{x(abs(a)):.1f}" x2="{x(abs(b)):.1f}" y1="{cy}" y2="{cy}" stroke="{BEFORE}" opacity=".5"/>')
        if b is not None:
            out.append(f'<circle cx="{x(abs(b)):.1f}" cy="{cy}" r="4" fill="#FFFFFF" stroke="{BEFORE}" stroke-width="2"/>')
        if a is not None:
            out.append(f'<circle cx="{x(abs(a)):.1f}" cy="{cy}" r="4.5" fill="{ACCENT}"/>')
    base = row_h * len(rows)
    for t in _ticks(0, top * 1.08, 6):
        out.append(_text(x(t), base + 14, _fmt(t, 2), 10.5, INK3, "middle"))
    out.append(_text((x.r0 + x.r1) / 2, base + 30, "|standardized mean difference| — ○ before, ● after weighting", 11, INK2,
                     "middle", 600))
    return _svg(width, height, "".join(out), "Covariate balance")


def hist(edges, counts, label="", width=760):
    if not edges or len(edges) != len(counts) + 1:
        return ""
    shown = [5 if 0 < k < 10 else k for k in counts]
    m_l, m_b, h = 48, 36, 200
    x, y = _Scale(edges[0], edges[-1], m_l, width - 14), _Scale(0, max(shown + [10]) * 1.08, h - m_b, 10)
    out = []
    for i, k in enumerate(shown):
        x0, x1 = x(edges[i]) + 1, x(edges[i + 1]) - 1
        color = "#DDDDDD" if 0 < counts[i] < 10 else ACCENT
        out.append(f'<rect x="{x0:.1f}" y="{y(k):.1f}" width="{max(1, x1 - x0):.1f}" height="{max(0, y(0) - y(k)):.1f}" rx="2" '
                   f'fill="{color}" opacity=".85"/>')
    for t in _ticks(edges[0], edges[-1], 6):
        out.append(_text(x(t), h - m_b + 15, _fmt(t), 10.5, INK3, "middle"))
    out.append(_text((m_l + width) / 2, h - 6, label, 11, INK2, "middle", 600))
    return _svg(width, h, "".join(out), label)


def line(points, x_labels, unit="", width=760, band=None):
    """A trend: ``points`` [(value, lo, hi)] at categorical x positions."""
    vals = [v for p in points for v in p if _num(v) is not None]
    if len(points) < 2 or not vals:
        return ""
    m_l, h = 50, 220
    xs = [m_l + 20 + i * (width - m_l - 60) / (len(points) - 1) for i in range(len(points))]
    lo, hi = min(vals + [0]), max(vals)
    y = _Scale(lo, hi + (hi - lo) * 0.15 or 1, h - 40, 14)
    out = []
    for t in _ticks(y.d0, y.d1, 4):
        out.append(f'<line x1="{m_l}" x2="{width - 14}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
        out.append(_text(m_l - 8, y(t) + 4, f"{_fmt(t, 0)}{unit}", 10.5, INK3, "end"))
    if all(_num(p[1]) is not None and _num(p[2]) is not None for p in points):
        d = "".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{y(p[2]):.1f}" for i, p in enumerate(points))
        d += "".join(f"L{xs[i]:.1f},{y(points[i][1]):.1f}" for i in range(len(points) - 1, -1, -1))
        out.append(f'<path d="{d}Z" fill="{ACCENT}" opacity=".12"/>')
    out.append('<path d="' + "".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{y(p[0]):.1f}" for i, p in enumerate(points))
               + f'" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
    for i, p in enumerate(points):
        out.append(f'<circle cx="{xs[i]:.1f}" cy="{y(p[0]):.1f}" r="4.5" fill="{ACCENT}"/>')
        out.append(_text(xs[i], h - 24, x_labels[i], 10.5, INK3, "middle"))
        out.append(_text(xs[i], y(p[0]) - 9, f"{_fmt(p[0], 1)}{unit}", 10.5, INK, "middle"))
    return _svg(width, h, "".join(out), "Trend")


def roc(points, auc=None, size=300):
    if len(points) < 2:
        return ""
    m = 40
    s = _Scale(0, 1, m, size - 10)
    sy = _Scale(0, 1, size - m, 10)
    d = "".join(f"{'M' if i == 0 else 'L'}{s(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(points))
    out = [f'<line x1="{s(0)}" y1="{sy(0)}" x2="{s(1)}" y2="{sy(1)}" stroke="{INK3}" stroke-dasharray="4 4"/>',
           f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="2.2"/>',
           _text(s(0.55), sy(0.2), f"AUC {_fmt(auc, 2)}" if auc is not None else "", 12, INK, weight=650),
           _text((s(0) + s(1)) / 2, size - 8, "False positive rate", 11, INK2, "middle", 600)]
    for t in (0, 0.5, 1):
        out.append(_text(s(t), size - m + 14, f"{t * 100:.0f}%", 10.5, INK3, "middle"))
        out.append(_text(m - 6, sy(t) + 4, f"{t * 100:.0f}%", 10.5, INK3, "end"))
    return _svg(size, size, "".join(out), "ROC curve")


def overlap(q, treated, control, names, width=760):
    """Mirrored densities of the propensity scores from their percentiles."""
    if not q or not treated or not control:
        return ""
    top_v = min(1.0, max(treated + control) * 1.05)
    bins, h, mid = 44, 180, 90

    def cdf(v, x0):
        if x0 <= v[0]:
            return q[0]
        for i in range(len(v) - 1):
            if x0 <= v[i + 1]:
                return q[i] + (q[i + 1] - q[i]) * (x0 - v[i]) / max(v[i + 1] - v[i], 1e-9)
        return q[-1]

    def dens(v):
        step = top_v / bins
        raw = [(cdf(v, (j + 1) * step) - cdf(v, j * step)) / step if (j + 1) * step > v[0] and j * step < v[-1] else 0
               for j in range(bins)]
        peak = max(raw) or 1
        return [(j + 0.5) * step for j in range(bins)], [r / peak for r in raw]

    x = _Scale(0, top_v, 14, width - 14)
    out = []
    for v, sign, color in ((treated, -1, SERIES[0]), (control, 1, SERIES[1])):
        xs, ds = dens(v)
        d = f"M{x(xs[0]):.1f},{mid}" + "".join(f"L{x(a):.1f},{mid + sign * b * 76:.1f}" for a, b in zip(xs, ds)) + f"L{x(xs[-1]):.1f},{mid}Z"
        out.append(f'<path d="{d}" fill="{color}" opacity=".25" stroke="{color}" stroke-width="1.5"/>')
    out.append(f'<line x1="14" x2="{width - 14}" y1="{mid}" y2="{mid}" stroke="{AXIS}"/>')
    out.append(_text(width - 14, 14, f"▲ {names[0]}", 11, INK2, "end"))
    out.append(_text(width - 14, h - 22, f"▼ {names[1]}", 11, INK2, "end"))
    for t in _ticks(0, top_v, 6):
        out.append(_text(x(t), h - 6, _fmt(t, 2), 10.5, INK3, "middle"))
    return _svg(width, h, "".join(out), "Overlap of the propensity scores")


def hexmap(m, width=560):
    """The map of a "patients like this" study: stays as hexagons (at least 10 each), the similar ones in the accent,
    their ellipse and the patient as a star — one scale for both axes."""
    hexes = [h for h in (m or {}).get("hexes") or [] if isinstance(h, dict)]
    if not hexes:
        return ""
    rad = float(m.get("radius") or 0.1)
    xs = [h["x"] + d for h in hexes for d in (-rad, rad)]
    ys = [h["y"] + d for h in hexes for d in (-rad, rad)]
    p = m.get("patient") or {}
    if _num(p.get("x")) is not None:
        xs += [p["x"]]
        ys += [p["y"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    k = (width - 32) / ((x1 - x0) or 1)
    height = min(420, max(200, int(k * ((y1 - y0) or 1)) + 40))
    k = min(k, (height - 40) / ((y1 - y0) or 1))
    ox, oy = 16 + (width - 32 - k * (x1 - x0)) / 2, 20 + (height - 40 - k * (y1 - y0)) / 2
    sx, sy = (lambda v: ox + k * (v - x0)), (lambda v: height - oy - k * (v - y0))
    top = max(h["n"] for h in hexes)
    body = []
    for h in hexes:
        cx, cy, r = sx(h["x"]), sy(h["y"]), rad * k * 0.95
        pts = " ".join(f"{cx + r * math.cos(math.radians(60 * i - 30)):.1f},{cy + r * math.sin(math.radians(60 * i - 30)):.1f}"
                       for i in range(6))
        color, alpha = (ACCENT, 0.3 + 0.5 * min(1, h["k"] / max(10, h["n"]))) if h.get("k") else \
            (INK3, 0.06 + 0.5 * math.sqrt(h["n"] / top))
        body.append(f'<polygon points="{pts}" fill="{color}" fill-opacity="{alpha:.2f}" stroke="#FFFFFF"/>')
    e = m.get("ellipse") or {}
    if _num(e.get("cx")) is not None:
        body.append(f'<ellipse cx="{sx(e["cx"]):.1f}" cy="{sy(e["cy"]):.1f}" rx="{max(4, e["rx"] * k):.1f}" '
                    f'ry="{max(4, e["ry"] * k):.1f}" transform="rotate({-e["angle"]:.1f} {sx(e["cx"]):.1f} '
                    f'{sy(e["cy"]):.1f})" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 4"/>')
    if _num(p.get("x")) is not None:
        px, py = sx(p["x"]), sy(p["y"])
        star = " ".join(f"{px + (4.5 if i % 2 else 11) * math.cos(math.pi / 5 * i - math.pi / 2):.1f},"
                        f"{py + (4.5 if i % 2 else 11) * math.sin(math.pi / 5 * i - math.pi / 2):.1f}" for i in range(10))
        body.append(f'<polygon points="{star}" fill="{ACCENT}" stroke="#FFFFFF" stroke-width="1.5"/>')
    axes = m.get("axes") or ["", ""]
    if axes[0]:
        body.append(_text(width - 16, height - 6, f"{axes[0]} →", 11, INK2, "end", 600))
    if axes[1]:
        body.append(_text(16, 13, f"↑ {axes[1]}", 11, INK2, "start", 600))
    return _svg(width, height, "".join(body), "Map of the stays, the patients like this and the patient")


def trajectory(groups, unit="", width=760, height=260):
    """Medians (lines) and IQRs (bands) of a measurement per interval after time zero, one colour per group."""
    groups = [g for g in groups or [] if isinstance(g, dict) and g.get("points")]
    if not groups:
        return ""
    pts = [p for g in groups for p in g["points"]]
    x1 = max(p["t1"] for p in pts) or 1
    lo = min(p["q1"] for p in pts)
    hi = max(p["q3"] for p in pts)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    m = {"l": 48, "r": 16, "t": 12, "b": 34}
    sx = lambda v: m["l"] + (width - m["l"] - m["r"]) * v / x1
    sy = lambda v: height - m["b"] - (height - m["t"] - m["b"]) * (v - lo) / ((hi - lo) or 1)
    body = []
    for t in _ticks(lo, hi, 5):
        body.append(f'<line x1="{m["l"]}" x2="{width - m["r"]}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}"/>')
        body.append(_text(m["l"] - 6, sy(t) + 4, _fmt(t), 11, INK3, "end"))
    for t in _ticks(0, x1, 6):
        body.append(_text(sx(t), height - m["b"] + 16, f"{_fmt(t, 0)} h", 11, INK3, "middle"))
    for i, g in enumerate(groups):
        color = SERIES[i % len(SERIES)]
        mid = [((p["t0"] + p["t1"]) / 2, p) for p in g["points"]]
        band = " ".join(f"{sx(t):.1f},{sy(p['q3']):.1f}" for t, p in mid) + " " + \
            " ".join(f"{sx(t):.1f},{sy(p['q1']):.1f}" for t, p in reversed(mid))
        body.append(f'<polygon points="{band}" fill="{color}" fill-opacity="0.12"/>')
        line = " ".join(f"{sx(t):.1f},{sy(p['median']):.1f}" for t, p in mid)
        body.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        t, p = mid[-1]
        if len(groups) > 1:
            body.append(_text(min(width - m["r"], sx(t) + 4), sy(p["median"]) - 6, g.get("label", ""), 11, color, "end", 600))
    if unit:
        body.append(_text(10, m["t"] + 10, unit, 11, INK2, "start", 600))
    return _svg(width, height, "".join(body), "Trajectory over time")
