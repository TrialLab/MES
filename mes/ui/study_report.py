"""
The study card on paper: every tab of a real-world data study's card (ui/study_card.py) as static HTML with SVG charts
(ui/study_svg.py) — a standalone study report to print or open in Word, and the studies section of the answer's HTML
report. The same spec as the interactive card, so both always say the same.
"""

import html

from mes.tools.rwd import DATASET, about, database_of
from mes.ui import study_card
from mes.ui import study_svg as V

TAB_NAMES = {"protocol": "Design"}          # in the answer's report "Protocol" names the user's edited versions


def _e(text):
    return html.escape(str(text if text is not None else ""))


def _table(columns, rows, smd=False):
    head = "".join(f"<th>{_e(c.get('label'))}</th>" for c in columns)
    body = []
    for r in rows:
        if r.get("__heading"):
            body.append(f'<tr class="h"><td colspan="{len(columns)}">{_e(r.get("label"))}</td></tr>')
            continue
        cells = []
        for c in columns:
            v = r.get(c["key"])
            if smd and c.get("smd") and isinstance(v, (int, float)):
                cells.append(f'<td class="n{" hi" if abs(v) >= 0.1 else ""}">{v:.2f}</td>')
            else:
                cells.append(f'<td class="{"n" if c.get("numeric") else ""}">{_e("—" if v is None else v)}</td>')
        body.append(f'<tr class="{"lvl" if r.get("__level") else ""}">{"".join(cells)}</tr>')
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _chart(c):
    t = c.get("type")
    if t in ("forest", "arms", "dots", "subgroups", "ranges", "bars"):
        rows = []
        if t == "forest":
            for r in c.get("rows") or []:
                rows.append({**r, "hollow": r.get("sensitivity"), "color": V.ACCENT if r.get("primary") else V.INK2})
            return V.intervals(rows, log=c.get("log"), nul=c.get("nul"), axis=c.get("axis", ""), zero=c.get("zero"))
        if t == "arms":
            for i, r in enumerate(c.get("rows") or []):
                color = V.SERIES[0] if r.get("role") == "treated" else V.SERIES[1]
                crude = r.get("crude") or {}
                rows.append({"label": r.get("label"), "lo": crude.get("lo"), "hi": crude.get("hi"), "value": r.get("weighted"),
                             "marks": [{"value": crude.get("value"), "hollow": True, "color": color}], "color": color,
                             "text": r.get("text"), "textSub": r.get("textSub")})
            return V.intervals(rows, axis=c.get("axis", "Risk (%)"), zero=True)
        if t == "dots":
            for g in c.get("groups") or []:
                rows.append({"head": True, "label": g.get("variable")})
                rows += [{"label": r.get("level"), "value": r.get("value"), "lo": r.get("lo"), "hi": r.get("hi"),
                          "color": V.ACCENT, "text": f"{V._fmt(r.get('value'), 1)}%"} for r in g.get("rows") or []]
            overall = (c.get("overall") or {}).get("value")
            return V.intervals(rows, axis=c.get("axis", ""), ref=overall, zero=True)
        if t == "ranges":
            for g in c.get("groups") or []:
                rows.append({"head": True, "label": g.get("variable")})
                rows += [{"label": r.get("level"), "value": r.get("median"), "lo": r.get("q1"), "hi": r.get("q3"), "thick": True,
                          "median": True, "color": V.ACCENT, "text": V._fmt(r.get("median"))} for r in g.get("rows") or []]
            return V.intervals(rows, axis=c.get("axis", ""))
        if t == "subgroups":
            o = c.get("overall") or {}
            if o.get("value") is not None:
                rows.append({"label": "All patients", "value": o.get("value"), "lo": o.get("lo"), "hi": o.get("hi"), "primary": True,
                             "text": f"{V._fmt(o.get('value'), 1)} pp"})
            rows += [{"label": r.get("label"), "value": r.get("value"), "lo": r.get("lo"), "hi": r.get("hi"),
                      "text": f"{V._fmt(r.get('value'), 1)} pp"} for r in c.get("rows") or []]
            return V.intervals(rows, nul=0, axis=c.get("axis", ""))
        col = next((x for x in c.get("columns") or [] if x["key"] == c.get("default")), (c.get("columns") or [{}])[0])
        rows = [{"label": r.get("label"), "value": (r.get("values") or {}).get(col.get("key")),
                 "lo": (r.get("values") or {}).get(col.get("lo")), "hi": (r.get("values") or {}).get(col.get("hi")),
                 "color": V.ACCENT, "text": V._fmt((r.get("values") or {}).get(col.get("key")))}
                for r in (c.get("rows") or [])[:30]]
        return V.intervals(rows, axis=col.get("label", ""), zero=True)
    if t == "received":
        rows = [{"label": r.get("label"), "value": r.get("value"), "lo": r.get("lo"), "hi": r.get("hi"), "color": V.ACCENT,
                 "marks": [{"value": r.get("cohort"), "hollow": True, "color": V.INK3}],
                 "text": f"{V._fmt(r.get('value'), 1)}%" if r.get("value") is not None else "<10"}
                for r in c.get("rows") or []]
        return V.intervals(rows, axis="Patients (%) — dot: patients like this; ring: everyone", zero=True)
    if t == "simmap":
        return V.hexmap(c)
    if t == "trajectory":
        return V.trajectory(c.get("groups") or [], c.get("unit", ""))
    if t == "km":
        return V.km(c.get("groups") or [], c.get("horizon"), c.get("ordered"), c.get("x_label", "Days"))
    if t == "love":
        return V.love(c.get("rows") or [], c.get("threshold") or 0.1)
    if t == "overlap" and c.get("q"):
        return V.overlap(c["q"], c.get("treated"), c.get("control"), c.get("names") or ["treated", "comparison"])
    if t == "hist":
        return V.hist(c.get("edges") or [], c.get("counts") or [], c.get("label", ""))
    if t == "trend":
        rows = c.get("rows") or []
        return V.line([(r.get("value"), r.get("lo"), r.get("hi")) for r in rows], [r.get("period", "") for r in rows],
                      c.get("unit", ""))
    if t == "roc":
        return V.roc(c.get("points") or [], c.get("auc"))
    if t == "flow":
        steps = c.get("steps") or []
        boxes = []
        for i, (label, n) in enumerate(steps):
            prev = steps[i - 1][1] if i else None
            gone = f' <span class="x">− {prev - n:,.0f} excluded</span>' if isinstance(prev, (int, float)) and isinstance(n, (int, float)) and prev > n else ""
            boxes.append(f'<div class="box"><b>{"<10" if n is None else f"{n:,.0f}"}</b> {_e(label)}{gone}</div>')
        for g in c.get("split") or []:
            boxes.append(f'<div class="box split"><b>{g.get("n") or 0:,.0f}</b> {_e(g.get("label"))} {_e(g.get("sub", ""))}</div>')
        return '<div class="flow">' + "".join(boxes) + "</div>"
    if t == "timeline":
        segs = "".join(f'<span class="seg {s.get("kind", "")}" style="flex:{s.get("w", 1)}">{_e(s.get("label"))}</span>'
                       for s in c.get("segments") or [])
        marks = " · ".join(f"<b>{_e(m.get('label'))}</b> {_e(m.get('sub', ''))}" for m in c.get("marks") or [])
        return f'<div class="tl">{segs}</div><div class="cap">{marks}</div>'
    if t == "steps":
        rows = c.get("rows") or []
        return _table([{"key": "n", "label": "#"}, {"key": "action", "label": "Action"}, {"key": "seconds", "label": "s", "numeric": True},
                       {"key": "plan", "label": "Plan"}],
                      [{"n": i + 1, "action": r.get("action"), "seconds": V._fmt(r.get("seconds"), 1), "plan": r.get("plan")}
                       for i, r in enumerate(rows)])
    return ""


def _block(b):
    kind = b.get("type")
    title = f'<h4>{_e(b.get("title"))}</h4>' if b.get("title") else ""
    sub = f'<div class="sub">{_e(b.get("subtitle"))}</div>' if b.get("subtitle") else ""
    cap = f'<div class="cap">{_e(b.get("caption"))}</div>' if b.get("caption") else ""
    if kind == "grid":
        return "".join(_block(x) for x in b.get("blocks") or [])
    if kind == "chart":
        c = b.get("chart") or {}
        drawn = _chart(c)
        if not drawn and c.get("table"):
            drawn = _table(c["table"]["columns"], c["table"]["rows"])
        return f'<div class="block">{title}{sub}{drawn}{cap}</div>' if drawn else ""
    if kind == "table":
        columns = [col for col in b.get("columns") or [] if not col.get("view")] + \
                  [col for col in b.get("columns") or [] if col.get("view")]
        return f'<div class="block">{title}{sub}{_table(columns, b.get("rows") or [], smd=b.get("smd"))}</div>'
    if kind == "kv":
        rows = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in b.get("rows") or [])
        return f'<div class="block">{title}<table class="kv">{rows}</table></div>'
    if kind == "notes":
        items = "".join(f"<li>{_e(x)}</li>" for x in b.get("items") or [])
        return f'<div class="block notes {b.get("tone", "info")}">{title}<ul>{items}</ul></div>'
    if kind == "chips":
        items = "".join(f"<li>{_e(x.get('text'))} <span>{_e(x.get('detail', ''))}</span></li>" for x in b.get("items") or [])
        return f'<div class="block">{title}<ul>{items}</ul></div>'
    if kind == "scorecard":
        icon = {"ok": "✓", "warn": "!", "bad": "✕", "info": "i"}
        items = "".join(f'<li class="{x.get("status")}"><b>{icon.get(x.get("status"), "i")} {_e(x.get("label"))}</b> — '
                        f'{_e(x.get("detail", ""))}</li>' for x in b.get("items") or [])
        return f'<div class="block">{title}<ul class="score">{items}</ul></div>'
    if kind == "defs":
        items = []
        for d in b.get("items") or []:
            codes = "; ".join(f"{_e(k)}: {_e(', '.join(map(str, v[:40])))}" for k, v in (d.get("codes") or {}).items() if v)
            meds = ", ".join(map(_e, d.get("medications") or []))
            items.append(f"<li><b>{_e(d.get('role'))} — {_e(d.get('label'))}</b> ({_e(d.get('kind'))}). {_e(d.get('detail', ''))}"
                         + (f"<br><span>{codes}</span>" if codes else "") + (f"<br><span>Matched: {meds}</span>" if meds else "")
                         + "</li>")
        return f'<div class="block">{title}{sub}<ul>{"".join(items)}</ul></div>'
    if kind == "text":
        return f'<div class="block">{title}<p>{_e(b.get("text"))}</p></div>'
    return ""


def card_html(spec, tab_names=None):
    """The card as HTML: header, the key numbers, then every tab in turn."""
    names = {**{"results": "Results"}, **(tab_names or {})}
    kick = " · ".join(_e(x) for x in (spec.get("dataset"), spec.get("design"),
                                      f"{spec['n']:,.0f} {spec.get('unit', 'patients')}" if spec.get("n") else "") if x)
    kpis = "".join(f'<div class="kpi"><div class="l">{_e(k.get("label"))}</div><div class="v">{_e(k.get("value"))}'
                   f'<small>{_e(k.get("unit", ""))}</small></div><div class="c">{_e(k.get("ci", ""))}</div>'
                   f'<div class="c">{_e(k.get("note", ""))}</div></div>' for k in spec.get("kpis") or [])
    badges = "".join(f'<span class="badge {b.get("kind")}">{_e(b.get("text"))}</span>' for b in spec.get("badges") or [])
    tabs = []
    for tab in spec.get("tabs") or []:
        body = "".join(_block(b) for b in tab.get("blocks") or [])
        if body:
            tabs.append(f'<h3>{_e(names.get(tab["id"], tab.get("label")))}</h3>{body}')
    return (f'<div class="study"><div class="kick">[{_e(spec.get("cite"))}] {kick}</div><h2>{_e(spec.get("title"))}</h2>'
            f'<div class="sub">{_e(spec.get("subtitle"))}</div><div class="badges">{badges}</div>'
            f'<div class="kpis">{kpis}</div>{"".join(tabs)}</div>')


CSS = """
.study { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1A1A1A; }
.study .kick { color: #0F766E; font-weight: 600; font-size: 13px; }
.study h2 { font-size: 20px; margin: 4px 0 2px; border: 0; }
.study h3 { font-size: 15px; margin: 22px 0 6px; padding-bottom: 4px; border-bottom: 2px solid #CF4520; }
.study h4 { font-size: 13px; margin: 14px 0 2px; }
.study .sub, .study .cap { color: #666666; font-size: 12px; }
.study .badges { margin: 6px 0; } .study .badge { display: inline-block; font-size: 11px; border-radius: 999px; padding: 1px 8px;
  margin: 0 4px 4px 0; background: #F2F2F2; } .badge.ok { background: #EDF5EE; color: #2E7D32; }
.badge.warn { background: #FFF6E5; color: #9A5B0B; } .badge.bad { background: #FDEEEC; color: #B42318; }
.study .kpis { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.study .kpi { border: 1px solid #DDDDDD; border-radius: 10px; padding: 6px 10px; min-width: 140px; }
.study .kpi .l { font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: #555555; font-weight: 700; }
.study .kpi .v { font-size: 20px; font-weight: 700; } .study .kpi .v small { font-size: 12px; color: #555555; margin-left: 3px; }
.study .kpi .c { font-size: 11px; color: #666666; }
.study table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 6px 0; }
.study th, .study td { border: 1px solid #DDDDDD; padding: 4px 7px; text-align: left; vertical-align: top; }
.study th { background: #F7F7F7; } .study td.n { text-align: right; } .study td.hi { color: #B42318; font-weight: 600; }
.study tr.h td { background: #FBFBFA; font-weight: 600; } .study tr.lvl td:first-child { padding-left: 18px; }
.study table.kv th { width: 24%; }
.study .notes { border-radius: 8px; padding: 4px 12px; background: #F7F7F7; font-size: 13px; }
.study .notes.warn { background: #FFF6E5; } .study ul { font-size: 13px; }
.study ul.score li.ok b { color: #2E7D32; } .study ul.score li.warn b { color: #9A5B0B; } .study ul.score li.bad b { color: #B42318; }
.study .flow .box { border: 1px solid #DDDDDD; border-radius: 8px; padding: 5px 10px; margin: 4px 0; font-size: 12px; }
.study .flow .box.split { display: inline-block; margin-right: 8px; border-top: 3px solid #CF4520; }
.study .flow .x { color: #B42318; } .study .tl { display: flex; border-radius: 8px; overflow: hidden; border: 1px solid #DDDDDD; }
.study .tl .seg { padding: 6px; font-size: 11px; text-align: center; background: #F7F7F7; }
.study .tl .seg.exposure { background: #F7D9CC; } .study .tl .seg.follow { background: #DCE6F2; }
.study svg { margin: 4px 0; }
@media print { .study h3 { break-after: avoid; } .study .block { break-inside: avoid; } }
"""


def report_html(result, study):
    """A standalone report of one study: its card, every tab, and its SQL or code."""
    spec = study_card.card(result, study)
    if spec is None:
        return ""
    code = study.get("code") or study.get("sql") or ""
    code_html = (f"<h3>{'Code' if study.get('code') else 'SQL'}</h3><pre>{_e(code)}</pre>") if code else ""
    centers = about(database_of(study))["centers"]
    data = _e(study.get("dataset") or DATASET) + (f" ({_e(centers)})" if centers else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_e(spec.get('title'))}</title>
<style>body {{ max-width: 900px; margin: 32px auto; padding: 0 24px; font-family: -apple-system, "Segoe UI", Helvetica,
Arial, sans-serif; color: #1A1A1A; }} pre {{ font-size: 11px; background: #F7F7F7;
border: 1px solid #DDDDDD; border-radius: 8px; padding: 8px; white-space: pre-wrap; }} .note {{ color: #666666; font-size: 12px; }}
{CSS}</style></head><body>
<div class="note">MES · real-world evidence study · question: {_e(result.question)}</div>
{card_html(spec, {"protocol": "Protocol"})}
<div class="study">{code_html}</div>
<div class="note">Observational analysis of {data}: residual confounding cannot be excluded.
MES is a research prototype for evidence synthesis, not a medical device.</div></body></html>"""
