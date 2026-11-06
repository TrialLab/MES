"""
The Results of the descriptive real-world data studies (ui/study_card.py): a characterization (the largest differences
between two groups and their Table 1, or the Table 1 of one cohort), a trajectory (the median and IQR of a measurement
per interval, by group) and treatment pathways (the flow of first, second and third line, who received each treatment
and when). Pure functions of the stored record; every chart carries its numbers as a table.
"""

from mes.tools.rwd import library as L
from mes.ui.study_results import num, table


def _pct(v):
    v = num(v)
    return "—" if v is None else f"{v:.1f}%"


# ============================================================================ characterize
def characterize_results(spec, res):
    from mes.ui.study_card import table1_block
    blocks = []
    g = res.get("groups") or {}
    diffs = [d for d in (res.get("tables") or {}).get("differences") or [] if isinstance(d, dict) and num(d.get("smd"))
             is not None]
    if diffs:
        a, b = g.get("treated", "group 1"), g.get("control", "group 2")
        rows = [{"label": d.get("label"), "value": num(d["smd"]), "text": f"{d['smd']:+.2f}", "textSub": f"{d.get('a')} vs "
                 f"{d.get('b')}", "full": f"{d.get('label')}: {d.get('a')} vs {d.get('b')} (SMD {d['smd']:+.2f})"}
                for d in diffs[:15]]
        t = table([("label", "Characteristic", False), ("a", a, True), ("b", b, True), ("smd", "SMD", True)],
                  [{"label": d.get("label"), "a": d.get("a"), "b": d.get("b"), "smd": f"{d['smd']:+.2f}"} for d in diffs])
        blocks.append({"type": "chart", "id": "differences", "title": f"How {a} differ from {b}",
                       "subtitle": "the largest standardized mean differences (|SMD| ≥ 0.1: a meaningful difference)",
                       "chart": {"type": "forest", "rows": rows, "nul": 0, "axis": "Standardized mean difference",
                                 "dir": [f"higher in {b}", f"higher in {a}"], "table": t}})
    t1 = res.get("table1") or {}
    if t1.get("rows"):
        blocks.append(_single_table(t1) if t1.get("single") else table1_block(t1))
    return blocks


def _single_table(t1):
    n = (t1.get("n") or [None])[0]
    rows = []
    for r in t1["rows"]:
        if r.get("heading"):
            rows.append({"label": r.get("label", ""), "__heading": True})
            continue
        label = r.get("label", "") + (f", {r['unit']} — mean (SD)" if r.get("unit") and r.get("kind") == "numeric" else
                                      " — mean (SD)" if r.get("kind") == "numeric" else "")
        rows.append({"label": label, "value": r.get("treated_raw"), "missing": f"{100 * num(r.get('missing')):.0f}%"
                     if num(r.get("missing")) else "", "__level": bool(r.get("level"))})
    return {"type": "table", "id": "table1", "title": "Characteristics (Table 1)",
            "subtitle": "n (%) or mean (SD); cells under 10 patients are not shown",
            "columns": [{"key": "label", "label": "Characteristic"},
                        {"key": "value", "label": f"All patients (n = {int(n):,})" if num(n) is not None else "All patients",
                         "numeric": True}, {"key": "missing", "label": "Missing", "numeric": True}],
            "rows": rows, "max_height": 560}


# ============================================================================ trajectory
def trajectory_results(spec, res):
    c = (res.get("charts") or {}).get("trajectory") or {}
    groups = [g for g in c.get("groups") or [] if isinstance(g, dict) and g.get("points")]
    if not groups:
        return []
    unit = c.get("unit") or ""
    from mes.tools.rwd.similar import numbers

    def cells(p):
        a, b, c_ = numbers([p["median"], p["q1"], p["q3"]])
        return {"median": a, "iqr": f"{b}–{c_}"}
    rows = [{"group": g["label"], "interval": f"{p['t0']:g}–{p['t1']:g} h", "n": f"{int(p['n']):,}", **cells(p)}
            for g in groups for p in g["points"]]
    t = table([("group", "Group", False), ("interval", "Interval", False), ("n", "Patients", True),
               ("median", f"Median{f' ({unit})' if unit else ''}", True), ("iqr", "IQR", True)], rows)
    return [{"type": "chart", "id": "trajectory", "title": f"{c.get('variable', 'The measurement')} after "
             f"{c.get('t0_label', 'time zero')}",
             "subtitle": f"median (line) and IQR (band) of each patient's mean per {c.get('bin_h', 6):g}-h interval",
             "caption": "Intervals with fewer than 10 patients are left out. Patients who died or left drop out over time: "
                        "a later interval describes those still there.",
             "chart": {"type": "trajectory", **{k: c.get(k) for k in ("unit", "bin_h", "horizon_h", "variable")},
                       "groups": groups, "table": t}}]


# ============================================================================ pathways
def pathways_results(spec, res):
    tables = res.get("tables") or {}
    rows = [r for r in tables.get("treatments") or [] if isinstance(r, dict)]
    blocks = []
    s = (res.get("charts") or {}).get("sankey") or {}
    links = [x for x in s.get("links") or [] if isinstance(x, dict) and num(x.get("n")) is not None]
    if links:
        t = table([("from", "From", False), ("to", "To", False), ("n", "Patients", True)],
                  [{"from": f"{x['from'][1]} (line {x['from'][0]})", "to": f"{x['to'][1]} (line {x['to'][0]})",
                    "n": f"{int(x['n']):,}"} for x in links])
        blocks.append({"type": "chart", "id": "sankey", "title": "Treatment lines",
                       "subtitle": f"the first, second and third treatment started within {s.get('horizon_h', 72):g} h "
                                   "(by the first dose)",
                       "caption": "Links of fewer than 10 patients are merged into “other” or left out.",
                       "chart": {"type": "sankey", "links": links, "names": s.get("names") or [], "n": s.get("n"),
                                 "table": t}})
    if rows:
        t = table([("treatment", "Treatment", False), ("given", "Received (95% CI)", True), ("first", "First", True),
                   ("time", "First dose, h (IQR)", True)],
                  [{"treatment": r.get("treatment"), "given": f"{_pct(r.get('given'))} ({_pct(r.get('lo'))}–{_pct(r.get('hi'))})"
                    if r.get("given") is not None else "<10", "first": _pct(r.get("first")),
                    "time": f"{r['median_h']:g} ({r['q1_h']:g}–{r['q3_h']:g})" if r.get("median_h") is not None else "—"}
                   for r in rows])
        blocks.append({"type": "chart", "id": "given", "title": "Who received each treatment",
                       "subtitle": "received within the window (dot, 95% CI) and as the first treatment (ring)",
                       "chart": {"type": "received", "rows": [{"label": r.get("treatment"), "value": num(r.get("given")),
                                                               "lo": num(r.get("lo")), "hi": num(r.get("hi")),
                                                               "cohort": num(r.get("first"))} for r in rows],
                                 "legend": ["received (95% CI)", "first treatment"], "table": t}})
        timed = [r for r in rows if num(r.get("median_h")) is not None]
        if timed:
            groups = [{"variable": "First dose", "rows": [{"level": r["treatment"], "median": num(r["median_h"]),
                                                           "q1": num(r.get("q1_h")), "q3": num(r.get("q3_h")),
                                                           "n": r.get("events")} for r in timed]}]
            t = table([("level", "Treatment", False), ("median", "Median (h)", True), ("iqr", "IQR (h)", True)],
                      [{"level": r["treatment"], "median": f"{r['median_h']:g}", "iqr": f"{r['q1_h']:g}–{r['q3_h']:g}"}
                       for r in timed])
            blocks.append({"type": "chart", "id": "first_dose", "title": "Hours to the first dose",
                           "subtitle": "median and IQR among the patients who received it",
                           "chart": {"type": "ranges", "groups": groups, "unit": "h", "axis": "Hours after time zero",
                                     "table": t}})
    seqs = [q for q in tables.get("sequences") or [] if isinstance(q, dict)]
    if seqs:
        blocks.append({"type": "table", "id": "sequences", "title": "The commonest sequences",
                       **table([("sequence", "Order of the first doses", False), ("n", "Patients", True),
                                ("share", "Share", True)],
                               [{"sequence": q.get("sequence"), "n": f"{int(q['n']):,}", "share": _pct(q.get("share"))}
                                for q in seqs])})
    return blocks


# ============================================================================ the card's key numbers
def kpis(study):
    res = study.get("result") or {}
    spec = study.get("spec") or {}
    analysis = res.get("analysis") or spec.get("analysis")
    est = res.get("estimates") or []
    out = []
    if analysis == "characterize":
        g = res.get("groups") or {}
        if num(g.get("n_treated")) is not None:
            out.append({"label": "Patients", "value": f"{int(g['n_treated']):,} / {int(g.get('n_control') or 0):,}",
                        "note": f"{g.get('treated')} / {g.get('control')}"})
        rate = next((e for e in est if e.get("measure") == "rate"), None)
        if rate and num(rate.get("value")) is not None:
            out.append({"label": L.OUTCOMES.get(spec.get("outcome"), ("Outcome",))[0], "value": f"{rate['value']:.1f}",
                        "unit": "%", "tone": "primary", "ci": f"95% CI {rate['lo']:.1f}–{rate['hi']:.1f}",
                        "bar": {"value": rate["value"], "lo": rate["lo"], "hi": rate["hi"], "min": 0, "max": 100}})
        diffs = (res.get("tables") or {}).get("differences") or []
        if diffs and num(diffs[0].get("smd")) is not None:
            out.append({"label": "Largest difference", "value": f"{diffs[0]['smd']:+.2f}", "unit": "SMD",
                        "note": str(diffs[0].get("label"))[:60]})
        if not out and num(res.get("n")) is not None:
            out.append({"label": "Patients", "value": f"{int(res['n']):,}", "note": res.get("population", "")})
    elif analysis == "trajectory":
        from mes.tools.rwd.similar import numbers
        for e in [e for e in est if e.get("measure") == "median" and num(e.get("value")) is not None][:4]:
            values = [num(e["value"])] + ([num(e["lo"]), num(e["hi"])] if num(e.get("lo")) is not None else [])
            shown = numbers(values)
            out.append({"label": f"{e.get('group') or 'all'} · {e.get('interval') or ''}".strip(" ·")[:60],
                        "value": shown[0], "unit": e.get("unit", ""), "tone": "primary" if e.get("primary") else "",
                        "ci": f"IQR {shown[1]}–{shown[2]}" if len(shown) == 3 else "",
                        "note": f"{int(e['n']):,} patients" if num(e.get("n")) else ""})
    elif analysis == "top":
        rows = [r for r in (res.get("tables") or {}).get("ranking") or [] if isinstance(r, dict)]
        if rows and num(rows[0].get("share")) is not None:
            out.append({"label": "Commonest", "value": str(rows[0]["item"])[:40], "tone": "primary",
                        "note": f"{rows[0]['share']:.1f}% of the patients"})
        for r in rows[1:3]:
            if num(r.get("share")) is not None:
                out.append({"label": str(r["item"])[:40], "value": f"{r['share']:.1f}", "unit": "%",
                            "bar": {"value": r["share"], "lo": r.get("lo"), "hi": r.get("hi"), "min": 0, "max": 100}})
        if num(res.get("n")) is not None:
            out.append({"label": "Patients", "value": f"{int(res['n']):,}", "note": res.get("population", "")})
    elif analysis == "pathways":
        rows = [r for r in (res.get("tables") or {}).get("treatments") or [] if isinstance(r, dict)]
        firsts = sorted([r for r in rows if num(r.get("first")) is not None], key=lambda r: -r["first"])
        if firsts:
            out.append({"label": "Most often first", "value": firsts[0]["treatment"], "tone": "primary",
                        "note": f"first in {firsts[0]['first']:.1f}% of the patients"})
        for r in sorted([r for r in rows if num(r.get("given")) is not None], key=lambda r: -r["given"])[:2]:
            out.append({"label": f"Received {r['treatment']}", "value": f"{r['given']:.1f}", "unit": "%",
                        "ci": f"95% CI {r['lo']:.1f}–{r['hi']:.1f}" if num(r.get("lo")) is not None else "",
                        "bar": {"value": r["given"], "lo": r.get("lo"), "hi": r.get("hi"), "min": 0, "max": 100}})
        if num(res.get("n")) is not None:
            out.append({"label": "Patients", "value": f"{int(res['n']):,}", "note": res.get("population", "")})
    return [k for k in out if k][:4]


def top_results(spec, res):
    """The commonest items: the share of patients with each (dot, 95% CI; two groups: dot and ring) and the table."""
    rows = [r for r in (res.get("tables") or {}).get("ranking") or [] if isinstance(r, dict)]
    if not rows:
        return []
    names = (res.get("groups") or {}).get("names") or []
    two = len(names) == 2 and any(num(r.get("a")) is not None or num(r.get("b")) is not None for r in rows)
    kind = {"drugs": "medications", "diagnoses": "diagnoses", "procedures": "procedures"}.get(spec.get("ranked"), "items")
    columns = [("item", kind.capitalize()[:-1] if kind.endswith("s") else kind, False), ("code", "Code", False),
               ("patients", "Patients", True), ("share", "Share (95% CI)", True)]
    if two:
        columns += [("a", names[0], True), ("b", names[1], True)]
    if spec.get("ranked") != "diagnoses":
        columns = [c for c in columns if c[0] != "code"]
    t = table(columns, [{"item": r.get("item"), "code": r.get("code") or "", "patients": f"{int(r['patients']):,}",
                         "share": f"{_pct(r.get('share'))} ({_pct(r.get('lo'))}–{_pct(r.get('hi'))})",
                         "a": _pct(r.get("a")) if num(r.get("a")) is not None else "<10",
                         "b": _pct(r.get("b")) if num(r.get("b")) is not None else "<10"} for r in rows])
    chart = {"type": "received", "rows": [{"label": str(r.get("item"))[:60], "value": num(r.get("a") if two else r.get("share")),
                                           "lo": None if two else num(r.get("lo")), "hi": None if two else num(r.get("hi")),
                                           "cohort": num(r.get("b")) if two else None} for r in rows[:20]],
             "legend": [names[0], names[1]] if two else ["patients with it (95% CI)", ""], "table": t}
    return [{"type": "chart", "id": "top", "title": f"The commonest {kind}",
             "subtitle": ("the share of patients with each — " + f"{names[0]} (dot) and {names[1]} (ring)") if two else
             "the share of patients with each (95% CI)",
             "caption": "Items of fewer than 10 patients are not shown.", "chart": chart}]


RESULTS = {"characterize": characterize_results, "trajectory": trajectory_results, "pathways": pathways_results,
           "top": top_results}
