"""
The card of a "patients like this" study (ui/study_card.py): the outcome of the most similar stays beside the whole
population's and by treatment among them, the risk a model predicts for the patient, what similar patients received
(Results); the patient beside them characteristic by characteristic (Profile); where the patient sits among all stays
(Map). Pure functions of the stored record; every chart carries its numbers as a table.
"""

from mes.tools.rwd import library as L
from mes.tools.rwd.patient import short
from mes.ui.study_results import num, table, text_of


# outcomes in a few words, for the card's key numbers
OUTCOME_SHORT = {"death_hosp": "In-hospital death", "death_icu": "ICU death", "death_28d": "28-day death",
                 "death_30d": "30-day death", "death_90d": "90-day death", "death_1y": "1-year death", "aki_7d": "AKI by day 7",
                 "aki_stage2_7d": "AKI stage ≥ 2", "rrt_any": "RRT", "invasive_vent_any": "Ventilation",
                 "icu_readmit_30d": "ICU readmission", "hosp_readmit_30d": "Readmission", "los_icu_days": "ICU stay",
                 "los_hosp_days": "Hospital stay"}


def _rate_row(e, level):
    return {"level": level, "value": num(e.get("value")), "lo": num(e.get("lo")), "hi": num(e.get("hi")),
            "n": e.get("n"), "events": e.get("events")}


def _pct(v):
    v = num(v)
    return "—" if v is None else f"{v:.1f}%"


def outcome_block(spec, res):
    """The outcome of the similar patients, of everyone, and of the similar patients by treatment (one chart)."""
    est = res.get("estimates") or []
    rates = [e for e in est if e.get("measure") == "rate"]
    if not rates:
        return None
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("the outcome",))[0]
    rows = []
    for e in rates:
        label = str(e.get("label", ""))
        level = label.split(": ", 1)[1] if ": " in label else label
        rows.append(_rate_row(e, level[:1].upper() + level[1:]))
    overall = next((e for e in rates if "all " in str(e.get("label", "")) and "population" in str(e.get("label", ""))), None)
    t = table([("level", "Patients", False), ("n", "Patients (n)", True), ("events", "Events", True),
               ("rate", "Rate (95% CI)", True)],
              [{"level": r["level"], "n": f"{int(r['n']):,}" if num(r.get("n")) is not None else "—",
                "events": f"{int(r['events']):,}" if num(r.get("events")) is not None else "<10",
                "rate": f"{_pct(r['value'])} ({_pct(r['lo'])}–{_pct(r['hi'])})" if r["value"] is not None else "—"}
               for r in rows])
    return {"type": "chart", "id": "similar_outcome", "title": outcome, "subtitle": "patients like this, everyone, and "
            "patients like this by treatment — rate with 95% CI; the line: everyone",
            "caption": "By treatment the rates are crude: treatment was not assigned at random.",
            "chart": {"type": "dots", "groups": [{"variable": "Patients", "rows": rows}], "unit": "%", "measure": "Rate",
                      "axis": f"{outcome} (%)", "overall": {"value": num(overall.get("value")), "label": "everyone"}
                      if overall else None, "table": t}}


def received_block(res):
    rows = [r for r in (res.get("tables") or {}).get("treatments") or [] if isinstance(r, dict)]
    if not rows:
        return None
    t = table([("label", "Treatment", False), ("similar", "Patients like this", True), ("cohort", "Everyone", True)],
              [{"label": r.get("label"), "similar": "<10" if r.get("similar") is None and r.get("n") else
                f"{_pct(r.get('similar'))} ({_pct(r.get('lo'))}–{_pct(r.get('hi'))})" if r.get("lo") is not None else
                _pct(r.get("similar")), "cohort": _pct(r.get("cohort"))} for r in rows])
    return {"type": "chart", "id": "received", "title": "What patients like this received",
            "subtitle": "in the first 24 h — patients like this (dot, 95% CI) and everyone (ring)",
            "chart": {"type": "received", "rows": [{"label": r.get("label"), "value": num(r.get("similar")),
                                                     "lo": num(r.get("lo")), "hi": num(r.get("hi")),
                                                     "cohort": num(r.get("cohort"))} for r in rows], "table": t}}


def risk_block(res):
    e = next((x for x in res.get("estimates") or [] if x.get("measure") == "risk"), None)
    if not e or num(e.get("value")) is None:
        return None
    auc = num((res.get("diagnostics") or {}).get("risk_auc"))
    return {"type": "kv", "id": "risk", "title": "The risk a model predicts for this patient",
            "rows": [["Predicted", text_of(e)],
                     ["Model", "logistic regression of the population on the patient's characteristics" +
                      (f"; cross-validated c-statistic {auc:.2f}" if auc is not None else "")],
                     ["Read with", "the observed outcome of the patients like this above: the model averages over "
                                   "everyone, the similar patients are the nearest ones"]]}


def results(spec, res):
    blocks = [b for b in (outcome_block(spec, res), risk_block(res), received_block(res)) if b]
    rest = [e for e in res.get("estimates") or [] if e.get("measure") == "RD"]
    if rest:
        blocks.append({"type": "table", "id": "similar_rd", "title": "Crude difference by treatment among patients like this",
                       **table([("label", "Estimate", False), ("text", "Value", True)],
                               [{"label": str(e.get("label", "")).split(": ", 1)[-1], "text": text_of(e)} for e in rest])})
    return blocks


def profile_blocks(study):
    res = study.get("result") or {}
    rows = [r for r in (res.get("tables") or {}).get("profile") or [] if isinstance(r, dict)]
    if not rows:
        return []
    chart_rows = []
    for r in rows:
        row = {"label": r.get("label"), "kind": r.get("kind"), "unit": r.get("unit") or "", "weight": num(r.get("weight")),
               "must": bool(r.get("must")), "patient": r.get("patient"), "similar_text": r.get("similar"),
               "cohort_text": r.get("cohort"), "missing": num(r.get("missing_similar"))}
        if r.get("kind") == "numeric":
            row.update(range=r.get("range"), similar=[num(r.get("q1")), num(r.get("median")), num(r.get("q3"))],
                       cohort=r.get("cohort_iqr"))
        else:
            row.update(share_similar=num(r.get("share_similar")), share_cohort=num(r.get("share_cohort")))
        chart_rows.append(row)
    t = table([("label", "Characteristic", False), ("patient", "This patient", True), ("similar", "Patients like this", True),
               ("cohort", "Everyone", True)],
              [{"label": r.get("label"), "patient": str(r.get("patient")), "similar": r.get("similar"),
                "cohort": r.get("cohort")} for r in rows])
    d = res.get("diagnostics") or {}
    sub = "median (IQR) or % with the patient's value"
    if num(d.get("median_similarity")) is not None:
        sub += f" · median similarity {100 * d['median_similarity']:.0f}% (1 = identical)"
    return [{"type": "chart", "id": "profile", "title": "This patient beside patients like this", "subtitle": sub,
             "caption": "Grey band: the interquartile range of everyone; coloured band: of the patients like this; "
                        "diamond: the patient.",
             "chart": {"type": "profile", "rows": chart_rows, "table": t}}]


def map_blocks(study):
    res = study.get("result") or {}
    m = (res.get("charts") or {}).get("map")
    if not isinstance(m, dict) or not m.get("hexes"):
        return []
    t = table([("x", "Component 1", True), ("y", "Component 2", True), ("n", "Stays", True), ("k", "Of them similar", True)],
              [{"x": f"{h['x']:.2f}", "y": f"{h['y']:.2f}", "n": f"{h['n']:,}", "k": "<10" if h.get("k") is None else
                f"{h['k']:,}"} for h in sorted(m["hexes"], key=lambda h: -h["n"])[:60]])
    axes = m.get("axes") or ["", ""]
    return [{"type": "chart", "id": "simmap", "title": "Where this patient sits among everyone",
             "subtitle": "the patient's characteristics in two dimensions (principal components)" +
                         (f" — across: {axes[0]}; up: {axes[1]}" if axes[0] else ""),
             "caption": "Hexagons: stays (at least 10 each, darker = more); ellipse: the patients like this (mean ± 2 SD); "
                        "star: the patient. A map of aggregates — no single stay is drawn.",
             "chart": {"type": "simmap", **m, "table": t}}]


def kpis(study):
    res = study.get("result") or {}
    spec = study.get("spec") or {}
    est = res.get("estimates") or []
    d = res.get("diagnostics") or {}
    out = []
    k, pool = num(d.get("k")), num(d.get("n_pool"))
    if k is not None:
        sim = num(d.get("median_similarity"))
        out.append({"label": "Patients like this", "value": f"{int(k):,}",
                    "note": f"of {int(pool):,}" + (f" · similarity {100 * sim:.0f}%" if sim is not None else "")
                    if pool is not None else ""})
    outcome = OUTCOME_SHORT.get(spec.get("outcome")) or L.OUTCOMES.get(spec.get("outcome"), ("Outcome",))[0]
    near = next((e for e in est if e.get("primary")), None)
    everyone = next((e for e in est if e.get("measure") == "rate" and "population" in str(e.get("label", ""))), None)
    risk = next((e for e in est if e.get("measure") == "risk"), None)
    for e, label, tone in ((near, f"{outcome} · patients like this", "primary"), (everyone, "Everyone", ""),
                           (risk, "Predicted for the patient", "")):
        if e is None or num(e.get("value")) is None:
            continue
        if e.get("measure") == "median":
            out.append({"label": label, "value": f"{num(e['value']):.1f}", "unit": e.get("unit", ""), "tone": tone,
                        "ci": f"IQR {num(e.get('lo')):.1f}–{num(e.get('hi')):.1f}" if num(e.get("lo")) is not None else ""})
            continue
        lo, hi = num(e.get("lo")), num(e.get("hi"))
        out.append({"label": label, "value": f"{num(e['value']):.1f}", "unit": "%", "tone": tone,
                    "ci": f"95% CI {lo:.1f}–{hi:.1f}" if lo is not None and hi is not None else "",
                    "bar": {"value": num(e["value"]), "lo": lo, "hi": hi, "min": 0, "max": 100}})
    return out[:4]


def title(study):
    spec, res = study.get("spec") or {}, study.get("result") or {}
    who = short(spec.get("profile")) or "the described patient"
    outcome = L.OUTCOMES.get(spec.get("outcome"), ("",))[0]
    population = "; ".join(c.get("label") or c.get("concept", "") for c in spec.get("include") or [] if isinstance(c, dict))
    return f"Patients like this: {who}", " · ".join(x for x in (outcome, population or res.get("population", "")) if x)


def protocol_rows(spec):
    """The described patient as rows of the protocol: each characteristic, its weight, whether it must match."""
    from mes.tools.rwd.report import profile_text
    rows = [["Patient", profile_text(spec) or "—"]]
    weights = [f"{str(f.get('concept'))[4:]} ×{f['weight']:g}" for f in spec.get("profile") or []
               if isinstance(f, dict) and num(f.get("weight")) not in (None, 1.0)]
    if weights:
        rows.append(["Weights", ", ".join(weights)])
    rows.append(["Similar patients", f"the {int(num(spec.get('k')) or 200):,} nearest by a weighted distance over the "
                                     "patient's characteristics (numbers in robust SDs, capped at 3; yes/no and "
                                     "categories by agreement; a missing value counts half)"])
    return rows
