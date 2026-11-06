"""
Offline tests of the study card and the charts: the chart data the analyses add (Table 1, the band and numbers at
risk of survival curves, the propensity score distributions, the ROC curve and calibration, the definitions of the
concepts), the card built from every kind of study record (also broken, old and edited ones), the answer-level views,
the dark styles' scoping, and the chart library's arithmetic in Node.

    python -m pytest mes/tests/test_viz.py -q
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mes.tests.test_rwd import _synthetic_store  # noqa: E402
from mes.tools.result_schema import EvidenceItem, MESResult, Passage  # noqa: E402
from mes.tools.rwd import analyses, definitions  # noqa: E402
from mes.tools.rwd import stats as S  # noqa: E402
from mes.tools.rwd.compiler import Compiler  # noqa: E402
from mes.tools.rwd.concepts import ConceptIndex  # noqa: E402
from mes.tools.rwd.db import Store  # noqa: E402
from mes.tools.rwd.spec import StudySpec, validate  # noqa: E402
from mes.ui import answer_viz, study_card, study_results  # noqa: E402
from mes.ui import components as C  # noqa: E402
from mes.ui.theme import dark_scoped  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "ui" / "web"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("mimic_viz")
    with np.errstate(over="ignore"):        # pandas' NaN → NaT conversion of the synthetic deaths sometimes flags one
        icu, treated, early = _synthetic_store(root)
    store = Store(root)
    return store, ConceptIndex(store), icu


def _run(synth, spec):
    store, index, _icu = synth
    spec = StudySpec.from_dict(spec)
    errors, _ = validate(spec, index)
    assert not errors, errors
    cohort = Compiler(index, store).build(spec)
    return spec, cohort, analyses.run(spec, cohort, "sackett")


COMPARE = {"analysis": "compare", "base": "icu", "exposure": {"concept": "drug:testdrug"}, "outcome": "death_1y",
           "grace_h": 24, "covariates": ["sofa", "age", "gender"]}


# ============================================================================ the data the charts draw
def test_table1_is_exact_before_and_balanced_after_weighting(synth):
    spec, cohort, res = _run(synth, COMPARE)
    df, t1 = cohort.df, res["table1"]
    assert "testdrug" in t1["groups"][0] and t1["n"] == [int(df.treated.sum()), int((1 - df.treated).sum())]
    rows = {(r["variable"], r["label"]): r for r in t1["rows"] if not r.get("heading")}
    age = rows[("age", "Age")]
    mean1 = df.loc[df.treated == 1, "age"].mean()
    assert age["treated_raw"].startswith(f"{mean1:.1f} (")
    sofa = rows[("sofa", "SOFA score, first 24 h")]
    assert abs(sofa["smd_raw"]) > 0.3 and abs(sofa["smd_weighted"]) < 0.1        # sicker patients were treated
    men = rows[("gender", "M")]
    k = int((df.loc[df.treated == 1, "gender"] == "M").sum())
    assert men["treated_raw"] == f"{k:,} ({100 * k / int(df.treated.sum()):.1f}%)" and men["level"]
    assert any(r.get("heading") and r["label"] == "Sex" for r in t1["rows"])
    json.dumps(t1)


def test_small_cells_of_table1_are_not_shown():
    df = pd.DataFrame({"gender": ["M"] * 5 + ["F"] * 195, "age": np.linspace(20, 80, 200)})
    t = np.array([1] * 100 + [0] * 100)
    rows = analyses.baseline_table(df, ["gender", "age"], t)["rows"]
    men = next(r for r in rows if r["label"] == "M")
    assert men["treated_raw"] == "<10" and "weighted" not in json.dumps([k for r in rows for k in r if k.endswith("weighted")])


def test_survival_curves_carry_a_band_and_the_patients_at_risk(synth):
    _spec, cohort, res = _run(synth, COMPARE)
    curves = res["tables"]["km"]
    assert res["charts"]["km"]["offset_days"] == 1.0 and "landmark" in res["charts"]["km"]["start"]
    for name, c in curves.items():
        assert len(c["lo"]) == len(c["t"]) == len(c["hi"])
        assert all(lo - 1e-9 <= s <= hi + 1e-9 for lo, s, hi in zip(c["lo"], c["s"], c["hi"]))
        days = [d for d, _n in c["at_risk"]]
        assert days[0] == 0 and days == sorted(days) and days[-1] <= 364
        counts = [n for _d, n in c["at_risk"] if n is not None]
        assert counts[0] == c["n"] and counts == sorted(counts, reverse=True)
    assert S.risk_times(27) == [0, 7, 14, 21, 27] and S.risk_times(365) == [0, 90, 180, 270, 360]
    assert S.risk_times(0) == [0.0] and S.risk_times(28) == [0, 7, 14, 21, 28]


def test_at_risk_counts_below_ten_are_hidden():
    time = np.array([1.0] * 12 + [30.0] * 5)
    curves, _p = S.kaplan_meier(time, np.ones(17, int), np.array(["a"] * 17), horizon=30, detail=True)
    assert dict((d, n) for d, n in curves["a"]["at_risk"])[0] == 17
    assert all(n is None for d, n in curves["a"]["at_risk"] if d >= 2)
    plain, _ = S.kaplan_meier(time, np.ones(17, int), np.array(["a"] * 17), horizon=30)
    assert set(plain["a"]) == {"t", "s", "at_horizon", "n"}                  # the analyst's sandbox sees the same


def test_propensity_score_distributions_are_quantiles_only(synth):
    _spec, _cohort, res = _run(synth, COMPARE)
    ps = res["charts"]["ps"]
    assert len(ps["q"]) == len(ps["treated"]) == len(ps["control"]) == 39
    for side in ("treated", "control"):
        v = ps[side]
        assert v == sorted(v) and 0 <= v[0] and v[-1] <= 1
    assert np.median(ps["treated"]) > np.median(ps["control"])                 # treatment follows severity
    assert S.ps_distribution(np.array([0.2, 0.3]), np.array([1, 1])) is None


def test_prognostic_models_report_roc_and_calibration(synth):
    _spec, cohort, res = _run(synth, {"analysis": "prognostic", "base": "icu", "outcome": "death_28d",
                                      "covariates": ["sofa", "age", "lactate_max"]})
    roc, cal = res["charts"]["roc"], res["charts"]["calibration"]
    assert roc[0] == [0.0, 0.0] and roc[-1] == [1.0, 1.0] and len(roc) <= 60
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(roc, roc[1:]))
    assert sum(b["n"] for b in cal) == len(cohort.df) and all(b["n"] >= 10 for b in cal)
    overall = 100 * cohort.df.death_28d.mean()
    assert abs(sum(b["obs"] * b["n"] for b in cal) / len(cohort.df) - overall) < 0.05
    assert "calibration" not in S.discrimination(np.array([0, 1] * 20), np.linspace(0, 1, 40), bins=10)   # 4 per tenth


def test_definitions_say_what_each_concept_matches(synth):
    _store, index, _icu = synth
    spec = {"analysis": "compare", "include": [{"concept": "phe:septic_shock_dx"}, {"concept": "var:sepsis3"}],
            "exclude": [{"concept": "dx10:N18", "negate": False}], "exposure": {"concept": "drug:testdrug",
                                                                              "window_h": [0, 24]},
            "comparator": {"concept": "drugclass:hydrocortisone"}, "outcome": "death_28d", "variable": "lab:50813"}
    rows = definitions.describe(spec, index)
    by_role = {r["role"]: r for r in rows}
    assert [r["role"] for r in rows] == ["Inclusion", "Inclusion", "Exclusion", "Treatment / exposure", "Comparator",
                                         "Outcome", "Variable"]
    assert rows[0]["codes"] == {"ICD-10": ["R6521"], "ICD-9": ["78552"]} and "discharge" in rows[0]["detail"]
    assert by_role["Treatment / exposure"]["medications"] == ["testdrug 10mg"] and "0 to 24 h" in by_role[
        "Treatment / exposure"]["rule"]
    assert by_role["Comparator"]["codes"]["Ingredients"] == ["hydrocortisone"] and "systemic routes" in by_role[
        "Comparator"]["detail"]
    assert "28 days" in by_role["Outcome"]["detail"] and "dod" in by_role["Outcome"]["detail"]
    assert rows[1]["detail"].startswith("mimic-code sepsis3")
    assert definitions.describe({}, None) == [] and definitions.describe({"exposure": "junk"}, None) == []
    json.dumps(rows)


# ============================================================================ the study card
def _result(studies, evidence=()):
    return MESResult(question="q", mode="sackett", sources=["literature", "rwd"], plan={}, evidence=list(evidence),
                     citation_order=["R1"], rwd_studies=studies)


SPECS = [COMPARE,
         {"analysis": "rate", "base": "icu", "include": [{"concept": "var:sepsis3"}], "outcome": "death_28d",
          "stratify": ["careunit", "sofa"]},
         {"analysis": "utilization", "base": "icu", "exposure": {"concept": "drug:testdrug"}, "stratify": ["careunit"]},
         {"analysis": "distribution", "base": "icu", "variable": "lactate_max", "stratify": ["careunit"]},
         {"analysis": "survival", "base": "icu", "outcome": "death_1y", "stratify": ["sofa"]},
         {"analysis": "trend", "base": "icu", "outcome": "death_28d"},
         {"analysis": "prognostic", "base": "icu", "outcome": "death_28d", "covariates": ["sofa", "age"]},
         {"analysis": "association", "base": "icu", "outcome": "death_28d", "exposure": {"concept": "var:lactate_max"},
          "covariates": ["sofa"]},
         {"analysis": "association", "base": "icu", "outcome": "death_28d", "exposure": {"concept": "var:sepsis3"},
          "covariates": ["sofa", "age"]}]


@pytest.fixture(scope="module")
def studies(synth):
    out = []
    for i, spec in enumerate(SPECS):
        s, cohort, res = _run(synth, spec)
        out.append({"key": f"s{i}", "kind": "template", "status": "done", "evidence_id": "R1", "spec": s.to_dict(),
                    "result": json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))),
                    "sql": cohort.sql, "timings": {"total": 1.2}, "dataset": "MIMIC-IV v2.2",
                    "definitions": definitions.describe(s.to_dict(), synth[1])})
    return out


def _blocks(spec):
    for tab in spec["tabs"]:
        for b in tab["blocks"]:
            yield from (b.get("blocks") or [b])


def test_every_analysis_gets_a_card_with_its_charts(studies):
    from mes.ui.viz import clean
    expect = {"compare": {"arms", "km", "forest", "love", "overlap", "flow", "timeline"},
              "rate": {"dots", "flow"}, "utilization": {"dots"}, "distribution": {"hist", "ranges"},
              "survival": {"km"}, "trend": {"trend"}, "prognostic": {"roc", "calibration", "forest"},
              "association": {"forest"}}
    for study in studies:
        spec = study_card.card(_result([study]), study)
        assert spec and spec["cite"] == "1" and spec["kpis"] and spec["title"], study["spec"]["analysis"]
        charts = {b["chart"]["type"] for b in _blocks(spec) if b["type"] == "chart"}
        assert expect[study["spec"]["analysis"]] <= charts, (study["spec"]["analysis"], charts)
        for b in _blocks(spec):
            if b["type"] == "chart" and b["chart"]["type"] not in ("timeline", "steps", "overlap"):
                assert b["chart"]["table"]["rows"], (study["spec"]["analysis"], b["id"])        # the table view
        json.dumps(clean(spec))
    compare = study_card.card(_result([studies[0]]), studies[0])
    assert [t["id"] for t in compare["tabs"]] == ["results", "protocol", "cohort", "balance", "definitions"]
    assert any(b["type"] == "table" and b["id"] == "table1" and b.get("toggle") for b in _blocks(compare))
    checks = next(b for b in _blocks(compare) if b["type"] == "scorecard")["items"]
    assert {i["label"] for i in checks} >= {"New users only", "Covariate balance after weighting",
                                           "Robustness to unmeasured confounding (E-value)"}
    assert next(i for i in checks if i["label"] == "Covariate balance after weighting")["status"] == "ok"


def test_a_card_survives_broken_and_old_records(studies):
    rng = random.Random(4)
    assert study_card.card(_result([]), {"status": "failed"}) is None
    assert study_card.card(_result([]), {"status": "not_feasible", "spec": {"reason": "x"}}) is None
    old = json.loads(json.dumps(studies[0]))
    for key in ("table1", "charts"):
        old["result"].pop(key, None)
    old["result"]["diagnostics"]["ps_treated"] = [0.1, 0.3, 0.6]
    old["result"]["diagnostics"]["ps_control"] = [0.05, 0.2, 0.5]
    old.pop("definitions")
    old["result"].pop("card_data")
    spec = study_card.card(_result([old]), old)
    assert any(b["type"] == "chart" and b["chart"]["type"] == "overlap" and b["chart"].get("ranges") for b in _blocks(spec))
    assert any(b["type"] == "defs" for b in _blocks(spec))                   # from the library alone
    note = [b for b in spec["tabs"][0]["blocks"] if b["type"] == "notes" and "earlier version" in b["title"]]
    assert len(note) == 1 and "Table 1" in note[0]["items"][0] and "propensity" in note[0]["items"][0]
    unmarked = json.loads(json.dumps(studies[0]))
    unmarked["result"].pop("card_data")                  # stored before the mark but with the data: no note
    for study in studies + [unmarked]:                   # a study of this version: no note
        assert not any("earlier version" in str(b.get("title")) for b in _blocks(study_card.card(_result([study]), study)))
    # what a re-run adds: survival bands, the ROC curve, Table 1 of a yes/no exposure; nothing to a continuous
    # exposure's association or to a rate
    for i, expect in ((4, ["confidence bands"]), (6, ["ROC"]), (7, []), (8, ["Table 1"]), (1, [])):
        record = json.loads(json.dumps(studies[i]))
        for key in ("card_data", "charts", "table1"):
            record["result"].pop(key, None)
        notes = [b["items"][0] for b in _blocks(study_card.card(_result([record]), record)) if "earlier version" in str(b.get("title"))]
        assert [all(w in n for w in expect) for n in notes] == ([True] if expect else []), (i, notes)
    garbage = [None, "", 0, [], {}, "x", float("nan"), [None, "a"], {"a": None}, -1, 1e300]
    for _ in range(300):                                                       # random damage to random records
        record = json.loads(json.dumps(rng.choice(studies)))
        for _k in range(rng.randint(1, 6)):
            target = record if rng.random() < 0.3 else record["result"]
            if target and isinstance(target, dict):
                key = rng.choice(list(target))
                target[key] = rng.choice(garbage)
        out = study_card.card(_result([record]), record)                       # never raises
        assert out is None or isinstance(out, dict)


def test_analyst_studies_get_results_approach_review_and_steps():
    study = {"key": "an1", "kind": "analyst", "status": "done", "evidence_id": "R1", "llm": "claude",
             "code": "x = 1", "cells": [{"ok": True}, {"ok": False}],
             "steps": [{"action": "search", "seconds": 1.5, "plan": "find"}, {"action": "python", "seconds": 4.0},
                       {"action": "python", "seconds": 2.0, "plan": "fix"}],
             "review": {"verdict": "minor_issues"}, "reviews": [{"verdict": "minor_issues", "summary": "ok",
                                                                 "issues": [{"severity": "minor", "problem": "p", "fix": "f"}]}],
             "reproduction": {"reproduced": True, "detail": "same numbers"},
             "code_audit": [{"kind": "ICD code", "written": "A41%", "matches": "sepsis", "patients": 120}],
             "result": {"analysis": "custom", "title": "Transfusion", "n": 900, "estimates": [
                 {"label": "Transfusion", "measure": "rate", "value": 22.8, "lo": 22.5, "hi": 23.2, "n": 900, "events": 205,
                  "primary": True},
                 {"label": "RR", "measure": "RR", "value": 2.5, "lo": 2.4, "hi": 2.6}],
                 "tables": {"by unit": [{"unit": "MICU", "n": 400, "rate_%": 20.1, "lo_%": 18.0, "hi_%": 22.3},
                                        {"unit": "SICU", "n": "<10", "rate_%": 30.0, "lo_%": 25.0, "hi_%": 35.0}]},
                 "analyst": {"exposure": "septic shock", "limitations": ["one center"]}, "warnings": ["w"],
                 "attrition": [["all", 1000], ["adults", 900]]}}
    spec = study_card.card(_result([study]), study)
    assert [t["id"] for t in spec["tabs"]] == ["results", "approach", "review", "cohort", "steps"]
    bars = next(b for b in _blocks(spec) if b["type"] == "chart" and b["chart"]["type"] == "bars")
    col = next(c for c in bars["chart"]["columns"] if c["key"] == "rate_%")
    assert (col["lo"], col["hi"], col["unit"], bars["chart"]["default"]) == ("lo_%", "hi_%", "%", "rate_%")
    steps = next(b for b in _blocks(spec) if b["type"] == "chart" and b["chart"]["type"] == "steps")["chart"]["rows"]
    assert [s["ok"] for s in steps] == [True, True, False]
    assert {b["text"] for b in spec["badges"]} >= {"◐ review: minor issues", "✓ reproduced in a fresh session"}


def test_numbers_of_the_card_read_like_the_answer():
    assert study_results.compact({"measure": "RD", "value": 6.04, "lo": 0.12, "hi": 10.7}) == ("+6.0 pp", "+0.1 to +10.7")
    assert study_results.compact({"measure": "HR", "value": 1.2155, "lo": 1.03, "hi": 1.43, "p": 0.0204}) == \
        ("1.22", "1.03–1.43 · p 0.020")
    assert study_results.compact({"measure": "rate", "value": 18.4}) == ("18.4%", "")
    assert study_results.compact({"measure": "OR", "value": None}) == ("—", "")
    assert study_results.plain_name("Hydrocortisone (systemic)") == "Hydrocortisone"
    assert study_results.short("Death within 28 days: risk ratio (IPTW)", "Death within 28 days") == "Risk ratio (IPTW)"


# ============================================================================ the answer
def _answer():
    ev = [EvidenceItem(id="L1", source_type="literature", title="Meta", year=2020, design="Meta-analysis", evidence_level=1,
                       meta={"landmark": True, "cites_per_year": 120.0}),
          EvidenceItem(id="L2", source_type="literature", title="RCT", year=2015, design="Randomized controlled trial",
                       evidence_level=2, meta={"cites_per_year": "12.5"}),
          EvidenceItem(id="T1", source_type="trial", title="Trial", year=2018, design="Phase 3 randomized trial",
                       evidence_level=2, meta={"enrollment": "800"}),
          EvidenceItem(id="K1", source_type="kg", title="A treats B", design="Curated relation", evidence_level=5,
                       meta={"edges": [["A", "treats", "B"], ["B", "causes", "C"], ["bad"]], "reliability": "High"}),
          EvidenceItem(id="R1", source_type="rwd", title="MIMIC", design="Target trial emulation", evidence_level=3,
                       meta={"n": 5000}, passages=[Passage(text="x")])]
    return MESResult(question="q", mode="osler", sources=["literature"], plan={}, evidence=ev,
                     citation_order=["L1", "T1", "K1", "R1"], created_at="2026-11-05T10:00:00",
                     verification={"n_numbers": 10, "n_numbers_grounded": 9, "citation_coverage": 0.8,
                                   "counts": {"unsupported_number": 1}})


def test_the_glance_map_and_network_of_an_answer():
    result = _answer()
    g = answer_viz.glance(result, "t1")
    kinds = [c["kind"] for c in g["cards"]]
    assert g["anchor"] == "mes-t1" and kinds == ["sources", "grounding", "levels", "stat"]
    levels = {r["tier"]: r["n"] for r in g["cards"][2]["rows"]}
    assert levels == {1: 1, 2: 1, 3: 1, 4: 0, 6: 1} and g["cards"][1]["flagged"] == 1
    m = answer_viz.landscape(result, "t1")
    by = {it["title"]: it for it in m["items"]}
    assert set(by) == {"Meta", "RCT", "Trial", "MIMIC"} and by["MIMIC"]["year"] == 2026
    assert by["Meta"]["cited"] and not by["RCT"]["cited"] and by["Meta"]["cite_id"] == "mes-t1-L1" and by["RCT"]["size"] == 12.5
    net = answer_viz.network([e for e in result.evidence if e.source_type == "kg"])
    assert len(net["edges"]) == 2 and {n["id"] for n in net["nodes"]} == {"a", "b", "c"}
    assert answer_viz.glance(MESResult(question="q", mode="osler", sources=[], plan={}, evidence=[]), "x") is None
    chip = C.chip(1, result.evidence[0], "mes-t1-L1")
    assert 'data-tier="1"' in chip and 'href="#mes-t1-L1"' in chip


def test_tildes_never_become_strikethrough():
    assert C.esc("95%CI 0.82~1.10, P=0.50") == "95%CI 0.82&#126;1.10, P=0.50"
    assert "~" not in C.answer_html("RR 0.93 (0.88~0.99) [L1]", {}, [], [], "a")


def test_dark_styles_apply_only_to_a_page_drawn_dark():
    css = (":root { --a: 1; }\n.x, .y:has(.a, .b) { color: red; }\n@media (max-width: 9px) { .z { top: 0; } }\n"
           "@keyframes k { to { opacity: 0; } }\n/* note { } */ html body .w { left: 0; }")
    out = dark_scoped(css)
    assert "html.mes-dark {--a: 1;}" in out
    assert ":where(html.mes-dark) .x, :where(html.mes-dark) .y:has(.a, .b) {color: red;}" in out
    assert "@media (max-width: 9px) {:where(html.mes-dark) .z {top: 0;}}" in out
    assert "@keyframes k { to { opacity: 0; } }" in out and "html.mes-dark body .w" in out and "note" not in out
    real = dark_scoped((Path(__file__).resolve().parents[1] / "ui" / "style_dark.css").read_text())
    assert real.count("{") == real.count("}") and all(line.startswith(("html.mes-dark", ":where(html.mes-dark)", "@"))
                                                      for line in real.splitlines() if line and not line[0].isspace())


# ============================================================================ the chart library (Node)
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_chart_library_parses_and_its_arithmetic_is_right(tmp_path):
    from mes.ui import viz
    module = viz._js() + ("\nexport { ticks, logTicks, nullDomain, fmt, signed, stepAt, quantileDensity, cdf, scale, fitLines, "
                          "fitText };\n")
    (tmp_path / "viz.mjs").write_text(module, encoding="utf-8")
    (tmp_path / "check.mjs").write_text("""
import { ticks, logTicks, nullDomain, fmt, signed, stepAt, quantileDensity, cdf, scale, fitLines, fitText } from "./viz.mjs";
// SVG text 6 px per character (the labels are measured in the browser)
const el = () => ({ textContent: "", setAttribute() {}, appendChild(c) { return c; }, remove() {},
                    getComputedTextLength() { return this.textContent.length * 6; } });
globalThis.document = { createElementNS: () => el() };
const svg = el();
const out = {
  lines: [fitLines(svg, "Systematic reviews · guidelines", "t", 120), fitLines(svg, "Randomized trials", "t", 120),
          fitLines(svg, "Reviews · case reports · other", "t", 120), fitLines(svg, "an unbreakablewordthatisfartoolong x", "t", 60)],
  cut: [fitText(svg, "Charlson comorbidity index", "t", 60), fitText(svg, "Age", "t", 60)],
  ticks: ticks(0, 27, 5), logs: logTicks(0.6, 1.7), wide: logTicks(0.2, 5), dom: nullDomain([1.2, 0.9, 1.5], 1, true),
  lin: nullDomain([-2, 6], 0, false), fmt: [fmt(1234.5), fmt(-0.004), fmt(12.345), fmt(null)], signed: [signed(6.04), signed(-2.9)],
  step: [stepAt([0, 1, 5], [1, 0.9, 0.7], 3), stepAt([0, 1, 5], [1, 0.9, 0.7], 9), stepAt([0, 1, 5], [1, 0.9, 0.7], -1)],
  cdf: [cdf([0.1, 0.5, 0.9], [1, 2, 3], 2.5), cdf([0.1, 0.5, 0.9], [1, 2, 3], 9)],
  dens: quantileDensity([0.1, 0.5, 0.9], [1, 2, 3], 0, 4, 8).map((p) => Math.round(p[1] * 100) / 100),
  log: scale(0.5, 2, 0, 100, true)(1),
};
console.log(JSON.stringify(out));
""", encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "check.mjs")], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["ticks"] == [0, 10, 20] and out["logs"] == [0.67, 0.8, 1, 1.25, 1.5]
    assert out["wide"] == [0.25, 0.5, 1, 2, 4]
    lo, hi = out["dom"]
    assert lo < 0.9 and hi > 1.5 and abs(lo * hi - 1) < 1e-9                  # symmetric around 1 on a log scale
    assert out["lin"][0] == -out["lin"][1] and out["lin"][1] > 6
    assert out["fmt"] == ["1,235", "−0.004", "12.3", "—"] and out["signed"] == ["+6.0", "−2.9"]
    assert out["step"] == [0.9, 0.7, 1] and out["cdf"] == [0.7, 0.9] and abs(out["log"] - 50) < 1e-9
    assert max(out["dens"]) == 1 and out["dens"][0] == 0 and out["dens"][-1] == 0
    # a long label on two lines of the room it has, broken at its " · " (the dot dropped), never cut off unseen
    assert out["lines"][:3] == [["Systematic reviews", "guidelines"], ["Randomized trials"],
                                ["Reviews", "case reports · other"]]
    assert all(len(line) * 6 <= 60 and line.endswith("…") for line in out["lines"][3][:1])
    assert out["cut"] == ["Charlson…", "Age"]


# ============================================================================ the design editor's table
def test_the_criteria_table_round_trips_and_refuses_what_it_cannot_mean():
    from mes.ui import design_rows as D
    spec = {"include": [{"concept": "var:sepsis3"}, {"concept": "var:lactate_max", "op": "between", "value": [2, 4]}],
            "exclude": [{"concept": "phe:ckd", "negate": True, "window_h": [-72, 24]}],
            "exposure": {"concept": "drugclass:hydrocortisone", "window_h": [0, 24]},
            "comparator": {"concept": "var:careunit", "op": "=", "value": "MICU"}}
    rows = D.rows_of(spec)
    assert [r["Role"] for r in rows] == ["Inclusion", "Inclusion", "Exclusion", "Treatment / exposure", "Comparator"]
    assert rows[1]["Value"] == "2–4" and rows[2]["Without"] and (rows[3]["From (h)"], rows[3]["To (h)"]) == ("0", "24")
    assert rows[0]["Comparison"] == D.NONE and rows[0]["From (h)"] == ""
    back, errors = D.criteria_of(rows)
    assert not errors
    assert back["include"][1]["value"] == [2.0, 4.0] and back["exclude"][0]["negate"] is True
    assert back["exposure"]["window_h"] == [0.0, 24.0] and back["comparator"]["value"] == "MICU"
    assert D.criteria_of([{"Role": None, "Concept": None, "Comparison": None, "Value": None, "From (h)": float("nan"),
                           "To (h)": None, "Without": False}]) == ({"include": [], "exclude": [], "exposure": None,
                                                                   "comparator": None}, [])
    bad = [{"Role": "Inclusion", "Concept": "sepsis"},                                    # not a concept id
           {"Role": "Inclusion", "Concept": "var:age", "Comparison": ">", "Value": "old"},  # not a number
           {"Role": "Inclusion", "Concept": "var:age", "Comparison": "between", "Value": "5"},
           {"Role": "Inclusion", "Concept": "var:age", "Value": "5"},                      # a value without comparison
           {"Role": "Exclusion", "Concept": "phe:ckd", "From (h)": 24, "To (h)": 0},       # a window that ends first
           {"Role": "Treatment / exposure", "Concept": "drug:a"}, {"Role": "Treatment / exposure", "Concept": "drug:b"},
           {"Role": "Nobody", "Concept": "var:age"}]
    _out, errors = D.criteria_of(bad)
    assert len(errors) == 7 and all(e.startswith("row ") for e in errors)
    out, errors = D.criteria_of([{"Role": "Inclusion", "Concept": "var:age", "Comparison": "between", "Value": "40 to 60"},
                                 {"Role": "Exclusion", "Concept": "phe:ckd", "From (h)": -24, "To (h)": None}])
    assert not errors and out["include"][0]["value"] == [40.0, 60.0] and out["exclude"][0]["window_h"] == [-24.0, None]


# ============================================================================ the study on paper
def test_reports_draw_the_card_as_svg(studies):
    from mes.ui import exports, study_report
    for study in studies:
        result = _result([study])
        page = study_report.report_html(result, study)
        assert page.startswith("<!doctype html>") and "<svg" in page and "<script" not in page
        assert "Protocol" in page and "SELECT" in page                         # its protocol and its SQL
    compare = studies[0]
    answer = exports.report_html(_result([compare]))
    assert answer.count("<svg") >= 4 and "Real-world data studies" in answer
    assert "Protocol" not in answer                                            # the word names edited versions there
    assert study_report.report_html(_result([]), {"status": "failed"}) == ""


def _damage(obj, rng, rate=0.08):
    """``obj`` with nested values replaced at random (types a stored record could end up with)."""
    garbage = [None, "", 0, -1, 1e300, float("nan"), [], {}, "x", [None], [[1]], {"a": 1}, True]
    if isinstance(obj, dict):
        return {k: (rng.choice(garbage) if rng.random() < rate else _damage(v, rng, rate)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rng.choice(garbage) if rng.random() < rate else _damage(v, rng, rate) for v in obj]
    return obj


def test_deeply_damaged_records_still_give_a_card(studies):
    from mes.ui.viz import clean
    rng = random.Random(11)
    cards = 0
    for _ in range(200):
        record = _damage(json.loads(json.dumps(rng.choice(studies))), rng)
        record.update(status="done", kind="template")          # a card is expected
        spec = study_card.card(_result([record]), record)
        if spec is not None:
            cards += 1
            json.dumps(clean(spec))
    assert cards >= 190                                       # a damaged part costs its tab, not the card
