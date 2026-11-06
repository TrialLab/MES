"""
Real-world data the way the page names it: what MES does with real-world data says "real-world data" / "RWD"; the
database is named where it is chosen (the source toggle), in its status and on the results computed from it — each
study's own dataset, so another database names itself. And the app as run.sh starts it (mes/app.py): MES's landing
under the header named MES.

    python -m pytest mes/tests/test_rwd_names.py -q
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from mes.tools import rwd  # noqa: E402
from mes.ui import landing, page, study_card, teams, trial_compare  # noqa: E402

DB = re.compile(r"MIMIC", re.I)


def test_database_of_a_study():
    assert rwd.database_of({"dataset": "MIMIC-IV v2.2"}) == "MIMIC-IV" == rwd.DATABASE
    assert rwd.database_of({"dataset": "eICU-CRD v2.0"}) == "eICU-CRD"
    assert rwd.database_of({}) == rwd.database_of(None) == rwd.database_of("x") == rwd.DATABASE
    assert rwd.about("MIMIC-IV")["centers"] and rwd.about("eICU-CRD") == {"where": "", "centers": "", "limits": ""}


def test_what_mes_does_names_no_database():
    rwd_agent = teams.catalog()["agents"]["rwd"]
    texts = [landing.hero(), landing.capabilities({}), landing.how_it_works(),
             page.STAGE_LABELS["rwd"], page.SEARCH_NAMES["rwd"], page.DONE_TEXT["rwd"], page.INTERRUPTED,
             teams.SHORT["rwd"], rwd_agent["subtitle"], rwd_agent["desc"], teams.catalog()["agents"]["synthesizer"]["llm"],
             json.dumps([t["description"] for t in teams.builtin_templates()]), trial_compare.TOOL]
    assert not [t for t in texts if DB.search(t)]
    assert "real-world data" in landing.hero() and "RWD study" in landing.how_it_works()
    # the source is "Real-world data"; the database is named where it is chosen (its menu) and in its status
    assert page.SOURCE_OPTIONS["rwd"].endswith("Real-world data") and page.RWD_STATUS.startswith(rwd.DATABASE)
    assert [d["name"] for d in rwd.DATABASES.values()] == ["MIMIC-IV", "INSIGHT"]
    assert [k for k, d in rwd.DATABASES.items() if d["supported"]] == ["mimic"]


def test_progress_messages_old_and_new():
    for message in ("RWD: designing the study", "RWD analyst · step 3: designing the study",
                    "RWD analyst (Claude): designing the study", "MIMIC-IV analyst · step 2: designing the study"):
        assert page.PROGRESS_PREFIX.sub("", message) == "designing the study"


def _study(dataset, analysis="catalog"):
    rows = [{"term": "ecmo", "concept": "px10:5A15", "label": "ECMO", "kind": "procedure", "patients": 120}]
    return {"key": "k", "status": "done", "dataset": dataset, "kind": "template",
            "spec": {"analysis": analysis, "base": "icu"},
            "result": {"analysis": analysis, "title": "What the database records about ecmo", "n": None,
                       "tables": {"concepts": rows, "overview": [{"what": "patients", "n": 1000}]}, "estimates": []}}


def test_results_name_their_own_database():
    for dataset, name in (("MIMIC-IV v2.2", "MIMIC-IV"), ("eICU-CRD v2.0", "eICU-CRD")):
        card = study_card.card(None, _study(dataset))
        assert card["dataset"] == dataset and card["tabs"][0]["label"] == f"What {name} has"
        titles = [b.get("title", "") for b in card["tabs"][0]["blocks"]]
        assert f"{dataset} at a glance" in titles and f"What {name} records" in titles
    other = json.dumps(study_card.card(None, _study("eICU-CRD v2.0")))
    assert not re.search(r"MIMIC|center", other)                   # nothing of MIMIC-IV on another database's card
    mimic = study_card.card(None, _study("MIMIC-IV v2.2"))
    assert {"text": "observational · one academic center", "kind": "neutral"} in mimic["badges"]
    spec = trial_compare.card({"rows": [], "n_trial": 100, "n_cohort": 50, "nct": "NCT00000001"}, "eICU-CRD v2.0")
    assert spec["dataset"] == "ClinicalTrials.gov · eICU-CRD v2.0" and "MIMIC" not in json.dumps(spec)
    assert "MIMIC-IV patients" in json.dumps(trial_compare.card({"rows": [], "nct": "NCT00000001"}))


def test_the_app():
    """mes/app.py (run.sh): the header named MES above MES's landing, no exception."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "mes" / "app.py"), default_timeout=180)
    app.run()
    assert [e.value for e in app.exception] == []
    shown = [m.value for m in app.markdown if not m.value.lstrip().startswith("<style")]
    assert any("<strong>MES</strong><span>Multi-agent Evidence Synthesis</span>" in m for m in shown)
    assert any("What would you like to <em>investigate</em>?" in m for m in shown)
