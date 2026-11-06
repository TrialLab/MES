"""
Smoke tests of the question boxes — the MES landing box and the follow-up box — and of the data contract
between ``ui/composer.py`` and its browser half.

The component's JavaScript cannot run here: AppTest has no browser, so the stub in
``_composer_app.py`` takes its place. What is covered is the Python side — each call site renders,
hands over the arguments it means to, and does the right thing with a sent question. The browser
half (typing, the mode menu, Enter, the meter) is driven in a real browser by
``scripts/ui_browser_monkey.py``.

    python -m pytest mes/tests/test_composer.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # repository root

APP = str(Path(__file__).with_name("_composer_app.py"))

# What the browser half reads out of `data`. A later version may add keys — dropping one, or
# renaming it, breaks a box without any Python error, which is what this set is here to catch.
DATA_KEYS = {"key", "mode", "modes", "placeholder", "value", "nonce", "disabled", "autofocus",
             "compact", "send_title", "default_question", "routed", "keep", "dark",
             "label", "depth", "depths", "depth_label"}

MODE_KEYS = {"key", "label", "desc", "time"}

# What a sent question, an edit, a menu action and a depth change come back under. A box that loses one
# of these stops reporting something its call site acts on, without any Python error.
RESULT_KEYS = {"submit", "edited", "action", "depth"}


def run(site, submit=None, edited=None, action=None, depth=None, **state):
    """Run the harness at one site and give back the finished app."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP, default_timeout=120)
    app.session_state["site"] = site
    app.session_state["submit_payload"] = submit
    app.session_state["edited_payload"] = edited
    app.session_state["action_payload"] = action
    app.session_state["depth_payload"] = depth
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    assert [e.value for e in app.exception] == []
    return app


def calls(app, key=None):
    """The `data` of every composer call, optionally only the box with this key."""
    made = app.session_state.get("composer_calls") or []
    return [c for c in made if key is None or c["key"] == key]


def sent(question="Does metformin lower mortality?", mode="snow", at=1700000000000, depth=None):
    return {"question": question, "mode": mode, "depth": depth, "at": at}


# ----------------------------------------------------------------- the data contract
def test_every_argument_reaches_the_component():
    data = calls(run("probe"))[0]
    assert DATA_KEYS <= set(data), f"missing from the hand-over: {DATA_KEYS - set(data)}"
    assert data["key"] == "probe_box"
    assert data["mode"] == "sackett"
    assert data["placeholder"] == "Ask a medical question…"
    assert data["value"] == "seeded question"
    assert data["nonce"] == "nonce-1"
    assert (data["disabled"], data["autofocus"], data["compact"]) == (True, True, True)
    assert data["send_title"] == "Go"
    assert data["default_question"] == "A question in grey"
    assert data["keep"] is True
    assert data["routed"] == "seeded question"
    assert isinstance(data["dark"], bool)
    assert (data["label"], data["depth_label"]) == ("Task", "Depth")
    assert data["depth"] == "deep"


def test_each_mode_carries_what_the_menu_draws():
    for key in ("modes", "depths"):
        items = [m for m in calls(run("probe"))[0][key] if m.get("kind", "mode") == "mode"]
        assert items
        for item in items:
            assert MODE_KEYS <= set(item), f"missing from a {key} entry: {MODE_KEYS - set(item)}"
            assert all(isinstance(item[k], str) and item[k] for k in MODE_KEYS)


def test_a_box_without_depths_draws_only_the_one_dropdown():
    """MES's own boxes: the first dropdown *is* the depth, so there is no second one."""
    assert calls(run("landing"), "mes_composer")[0]["depths"] == []


def test_nothing_sent_gives_back_no_question():
    result = run("probe").session_state["probe_result"]
    assert set(result) == RESULT_KEYS
    assert all(value is None for value in result.values())


def test_a_sent_question_is_passed_through_unchanged():
    payload = sent()
    assert run("probe", submit=payload).session_state["probe_result"]["submit"] == payload


def test_the_menu_can_carry_a_heading_and_an_action_alongside_the_modes():
    """What a task picker needs: group headings, and a last item that runs something."""
    items = calls(run("probe"))[0]["modes"]
    assert [i.get("kind") for i in items if i.get("kind")] == ["group", "action"]
    assert {i["key"] for i in items if i.get("kind") != "group"} >= {"osler", "sackett", "snow"}


# ----------------------------------------------------------------- the MES landing box
def test_landing_box_renders():
    data = calls(run("landing"), "mes_composer")
    assert len(data) == 1
    assert data[0]["mode"] == "osler"                     # config.DEFAULT_MODE
    assert [m["key"] for m in data[0]["modes"]] == ["osler", "sackett", "snow"]
    assert data[0]["compact"] is False
    assert data[0]["autofocus"] is True
    assert data[0]["default_question"]                    # the question in grey can be sent as it is


def test_landing_box_without_a_source_cannot_send():
    data = calls(run("landing", _keep_mes_sources=[]), "mes_composer")[0]
    assert data["disabled"] is True
    assert "evidence source" in data["send_title"]


def test_landing_submit_starts_the_question_in_the_chosen_mode():
    app = run("landing", submit=sent(mode="snow"))
    thread = app.session_state["mes_thread"]
    assert len(thread) == 1
    assert thread[0]["question"] == "Does metformin lower mortality?"
    assert thread[0]["mode"] == "snow"
    assert app.session_state["mes_mode"] == "snow"


def test_the_source_bar_opens_literature_and_real_world_data():
    """Literature opens to PubMed, its open-access full text and the PDF library (all on); real-world data to MIMIC-IV
    (on, like every source of the default team, where its store is built) and the databases not connected yet
    (locked, off). What is switched off goes with the question."""
    from mes.tools import rwd_tools
    rwd = ["rwd"] if rwd_tools.available() else []
    app = run("landing")
    boxes = {c.key: (c.value, c.disabled) for c in app.checkbox if c.key and c.key.startswith("mes_src_")}
    assert [boxes[f"mes_src_lit_{p}"] for p in ("pubmed", "fulltext", "pdfs")] == [(True, False)] * 3
    assert boxes["mes_src_db_insight"] == (False, True)
    assert boxes["mes_src_db_mimic"] == (bool(rwd), not rwd)                   # real-world data: on where it can run
    assert app.pills(key="mes_src_mid").value == ["trials", "kg"]
    app.checkbox(key="mes_src_lit_pubmed").uncheck().run()
    assert app.session_state["mes_sources"] == ["literature", "trials", "kg"] + rwd   # the PDF library is still on
    assert app.checkbox(key="mes_src_lit_fulltext").disabled                    # full text reads PubMed's articles
    app.session_state["submit_payload"] = sent()
    app.run()
    assert app.session_state["mes_thread"][0]["literature"] == ["pdfs"]
    app = run("landing")
    app.checkbox(key="mes_src_lit_pubmed").uncheck().run()
    app.checkbox(key="mes_src_lit_pdfs").uncheck().run()
    assert app.session_state["mes_sources"] == ["trials", "kg"] + rwd           # no part left: no literature
    app.checkbox(key="mes_src_lit_pubmed").check().run()                        # PubMed back: with its full text
    assert app.session_state["mes_sources"][0] == "literature"
    assert [c.value for c in app.checkbox if c.key in ("mes_src_lit_pubmed", "mes_src_lit_fulltext")] == [True, True]


def test_the_same_send_is_not_asked_twice():
    """The box is rendered again after the rerun and hands the same payload back."""
    app = run("landing", submit=sent())
    assert len(app.session_state["mes_thread"]) == 1
    assert app.session_state["mes_last_sent"] == sent()["at"]


# ----------------------------------------------------------------- the follow-up box
def test_followup_box_renders_under_the_answer():
    app = run("followup")
    data = calls(app, "mes_followup_1")
    assert len(data) == 1
    assert data[0]["compact"] is True                     # the follow-up box is the small one
    assert data[0]["mode"] == "sackett"                   # the mode of the question above it
    assert not data[0]["default_question"]                # nothing to send from an empty follow-up


def test_followup_submit_appends_a_turn_and_keeps_the_sources():
    app = run("followup", submit=sent(mode="osler"))
    thread = app.session_state["mes_thread"]
    assert len(thread) == 2
    assert thread[1]["question"] == "Does metformin lower mortality?"
    assert thread[1]["mode"] == "osler"
    assert thread[1]["sources"] == ["literature"]         # inherited from the question above
    assert app.session_state["mes_followup_mode"] == "osler"


# ----------------------------------------------------------------- the depth menu
def test_the_depth_menu_is_an_effort_scale():
    """A menu whose items all carry a level is drawn animated (level bars, meter): MES's mode menu is one."""
    landing = calls(run("landing"), "mes_composer")[0]
    assert [(m["label"], m["level"]) for m in landing["modes"]] == [("Fast", 0), ("Balanced", 1), ("Deep", 2)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
