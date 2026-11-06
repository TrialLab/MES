"""
The WCM palette: the theme MES starts with (.streamlit/config.toml), and MES's own styles use the theme's accent.

    python -m pytest mes/tests/test_theme.py -q
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MES_THEME = ROOT / ".streamlit" / "config.toml"
UI = ROOT / "mes" / "ui"


def theme(path):
    return tomllib.loads(path.read_text(encoding="utf-8"))["theme"]


def test_mes_styles_use_the_theme_accent():
    """The question box, the page, the app shell and the agent canvas draw their accent in the theme's primary
    colour."""
    accent = theme(MES_THEME)["primaryColor"].upper()
    assert f"--accent: {accent};" in (UI / "composer.py").read_text(encoding="utf-8")
    assert f"--mes-accent: {accent};" in (UI / "style.css").read_text(encoding="utf-8")
    assert f"--accent:         {accent};" in (UI / "shell.css").read_text(encoding="utf-8")
    assert f"--accent: {accent};" in (UI / "team_canvas" / "canvas.css").read_text(encoding="utf-8")


def test_no_old_red_accent_is_left():
    """The red of an earlier palette (#8B1A1A / #A4262C) is gone from MES's own styles and exports."""
    files = [UI / "composer.py", UI / "style.css", UI / "style_dark.css", UI / "shell.css", UI / "exports.py",
             UI / "rwd_panel.py"]
    for path in files:
        text = path.read_text(encoding="utf-8").upper()
        assert "#8B1A1A" not in text and "#A4262C" not in text, path.name
