"""The MES page with the fakes of ui_monkey.py (pipeline, language model, question box) — run by ui_monkey."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

import streamlit as st  # noqa: E402

from mes.scripts import ui_monkey  # noqa: E402

st.set_page_config(page_title="MES", layout="wide", initial_sidebar_state="collapsed")
ui_monkey.install()

from mes.ui import page  # noqa: E402

page.render()
