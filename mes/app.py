"""
MES — the evidence-synthesis web app (Streamlit).

    ./run.sh                          # from the repository root (checks the setup first)
    streamlit run mes/app.py          # the same, without the checks
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# the repository root (package `mes`); not this folder, whose config.py / utils.py would shadow
# other modules
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() not in (HERE, ROOT)]

import streamlit as st  # noqa: E402

from mes.ui import header  # noqa: E402

st.set_page_config(page_title="MES", page_icon=header.favicon(), layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(HERE / 'ui' / 'shell.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
header.render_header()

from mes.ui import page  # noqa: E402

page.render()
