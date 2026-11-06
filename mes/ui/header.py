"""The header of the MES app: the WCM logo and the product's name (ui/shell.css holds the spacing)."""

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
# The WCM Enterprise logo, full colour, for the white header. A dark surface would need the all-white version of the
# same lockup instead.
LOGO = ASSETS / "WCM_MB_LOGO_HZSS2L_CLR_RGB.png"

NAME, TAGLINE = "MES", "Multi-agent Evidence Synthesis"


@st.cache_resource
def _logo_src():
    """The logo as a data URI, so the header needs no static file route."""
    return "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()


def favicon():
    """The shield of the horizontal WCM logo, cropped to a square (``None`` when Pillow or the file is missing)."""
    try:
        from PIL import Image
        img = Image.open(LOGO)
        return img.crop((0, 0, img.height, img.height))
    except Exception:
        return None


def render_header():
    """The logo, a 1px rule, then the product name; a divider under them."""
    col_logo, = st.columns([3], vertical_alignment="center")
    with col_logo:
        st.markdown(
            f'<div class="brand">'
            f'<img class="brand-logo" src="{_logo_src()}" alt="Weill Cornell Medicine">'
            f'<span class="brand-divider"></span>'
            f'<span class="brand-name"><strong>{NAME}</strong><span>{TAGLINE}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.divider()
