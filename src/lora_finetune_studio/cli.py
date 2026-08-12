"""Console launcher for the Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).parents[2] / "streamlit_app.py"
    sys.argv = ["streamlit", "run", str(app)]
    raise SystemExit(stcli.main())
