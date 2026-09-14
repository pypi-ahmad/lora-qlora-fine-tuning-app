"""Console launcher for the Streamlit application.

Backs the `lora-finetune-studio` console script declared in pyproject.toml
([project.scripts]). This is a thin alternative to the platform launcher scripts
(Launch LoRA Studio.cmd/.sh): it starts Streamlit directly and skips their
environment-preparation steps (uv sync, Unsloth runtime setup, health-check wait).
Next file to read: streamlit_app.py, the actual UI entry point this launches.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).parents[2] / "streamlit_app.py"
    # Streamlit's CLI reads its arguments from sys.argv rather than accepting them as
    # a function call, so we rewrite argv before delegating to it.
    sys.argv = ["streamlit", "run", str(app)]
    raise SystemExit(stcli.main())
