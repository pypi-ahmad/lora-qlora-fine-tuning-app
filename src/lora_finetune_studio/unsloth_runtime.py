"""Discovery for the repository-local native Windows Unsloth runtime.

This is a read-only probe of the separate `.venv-unsloth` environment (see SETUP.md);
it never installs or modifies that environment. jobs.launch_run reads
inspect_unsloth_runtime().python to pick the worker interpreter when a run has
use_unsloth enabled.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
# Fixed layout assumption: the Windows launcher always creates the Unsloth venv at
# this exact path, so no configuration or search is needed.
UNSLOTH_PYTHON = PROJECT_ROOT / ".venv-unsloth" / "Scripts" / "python.exe"


@dataclass(frozen=True, slots=True)
class UnslothRuntimeStatus:
    available: bool
    python: Path
    version: str | None = None
    detail: str = ""


@lru_cache(maxsize=1)
def inspect_unsloth_runtime() -> UnslothRuntimeStatus:
    # Cached for the lifetime of the Streamlit process: the result will not reflect
    # an Unsloth runtime installed or repaired after this process started (a restart
    # is required to pick that up, e.g. after re-running the Windows launcher).
    if os.name != "nt":
        return UnslothRuntimeStatus(
            available=False,
            python=UNSLOTH_PYTHON,
            detail="Native Unsloth integration is currently available on Windows only.",
        )
    if not UNSLOTH_PYTHON.is_file():
        return UnslothRuntimeStatus(
            available=False,
            python=UNSLOTH_PYTHON,
            detail="Launch the app with Launch LoRA Studio.cmd to prepare Unsloth.",
        )
    try:
        result = subprocess.run(
            [
                str(UNSLOTH_PYTHON),
                "-c",
                "from importlib.metadata import version; print(version('unsloth'))",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return UnslothRuntimeStatus(
            available=False,
            python=UNSLOTH_PYTHON,
            detail=f"Unsloth runtime check failed: {error}",
        )
    version = result.stdout.strip()
    return UnslothRuntimeStatus(
        available=bool(version),
        python=UNSLOTH_PYTHON,
        version=version or None,
        detail="Ready" if version else "Unsloth package was not found.",
    )
