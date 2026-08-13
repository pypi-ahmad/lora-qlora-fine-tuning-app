"""Discovery for the repository-local native Windows Unsloth runtime."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
UNSLOTH_PYTHON = PROJECT_ROOT / ".venv-unsloth" / "Scripts" / "python.exe"


@dataclass(frozen=True, slots=True)
class UnslothRuntimeStatus:
    available: bool
    python: Path
    version: str | None = None
    detail: str = ""


@lru_cache(maxsize=1)
def inspect_unsloth_runtime() -> UnslothRuntimeStatus:
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
