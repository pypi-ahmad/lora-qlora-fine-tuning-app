"""Application lifecycle helpers for the local Streamlit server."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable


def schedule_application_exit(
    delay_seconds: float = 1.0,
    exit_function: Callable[[int], object] = os._exit,
) -> None:
    """Terminate this process after allowing the shutdown message to render."""
    timer = threading.Timer(delay_seconds, exit_function, args=(0,))
    timer.daemon = True
    timer.start()
