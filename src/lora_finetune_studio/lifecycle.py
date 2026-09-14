"""Application lifecycle helpers for the local Streamlit server.

Used by the sidebar Stop control to end the Streamlit process itself (not any
training worker; cancelling a run is jobs.cancel_run's responsibility).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable


def schedule_application_exit(
    delay_seconds: float = 1.0,
    exit_function: Callable[[int], object] = os._exit,
) -> None:
    """Terminate this process after allowing the shutdown message to render."""
    # Deferred via a daemon timer thread, not called inline: the caller is a
    # Streamlit script handling the click that triggered this, and it still needs to
    # finish rendering the confirmation message before the process disappears.
    # Defaults to os._exit (skips atexit/finally handlers) rather than sys.exit,
    # because sys.exit only raises SystemExit on the calling thread and would not
    # stop Streamlit's own server threads.
    timer = threading.Timer(delay_seconds, exit_function, args=(0,))
    timer.daemon = True
    timer.start()
