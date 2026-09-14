"""Wait for a training worker to exit, then safely advance the queue.

Runs as its own short-lived subprocess (see schedule_queue_handoff), started by a
training worker (worker.py) right before that worker's own process ends. It cannot
be a background thread inside the worker: the worker process is about to exit, and
a thread dies with its parent process, whereas this needs to keep running after the
worker is gone in order to wait for it to fully release the GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil

from .jobs import dispatch_next_run

BASE_PYTHON_ENV = "LORA_STUDIO_PYTHON"


def schedule_queue_handoff(parent_pid: int) -> None:
    """Start a lightweight process that dispatches after this worker exits."""
    # BASE_PYTHON_ENV (set by jobs.launch_run) always points at the main .venv
    # interpreter, even when the exiting worker itself ran under the isolated
    # .venv-unsloth interpreter — the next dispatched run must not inherit an
    # Unsloth-specific interpreter it may not need.
    base_python = os.environ.get(BASE_PYTHON_ENV, sys.executable)
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # Detached with all streams discarded and no stdin: this process outlives the
    # caller and nothing reads its output, so leaving pipes open would risk a
    # blocked write if a buffer filled.
    subprocess.Popen(
        [
            base_python,
            "-m",
            "lora_finetune_studio.queue_dispatcher",
            str(parent_pid),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
        cwd=Path.cwd(),
        close_fds=True,
        env=os.environ.copy(),
    )


def wait_for_parent_and_dispatch(parent_pid: int) -> int:
    """Wait for VRAM-owning parent termination before launching the next run.

    Only one training worker may hold the GPU at a time, and the OS does not free a
    process's VRAM allocations until the process has actually exited — so dispatching
    the next run must wait for that exit rather than racing it. The 60-second timeout
    bounds how long this handoff process waits if the parent hangs on shutdown; on
    timeout it gives up on dispatching rather than waiting indefinitely (the next
    normal reconciliation, e.g. on Streamlit restart, can still recover the queue).
    """
    try:
        psutil.Process(parent_pid).wait(timeout=60)
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied, psutil.TimeoutExpired:
        return 1
    try:
        dispatch_next_run()
    except OSError, RuntimeError, ValueError:
        return 1
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    try:
        parent_pid = int(sys.argv[1])
    except ValueError:
        return 2
    if parent_pid <= 0:
        return 2
    return wait_for_parent_and_dispatch(parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
