"""Wait for a training worker to exit, then safely advance the queue."""

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
    base_python = os.environ.get(BASE_PYTHON_ENV, sys.executable)
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
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
    """Wait for VRAM-owning parent termination before launching the next run."""
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
