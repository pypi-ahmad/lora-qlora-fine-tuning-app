"""Subprocess entry point for a durable training run.

Invoked as `python -m lora_finetune_studio.worker CONFIG_PATH` by jobs.launch_run,
using either the main .venv interpreter or the isolated .venv-unsloth interpreter.
Must not be imported by the main Streamlit process: when Unsloth is requested, the
`unsloth` package has to be imported before `.training` (see main() below), and
mixing that import order into the long-lived Streamlit process would affect every
other page, not just this run. Next file to read: training.py, which does the
actual dataset/model/trainer work this module wraps in status reporting.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from .jobs import write_json_atomic
from .models import JobState, JobStatus, TrainingConfig
from .queue_dispatcher import schedule_queue_handoff
from .sources import get_hf_token


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m lora_finetune_studio.worker CONFIG_PATH")
        return 2
    config_path = Path(sys.argv[1]).resolve()
    status_path = config_path.with_name("status.json")
    config = TrainingConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    write_json_atomic(
        status_path,
        JobStatus(
            state=JobState.RUNNING,
            message="Loading model and dataset",
            pid=os.getpid(),
            artifact_dir=config.output_dir,
        ).to_dict(),
    )
    try:
        # Unsloth must be imported before transformers/peft/trl get pulled in by
        # .training, because Unsloth patches those libraries at import time; doing
        # this import here (rather than at module top) keeps it conditional on
        # config.use_unsloth so a standard run never touches the optional package.
        if config.use_unsloth:
            __import__("unsloth")
        from .training import train

        metrics = train(config, status_path)
    except Exception as error:  # noqa: BLE001
        # Broad catch is deliberate: this is the top-level boundary of an isolated
        # subprocess, so any exception must become a FAILED status write rather than
        # an unhandled crash the UI would never see. The HF token, if any, is
        # stripped from the short message shown in status.json/the UI; the
        # unredacted traceback still goes to training.log for debugging.
        token = get_hf_token()
        message = str(error).replace(token, "[REDACTED]") if token else str(error)
        write_json_atomic(
            status_path,
            JobStatus(
                state=JobState.FAILED,
                message="Training failed",
                pid=os.getpid(),
                error=message,
                artifact_dir=config.output_dir,
            ).to_dict(),
        )
        traceback.print_exc()
        # Best-effort: if spawning the handoff process itself fails, this run's own
        # FAILED status above is already durable, but the next queued run will not
        # be auto-dispatched until something else calls dispatch_next_run (e.g. the
        # app restarting or a user action that triggers it).
        try:
            schedule_queue_handoff(os.getpid())
        except OSError:
            traceback.print_exc()
        return 1
    write_json_atomic(
        status_path,
        JobStatus(
            state=JobState.COMPLETED,
            message="Training completed",
            progress=1.0,
            pid=os.getpid(),
            metrics=metrics,
            artifact_dir=config.output_dir,
        ).to_dict(),
    )
    try:
        schedule_queue_handoff(os.getpid())
    except OSError:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
