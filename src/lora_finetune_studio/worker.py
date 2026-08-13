"""Subprocess entry point for a durable training run."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from .jobs import write_json_atomic
from .models import JobState, JobStatus, TrainingConfig
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
        if config.use_unsloth:
            __import__("unsloth")
        from .training import train

        metrics = train(config, status_path)
    except Exception as error:  # noqa: BLE001
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
