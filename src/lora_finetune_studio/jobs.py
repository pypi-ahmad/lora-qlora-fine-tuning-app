"""Single-job process manager with durable, token-free status files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import psutil

from .models import JobState, JobStatus, TrainingConfig, run_path

RUNS_ROOT = Path(".runs")


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def read_status(run_id: str) -> JobStatus:
    path = run_path(run_id, RUNS_ROOT) / "status.json"
    return JobStatus.from_dict(json.loads(path.read_text(encoding="utf-8")))


def read_config(run_id: str) -> TrainingConfig:
    path = run_path(run_id, RUNS_ROOT) / "config.json"
    return TrainingConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _is_training_worker(pid: int, config_path: Path) -> bool:
    """Return whether a PID is this run's isolated training worker."""
    try:
        command = psutil.Process(pid).cmdline()
    except psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess:
        return False
    expected_config = os.path.normcase(str(config_path.resolve()))
    normalized_command = {
        os.path.normcase(str(Path(item).resolve())) for item in command
    }
    return (
        "lora_finetune_studio.worker" in command
        and "-m" in command
        and expected_config in normalized_command
    )


def active_run() -> str | None:
    if not RUNS_ROOT.exists():
        return None
    for status_path in RUNS_ROOT.glob("*/status.json"):
        try:
            status = JobStatus.from_dict(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except OSError, ValueError, TypeError:
            continue
        if (
            status.state in {JobState.QUEUED, JobState.RUNNING}
            and status.pid
            and psutil.pid_exists(status.pid)
        ):
            return status_path.parent.name
    return None


def create_run(config: TrainingConfig) -> str:
    if active_run():
        raise RuntimeError("Another training job is already active.")
    run_id = uuid.uuid4().hex[:12]
    directory = run_path(run_id, RUNS_ROOT)
    directory.mkdir(parents=True)
    config.output_dir = str((directory / "output").resolve())
    write_json_atomic(directory / "config.json", config.to_dict())
    write_json_atomic(
        directory / "status.json",
        JobStatus(
            state=JobState.QUEUED, message="Queued", artifact_dir=config.output_dir
        ).to_dict(),
    )
    return run_id


def launch_run(run_id: str) -> int:
    directory = run_path(run_id, RUNS_ROOT)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Training configuration was not found.")
    log_handle = (directory / "training.log").open("a", encoding="utf-8")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lora_finetune_studio.worker",
            str(config_path.resolve()),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        cwd=Path.cwd(),
        close_fds=True,
    )
    log_handle.close()
    status = read_status(run_id)
    status.pid = process.pid
    status.state = JobState.RUNNING
    status.message = "Starting training worker"
    write_json_atomic(directory / "status.json", status.to_dict())
    return process.pid


def cancel_run(run_id: str) -> None:
    status = read_status(run_id)
    if status.pid and psutil.pid_exists(status.pid):
        config_path = run_path(run_id, RUNS_ROOT) / "config.json"
        if not _is_training_worker(status.pid, config_path):
            raise RuntimeError(
                "Refusing to stop a process that is not this run's training worker."
            )
        process = psutil.Process(status.pid)
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            process.kill()
    status.state = JobState.CANCELLED
    status.message = "Cancelled by user"
    status.progress = 0.0
    write_json_atomic(run_path(run_id, RUNS_ROOT) / "status.json", status.to_dict())


def cancel_active_run() -> str | None:
    """Cancel the active owned training worker, if one exists."""
    run_id = active_run()
    if run_id:
        cancel_run(run_id)
    return run_id


def resume_run(run_id: str) -> int:
    if active_run():
        raise RuntimeError("Another training job is already active.")
    directory = run_path(run_id, RUNS_ROOT)
    checkpoints = sorted(
        (directory / "output").glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError("No checkpoint is available for this run.")
    config_path = directory / "config.json"
    config = TrainingConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    config.resume_from_checkpoint = str(checkpoints[-1].resolve())
    write_json_atomic(config_path, config.to_dict())
    return launch_run(run_id)


def read_log(run_id: str, max_chars: int = 12_000) -> str:
    path = run_path(run_id, RUNS_ROOT) / "training.log"
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[-max_chars:]
