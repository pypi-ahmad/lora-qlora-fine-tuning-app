"""Single-job process manager with durable, token-free status files."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil

from .models import JobState, JobStatus, TrainingConfig, run_path
from .unsloth_runtime import PROJECT_ROOT, inspect_unsloth_runtime

RUNS_ROOT = Path(".runs")
PROCESS_LOOKUP_ERRORS = (
    psutil.AccessDenied,
    psutil.NoSuchProcess,
    psutil.ZombieProcess,
)
STATUS_READ_ERRORS = (OSError, ValueError, TypeError)
QUEUE_VERSION = 1
BASE_PYTHON_ENV = "LORA_STUDIO_PYTHON"


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


@contextmanager
def _queue_lock() -> Iterator[None]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RUNS_ROOT / ".queue.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_queue_unlocked(run_ids: list[str]) -> None:
    write_json_atomic(
        RUNS_ROOT / "queue.json",
        {"version": QUEUE_VERSION, "run_ids": run_ids},
    )


def _read_queue_unlocked() -> list[str]:
    queue_path = RUNS_ROOT / "queue.json"
    if queue_path.exists():
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        if data.get("version") != QUEUE_VERSION or not isinstance(
            data.get("run_ids"), list
        ):
            raise ValueError("Training queue metadata is invalid.")
        saved_ids = [str(run_id) for run_id in data["run_ids"]]
    else:
        saved_ids = []

    queued_by_time: list[tuple[int, str]] = []
    for status_path in RUNS_ROOT.glob("*/status.json"):
        try:
            status = JobStatus.from_dict(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
            TrainingConfig.from_dict(
                json.loads(
                    status_path.with_name("config.json").read_text(encoding="utf-8")
                )
            )
        except STATUS_READ_ERRORS:
            continue
        if status.state is JobState.QUEUED:
            run_id = status_path.parent.name
            try:
                run_path(run_id, RUNS_ROOT)
            except ValueError:
                continue
            queued_by_time.append((status_path.stat().st_mtime_ns, run_id))

    queued_ids = {run_id for _, run_id in queued_by_time}
    ordered = list(
        dict.fromkeys(run_id for run_id in saved_ids if run_id in queued_ids)
    )
    for _, run_id in sorted(queued_by_time):
        if run_id not in ordered:
            ordered.append(run_id)
    if ordered != saved_ids or not queue_path.exists():
        _write_queue_unlocked(ordered)
    return ordered


def queued_runs() -> list[str]:
    """Return waiting run IDs in durable first-in-first-out order."""
    with _queue_lock():
        return _read_queue_unlocked()


def list_runs() -> list[str]:
    """Return active, waiting, and historical runs in useful display order."""
    active_id = active_run()
    waiting_ids = queued_runs()
    history: list[tuple[int, str]] = []
    excluded = set(waiting_ids)
    if active_id:
        excluded.add(active_id)
    for status_path in RUNS_ROOT.glob("*/status.json"):
        try:
            JobStatus.from_dict(json.loads(status_path.read_text(encoding="utf-8")))
        except STATUS_READ_ERRORS:
            continue
        run_id = status_path.parent.name
        if run_id not in excluded:
            history.append((status_path.stat().st_mtime_ns, run_id))
    ordered = [active_id] if active_id else []
    ordered.extend(waiting_ids)
    ordered.extend(run_id for _, run_id in sorted(history, reverse=True))
    return ordered


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
    except PROCESS_LOOKUP_ERRORS:
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
        except STATUS_READ_ERRORS:
            continue
        if status.state is not JobState.QUEUED and status.pid:
            config_path = status_path.with_name("config.json")
            if psutil.pid_exists(status.pid) and _is_training_worker(
                status.pid, config_path
            ):
                return status_path.parent.name
    return None


def create_run(config: TrainingConfig) -> str:
    with _queue_lock():
        queue = _read_queue_unlocked()
        run_id = uuid.uuid4().hex[:12]
        directory = run_path(run_id, RUNS_ROOT)
        directory.mkdir(parents=True)
        config.output_dir = str((directory / "output").resolve())
        write_json_atomic(directory / "config.json", config.to_dict())
        write_json_atomic(
            directory / "status.json",
            JobStatus(
                state=JobState.QUEUED,
                message="Waiting in training queue",
                artifact_dir=config.output_dir,
            ).to_dict(),
        )
        queue.append(run_id)
        _write_queue_unlocked(queue)
    return run_id


def launch_run(run_id: str) -> int:
    directory = run_path(run_id, RUNS_ROOT)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Training configuration was not found.")
    log_handle = (directory / "training.log").open("a", encoding="utf-8")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    config = read_config(run_id)
    worker_environment = os.environ.copy()
    worker_python = worker_environment.get(BASE_PYTHON_ENV, sys.executable)
    worker_environment[BASE_PYTHON_ENV] = worker_python
    if config.use_unsloth:
        runtime = inspect_unsloth_runtime()
        if not runtime.available:
            log_handle.close()
            raise RuntimeError(runtime.detail)
        worker_python = str(runtime.python)
        source_path = str(PROJECT_ROOT / "src")
        existing_python_path = worker_environment.get("PYTHONPATH")
        worker_environment["PYTHONPATH"] = (
            source_path
            if not existing_python_path
            else os.pathsep.join((source_path, existing_python_path))
        )
    process = subprocess.Popen(
        [
            worker_python,
            "-m",
            "lora_finetune_studio.worker",
            str(config_path.resolve()),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        cwd=Path.cwd(),
        close_fds=True,
        env=worker_environment,
    )
    log_handle.close()
    status = read_status(run_id)
    status.pid = process.pid
    status.state = JobState.RUNNING
    status.message = "Starting training worker"
    write_json_atomic(directory / "status.json", status.to_dict())
    return process.pid


def dispatch_next_run() -> str | None:
    """Launch the first waiting run when no training worker is active."""
    with _queue_lock():
        _fail_stale_running_jobs_unlocked()
        if active_run():
            return None
        queue = _read_queue_unlocked()
        while queue:
            run_id = queue.pop(0)
            _write_queue_unlocked(queue)
            try:
                launch_run(run_id)
            except (OSError, RuntimeError, ValueError) as error:
                config = read_config(run_id)
                write_json_atomic(
                    run_path(run_id, RUNS_ROOT) / "status.json",
                    JobStatus(
                        state=JobState.FAILED,
                        message="Training worker could not start",
                        error=str(error),
                        artifact_dir=config.output_dir,
                    ).to_dict(),
                )
                continue
            return run_id
        return None


def enqueue_run(config: TrainingConfig) -> str:
    """Persist a training configuration and start the queue when idle."""
    run_id = create_run(config)
    dispatch_next_run()
    return run_id


def _fail_stale_running_jobs_unlocked() -> None:
    for status_path in RUNS_ROOT.glob("*/status.json"):
        try:
            status = JobStatus.from_dict(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except STATUS_READ_ERRORS:
            continue
        config_path = status_path.with_name("config.json")
        if status.state is JobState.RUNNING and (
            not status.pid
            or not psutil.pid_exists(status.pid)
            or not _is_training_worker(status.pid, config_path)
        ):
            status.state = JobState.FAILED
            status.message = "Training worker stopped unexpectedly"
            status.error = "The training worker stopped without a terminal status."
            status.pid = None
            write_json_atomic(status_path, status.to_dict())


def cancel_run(run_id: str, *, dispatch_next: bool = True) -> None:
    with _queue_lock():
        status = read_status(run_id)
        if status.state not in {JobState.QUEUED, JobState.RUNNING}:
            return
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
        _read_queue_unlocked()
    if dispatch_next:
        dispatch_next_run()


def cancel_active_run(*, dispatch_next: bool = True) -> str | None:
    """Cancel the active owned training worker, if one exists."""
    run_id = active_run()
    if run_id:
        cancel_run(run_id, dispatch_next=dispatch_next)
    return run_id


def resume_run(run_id: str) -> str:
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
    with _queue_lock():
        queue = _read_queue_unlocked()
        write_json_atomic(config_path, config.to_dict())
        write_json_atomic(
            directory / "status.json",
            JobStatus(
                state=JobState.QUEUED,
                message="Waiting in training queue to resume",
                artifact_dir=config.output_dir,
            ).to_dict(),
        )
        queue = [queued_id for queued_id in queue if queued_id != run_id]
        queue.append(run_id)
        _write_queue_unlocked(queue)
    dispatch_next_run()
    return run_id


def read_log(run_id: str, max_chars: int = 12_000) -> str:
    path = run_path(run_id, RUNS_ROOT) / "training.log"
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    return content[-max_chars:]
