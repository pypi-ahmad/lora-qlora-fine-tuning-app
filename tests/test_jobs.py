from pathlib import Path
from types import SimpleNamespace

import pytest

from lora_finetune_studio import jobs
from lora_finetune_studio.models import DatasetSpec, JobState, TrainingConfig
from lora_finetune_studio.unsloth_runtime import UnslothRuntimeStatus


def test_create_run_persists_safe_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )

    run_id = jobs.create_run(config)
    status = jobs.read_status(run_id)
    saved_config = jobs.read_config(run_id)

    assert status.state is JobState.QUEUED
    assert saved_config.model_id == "owner/model"
    assert saved_config.datasets[0].repo_id == "owner/data"
    assert (tmp_path / run_id / "config.json").is_file()
    assert "HF_TOKEN" not in (tmp_path / run_id / "config.json").read_text(
        encoding="utf-8"
    )


def test_create_run_appends_jobs_in_fifo_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )

    first_run = jobs.create_run(config)
    second_run = jobs.create_run(config)

    assert jobs.queued_runs() == [first_run, second_run]


def test_dispatch_starts_only_first_waiting_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    first_run = jobs.create_run(config)
    second_run = jobs.create_run(config)
    launched_commands: list[list[str]] = []

    def fake_popen(command, **_options):
        launched_commands.append(command)
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda pid: pid == 1234)
    monkeypatch.setattr(jobs, "_is_training_worker", lambda _pid, _path: True)

    assert jobs.dispatch_next_run() == first_run
    assert jobs.dispatch_next_run() is None
    assert jobs.queued_runs() == [second_run]
    assert jobs.read_status(first_run).state is JobState.RUNNING
    assert len(launched_commands) == 1


def test_cancelling_running_job_starts_next_waiting_run(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    first_run = jobs.create_run(config)
    second_run = jobs.create_run(config)
    pids = iter((1234, 5678))
    live_pids = {1234}

    def fake_popen(_command, **_options):
        pid = next(pids)
        live_pids.add(pid)
        return SimpleNamespace(pid=pid)

    class RunningProcess(FakeWorkerProcess):
        def terminate(self) -> None:
            super().terminate()
            live_pids.discard(1234)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(jobs.psutil, "pid_exists", live_pids.__contains__)
    first_config = tmp_path / first_run / "config.json"
    first_process = RunningProcess(
        [
            "python",
            "-m",
            "lora_finetune_studio.worker",
            str(first_config.resolve()),
        ]
    )
    monkeypatch.setattr(jobs.psutil, "Process", lambda _pid: first_process)

    jobs.dispatch_next_run()
    jobs.cancel_run(first_run)

    assert jobs.read_status(first_run).state is JobState.CANCELLED
    assert jobs.read_status(second_run).state is JobState.RUNNING
    assert jobs.queued_runs() == []


def test_dispatch_marks_dead_worker_failed_before_continuing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    stale_run = jobs.create_run(config)
    waiting_run = jobs.create_run(config)
    stale_status = jobs.read_status(stale_run)
    stale_status.state = JobState.RUNNING
    stale_status.pid = 1234
    jobs.write_json_atomic(tmp_path / stale_run / "status.json", stale_status.to_dict())
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda _pid: False)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=5678)
    )

    assert jobs.dispatch_next_run() == waiting_run
    assert jobs.read_status(stale_run).state is JobState.FAILED
    assert "stopped" in (jobs.read_status(stale_run).error or "").lower()


def test_resume_appends_checkpoint_run_to_queue_when_worker_is_active(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    active_id = jobs.create_run(config)
    resumable_id = jobs.create_run(config)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=1234)
    )
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda pid: pid == 1234)
    monkeypatch.setattr(jobs, "_is_training_worker", lambda _pid, _path: True)
    jobs.dispatch_next_run()
    jobs.cancel_run(resumable_id)
    checkpoint = tmp_path / resumable_id / "output" / "checkpoint-5"
    checkpoint.mkdir(parents=True)

    jobs.resume_run(resumable_id)

    assert jobs.queued_runs() == [resumable_id]
    assert jobs.read_status(resumable_id).state is JobState.QUEUED
    assert jobs.read_config(resumable_id).resume_from_checkpoint == str(
        checkpoint.resolve()
    )
    assert jobs.active_run() == active_id


def test_list_runs_orders_active_then_waiting_then_history(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    history_id = jobs.create_run(config)
    history_status = jobs.read_status(history_id)
    history_status.state = JobState.COMPLETED
    jobs.write_json_atomic(
        tmp_path / history_id / "status.json", history_status.to_dict()
    )
    active_id = jobs.create_run(config)
    waiting_id = jobs.create_run(config)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=1234)
    )
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda pid: pid == 1234)
    monkeypatch.setattr(jobs, "_is_training_worker", lambda _pid, _path: True)
    jobs.dispatch_next_run()

    assert jobs.list_runs() == [active_id, waiting_id, history_id]


def test_dispatch_marks_launch_failure_and_continues_queue(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    datasets = [DatasetSpec(source="hub", repo_id="owner/data", format="text")]
    failed_id = jobs.create_run(
        TrainingConfig(model_id="owner/model", datasets=datasets, use_unsloth=True)
    )
    next_id = jobs.create_run(TrainingConfig(model_id="owner/model", datasets=datasets))
    monkeypatch.setattr(
        jobs,
        "inspect_unsloth_runtime",
        lambda: UnslothRuntimeStatus(
            False, tmp_path / "missing.exe", detail="Unsloth is missing."
        ),
    )
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=1234)
    )

    assert jobs.dispatch_next_run() == next_id
    failed_status = jobs.read_status(failed_id)
    assert failed_status.state is JobState.FAILED
    assert failed_status.error == "Unsloth is missing."


def test_dispatch_waits_for_terminal_worker_process_to_exit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    )
    finishing_id = jobs.create_run(config)
    waiting_id = jobs.create_run(config)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=1234)
    )
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda pid: pid == 1234)
    monkeypatch.setattr(jobs, "_is_training_worker", lambda _pid, _path: True)
    jobs.dispatch_next_run()
    finishing_status = jobs.read_status(finishing_id)
    finishing_status.state = JobState.COMPLETED
    jobs.write_json_atomic(
        tmp_path / finishing_id / "status.json", finishing_status.to_dict()
    )

    assert jobs.dispatch_next_run() is None
    assert jobs.queued_runs() == [waiting_id]


def test_atomic_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    jobs.write_json_atomic(path, {"value": 1})
    jobs.write_json_atomic(path, {"value": 2})

    assert path.read_text(encoding="utf-8").strip().endswith("2\n}")


class FakeWorkerProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.terminated = False

    def cmdline(self) -> list[str]:
        return self.command

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: int) -> None:
        assert timeout == 10


def _running_test_run(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    run_id = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        )
    )
    status = jobs.read_status(run_id)
    status.state = JobState.RUNNING
    status.pid = 1234
    jobs.write_json_atomic(tmp_path / run_id / "status.json", status.to_dict())
    return run_id, tmp_path / run_id / "config.json"


def test_cancel_run_stops_only_its_training_worker(tmp_path: Path, monkeypatch) -> None:
    run_id, config_path = _running_test_run(tmp_path, monkeypatch)
    process = FakeWorkerProcess(
        [
            "python",
            "-m",
            "lora_finetune_studio.worker",
            str(config_path.resolve()),
        ]
    )
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda _pid: True)
    monkeypatch.setattr(jobs.psutil, "Process", lambda _pid: process)

    jobs.cancel_run(run_id)

    assert process.terminated
    assert jobs.read_status(run_id).state is JobState.CANCELLED


def test_cancel_run_rejects_unrelated_process(tmp_path: Path, monkeypatch) -> None:
    run_id, _config_path = _running_test_run(tmp_path, monkeypatch)
    process = FakeWorkerProcess(["python", "unrelated.py"])
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda _pid: True)
    monkeypatch.setattr(jobs.psutil, "Process", lambda _pid: process)

    with pytest.raises(RuntimeError, match="not this run's training worker"):
        jobs.cancel_run(run_id)

    assert not process.terminated
    assert jobs.read_status(run_id).state is JobState.RUNNING


def test_cancel_active_run_delegates_to_active_run(monkeypatch) -> None:
    cancelled: list[tuple[str, bool]] = []
    monkeypatch.setattr(jobs, "active_run", lambda: "active-run")

    def fake_cancel(run_id: str, *, dispatch_next: bool = True) -> None:
        cancelled.append((run_id, dispatch_next))

    monkeypatch.setattr(jobs, "cancel_run", fake_cancel)

    assert jobs.cancel_active_run(dispatch_next=False) == "active-run"
    assert cancelled == [("active-run", False)]


def test_launch_run_uses_unsloth_interpreter_and_source_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    python = tmp_path / "python.exe"
    monkeypatch.setattr(
        jobs,
        "inspect_unsloth_runtime",
        lambda: UnslothRuntimeStatus(True, python, "2026.8.15", "Ready"),
    )
    captured: dict = {}

    def fake_popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    run_id = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
            use_unsloth=True,
        )
    )

    jobs.launch_run(run_id)

    assert captured["command"][0] == str(python)
    python_path = captured["options"]["env"]["PYTHONPATH"].split(jobs.os.pathsep)
    assert python_path[0] == str(jobs.PROJECT_ROOT / "src")


def test_launch_run_keeps_current_interpreter_when_unsloth_is_off(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    captured: dict = {}

    def fake_popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    run_id = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        )
    )

    jobs.launch_run(run_id)

    assert captured["command"][0] == jobs.sys.executable
    assert captured["options"]["env"]["LORA_STUDIO_PYTHON"] == jobs.sys.executable


def test_launch_run_preserves_base_interpreter_across_worker_handoffs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    monkeypatch.setenv("LORA_STUDIO_PYTHON", "C:/project/.venv/Scripts/python.exe")
    captured: dict = {}

    def fake_popen(command, **options):
        captured["command"] = command
        captured["options"] = options
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    run_id = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        )
    )

    jobs.launch_run(run_id)

    assert captured["command"][0] == "C:/project/.venv/Scripts/python.exe"
    assert (
        captured["options"]["env"]["LORA_STUDIO_PYTHON"]
        == "C:/project/.venv/Scripts/python.exe"
    )


def test_launch_run_rejects_missing_unsloth_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        jobs,
        "inspect_unsloth_runtime",
        lambda: UnslothRuntimeStatus(
            False, tmp_path / "missing.exe", detail="Unsloth is missing."
        ),
    )
    run_id = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
            use_unsloth=True,
        )
    )

    with pytest.raises(RuntimeError, match="Unsloth is missing"):
        jobs.launch_run(run_id)
