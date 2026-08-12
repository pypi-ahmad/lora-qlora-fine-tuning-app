from pathlib import Path

import pytest

from lora_finetune_studio import jobs
from lora_finetune_studio.models import DatasetSpec, JobState, TrainingConfig


def test_create_run_persists_safe_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    config = TrainingConfig(
        model_id="owner/model",
        dataset=DatasetSpec(source="hub", repo_id="owner/data"),
    )

    run_id = jobs.create_run(config)
    status = jobs.read_status(run_id)
    saved_config = jobs.read_config(run_id)

    assert status.state is JobState.QUEUED
    assert saved_config.model_id == "owner/model"
    assert saved_config.dataset.repo_id == "owner/data"
    assert (tmp_path / run_id / "config.json").is_file()
    assert "HF_TOKEN" not in (tmp_path / run_id / "config.json").read_text(
        encoding="utf-8"
    )


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
            dataset=DatasetSpec(source="hub", repo_id="owner/data"),
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
    cancelled: list[str] = []
    monkeypatch.setattr(jobs, "active_run", lambda: "active-run")
    monkeypatch.setattr(jobs, "cancel_run", cancelled.append)

    assert jobs.cancel_active_run() == "active-run"
    assert cancelled == ["active-run"]
