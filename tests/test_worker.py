import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lora_finetune_studio import worker
from lora_finetune_studio.models import DatasetSpec, JobState, JobStatus, TrainingConfig


def test_worker_schedules_queue_handoff_after_terminal_status(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    status_path = tmp_path / "status.json"
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        output_dir=str(tmp_path / "output"),
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", str(config_path)])
    monkeypatch.setitem(
        sys.modules,
        "lora_finetune_studio.training",
        SimpleNamespace(train=lambda *_args: {"loss": 0.5}),
    )
    handoffs: list[int] = []
    monkeypatch.setattr(worker, "schedule_queue_handoff", handoffs.append)

    assert worker.main() == 0
    status = JobStatus.from_dict(json.loads(status_path.read_text(encoding="utf-8")))
    assert status.state is JobState.COMPLETED
    assert handoffs == [worker.os.getpid()]


def test_worker_failure_still_schedules_queue_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    status_path = tmp_path / "status.json"
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        output_dir=str(tmp_path / "output"),
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", str(config_path)])

    def fail_training(*_args):
        raise RuntimeError("out of memory")

    monkeypatch.setitem(
        sys.modules,
        "lora_finetune_studio.training",
        SimpleNamespace(train=fail_training),
    )
    handoffs: list[int] = []
    monkeypatch.setattr(worker, "schedule_queue_handoff", handoffs.append)

    assert worker.main() == 1
    status = JobStatus.from_dict(json.loads(status_path.read_text(encoding="utf-8")))
    assert status.state is JobState.FAILED
    assert handoffs == [worker.os.getpid()]
