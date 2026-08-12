from pathlib import Path

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

    assert status.state is JobState.QUEUED
    assert (tmp_path / run_id / "config.json").is_file()
    assert "HF_TOKEN" not in (tmp_path / run_id / "config.json").read_text(
        encoding="utf-8"
    )


def test_atomic_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    jobs.write_json_atomic(path, {"value": 1})
    jobs.write_json_atomic(path, {"value": 2})

    assert path.read_text(encoding="utf-8").strip().endswith("2\n}")
