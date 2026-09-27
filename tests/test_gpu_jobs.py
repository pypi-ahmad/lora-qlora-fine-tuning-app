import json
from pathlib import Path

import pytest

from lora_finetune_studio import jobs
from lora_finetune_studio.models import EvaluationConfig, JobState, TrainingConfig


def test_gpu_reservation_is_exclusive_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "PROJECT_ROOT", tmp_path)
    with jobs._queue_lock(".gpu.lock", blocking=False):  # noqa: SIM117
        with pytest.raises(OSError), jobs._queue_lock(".gpu.lock", blocking=False):
            pytest.fail("A second worker acquired the GPU lock")
    with jobs._queue_lock(".gpu.lock", blocking=False):
        pass


def test_mixed_jobs_share_fifo_and_evaluation_retry_keeps_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "dispatch_next_run", lambda: None)
    training = jobs.create_run(TrainingConfig(model_id="a/b"))
    evaluation = jobs.create_run(
        TrainingConfig(
            model_id="a/b",
            job_kind="evaluation",
            parent_run_id=training,
            evaluation=EvaluationConfig(rows=[{"prompt": "hi"}]),
        )
    )
    assert jobs.queued_runs() == [training, evaluation]
    status = jobs.read_status(evaluation)
    status.state = JobState.COMPLETED
    jobs.write_json_atomic(tmp_path / evaluation / "status.json", status.to_dict())
    result = Path(jobs.read_config(evaluation).output_dir) / "evaluation.json"
    jobs.write_json_atomic(result, {"results": [{"base": "saved"}]})
    jobs.retry_evaluation(evaluation)
    assert jobs.queued_runs() == [training, evaluation]
    assert json.loads(result.read_text())["results"][0]["base"] == "saved"
    with pytest.raises(ValueError, match="Only training"):
        jobs.resume_run(evaluation)


def test_failed_judgments_retry_without_regeneration(tmp_path, monkeypatch):
    from lora_finetune_studio import evaluation, hardware, inference, judge

    parent = TrainingConfig(
        model_id="a/b", output_dir=str(tmp_path / "parent" / "output")
    )
    config = TrainingConfig(
        model_id="a/b",
        output_dir=str(tmp_path / "evaluation" / "output"),
        job_kind="evaluation",
        parent_run_id="parent",
        evaluation=EvaluationConfig(
            rows=[{"prompt": "p", "reference": "answer"}], judge_enabled=True
        ),
    )
    monkeypatch.setattr(evaluation, "read_config", lambda _: parent)
    monkeypatch.setattr(
        evaluation,
        "read_status",
        lambda _: jobs.JobStatus(state=JobState.COMPLETED, message="done"),
    )
    monkeypatch.setattr(hardware, "release_unused_cuda_memory", lambda: None)
    calls = []

    def generate(*args, **kwargs):
        calls.append(kwargs["adapter_path"])
        return [{"response": "answer", "latency_seconds": 1.0, "output_tokens": 1}]

    monkeypatch.setattr(inference, "generate_rows", generate)
    verdicts = iter(
        [
            {"status": "error", "error": "Timeout"},
            {"status": "completed", "winner": "tie"},
        ]
    )
    monkeypatch.setattr(judge, "judge_pair", lambda *args, **kwargs: next(verdicts))
    status_path = tmp_path / "evaluation" / "status.json"
    first = evaluation.evaluate(config, status_path)
    assert first["judge_errors"] == 1
    assert first["adapter_exact_match"] == 1
    second = evaluation.evaluate(config, status_path)
    assert second["judge_errors"] == 0
    assert second["judge_completed"] == 1
    assert len(calls) == 2
