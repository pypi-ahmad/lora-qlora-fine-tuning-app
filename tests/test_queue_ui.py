from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from lora_finetune_studio import jobs, unsloth_runtime
from lora_finetune_studio.models import DatasetSpec, HardwareProfile, TrainingConfig
from lora_finetune_studio.unsloth_runtime import UnslothRuntimeStatus


def test_review_adds_training_to_queue_while_worker_is_active(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(
        unsloth_runtime,
        "inspect_unsloth_runtime",
        lambda: UnslothRuntimeStatus(
            True, Path(".venv-unsloth/Scripts/python.exe"), "test", "Ready"
        ),
    )
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=1234)
    )
    monkeypatch.setattr(jobs.psutil, "pid_exists", lambda pid: pid == 1234)
    monkeypatch.setattr(jobs, "_is_training_worker", lambda _pid, _path: True)
    config = TrainingConfig(
        model_id="owner/model",
        input_fingerprint="reviewed-fixture",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
    )
    from lora_finetune_studio import provenance

    monkeypatch.setattr(provenance, "prepare_run", lambda *_args: None)
    active_id = jobs.enqueue_run(config)

    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()
    app.session_state["training_config"] = config
    app.session_state["hardware_profile"] = HardwareProfile(
        cuda_available=True,
        gpu_name="Test GPU",
        vram_gb=8.0,
        ram_gb=32.0,
        free_disk_gb=100.0,
        bf16_supported=True,
        recommended_mode=None,
        recommended_max_billions=3.0,
    )
    app.session_state["model_parameters"] = None

    app.switch_page("app_pages/review.py").run()
    add_button = next(button for button in app.button if button.label == "Add to queue")
    assert not add_button.disabled
    add_button.click().run()

    queued_id = app.session_state["run_id"]
    assert queued_id != active_id
    assert jobs.queued_runs() == [queued_id]
    assert any(item.value == "Training queue" for item in app.subheader)
    assert any(selectbox.label == "Selected run" for selectbox in app.selectbox)
    assert any(button.label == "Remove from queue" for button in app.button)


def test_monitor_queues_chat_evaluation_without_loading_a_model(tmp_path, monkeypatch):
    from lora_finetune_studio.models import JobState, JobStatus

    monkeypatch.setattr(jobs, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "dispatch_next_run", lambda: None)
    parent = jobs.create_run(
        TrainingConfig(
            model_id="owner/model",
            datasets=[DatasetSpec(source="upload", format="messages")],
        )
    )
    parent_config = jobs.read_config(parent)
    jobs.write_json_atomic(
        tmp_path / parent / "status.json",
        JobStatus(
            state=JobState.COMPLETED,
            message="done",
            artifact_dir=parent_config.output_dir,
        ).to_dict(),
    )
    app = AppTest.from_file(
        Path(__file__).parents[1] / "streamlit_app.py", default_timeout=60
    ).run()
    app.session_state["run_id"] = parent
    app.switch_page("app_pages/monitor.py").run()
    assert not app.exception
    next(item for item in app.selectbox if item.label == "AI judge model").set_value(
        "agnes-3.0-flash"
    )
    next(item for item in app.text_area if item.label == "Comparison prompt").set_value(
        "Explain LoRA."
    )
    next(item for item in app.button if item.label == "Queue evaluation").click().run()
    assert not app.exception
    evaluation_id = app.session_state["run_id"]
    evaluation = jobs.read_config(evaluation_id)
    assert evaluation.job_kind == "evaluation"
    assert evaluation.parent_run_id == parent
    assert evaluation.evaluation.mode == "chat"
    assert not evaluation.evaluation.held_out
    assert not evaluation.evaluation.judge_enabled
    assert evaluation.evaluation.judge_model == "agnes-3.0-flash"
    assert jobs.read_status(evaluation_id).state is JobState.QUEUED
