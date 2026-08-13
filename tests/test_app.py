from pathlib import Path

from streamlit.testing.v1 import AppTest

from lora_finetune_studio import jobs, lifecycle, ollama, unsloth_runtime
from lora_finetune_studio.models import ComputeType, DatasetSpec, TrainingApproach
from lora_finetune_studio.sources import DatasetInspection
from lora_finetune_studio.unsloth_runtime import UnslothRuntimeStatus


def test_app_starts_without_ollama(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "list_models", list)
    monkeypatch.setattr(jobs, "dispatch_next_run", lambda: None)
    monkeypatch.setattr(
        unsloth_runtime,
        "inspect_unsloth_runtime",
        lambda: UnslothRuntimeStatus(
            True, Path(".venv-unsloth/Scripts/python.exe"), "2026.8.15", "Ready"
        ),
    )

    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "System"
    assert any("never installs drivers" in caption.value for caption in app.caption)
    stop_button = next(
        button for button in app.button if button.label == "Stop LoRA Studio"
    )
    stop_button.click().run()
    assert any(
        "cancels its active training worker" in item.value for item in app.warning
    )
    assert any(button.label == "Confirm stop" for button in app.button)
    keep_running = next(
        button for button in app.button if button.label == "Keep running"
    )
    keep_running.click().run()
    assert any(button.label == "Stop LoRA Studio" for button in app.button)

    scheduled_exits: list[bool] = []
    cancellation_options: list[bool] = []

    def cancel_for_shutdown(*, dispatch_next: bool = True) -> None:
        cancellation_options.append(dispatch_next)

    monkeypatch.setattr(jobs, "cancel_active_run", cancel_for_shutdown)
    monkeypatch.setattr(
        lifecycle,
        "schedule_application_exit",
        lambda: scheduled_exits.append(True),
    )
    next(
        button for button in app.button if button.label == "Stop LoRA Studio"
    ).click().run()
    next(
        button for button in app.button if button.label == "Confirm stop"
    ).click().run()
    assert scheduled_exits == [True]
    assert cancellation_options == [False]
    assert any("Stopping LoRA Studio" in item.value for item in app.info)

    pages = {
        "app_pages/dataset.py": "Dataset",
        "app_pages/model.py": "Model",
        "app_pages/gpu_memory.py": "GPU memory",
        "app_pages/training.py": "Training",
        "app_pages/review.py": "Review & run",
        "app_pages/monitor.py": "Monitor",
        "app_pages/ollama.py": "Ollama playground",
    }
    for path, title in pages.items():
        app.switch_page(path).run()
        assert not app.exception
        assert app.title[0].value == title

    app.switch_page("app_pages/gpu_memory.py").run()
    assert any(button.label == "Clear unused VRAM" for button in app.button)

    selected_spec = DatasetSpec(
        source="hub", repo_id="owner/data", format="text", text_column="text"
    )
    app.session_state["dataset_specs"] = [selected_spec]
    app.session_state["dataset_inspections"] = [
        DatasetInspection(columns=["text"], format="text", rows=10, preview=[])
    ]
    app.switch_page("app_pages/dataset.py").run()
    remove_dataset = next(button for button in app.button if button.label == "Remove")
    remove_dataset.click().run()
    assert app.session_state["dataset_specs"] == []

    app.session_state["model_ready"] = True
    app.session_state["dataset_specs"] = [selected_spec]
    app.session_state["dataset_inspections"] = [
        DatasetInspection(columns=["text"], format="text", rows=10, preview=[])
    ]
    app.switch_page("app_pages/training.py").run()
    unsloth_toggle = next(
        toggle for toggle in app.toggle if toggle.label == "Use Unsloth acceleration"
    )
    assert unsloth_toggle.value is True
    assert not unsloth_toggle.disabled

    approach = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Approach"
    )
    method = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Method"
    )
    assert approach.value == "Supervised Fine-Tuning"
    assert set(method.options) == {"LoRA", "QLoRA", "OFT", "QOFT"}
    compute_type = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Compute type"
    )
    assert set(compute_type.options) == {"Default (Auto)", "BF16", "FP16", "FP32"}

    for label in ("Epochs", "Maximum samples", "Maximum gradient norm"):
        mode = next(
            control for control in app.segmented_control if control.label == label
        )
        assert mode.value == "Default"

    learning_rate_mode = next(
        control for control in app.segmented_control if control.label == "Learning rate"
    )
    assert learning_rate_mode.value == "Default"
    assert app.session_state["training_learning_rate"] == 2e-4

    learning_rate_mode.select("Custom").run()
    for label in ("Epochs", "Maximum samples", "Maximum gradient norm"):
        mode = next(
            control for control in app.segmented_control if control.label == label
        )
        mode.select("Custom").run()
    compute_type = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Compute type"
    )
    compute_type.select("FP32").run()
    unsloth_toggle = next(
        toggle for toggle in app.toggle if toggle.label == "Use Unsloth acceleration"
    )
    assert unsloth_toggle.value is False
    assert unsloth_toggle.disabled
    custom_values = {
        "Custom learning rate": 5e-5,
        "Custom epochs": 4.5,
        "Custom maximum samples": 7,
        "Custom maximum gradient norm": 0.0,
    }
    for label, value in custom_values.items():
        next(
            number_input
            for number_input in app.number_input
            if number_input.label == label
        ).set_value(value)
    next(
        button for button in app.button if button.label == "Save training settings"
    ).click().run()
    saved_config = app.session_state["training_config"]
    assert saved_config.learning_rate == 5e-5
    assert saved_config.epochs == 4.5
    assert saved_config.max_samples == 7
    assert saved_config.max_grad_norm == 0.0
    assert saved_config.compute_type is ComputeType.FP32

    preset = next(
        control for control in app.segmented_control if control.label == "Preset"
    )
    preset.select("Smoke test").run()
    assert app.session_state["training_epochs_mode"] == "Default"
    assert app.session_state["training_epochs"] == 1.0
    assert app.session_state["training_max_samples_mode"] == "Default"
    assert app.session_state["training_max_samples"] == 100

    method = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Method"
    )
    method.select("OFT").run()
    unsloth_toggle = next(
        toggle for toggle in app.toggle if toggle.label == "Use Unsloth acceleration"
    )
    assert unsloth_toggle.value is False
    assert unsloth_toggle.disabled

    approach = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Approach"
    )
    approach.select("DPO Training").run()
    assert app.session_state["training_approach"] is TrainingApproach.DPO
    assert app.session_state["training_learning_rate_mode"] == "Default"
    assert app.session_state["training_learning_rate"] == 1e-5
    assert any("Remap or remove" in item.value for item in app.info)

    app.switch_page("app_pages/dataset.py").run()
    assert any(button.label == "Remap" for button in app.button)
