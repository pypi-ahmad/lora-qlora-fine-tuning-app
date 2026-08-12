from pathlib import Path

from streamlit.testing.v1 import AppTest

from lora_finetune_studio import jobs, lifecycle, ollama


def test_app_starts_without_ollama(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "list_models", list)

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
    monkeypatch.setattr(jobs, "cancel_active_run", lambda: None)
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
