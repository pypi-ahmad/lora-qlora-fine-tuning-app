from pathlib import Path

from streamlit.testing.v1 import AppTest

from lora_finetune_studio import ollama


def test_app_starts_without_ollama(monkeypatch) -> None:
    monkeypatch.setattr(ollama, "list_models", list)

    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "LoRA Fine-tune Studio"
