from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from lora_finetune_studio.models import JobStatus, TrainingConfig

ROOT = Path(__file__).parents[1]
FIXTURE_PATH = ROOT / "demo" / "fixtures" / "showcase.json"


def test_showcase_fixture_matches_production_contracts() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    config = TrainingConfig.from_dict(fixture["training_config"])
    status = JobStatus.from_dict(fixture["job_status"])

    assert config.validate() == []
    assert status.progress == 1.0

    source_rows = [
        json.loads(line)
        for line in (ROOT / "examples" / "sft_sample.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected = [
        {
            "user": row["messages"][0]["content"],
            "assistant": row["messages"][1]["content"],
        }
        for row in source_rows[:4]
    ]
    assert fixture["dataset_preview"] == expected


def test_showcase_is_read_only_and_starts_without_services() -> None:
    app = AppTest.from_file(
        ROOT / "demo" / "streamlit_app.py", default_timeout=30
    ).run()

    assert not app.exception
    assert app.title[0].value == "LoRA Fine-tune Studio"
    assert any("Synthetic read-only demonstration" in item.value for item in app.info)
    assert len(app.file_uploader) == 0
    start = next(button for button in app.button if button.label == "Start training")
    assert start.disabled is True
    assert [tab.label for tab in app.tabs] == [
        "1. Dataset",
        "2. Configure",
        "3. Review",
        "4. Monitor",
    ]
