from pathlib import Path

import pytest

from lora_finetune_studio.models import DatasetSpec, Preset, TrainingConfig, run_path


def test_config_round_trip() -> None:
    original = TrainingConfig(
        model_id="owner/model",
        dataset=DatasetSpec(source="hub", repo_id="owner/data"),
    )

    restored = TrainingConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.validate() == []


def test_config_requires_push_destination() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        dataset=DatasetSpec(source="hub", repo_id="owner/data"),
        push_to_hub=True,
    )

    assert (
        "Hub output repository is required when upload is enabled." in config.validate()
    )


def test_run_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid run ID"):
        run_path("../outside", tmp_path)


def test_preset_values_are_bounded() -> None:
    assert set(Preset) == {Preset.SMOKE, Preset.STANDARD, Preset.QUALITY}
