from pathlib import Path

import pytest

from lora_finetune_studio.models import (
    TRAINING_RECIPES,
    ComputeType,
    DatasetSpec,
    PeftMode,
    Preset,
    TrainingApproach,
    TrainingConfig,
    resolve_compute_type,
    run_path,
)


def test_config_round_trip() -> None:
    original = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub", repo_id="owner/data", format="text", text_column="text"
            )
        ],
    )

    restored = TrainingConfig.from_dict(original.to_dict())

    assert restored == original
    assert restored.validate() == []


def test_old_config_migrates_single_dataset_and_standard_backend() -> None:
    values = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
    ).to_dict()
    values["dataset"] = values.pop("datasets")[0]
    del values["use_unsloth"]
    del values["approach"]
    del values["beta"]
    del values["compute_type"]
    del values["max_grad_norm"]

    restored = TrainingConfig.from_dict(values)

    assert restored.use_unsloth is False
    assert restored.approach is TrainingApproach.SFT
    assert restored.beta == 0.1
    assert restored.compute_type is ComputeType.AUTO
    assert restored.max_grad_norm == 1.0
    assert len(restored.datasets) == 1
    assert restored.datasets[0].repo_id == "owner/data"


def test_config_requires_push_destination() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[DatasetSpec(source="hub", repo_id="owner/data", format="text")],
        push_to_hub=True,
    )

    assert (
        "Hub output repository is required when upload is enabled." in config.validate()
    )


def test_config_rejects_mixed_and_duplicate_datasets() -> None:
    repeated = DatasetSpec(source="hub", repo_id="owner/text", format="text")
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            repeated,
            repeated,
            DatasetSpec(source="hub", repo_id="owner/chat", format="messages"),
        ],
    )

    errors = config.validate()

    assert "Dataset 2 duplicates an existing dataset source." in errors
    assert "All datasets must use the same training format." in errors


@pytest.mark.parametrize("approach", list(TrainingApproach))
@pytest.mark.parametrize("method", list(PeftMode))
def test_recipe_matrix_accepts_all_advertised_pairs(
    approach: TrainingApproach, method: PeftMode
) -> None:
    if approach is TrainingApproach.SFT:
        dataset = DatasetSpec(
            source="hub",
            repo_id="owner/data",
            format="text",
            text_column="text",
        )
    else:
        dataset = DatasetSpec(
            source="hub",
            repo_id="owner/data",
            format="preference",
            prompt_column="prompt",
            chosen_column="chosen",
            rejected_column="rejected",
        )
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[dataset],
        approach=approach,
        peft_mode=method,
        batch_size=TRAINING_RECIPES[approach].minimum_batch_size,
    )

    assert method in TRAINING_RECIPES[approach].methods
    assert config.validate() == []


def test_config_rejects_approach_dataset_mismatch_and_unsloth_oft() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
        approach=TrainingApproach.DPO,
        peft_mode=PeftMode.OFT,
        use_unsloth=True,
    )

    errors = config.validate()

    assert "Unsloth acceleration supports only LoRA and QLoRA." in errors
    assert "DPO Training requires preference dataset format." in errors


def test_config_rejects_unsloth_fp32() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
        use_unsloth=True,
        compute_type=ComputeType.FP32,
    )

    assert "Unsloth acceleration does not support FP32 compute." in config.validate()


def test_kto_requires_actual_batch_size_above_one() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="preference",
                prompt_column="prompt",
                chosen_column="chosen",
                rejected_column="rejected",
            )
        ],
        approach=TrainingApproach.KTO,
        batch_size=1,
    )

    assert (
        "KTO Training requires a per-device batch size of at least 2."
        in config.validate()
    )


@pytest.mark.parametrize("learning_rate", [1e-8, 1.1e-2])
def test_config_rejects_learning_rate_outside_supported_range(
    learning_rate: float,
) -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
        learning_rate=learning_rate,
    )

    assert "Learning rate must be between 1e-7 and 1e-2." in config.validate()


@pytest.mark.parametrize("max_samples", [0, -1, 1.5, True])
def test_config_rejects_invalid_maximum_samples(max_samples) -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
        max_samples=max_samples,
    )

    assert "Maximum samples must be a positive integer." in config.validate()


@pytest.mark.parametrize("max_grad_norm", [-0.1, float("inf"), float("nan")])
def test_config_rejects_invalid_maximum_gradient_norm(max_grad_norm: float) -> None:
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="owner/data",
                format="text",
                text_column="text",
            )
        ],
        max_grad_norm=max_grad_norm,
    )

    assert "Maximum gradient norm must be finite and non-negative." in config.validate()


@pytest.mark.parametrize(
    ("requested", "bf16_supported", "expected"),
    [
        (ComputeType.AUTO, True, ComputeType.BF16),
        (ComputeType.AUTO, False, ComputeType.FP16),
        (ComputeType.BF16, False, ComputeType.FP16),
        (ComputeType.FP16, True, ComputeType.FP16),
        (ComputeType.FP32, True, ComputeType.FP32),
    ],
)
def test_compute_type_resolution(
    requested: ComputeType, bf16_supported: bool, expected: ComputeType
) -> None:
    assert resolve_compute_type(requested, bf16_supported=bf16_supported) is expected


def test_run_path_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid run ID"):
        run_path("../outside", tmp_path)


def test_preset_values_are_bounded() -> None:
    assert set(Preset) == {Preset.SMOKE, Preset.STANDARD, Preset.QUALITY}
