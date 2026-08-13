"""Shared data contracts for the UI and training worker."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class PeftMode(StrEnum):
    LORA = "LoRA"
    QLORA = "QLoRA"
    OFT = "OFT"
    QOFT = "QOFT"


class TrainingApproach(StrEnum):
    SFT = "Supervised Fine-Tuning"
    REWARD = "Reward Modeling"
    DPO = "DPO Training"
    KTO = "KTO Training"
    ORPO = "ORPO Training"


class ComputeType(StrEnum):
    AUTO = "Default (Auto)"
    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"


def resolve_compute_type(
    compute_type: ComputeType, *, bf16_supported: bool
) -> ComputeType:
    if compute_type in {ComputeType.AUTO, ComputeType.BF16}:
        return ComputeType.BF16 if bf16_supported else ComputeType.FP16
    return compute_type


@dataclass(frozen=True, slots=True)
class TrainingRecipe:
    methods: tuple[PeftMode, ...]
    dataset_formats: tuple[str, ...]
    learning_rate: float
    uses_beta: bool = False
    minimum_batch_size: int = 1


ALL_PEFT_METHODS = tuple(PeftMode)
TRAINING_RECIPES: dict[TrainingApproach, TrainingRecipe] = {
    TrainingApproach.SFT: TrainingRecipe(
        methods=ALL_PEFT_METHODS,
        dataset_formats=("messages", "text", "prompt_completion"),
        learning_rate=2e-4,
    ),
    TrainingApproach.REWARD: TrainingRecipe(
        methods=ALL_PEFT_METHODS,
        dataset_formats=("preference",),
        learning_rate=1e-3,
    ),
    TrainingApproach.DPO: TrainingRecipe(
        methods=ALL_PEFT_METHODS,
        dataset_formats=("preference",),
        learning_rate=1e-5,
        uses_beta=True,
    ),
    TrainingApproach.KTO: TrainingRecipe(
        methods=ALL_PEFT_METHODS,
        dataset_formats=("preference",),
        learning_rate=1e-5,
        uses_beta=True,
        minimum_batch_size=2,
    ),
    TrainingApproach.ORPO: TrainingRecipe(
        methods=ALL_PEFT_METHODS,
        dataset_formats=("preference",),
        learning_rate=1e-5,
        uses_beta=True,
    ),
}


class Preset(StrEnum):
    SMOKE = "Smoke test"
    STANDARD = "Standard"
    QUALITY = "Quality"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class HardwareProfile:
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    free_disk_gb: float
    bf16_supported: bool
    recommended_mode: PeftMode | None
    recommended_max_billions: float
    warning: str | None = None


@dataclass(slots=True)
class DatasetSpec:
    source: str
    repo_id: str | None = None
    local_path: str | None = None
    config_name: str | None = None
    split: str = "train"
    format: str = "auto"
    text_column: str | None = None
    prompt_column: str | None = None
    completion_column: str | None = None
    chosen_column: str | None = None
    rejected_column: str | None = None


@dataclass(slots=True)
class TrainingConfig:
    model_id: str
    model_revision: str = "main"
    datasets: list[DatasetSpec] = field(default_factory=list)
    approach: TrainingApproach = TrainingApproach.SFT
    peft_mode: PeftMode = PeftMode.QLORA
    use_unsloth: bool = False
    compute_type: ComputeType = ComputeType.AUTO
    preset: Preset = Preset.STANDARD
    output_dir: str = ""
    max_length: int = 1024
    epochs: float = 2.0
    max_steps: int = -1
    max_samples: int | None = None
    learning_rate: float = 2e-4
    beta: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    eval_enabled: bool = True
    eval_ratio: float = 0.1
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: str | None = None
    resume_from_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        values = dict(data)
        values.setdefault("use_unsloth", False)
        values.setdefault("approach", TrainingApproach.SFT)
        values.setdefault("beta", 0.1)
        values.setdefault("compute_type", ComputeType.AUTO)
        values.setdefault("max_grad_norm", 1.0)
        dataset_values = values.pop("datasets", None)
        legacy_dataset = values.pop("dataset", None)
        if dataset_values is None:
            dataset_values = [legacy_dataset] if legacy_dataset else []
        values["datasets"] = [DatasetSpec(**item) for item in dataset_values]
        values["approach"] = TrainingApproach(values["approach"])
        values["peft_mode"] = PeftMode(values["peft_mode"])
        values["compute_type"] = ComputeType(values["compute_type"])
        values["preset"] = Preset(values["preset"])
        return cls(**values)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model_id:
            errors.append("Model repository is required.")
        if not self.datasets:
            errors.append("At least one dataset is required.")
        recipe = TRAINING_RECIPES[self.approach]
        if self.peft_mode not in recipe.methods:
            errors.append(f"{self.peft_mode} is not supported for {self.approach}.")
        if self.use_unsloth and self.peft_mode in {PeftMode.OFT, PeftMode.QOFT}:
            errors.append("Unsloth acceleration supports only LoRA and QLoRA.")
        if self.use_unsloth and self.compute_type is ComputeType.FP32:
            errors.append("Unsloth acceleration does not support FP32 compute.")
        identities: set[tuple[str, str | None, str | None, str | None, str]] = set()
        formats: set[str] = set()
        for index, dataset in enumerate(self.datasets, start=1):
            prefix = f"Dataset {index}"
            if dataset.source not in {"hub", "upload"}:
                errors.append(f"{prefix} source is invalid.")
            if dataset.source == "hub" and not dataset.repo_id:
                errors.append(f"{prefix} repository is required.")
            if dataset.source == "upload" and not dataset.local_path:
                errors.append(f"{prefix} upload is required.")
            if dataset.format not in {
                "messages",
                "text",
                "prompt_completion",
                "preference",
            }:
                errors.append(f"{prefix} requires a saved column mapping.")
            else:
                formats.add(dataset.format)
            if dataset.format == "text" and not dataset.text_column:
                errors.append(f"{prefix} text column is required.")
            if dataset.format == "prompt_completion" and (
                not dataset.prompt_column or not dataset.completion_column
            ):
                errors.append(f"{prefix} prompt and completion columns are required.")
            elif (
                dataset.format == "prompt_completion"
                and dataset.prompt_column == dataset.completion_column
            ):
                errors.append(
                    f"{prefix} prompt and completion must use different columns."
                )
            if dataset.format == "preference" and (
                not dataset.prompt_column
                or not dataset.chosen_column
                or not dataset.rejected_column
            ):
                errors.append(
                    f"{prefix} prompt, chosen, and rejected columns are required."
                )
            elif (
                dataset.format == "preference"
                and len(
                    {
                        dataset.prompt_column,
                        dataset.chosen_column,
                        dataset.rejected_column,
                    }
                )
                != 3
            ):
                errors.append(
                    f"{prefix} prompt, chosen, and rejected must use different columns."
                )
            identity = (
                dataset.source,
                dataset.repo_id,
                dataset.local_path,
                dataset.config_name,
                dataset.split,
            )
            if identity in identities:
                errors.append(f"{prefix} duplicates an existing dataset source.")
            identities.add(identity)
        if len(formats) > 1:
            errors.append("All datasets must use the same training format.")
        incompatible = formats.difference(recipe.dataset_formats)
        if incompatible:
            errors.append(
                f"{self.approach} requires "
                f"{', '.join(recipe.dataset_formats)} dataset format."
            )
        if not 128 <= self.max_length <= 8192:
            errors.append("Maximum sequence length must be between 128 and 8192.")
        if not 0 < self.epochs <= 20:
            errors.append("Epochs must be between 0 and 20.")
        if self.max_samples is not None and (
            not isinstance(self.max_samples, int)
            or isinstance(self.max_samples, bool)
            or self.max_samples < 1
        ):
            errors.append("Maximum samples must be a positive integer.")
        if not 1e-7 <= self.learning_rate <= 1e-2:
            errors.append("Learning rate must be between 1e-7 and 1e-2.")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0:
            errors.append("Maximum gradient norm must be finite and non-negative.")
        if recipe.uses_beta and self.beta <= 0:
            errors.append("Beta must be greater than zero.")
        if self.batch_size < recipe.minimum_batch_size:
            errors.append(
                f"{self.approach} requires a per-device batch size of at least "
                f"{recipe.minimum_batch_size}."
            )
        if self.push_to_hub and not self.hub_model_id:
            errors.append("Hub output repository is required when upload is enabled.")
        return errors


@dataclass(slots=True)
class JobStatus:
    state: JobState
    message: str
    progress: float = 0.0
    pid: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    artifact_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatus:
        values = dict(data)
        values["state"] = JobState(values["state"])
        return cls(**values)


PRESETS: dict[Preset, dict[str, Any]] = {
    Preset.SMOKE: {
        "max_length": 512,
        "epochs": 1.0,
        "max_steps": 20,
        "max_samples": 100,
        "gradient_accumulation_steps": 4,
        "eval_enabled": False,
    },
    Preset.STANDARD: {
        "max_length": 1024,
        "epochs": 2.0,
        "max_steps": -1,
        "max_samples": None,
        "gradient_accumulation_steps": 8,
        "eval_enabled": True,
    },
    Preset.QUALITY: {
        "max_length": 2048,
        "epochs": 3.0,
        "max_steps": -1,
        "max_samples": None,
        "gradient_accumulation_steps": 16,
        "eval_enabled": True,
    },
}


def apply_preset(config: TrainingConfig, preset: Preset) -> TrainingConfig:
    values = config.to_dict()
    values.update(PRESETS[preset])
    values["preset"] = preset
    return TrainingConfig.from_dict(values)


def run_path(run_id: str, root: Path = Path(".runs")) -> Path:
    if not run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in run_id
    ):
        raise ValueError("Invalid run ID.")
    return root / run_id
