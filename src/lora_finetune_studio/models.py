"""Shared data contracts for the UI and training worker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class PeftMode(StrEnum):
    LORA = "LoRA"
    QLORA = "QLoRA"


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


@dataclass(slots=True)
class TrainingConfig:
    model_id: str
    model_revision: str = "main"
    dataset: DatasetSpec = field(default_factory=lambda: DatasetSpec(source="hub"))
    peft_mode: PeftMode = PeftMode.QLORA
    preset: Preset = Preset.STANDARD
    output_dir: str = ""
    max_length: int = 1024
    epochs: float = 2.0
    max_steps: int = -1
    max_samples: int | None = None
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
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
        values["dataset"] = DatasetSpec(**values["dataset"])
        values["peft_mode"] = PeftMode(values["peft_mode"])
        values["preset"] = Preset(values["preset"])
        return cls(**values)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model_id:
            errors.append("Model repository is required.")
        if self.dataset.source == "hub" and not self.dataset.repo_id:
            errors.append("Dataset repository is required.")
        if self.dataset.source == "upload" and not self.dataset.local_path:
            errors.append("Uploaded dataset is required.")
        if not 128 <= self.max_length <= 8192:
            errors.append("Maximum sequence length must be between 128 and 8192.")
        if not 0 < self.epochs <= 20:
            errors.append("Epochs must be between 0 and 20.")
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
