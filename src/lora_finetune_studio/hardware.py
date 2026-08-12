"""Local hardware detection and conservative training recommendations."""

from __future__ import annotations

import shutil
from pathlib import Path

import psutil
import torch

from .models import HardwareProfile, PeftMode


def detect_hardware(workspace: Path | None = None) -> HardwareProfile:
    workspace = workspace or Path.cwd()
    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    vram_gb = 0.0
    bf16_supported = False
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        vram_gb = properties.total_memory / 1024**3
        bf16_supported = torch.cuda.is_bf16_supported()

    ram_gb = psutil.virtual_memory().total / 1024**3
    free_disk_gb = shutil.disk_usage(workspace).free / 1024**3
    recommended_mode: PeftMode | None = None
    max_billions = 0.0
    warning: str | None = None

    if not cuda_available:
        warning = "CUDA GPU not detected. Local training is disabled."
    elif vram_gb < 6:
        recommended_mode, max_billions = PeftMode.QLORA, 1.0
    elif vram_gb < 10:
        recommended_mode, max_billions = PeftMode.QLORA, 3.0
    elif vram_gb < 16:
        recommended_mode, max_billions = PeftMode.QLORA, 7.0
    else:
        recommended_mode, max_billions = PeftMode.QLORA, 13.0

    return HardwareProfile(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        vram_gb=round(vram_gb, 1),
        ram_gb=round(ram_gb, 1),
        free_disk_gb=round(free_disk_gb, 1),
        bf16_supported=bf16_supported,
        recommended_mode=recommended_mode,
        recommended_max_billions=max_billions,
        warning=warning,
    )


def model_size_warning(
    parameter_count: int | None, profile: HardwareProfile
) -> str | None:
    if parameter_count is None or profile.recommended_max_billions == 0:
        return None
    billions = parameter_count / 1_000_000_000
    if billions > profile.recommended_max_billions:
        return (
            f"This {billions:.1f}B model exceeds the conservative "
            f"{profile.recommended_max_billions:g}B recommendation for this GPU."
        )
    return None
