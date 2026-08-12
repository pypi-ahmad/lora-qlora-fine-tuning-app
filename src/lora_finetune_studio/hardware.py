"""Local hardware detection and conservative training recommendations."""

from __future__ import annotations

import gc
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import psutil
import torch

from .models import HardwareProfile, PeftMode

MIN_QLORA_FREE_VRAM_GB = 3.5
REQUIRED_CUDA_VERSION = "13.0"
SUPPORTED_PYTHON = (3, 14)
SUPPORTED_OPERATING_SYSTEMS = ("Linux", "Windows")


@dataclass(frozen=True, slots=True)
class CudaMemoryStats:
    """Current CUDA memory usage for the first GPU."""

    free_gb: float
    total_gb: float
    allocated_gb: float
    reserved_gb: float


@dataclass(frozen=True, slots=True)
class SoftwareStatus:
    """Presence and version detail for one local integration."""

    name: str
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SystemScan:
    """Read-only local system and runtime inventory."""

    os_name: str
    os_release: str
    os_version: str
    python_version: str
    cuda_version: str | None
    cpu_threads: int
    available_ram_gb: float
    free_disk_gb: float
    native_runtime: str
    uv_venv_active: bool
    software: tuple[SoftwareStatus, ...]


def _package_status(display_name: str, package_name: str) -> SoftwareStatus:
    try:
        installed_version = version(package_name)
    except PackageNotFoundError:
        return SoftwareStatus(display_name, False, "Not installed")
    return SoftwareStatus(display_name, True, installed_version)


def scan_system(workspace: Path | None = None) -> SystemScan:
    """Inspect local resources and installed software without changing the system."""
    workspace = workspace or Path.cwd()
    os_name = platform.system() or "Unknown"
    cuda_version = torch.version.cuda
    uv_venv_active = sys.prefix != sys.base_prefix and Path(sys.prefix).name == ".venv"
    software = (
        SoftwareStatus("Python", True, platform.python_version()),
        SoftwareStatus(
            "uv",
            shutil.which("uv") is not None,
            "Command available" if shutil.which("uv") else "Not found",
        ),
        SoftwareStatus(
            "uv project environment",
            uv_venv_active,
            ".venv active" if uv_venv_active else ".venv not active",
        ),
        _package_status("PyTorch", "torch"),
        SoftwareStatus(
            "CUDA runtime",
            cuda_version is not None,
            cuda_version or "Not available",
        ),
        _package_status("bitsandbytes", "bitsandbytes"),
        _package_status("Transformers", "transformers"),
        _package_status("PEFT", "peft"),
        _package_status("TRL", "trl"),
        SoftwareStatus(
            "Ollama",
            shutil.which("ollama") is not None,
            "Command available" if shutil.which("ollama") else "Not found",
        ),
    )
    return SystemScan(
        os_name=os_name,
        os_release=platform.release() or "Unknown",
        os_version=platform.version() or "Unknown",
        python_version=platform.python_version(),
        cuda_version=cuda_version,
        cpu_threads=os.cpu_count() or 1,
        available_ram_gb=psutil.virtual_memory().available / 1024**3,
        free_disk_gb=shutil.disk_usage(workspace).free / 1024**3,
        native_runtime=f"Native {os_name}",
        uv_venv_active=uv_venv_active,
        software=software,
    )


def cuda_memory_stats() -> CudaMemoryStats:
    """Return global and process-local CUDA memory figures for GPU 0."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return CudaMemoryStats(
        free_gb=free_bytes / 1024**3,
        total_gb=total_bytes / 1024**3,
        allocated_gb=torch.cuda.memory_allocated(0) / 1024**3,
        reserved_gb=torch.cuda.memory_reserved(0) / 1024**3,
    )


def release_unused_cuda_memory() -> None:
    """Release unreachable objects and unused PyTorch CUDA cache blocks."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    gc.collect()
    torch.cuda.empty_cache()


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
