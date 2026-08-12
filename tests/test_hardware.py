from types import SimpleNamespace

import pytest

from lora_finetune_studio import hardware
from lora_finetune_studio.hardware import (
    cuda_memory_stats,
    model_size_warning,
    release_unused_cuda_memory,
    scan_system,
)
from lora_finetune_studio.models import HardwareProfile, PeftMode


def test_model_size_warning_uses_detected_limit() -> None:
    profile = HardwareProfile(
        cuda_available=True,
        gpu_name="Test GPU",
        vram_gb=8,
        ram_gb=32,
        free_disk_gb=100,
        bf16_supported=True,
        recommended_mode=PeftMode.QLORA,
        recommended_max_billions=3,
    )

    assert model_size_warning(7_000_000_000, profile)
    assert model_size_warning(1_000_000_000, profile) is None


def test_cuda_memory_stats_reports_global_and_process_usage(monkeypatch) -> None:
    gib = 1024**3
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        hardware.torch.cuda, "mem_get_info", lambda _device: (6 * gib, 8 * gib)
    )
    monkeypatch.setattr(
        hardware.torch.cuda, "memory_allocated", lambda _device: 1 * gib
    )
    monkeypatch.setattr(hardware.torch.cuda, "memory_reserved", lambda _device: 2 * gib)

    stats = cuda_memory_stats()

    assert stats.free_gb == 6
    assert stats.total_gb == 8
    assert stats.allocated_gb == 1
    assert stats.reserved_gb == 2


def test_release_unused_cuda_memory_runs_gc_and_empties_cache(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        hardware.torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )

    release_unused_cuda_memory()

    assert calls == ["gc", "empty_cache"]


def test_cuda_memory_helpers_reject_cpu_only_system(monkeypatch) -> None:
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA GPU is not available"):
        cuda_memory_stats()
    with pytest.raises(RuntimeError, match="CUDA GPU is not available"):
        release_unused_cuda_memory()


def test_system_scan_reports_live_resources_without_installing(
    tmp_path, monkeypatch
) -> None:
    gib = 1024**3
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.platform, "release", lambda: "11")
    monkeypatch.setattr(hardware.platform, "version", lambda: "test-build")
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=3 * gib),
    )
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100 * gib),
    )
    commands = {"uv": "uv.exe", "ollama": "ollama.exe"}
    monkeypatch.setattr(hardware.shutil, "which", commands.get)
    monkeypatch.setattr(hardware.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(hardware.sys, "base_prefix", str(tmp_path / "python"))

    result = scan_system(tmp_path)

    assert result.os_name == "Windows"
    assert result.os_release == "11"
    assert result.cpu_threads == 16
    assert result.available_ram_gb == 3
    assert result.free_disk_gb == 100
    assert result.uv_venv_active
    statuses = {item.name: item.available for item in result.software}
    assert statuses["uv"]
    assert statuses["uv project environment"]
    assert statuses["Ollama"]


def test_system_scan_recognizes_native_linux(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.8.0")
    monkeypatch.setattr(hardware.platform, "version", lambda: "test-kernel")

    result = scan_system(tmp_path)

    assert result.os_name == "Linux"
    assert result.native_runtime == "Native Linux"
    assert "Linux" in hardware.SUPPORTED_OPERATING_SYSTEMS
