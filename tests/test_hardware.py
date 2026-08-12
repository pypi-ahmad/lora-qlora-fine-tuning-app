from lora_finetune_studio.hardware import model_size_warning
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
