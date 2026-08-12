"""On-demand comparison inference for a completed adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .hardware import release_unused_cuda_memory


def generate_text(
    model_id: str,
    prompt: str,
    *,
    token: str | None,
    revision: str = "main",
    adapter_path: str | None = None,
    max_new_tokens: int = 128,
) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for model comparison.")
    try:
        return _generate_text(
            model_id,
            prompt,
            token=token,
            revision=revision,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
        )
    finally:
        try:
            release_unused_cuda_memory()
        except RuntimeError:
            pass


def _generate_text(
    model_id: str,
    prompt: str,
    *,
    token: str | None,
    revision: str,
    adapter_path: str | None,
    max_new_tokens: int,
) -> str:
    """Generate text in a short-lived frame so model references are released."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer_source = adapter_path or model_id
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_source, token=token, trust_remote_code=False, revision=revision
    )
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        trust_remote_code=False,
        use_safetensors=True,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
        dtype=dtype,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, Path(adapter_path))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return cast(
        str,
        tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ),
    )
