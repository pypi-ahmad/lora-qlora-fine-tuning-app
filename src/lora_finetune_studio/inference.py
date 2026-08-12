"""On-demand comparison inference for a completed adapter."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    del model, tokenizer, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    return text
