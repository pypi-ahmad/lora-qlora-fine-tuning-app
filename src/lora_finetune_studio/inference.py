"""On-demand comparison inference for a completed adapter.

Used by Monitor to generate a base-model response and an adapter response for the
same prompt. Reward-model adapters are out of scope here (their output is a scalar
score, not text) and must be excluded by the caller. Loads at most one model at a
time — see generate_text — so base and adapter comparisons never hold two full
models in VRAM simultaneously.
"""

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
        # Always attempt cleanup, including on a failed load or generation, so a
        # bad prompt/model does not leave VRAM occupied for the next comparison.
        # release_unused_cuda_memory re-raises RuntimeError only when CUDA itself is
        # unavailable, which cannot happen here since we already checked above.
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
    # Prefer the adapter's saved tokenizer (it may include chat-template or added
    # tokens) and fall back to the base model's tokenizer when there is no adapter.
    tokenizer_source = adapter_path or model_id
    # trust_remote_code=False and safetensors-only loading are deliberate trust
    # boundaries against untrusted model repositories; do not relax them here even
    # though it would allow loading more third-party model architectures.
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
    # do_sample=False (greedy decoding) so base-vs-adapter comparisons are
    # deterministic for a given prompt rather than varying between calls.
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    # Slice off the echoed prompt tokens: generate() returns prompt + continuation.
    return cast(
        str,
        tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ),
    )
