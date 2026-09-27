"""Batch text generation and reward scoring for isolated evaluation workers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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
    return generate_rows(
        model_id,
        [{"prompt": prompt}],
        token=token,
        revision=revision,
        adapter_path=adapter_path,
        mode="text",
        max_new_tokens=max_new_tokens,
    )[0]["response"]


def format_inputs(tokenizer: Any, row: dict, mode: str):
    if mode == "chat":
        if not tokenizer.chat_template:
            raise ValueError(
                "Chat mode requires a saved chat template. Select Text completion instead."
            )
        messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    return tokenizer(row["prompt"], return_tensors="pt")


def generate_rows(
    model_id: str,
    rows: list[dict],
    *,
    token: str | None,
    revision: str = "main",
    adapter_path: str | None = None,
    mode: str = "text",
    max_new_tokens: int = 128,
    quantized: bool = True,
    progress=None,
) -> list[dict]:
    """Load once for a batch; caller runs in the GPU-owning subprocess."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer_source = adapter_path or model_id
    tokenizer: Any = AutoTokenizer.from_pretrained(
        tokenizer_source, token=token, trust_remote_code=False, revision=revision
    )
    options: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": False,
        "use_safetensors": True,
        "device_map": {"": torch.cuda.current_device()},
        "dtype": dtype,
    }
    if quantized:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if mode == "reward":
        options["num_labels"] = 1
        model: Any = AutoModelForSequenceClassification.from_pretrained(
            model_id, **options
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **options)
    if adapter_path:
        model = PeftModel.from_pretrained(model, Path(adapter_path))
    model.eval()
    limit = getattr(model.config, "max_position_embeddings", 8192)
    results = []
    for index, row in enumerate(rows):
        started = time.monotonic()
        result: dict[str, Any]
        if mode == "reward":
            scores = []
            for response in (row["chosen"], row["rejected"]):
                value = row["prompt"] + response
                if isinstance(value, list):
                    inputs = tokenizer.apply_chat_template(
                        value,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = tokenizer(value, return_tensors="pt")
                if inputs["input_ids"].shape[1] > limit:
                    raise ValueError(
                        f"Evaluation row {index + 1} exceeds the {limit}-token context limit."
                    )
                with torch.inference_mode():
                    scores.append(
                        float(model(**inputs.to(model.device)).logits.flatten()[0])
                    )
            result = {
                "chosen_score": scores[0],
                "rejected_score": scores[1],
                "correct": scores[0] > scores[1],
            }
        else:
            inputs = format_inputs(tokenizer, row, mode)
            prompt_length = inputs["input_ids"].shape[1]
            if prompt_length + max_new_tokens > limit:
                raise ValueError(
                    f"Evaluation row {index + 1} exceeds the {limit}-token context budget including output."
                )
            with torch.inference_mode():
                output = cast(Any, model).generate(
                    **inputs.to(model.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                )
            continuation = output[0][prompt_length:]
            result = {
                "response": cast(
                    str, tokenizer.decode(continuation, skip_special_tokens=True)
                ),
                "output_tokens": len(continuation),
                "input_tokens": prompt_length,
            }
        result["latency_seconds"] = time.monotonic() - started
        result["compute_dtype"] = str(dtype)
        result["quantization"] = "nf4" if quantized else "none"
        results.append(result)
        if progress:
            progress(index + 1, len(rows))
    return results
