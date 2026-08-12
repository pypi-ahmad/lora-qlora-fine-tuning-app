"""TRL supervised fine-tuning worker implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from .jobs import write_json_atomic
from .models import JobState, JobStatus, PeftMode, TrainingConfig
from .sources import get_hf_token, load_training_dataset


class StatusCallback(TrainerCallback):
    def __init__(self, status_path: Path) -> None:
        self.status_path = status_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        logs = logs or {}
        metrics = {
            key: float(value)
            for key, value in logs.items()
            if isinstance(value, (int, float))
            and key in {"loss", "eval_loss", "learning_rate", "epoch"}
        }
        progress = min(state.global_step / max(state.max_steps, 1), 1.0)
        status = JobStatus(
            state=JobState.RUNNING,
            message=f"Training step {state.global_step} of {state.max_steps}",
            progress=progress,
            pid=os.getpid(),
            metrics=metrics,
        )
        write_json_atomic(self.status_path, status.to_dict())


def _normalize_dataset(dataset: Dataset, config: TrainingConfig) -> Dataset:
    spec = config.dataset
    if spec.format == "text" and spec.text_column and spec.text_column != "text":
        dataset = dataset.rename_column(spec.text_column, "text")
    elif spec.format == "prompt_completion":
        if not spec.prompt_column or not spec.completion_column:
            raise ValueError("Prompt and completion columns are required.")
        dataset = dataset.map(
            lambda row: {
                "prompt": str(row[spec.prompt_column]),
                "completion": str(row[spec.completion_column]),
            },
            remove_columns=dataset.column_names,
        )
    if config.max_samples and len(dataset) > config.max_samples:
        dataset = dataset.shuffle(seed=config.seed).select(range(config.max_samples))
    return dataset


def _split_dataset(
    dataset: Dataset, config: TrainingConfig
) -> tuple[Dataset, Dataset | None]:
    if not config.eval_enabled or len(dataset) < 10:
        return dataset, None
    split = dataset.train_test_split(test_size=config.eval_ratio, seed=config.seed)
    return split["train"], split["test"]


def train(config: TrainingConfig, status_path: Path) -> dict[str, Any]:
    token = get_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for local training.")

    raw_dataset = load_training_dataset(
        repo_id=config.dataset.repo_id,
        local_path=config.dataset.local_path,
        config_name=config.dataset.config_name,
        split=config.dataset.split,
        token=token,
    )
    dataset = _normalize_dataset(raw_dataset, config)
    train_dataset, eval_dataset = _split_dataset(dataset, config)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "token": token,
        "revision": config.model_revision,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": compute_dtype,
    }
    if config.peft_mode is PeftMode.QLORA:
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map={"": torch.cuda.current_device()},
        )

    tokenizer: Any = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        token=token,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(output_dir),
        max_length=config.max_length,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        logging_steps=1,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
        dataset_text_field="text",
        push_to_hub=False,
        hub_model_id=config.hub_model_id,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[StatusCallback(status_path)],
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(output_dir / "adapter")
    metrics = {
        key: float(value)
        for key, value in result.metrics.items()
        if isinstance(value, (int, float))
    }
    if eval_dataset is not None:
        metrics.update(
            {
                key: float(value)
                for key, value in trainer.evaluate().items()
                if isinstance(value, (int, float))
            }
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    if config.push_to_hub and config.hub_model_id:
        trained_model: Any = trainer.model
        trained_model.push_to_hub(config.hub_model_id, token=token)
        tokenizer.push_to_hub(config.hub_model_id, token=token)
    return metrics
