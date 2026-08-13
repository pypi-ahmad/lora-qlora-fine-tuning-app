"""Recipe-driven TRL training worker implementation."""

from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, OFTConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)

from .jobs import write_json_atomic
from .models import (
    ComputeType,
    DatasetSpec,
    JobState,
    JobStatus,
    PeftMode,
    TrainingApproach,
    TrainingConfig,
    resolve_compute_type,
)
from .sources import get_hf_token, load_training_dataset

_QOFT_PEFT_PATCHED = False


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


def _normalize_dataset(dataset: Dataset, spec: DatasetSpec) -> Dataset:
    if spec.format == "text" and spec.text_column and spec.text_column != "text":
        return dataset.map(
            lambda row: {"text": str(row[spec.text_column])},
            remove_columns=dataset.column_names,
        )
    if spec.format == "text":
        return dataset.select_columns(["text"])
    if spec.format == "prompt_completion":
        if not spec.prompt_column or not spec.completion_column:
            raise ValueError("Prompt and completion columns are required.")
        return dataset.map(
            lambda row: {
                "prompt": row[spec.prompt_column],
                "completion": row[spec.completion_column],
            },
            remove_columns=dataset.column_names,
        )
    if spec.format == "preference":
        if not spec.prompt_column or not spec.chosen_column or not spec.rejected_column:
            raise ValueError("Prompt, chosen, and rejected columns are required.")
        if (
            spec.prompt_column == "prompt"
            and spec.chosen_column == "chosen"
            and spec.rejected_column == "rejected"
        ):
            return dataset.select_columns(["prompt", "chosen", "rejected"])
        return dataset.map(
            lambda row: {
                "prompt": row[spec.prompt_column],
                "chosen": row[spec.chosen_column],
                "rejected": row[spec.rejected_column],
            },
            remove_columns=dataset.column_names,
        )
    if spec.format == "messages":
        return dataset.select_columns(["messages"])
    raise ValueError(f"Unsupported dataset format: {spec.format}")


def _dataset_label(spec: DatasetSpec) -> str:
    return spec.repo_id or spec.local_path or "unknown source"


def _load_and_combine_datasets(config: TrainingConfig, token: str | None) -> Dataset:
    normalized: list[Dataset] = []
    for index, spec in enumerate(config.datasets, start=1):
        try:
            dataset = load_training_dataset(
                repo_id=spec.repo_id,
                local_path=spec.local_path,
                config_name=spec.config_name,
                split=spec.split,
                token=token,
            )
            normalized.append(_normalize_dataset(dataset, spec))
        except Exception as error:
            raise RuntimeError(
                f"Dataset {index} ({_dataset_label(spec)}) could not be loaded: {error}"
            ) from error

    if not normalized:
        raise ValueError("At least one dataset is required.")
    try:
        combined = (
            concatenate_datasets(normalized) if len(normalized) > 1 else normalized[0]
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Selected datasets have incompatible schemas after normalization."
        ) from error
    if len(normalized) > 1:
        combined = combined.shuffle(seed=config.seed)
    if config.max_samples and len(combined) > config.max_samples:
        if len(normalized) == 1:
            combined = combined.shuffle(seed=config.seed)
        combined = combined.select(range(config.max_samples))
    return combined


def _split_dataset(
    dataset: Dataset, config: TrainingConfig
) -> tuple[Dataset, Dataset | None]:
    if not config.eval_enabled or len(dataset) < 10:
        return dataset, None
    split = dataset.train_test_split(test_size=config.eval_ratio, seed=config.seed)
    return split["train"], split["test"]


def _format_conversations(examples: dict[str, Any], tokenizer: Any) -> list[str]:
    messages = examples["messages"]
    conversations = (
        [messages] if messages and isinstance(messages[0], dict) else messages
    )
    return [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]


def _quantization_config(config: TrainingConfig, dtype: torch.dtype) -> Any | None:
    if config.peft_mode not in {PeftMode.QLORA, PeftMode.QOFT}:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def _peft_config(config: TrainingConfig) -> LoraConfig | OFTConfig:
    reward_modules = ["score"] if config.approach is TrainingApproach.REWARD else None
    task_type = "SEQ_CLS" if config.approach is TrainingApproach.REWARD else "CAUSAL_LM"
    if config.peft_mode in {PeftMode.LORA, PeftMode.QLORA}:
        return LoraConfig(
            task_type=task_type,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            modules_to_save=reward_modules,
        )
    return OFTConfig(
        task_type=task_type,
        oft_block_size=32,
        use_cayley_neumann=True,
        target_modules="all-linear",
        bias="none",
        modules_to_save=reward_modules,
    )


def _patch_qoft_peft_compatibility() -> None:
    """Bridge PEFT 0.20's mismatched 4-bit OFT dispatcher argument name."""
    global _QOFT_PEFT_PATCHED

    from peft.tuners.oft.bnb import Linear4bit

    if _QOFT_PEFT_PATCHED:
        return
    original = Linear4bit.__init__

    def compatible_init(
        self,
        base_layer,
        adapter_name,
        config=None,
        r=8,
        oft_config=None,
        **kwargs,
    ):
        selected_config = cast(OFTConfig, config or oft_config)
        return original(
            self,
            base_layer,
            adapter_name,
            selected_config,
            r=r,
            **kwargs,
        )

    Linear4bit.__init__ = compatible_init
    _QOFT_PEFT_PATCHED = True


def _load_model_and_tokenizer(
    config: TrainingConfig, token: str | None, compute_dtype: torch.dtype
) -> tuple[Any, Any, LoraConfig | OFTConfig | None]:
    if config.use_unsloth:
        if config.peft_mode not in {PeftMode.LORA, PeftMode.QLORA}:
            raise ValueError("Unsloth acceleration supports only LoRA and QLoRA.")
        fast_language_model = import_module("unsloth").FastLanguageModel
        load_options: dict[str, Any] = {
            "model_name": config.model_id,
            "max_seq_length": config.max_length,
            "dtype": compute_dtype,
            "load_in_4bit": config.peft_mode is PeftMode.QLORA,
            "load_in_16bit": config.peft_mode is PeftMode.LORA,
            "token": token,
            "revision": (
                config.model_revision if config.model_revision != "main" else None
            ),
            "use_exact_model_name": config.model_revision != "main",
            "trust_remote_code": False,
            "use_gradient_checkpointing": (
                "unsloth" if config.gradient_checkpointing else False
            ),
        }
        if config.approach is TrainingApproach.REWARD:
            load_options["num_labels"] = 1
        model, tokenizer = fast_language_model.from_pretrained(**load_options)
        if config.approach is TrainingApproach.REWARD:
            model = get_peft_model(model, _peft_config(config))
            return model, tokenizer, None
        model = fast_language_model.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            modules_to_save=(
                ["score"] if config.approach is TrainingApproach.REWARD else None
            ),
            use_gradient_checkpointing=(
                "unsloth" if config.gradient_checkpointing else False
            ),
            random_state=config.seed,
            max_seq_length=config.max_length,
        )
        return model, tokenizer, None

    model_kwargs: dict[str, Any] = {
        "token": token,
        "revision": config.model_revision,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": compute_dtype,
    }
    quantization = _quantization_config(config, compute_dtype)
    if quantization is not None:
        model_kwargs.update(
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        token=token,
        trust_remote_code=False,
    )
    if config.approach is TrainingApproach.REWARD:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id, num_labels=1, **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    return model, tokenizer, _peft_config(config)


def _apply_unsloth_trainer_patch(approach: TrainingApproach) -> None:
    if approach not in {TrainingApproach.DPO, TrainingApproach.KTO}:
        return
    unsloth = import_module("unsloth")
    patch_name = {
        TrainingApproach.DPO: "PatchDPOTrainer",
        TrainingApproach.KTO: "PatchKTOTrainer",
    }[approach]
    patch = getattr(unsloth, patch_name, None)
    if patch is not None:
        patch()


def _trainer_components(approach: TrainingApproach) -> tuple[type, type]:
    trl = import_module("trl")
    names = {
        TrainingApproach.SFT: ("SFTConfig", "SFTTrainer"),
        TrainingApproach.REWARD: ("RewardConfig", "RewardTrainer"),
        TrainingApproach.DPO: ("DPOConfig", "DPOTrainer"),
        TrainingApproach.KTO: ("KTOConfig", "KTOTrainer"),
        TrainingApproach.ORPO: ("ORPOConfig", "ORPOTrainer"),
    }
    config_name, trainer_name = names[approach]
    config_class = getattr(trl, config_name, None)
    trainer_class = getattr(trl, trainer_name, None)
    if config_class is not None and trainer_class is not None:
        return config_class, trainer_class
    experimental = import_module(f"trl.experimental.{approach.name.lower()}")
    return getattr(experimental, config_name), getattr(experimental, trainer_name)


def _trainer_config(
    config: TrainingConfig, compute_dtype: torch.dtype, *, has_evaluation: bool
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "output_dir": config.output_dir,
        "max_length": config.max_length,
        "num_train_epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "logging_steps": 1,
        "eval_strategy": "epoch" if has_evaluation else "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": config.seed,
        "push_to_hub": False,
        "hub_model_id": config.hub_model_id,
    }
    if config.approach is TrainingApproach.SFT:
        options["dataset_text_field"] = "text"
    if config.approach in {
        TrainingApproach.DPO,
        TrainingApproach.KTO,
        TrainingApproach.ORPO,
    }:
        options["beta"] = config.beta
    if config.use_unsloth:
        options.update(optim="adamw_8bit", dataset_num_proc=1)
    return options


def _resolve_compute_dtype(
    compute_type: ComputeType, *, bf16_supported: bool
) -> torch.dtype:
    resolved = resolve_compute_type(compute_type, bf16_supported=bf16_supported)
    return {
        ComputeType.BF16: torch.bfloat16,
        ComputeType.FP16: torch.float16,
        ComputeType.FP32: torch.float32,
    }[resolved]


def train(config: TrainingConfig, status_path: Path) -> dict[str, Any]:
    errors = config.validate()
    if errors:
        raise ValueError(" ".join(errors))
    token = get_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for local training.")
    if config.peft_mode is PeftMode.QOFT:
        _patch_qoft_peft_compatibility()
    if config.use_unsloth:
        _apply_unsloth_trainer_patch(config.approach)

    dataset = _load_and_combine_datasets(config, token)
    train_dataset, eval_dataset = _split_dataset(dataset, config)
    compute_dtype = _resolve_compute_dtype(
        config.compute_type,
        bf16_supported=torch.cuda.is_bf16_supported(),
    )
    model, tokenizer, peft_config = _load_model_and_tokenizer(
        config, token, compute_dtype
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_class, trainer_class = _trainer_components(config.approach)
    args = config_class(
        **_trainer_config(
            config, compute_dtype, has_evaluation=eval_dataset is not None
        )
    )
    trainer_options: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "callbacks": [StatusCallback(status_path)],
    }
    if peft_config is not None:
        trainer_options["peft_config"] = peft_config
    if (
        config.approach is TrainingApproach.SFT
        and "messages" in train_dataset.column_names
    ):
        trainer_options["formatting_func"] = lambda examples: _format_conversations(
            examples, tokenizer
        )
    trainer = trainer_class(**trainer_options)
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
