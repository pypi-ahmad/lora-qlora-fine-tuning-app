"""Deterministic data validation, grouping, and explicit cleanup."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from .models import DatasetSpec, TrainingApproach, TrainingConfig
from .sources import load_training_dataset


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def messages_valid(value: Any, *, assistant: bool = False) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(m, dict)
            and m.get("role") in {"system", "developer", "user", "assistant", "tool"}
            and isinstance(m.get("content"), str)
            and bool(m["content"].strip())
            for m in value
        )
        and (not assistant or any(m["role"] == "assistant" for m in value))
    )


def content_valid(value: Any, *, assistant: bool = False) -> bool:
    return (isinstance(value, str) and bool(value.strip())) or messages_valid(
        value, assistant=assistant
    )


def normalize_row(row: dict, spec: DatasetSpec) -> dict:
    mapping = {
        "text": {"text": spec.text_column or "text"},
        "messages": {"messages": "messages"},
        "prompt_completion": {
            "prompt": spec.prompt_column,
            "completion": spec.completion_column,
        },
        "preference": {
            "prompt": spec.prompt_column,
            "chosen": spec.chosen_column,
            "rejected": spec.rejected_column,
        },
    }.get(spec.format)
    if not mapping:
        raise ValueError("Unsupported dataset format.")
    result = {key: row.get(column) for key, column in mapping.items()}
    for key, value in result.items():
        if key == "messages":
            valid = messages_valid(value, assistant=True)
        elif key == "text":
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = content_valid(value, assistant=key != "prompt")
        if not valid:
            raise ValueError(f"Invalid or empty {key}.")
    if "chosen" in result and result["chosen"] == result["rejected"]:
        raise ValueError("Chosen and rejected responses are identical.")
    if "prompt" in result and any(
        isinstance(result[key], str) != isinstance(result["prompt"], str)
        for key in result
        if key != "prompt"
    ):
        raise ValueError(
            "Prompt and responses must use the same text or message format."
        )
    return result


def prompt_identity(row: dict) -> str:
    if "prompt" in row:
        prompt = row["prompt"]
    elif "messages" in row:
        prompt = [m for m in row["messages"] if m["role"] != "assistant"]
    elif "text" in row:
        prompt = row["text"]
    else:
        return digest(row)
    if (
        isinstance(prompt, list)
        and len(prompt) == 1
        and prompt[0].get("role") == "user"
    ):
        prompt = prompt[0]["content"]
    return digest(prompt)


@dataclass
class PreparedData:
    train: Dataset
    validation: Dataset | None
    report: dict[str, Any]
    membership: dict[str, list[dict[str, Any]]]


def prepare_data(config: TrainingConfig, token: str | None = None) -> PreparedData:
    report: dict[str, Any] = {
        "sources": [],
        "issues": [],
        "removed": [],
        "duplicates": 0,
        "errors": [],
        "warnings": [],
    }
    partitions: dict[str, list[dict]] = {"train": [], "validation": []}
    seen: dict[str, set[str]] = {"train": set(), "validation": set()}
    source_digests = []
    for role, specs in (
        ("train", config.datasets),
        ("validation", config.validation_datasets),
    ):
        for source_index, spec in enumerate(specs):
            dataset = load_training_dataset(
                repo_id=spec.repo_id,
                local_path=spec.local_path,
                config_name=spec.config_name,
                split=spec.split,
                revision=spec.revision,
                token=token,
            )
            source_hash = hashlib.sha256()
            accepted = 0
            for index, raw in enumerate(dataset):
                source_hash.update(digest(raw).encode())
                row_id = f"{role}:{source_index}:{index}"
                try:
                    row = normalize_row(raw, spec)
                    if spec.group_column:
                        group = raw.get(spec.group_column)
                        if (
                            group is None
                            or isinstance(group, str)
                            and not group.strip()
                        ):
                            raise ValueError("Group column is missing or empty.")
                except ValueError as error:
                    issue = {"row_id": row_id, "reason": str(error)}
                    report["issues"].append(issue)
                    if config.cleanup_invalid:
                        report["removed"].append(issue)
                    else:
                        report["errors"].append(f"{row_id}: {error}")
                    continue
                row_hash = digest(row)
                if row_hash in seen[role]:
                    report["duplicates"] += 1
                    if config.cleanup_duplicates:
                        report["removed"].append(
                            {"row_id": row_id, "reason": "Exact duplicate"}
                        )
                        continue
                seen[role].add(row_hash)
                partitions[role].append(
                    {
                        "row": row,
                        "id": row_id,
                        "hash": row_hash,
                        "prompt_hash": prompt_identity(row),
                        "group": digest(raw[spec.group_column])
                        if spec.group_column
                        else prompt_identity(row),
                    }
                )
                accepted += 1
            source_digests.append(source_hash.hexdigest())
            report["sources"].append(
                {
                    "role": role,
                    "source": spec.repo_id or spec.local_path,
                    "rows": len(dataset),
                    "accepted": accepted,
                    "hash": source_hash.hexdigest(),
                }
            )
    training = partitions["train"]
    validation = partitions["validation"]
    if config.max_samples and len(training) > config.max_samples:
        training = random.Random(config.seed).sample(training, config.max_samples)
    if validation:
        training_keys = {
            (key, r[key]) for r in training for key in ("hash", "prompt_hash", "group")
        }
        if any(
            (key, r[key]) in training_keys
            for r in validation
            for key in ("hash", "prompt_hash", "group")
        ):
            report["errors"].append(
                "Training and validation overlap. Edit the selected sources; no rows were moved."
            )
    elif config.eval_enabled and len(training) >= 10:
        # Union prompt and explicit group identities so neither can leak across splits.
        parent: dict[str, str] = {}

        def find(key: str) -> str:
            parent.setdefault(key, key)
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != key:
                next_key = parent[key]
                parent[key] = root
                key = next_key
            return root

        for row in training:
            parent[find("group:" + row["group"])] = find("prompt:" + row["prompt_hash"])
        groups = sorted({find("group:" + row["group"]) for row in training})
        random.Random(config.seed).shuffle(groups)
        if len(groups) >= 2:
            selected = set(
                groups[
                    : min(
                        len(groups) - 1, max(1, round(len(groups) * config.eval_ratio))
                    )
                ]
            )
            validation = [
                r for r in training if find("group:" + r["group"]) in selected
            ]
            training = [
                r for r in training if find("group:" + r["group"]) not in selected
            ]
        else:
            report["warnings"].append("Validation skipped: only one independent group.")
    if not training:
        report["errors"].append("No valid training rows remain.")
    if report["duplicates"] and not config.cleanup_duplicates:
        report["warnings"].append(
            f"{report['duplicates']} exact duplicate rows remain. Cleanup requires your explicit choice."
        )
    if config.eval_enabled and not validation:
        report["warnings"].append(
            "No validation set: best-checkpoint selection is unavailable."
        )
    report["training_rows"] = len(training)
    report["validation_rows"] = len(validation)
    settings = config.to_dict()
    for key in (
        "output_dir",
        "input_fingerprint",
        "resume_from_checkpoint",
        "job_kind",
        "parent_run_id",
        "evaluation",
    ):
        settings.pop(key, None)
    report["fingerprint"] = digest({"sources": source_digests, "settings": settings})
    membership = {
        role: [{k: v for k, v in row.items() if k != "row"} for row in rows]
        for role, rows in (("train", training), ("validation", validation))
    }
    return PreparedData(
        Dataset.from_list([r["row"] for r in training]),
        Dataset.from_list([r["row"] for r in validation]) if validation else None,
        report,
        membership,
    )


def token_report(dataset: Dataset, tokenizer: Any, config: TrainingConfig) -> dict:
    lengths, supervised, examples = [], [], []

    def with_eos(text: str) -> str:
        eos = tokenizer.eos_token or ""
        return text if not eos or text.endswith(eos) else text + eos

    rows: Iterator[dict[str, Any]] = (
        {**row, "chosen": answer} if "chosen" in row else row
        for row in dataset
        for answer in ([row["chosen"], row["rejected"]] if "chosen" in row else [None])
    )
    for row in rows:
        if "messages" in row:
            if not tokenizer.chat_template:
                raise ValueError("Conversational data requires a chat template.")
            encoded = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=config.loss_scope == "assistant",
                add_generation_prompt=False,
            )
            ids = encoded["input_ids"]
            mask = (
                encoded.get("assistant_masks", [])
                if config.loss_scope == "assistant"
                else [1] * len(ids)
            )
            if config.loss_scope == "assistant" and (
                len(mask) != len(ids) or not any(mask)
            ):
                raise ValueError(
                    "Assistant-only loss requires a template returning assistant masks."
                )
        elif "prompt" in row:
            answer = row.get("completion", row.get("chosen"))
            if isinstance(row["prompt"], list):
                encoded = tokenizer.apply_chat_template(
                    row["prompt"] + answer,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=config.loss_scope == "assistant",
                )
                ids = encoded["input_ids"]
                prefix = tokenizer.apply_chat_template(
                    row["prompt"], tokenize=True, add_generation_prompt=True
                )
            else:
                if config.loss_scope == "assistant":
                    raise ValueError(
                        "Assistant-only loss requires conversational data and assistant masks."
                    )
                ids = tokenizer(with_eos(row["prompt"] + answer))["input_ids"]
                prefix = tokenizer(row["prompt"])["input_ids"]
            mask = (
                ([0] * len(prefix) + [1] * max(0, len(ids) - len(prefix)))
                if config.loss_scope != "full"
                else [1] * len(ids)
            )
            if config.loss_scope == "assistant":
                assistant_mask = encoded.get("assistant_masks", [])
                if len(assistant_mask) != len(ids) or not any(assistant_mask):
                    raise ValueError(
                        "Assistant-only loss requires a template returning assistant masks."
                    )
                mask = [a * b for a, b in zip(mask, assistant_mask, strict=True)]
        else:
            ids = tokenizer(with_eos(row["text"]))["input_ids"]
            mask = [1] * len(ids)
        lengths.append(len(ids))
        supervised.append(sum(mask[: config.max_length]))
        if len(examples) < 3:
            examples.append(
                {
                    "tokens": tokenizer.convert_ids_to_tokens(ids[: config.max_length]),
                    "supervised": mask[: config.max_length],
                }
            )
    ordered = sorted(lengths)
    return {
        "rows": len(dataset),
        "sequences": len(lengths),
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "median": ordered[len(ordered) // 2] if ordered else 0,
        "truncated_rows": sum(n > config.max_length for n in lengths),
        "zero_supervised_rows": sum(n == 0 for n in supervised)
        if config.approach is TrainingApproach.SFT
        else 0,
        "loss_preview": "SFT token masks"
        if config.approach is TrainingApproach.SFT
        else "Response coverage only; the trainer applies its own sequence/pair objective and truncation.",
        "examples": examples,
    }
