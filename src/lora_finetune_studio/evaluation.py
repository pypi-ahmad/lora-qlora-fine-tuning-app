"""Persisted evaluation using local metrics and optional independent judgments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from .jobs import read_config, read_status, write_json_atomic
from .models import JobState, JobStatus, PeftMode, TrainingConfig
from .quality import content_valid, digest, messages_valid, prompt_identity
from .sources import get_hf_token


def validate_schema(schema: dict | None) -> None:
    """Validate a JSON Schema and reject references outside that schema.

    Args:
        schema: Optional Draft 2020-12 schema for generated JSON.

    Raises:
        ValueError: A reference points outside the supplied schema.
        SchemaError: The schema itself is invalid.
    """
    if schema is None:
        return
    Draft202012Validator.check_schema(schema)

    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    key in {"$ref", "$dynamicRef"}
                    and isinstance(child, str)
                    and not child.startswith("#")
                ):
                    raise ValueError(
                        "Evaluation schemas support only local references."
                    )
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)


def validate_rows(rows: list[dict], mode: str) -> None:
    """Check every evaluation row against its chat, text, or reward shape.

    Args:
        rows: Uploaded or manually entered evaluation examples.
        mode: ``chat``, ``text``, or ``reward``.

    Raises:
        TypeError: A row is not an object.
        ValueError: A row has invalid content or a non-string reference.
    """
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Evaluation row {index + 1} must be an object.")
        if mode == "reward":
            valid = all(
                content_valid(row.get(key), assistant=key != "prompt")
                for key in ("prompt", "chosen", "rejected")
            )
            valid = (
                valid
                and all(
                    isinstance(row[key], str) == isinstance(row["prompt"], str)
                    for key in ("chosen", "rejected")
                )
                and row["chosen"] != row["rejected"]
            )
        elif mode == "chat":
            valid = (
                (messages_valid(row.get("messages")) and "prompt" not in row)
                or "messages" not in row
                and isinstance(row.get("prompt"), str)
                and bool(row["prompt"].strip())
            )
        else:
            valid = (
                isinstance(row.get("prompt"), str)
                and bool(row["prompt"].strip())
                and "messages" not in row
            )
        if not valid or ("reference" in row and not isinstance(row["reference"], str)):
            raise ValueError(
                f"Evaluation row {index + 1} has invalid inputs or reference."
            )


def local_metrics(row: dict, response: str, schema: dict | None) -> dict:
    """Score exact match and JSON validity for one generated response.

    Missing references do not contribute an exact-match score.

    Args:
        row: Evaluation example with an optional reference.
        response: Generated response text.
        schema: Optional local JSON Schema for the parsed response.
    """
    metrics = {}
    if "reference" in row:
        metrics["exact_match"] = float(response.strip() == row["reference"].strip())
    try:
        value = json.loads(response)
        metrics["json_valid"] = 1.0
    except (ValueError, TypeError):
        metrics["json_valid"] = 0.0
        if schema is not None:
            metrics["schema_valid"] = 0.0
        return metrics
    if schema is not None:
        try:
            Draft202012Validator(schema).validate(value)
            metrics["schema_valid"] = 1.0
        except ValidationError:
            metrics["schema_valid"] = 0.0
    return metrics


def evaluate(config: TrainingConfig, status_path: Path) -> dict:
    """Evaluate a completed adapter and persist resumable row-level results.

    Args:
        config: Evaluation job with parent run and sample settings.
        status_path: File updated with evaluation progress.

    Returns:
        Aggregated local and optional judge metrics.

    Raises:
        ValueError: Inputs, parent state, schema, or held-out membership fail checks.
    """
    from .hardware import release_unused_cuda_memory
    from .inference import generate_rows
    from .judge import judge_pair

    options = config.evaluation
    if options is None or not config.parent_run_id:
        raise ValueError("Missing evaluation configuration.")
    errors = config.validate()
    if errors:
        raise ValueError(" ".join(errors))
    validate_rows(options.rows, options.mode)
    validate_schema(options.expected_schema)
    parent = read_config(config.parent_run_id)
    parent_status = read_status(config.parent_run_id)
    if parent_status.state is not JobState.COMPLETED or parent.job_kind != "training":
        raise ValueError("Evaluation requires a completed training run.")
    membership_path = Path(parent.output_dir).parent / "split_membership.json"
    if options.held_out and membership_path.exists():
        membership = json.loads(membership_path.read_text(encoding="utf-8"))
        known = {
            item["prompt_hash"] for values in membership.values() for item in values
        }
        if any(prompt_identity(row) in known for row in options.rows):
            raise ValueError(
                "Test prompts overlap training or validation. Select a separate test set."
            )
    rows = random.Random(config.seed).sample(
        options.rows, min(len(options.rows), options.sample_limit)
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "evaluation.json"
    fingerprint_options = config.to_dict()["evaluation"]
    if options.judge_model == "gpt-6-luna":
        # Preserve fingerprints of evaluations saved before model selection existed.
        fingerprint_options.pop("judge_model", None)
    fingerprint = digest(
        {
            "rows": rows,
            "evaluation": fingerprint_options,
            "parent": config.parent_run_id,
        }
    )
    saved = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    if saved and saved.get("fingerprint") != fingerprint:
        raise ValueError("Saved evaluation inputs changed; create a new evaluation.")
    results = saved.get("results", [{"row": row} for row in rows])
    record = {
        "fingerprint": fingerprint,
        "parent_run_id": config.parent_run_id,
        "settings": config.to_dict(),
        "results": results,
        "generation_settings": {
            "do_sample": False,
            "max_new_tokens": options.max_new_tokens,
            "quantized": parent.peft_mode in {PeftMode.QLORA, PeftMode.QOFT},
            "dtype": "bf16 if supported, otherwise fp16",
            "model_revision": parent.model_revision,
        },
        "provenance": (
            "manual comparison: not held out"
            if not options.held_out
            else "test prompts checked against saved split membership"
            if membership_path.exists()
            else "legacy: overlap cannot be verified"
        ),
    }
    quantized = parent.peft_mode in {PeftMode.QLORA, PeftMode.QOFT}
    if parent.peft_mode is PeftMode.QOFT:
        from .training import _patch_qoft_peft_compatibility

        _patch_qoft_peft_compatibility()
    for label in ["adapter"] if options.mode == "reward" else ["base", "adapter"]:
        if all(label in item for item in results):
            continue

        def progress(index, count, label=label):
            write_json_atomic(
                status_path,
                JobStatus(
                    state=JobState.RUNNING,
                    message=f"Evaluating {label}: {index}/{count}",
                    progress=index / count,
                    pid=os.getpid(),
                    artifact_dir=config.output_dir,
                ).to_dict(),
            )

        try:
            generations = generate_rows(
                parent.model_id,
                rows,
                token=get_hf_token(),
                revision=parent.model_revision,
                adapter_path=str(Path(parent.output_dir) / "adapter")
                if label == "adapter"
                else None,
                mode=options.mode,
                max_new_tokens=options.max_new_tokens,
                quantized=quantized,
                progress=progress,
            )
        finally:
            release_unused_cuda_memory()
        for item, generation in zip(results, generations, strict=True):
            item[label] = generation
            if "response" in generation:
                generation["metrics"] = local_metrics(
                    item["row"], generation["response"], options.expected_schema
                )
        write_json_atomic(result_path, record)
    if options.judge_enabled:
        for index, item in enumerate(results):
            if item.get("judge", {}).get("status") == "completed":
                continue
            write_json_atomic(
                status_path,
                JobStatus(
                    state=JobState.RUNNING,
                    message=f"AI judging {index + 1}/{len(results)}",
                    progress=index / len(results),
                    pid=os.getpid(),
                    artifact_dir=config.output_dir,
                ).to_dict(),
            )
            item["judge"] = judge_pair(
                item["row"],
                item["base"]["response"],
                item["adapter"]["response"],
                seed=config.seed + index,
                rubric=options.rubric,
                model=options.judge_model,
            )
            write_json_atomic(result_path, record)
    metrics = {"evaluated_examples": float(len(results))}
    for label in ("base", "adapter"):
        collected: dict[str, list[float]] = {}
        for item in results:
            generation = item.get(label, {})
            for key, value in {
                **generation.get("metrics", {}),
                **{k: v for k, v in generation.items() if isinstance(v, (float, int))},
            }.items():
                collected.setdefault(key, []).append(float(value))
        for key, values in collected.items():
            metrics[f"{label}_{key}"] = sum(values) / len(values)
    judgments = [
        item["judge"]
        for item in results
        if item.get("judge", {}).get("status") == "completed"
    ]
    metrics["judge_completed"] = float(len(judgments))
    metrics["judge_errors"] = float(
        sum(item.get("judge", {}).get("status") == "error" for item in results)
    )
    decisive = [item for item in judgments if item["winner"] != "abstain"]
    if decisive:
        metrics["judge_adapter_win_rate"] = sum(
            item["winner"] == "adapter" for item in decisive
        ) / len(decisive)
    record["metrics"] = metrics
    write_json_atomic(result_path, record)
    return metrics
