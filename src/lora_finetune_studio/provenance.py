"""Immutable input coordinates and reproducibility records."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from .jobs import write_json_atomic
from .models import TrainingConfig
from .quality import PreparedData, digest, prepare_data


def pin_revisions(config: TrainingConfig, token: str | None = None) -> None:
    """Replace Hub model and dataset revisions with immutable commit hashes.

    Args:
        config: Run configuration updated in place.
        token: Hugging Face token for gated repositories, if needed.

    Raises:
        ValueError: Hub metadata omits a model or dataset commit.
    """
    api = HfApi(token=token)
    revision = api.model_info(config.model_id, revision=config.model_revision).sha
    if not revision:
        raise ValueError("Hub did not return a model commit.")
    config.model_revision = revision
    for spec in config.datasets + config.validation_datasets:
        if spec.repo_id:
            revision = api.dataset_info(spec.repo_id, revision=spec.revision).sha
            if not revision:
                raise ValueError("Hub did not return a dataset commit.")
            spec.revision = revision


def prepare_run(config: TrainingConfig, token: str | None = None) -> PreparedData:
    """Pin sources and reject data that differs from the reviewed fingerprint.

    Args:
        config: New-format training or fit-check configuration.
        token: Hugging Face token passed to source loading.

    Raises:
        ValueError: Data review fails or inputs changed after review.
    """
    pin_revisions(config, token)
    prepared = prepare_data(config, token)
    if prepared.report["errors"]:
        raise ValueError("\n".join(prepared.report["errors"][:20]))
    if (
        config.input_fingerprint
        and prepared.report["fingerprint"] != config.input_fingerprint
    ):
        raise ValueError(
            "Inputs or settings changed since review. Run the quality check again."
        )
    config.input_fingerprint = prepared.report["fingerprint"]
    return prepared


def save_manifest(
    config: TrainingConfig, prepared: PreparedData, tokenizer, model
) -> None:
    """Persist provenance once and reject changes before checkpoint resume.

    Args:
        config: Configuration whose output directory identifies the run.
        prepared: Reviewed datasets, report, and split membership.
        tokenizer: Loaded tokenizer used to hash the chat template.
        model: Loaded model whose PEFT settings are recorded.

    Raises:
        ValueError: Saved membership or runtime inputs differ on resume.
    """
    import json

    import torch

    root = Path(config.output_dir).parent
    manifest_path = root / "manifest.json"
    versions = {}
    for package in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "huggingface-hub",
        "unsloth",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={Path.cwd().as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision, driver = "unknown", "unknown"
    manifest = {
        "version": 1,
        "input_fingerprint": config.input_fingerprint,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "sources": prepared.report["sources"],
        "template_hash": digest(tokenizer.chat_template),
        "settings": config.to_dict(),
        "packages": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "driver": driver,
        "application_revision": revision,
        "adapter_settings": {
            k: v.to_dict() for k, v in getattr(model, "peft_config", {}).items()
        },
    }
    if manifest_path.exists():
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        membership_path = root / "split_membership.json"
        if (
            not membership_path.exists()
            or json.loads(membership_path.read_text(encoding="utf-8"))
            != prepared.membership
        ):
            raise ValueError(
                "Cannot resume: split membership differs from the original run."
            )
        for key in ("input_fingerprint", "template_hash", "packages", "python", "cuda"):
            if original.get(key) != manifest[key]:
                raise ValueError(
                    f"Cannot resume: {key} differs from the original manifest."
                )
    else:
        # PEFT settings may contain sets; store portable JSON without credentials.
        write_json_atomic(
            manifest_path,
            json.loads(
                json.dumps(
                    manifest,
                    default=lambda v: sorted(v) if isinstance(v, set) else str(v),
                )
            ),
        )
        write_json_atomic(root / "split_membership.json", prepared.membership)
        write_json_atomic(root / "quality_report.json", prepared.report)
        if prepared.report["removed"]:
            for role, dataset in (
                ("train", prepared.train),
                ("validation", prepared.validation),
            ):
                if dataset is not None:
                    dataset.to_json(root / f"cleaned_{role}.jsonl", force_ascii=False)
