"""Hugging Face and local dataset boundary validation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

ALLOWED_UPLOAD_SUFFIXES = {".csv", ".json", ".jsonl"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass(slots=True)
class DatasetInspection:
    columns: list[str]
    format: str
    rows: int
    preview: list[dict[str, object]]


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN")


def parse_hf_repo(value: str, *, repo_type: str) -> str:
    candidate = value.strip().rstrip("/")
    if not candidate:
        raise ValueError("Repository is required.")
    if "://" not in candidate:
        parts = candidate.split("/")
        if len(parts) == 2 and all(parts):
            return candidate
        raise ValueError("Use a Hugging Face repository ID like owner/name.")

    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.hostname not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise ValueError("Only https://huggingface.co repository URLs are allowed.")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if repo_type == "dataset" and parts[:1] == ["datasets"]:
        parts = parts[1:]
    if repo_type == "model" and parts[:1] in (["models"], ["datasets"]):
        if parts[0] == "datasets":
            raise ValueError("Expected a model repository URL, not a dataset URL.")
        parts = parts[1:]
    if len(parts) != 2 or parsed.query or parsed.fragment:
        raise ValueError(
            "Use the repository root URL without files, revisions, or query parameters."
        )
    return "/".join(parts)


def token_identity(token: str | None = None) -> str | None:
    if not token:
        return None
    details = HfApi(token=token).whoami()
    return str(details.get("name") or details.get("fullname") or "authenticated user")


def model_parameter_count(
    repo_id: str, revision: str = "main", token: str | None = None
) -> int | None:
    info = HfApi(token=token).model_info(
        repo_id, revision=revision, expand=["safetensors"]
    )
    safetensors = info.safetensors
    return int(safetensors.total) if safetensors and safetensors.total else None


def validate_upload(filename: str, size: int) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError("Dataset upload must be CSV, JSON, or JSONL.")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("Dataset upload exceeds the 200 MB limit.")
    return suffix


def save_upload(filename: str, content: bytes, root: Path = Path(".uploads")) -> Path:
    suffix = validate_upload(filename, len(content))
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()[:16]
    path = root / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(content)
    return path.resolve()


def load_training_dataset(
    *,
    repo_id: str | None = None,
    local_path: str | None = None,
    config_name: str | None = None,
    split: str = "train",
    token: str | None = None,
) -> Dataset:
    if repo_id:
        return load_dataset(repo_id, config_name, split=split, token=token)
    if not local_path:
        raise ValueError("Dataset source is missing.")
    path = Path(local_path).resolve()
    if not path.is_file() or path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError("Local dataset path is invalid.")
    loader = "json" if path.suffix.lower() in {".json", ".jsonl"} else "csv"
    return load_dataset(loader, data_files=str(path), split="train")


def inspect_dataset(dataset: Dataset, limit: int = 5) -> DatasetInspection:
    columns = list(dataset.column_names)
    if "messages" in columns:
        detected = "messages"
    elif "text" in columns:
        detected = "text"
    elif {"prompt", "completion"}.issubset(columns):
        detected = "prompt_completion"
    else:
        detected = "needs_mapping"
    preview = [dict(dataset[index]) for index in range(min(limit, len(dataset)))]
    return DatasetInspection(
        columns=columns, format=detected, rows=len(dataset), preview=preview
    )
