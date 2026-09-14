"""Hugging Face and local dataset boundary validation.

This module is the trust boundary between user-supplied input (repository IDs/URLs,
uploaded files) and the rest of the app: every Hub identifier and uploaded file used
elsewhere is expected to have passed through parse_hf_repo/validate_upload first.
Next file to read: training.py, which reloads and normalizes whatever this module
resolved, inside the isolated worker process.
"""

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
    # Untrusted input: this accepts either a bare "owner/name" ID or a full Hub URL,
    # but deliberately rejects anything else (other hosts, non-HTTPS, nested file or
    # revision paths, query strings, fragments) so callers only ever receive a plain
    # two-segment repo ID, never a URL with attacker-controlled path segments to
    # thread through to a filesystem or subprocess call downstream.
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
    # Hub dataset URLs are prefixed with "datasets/" (e.g. huggingface.co/datasets/x/y)
    # while model URLs are not; strip that segment only for the type it belongs to,
    # and treat a dataset URL supplied where a model is expected as an explicit error
    # rather than silently accepting it.
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
    # The caller-supplied filename is used only to derive/validate the extension; it
    # never becomes part of the stored path. Storage name is a content hash instead,
    # which both prevents path-injection via a crafted filename and deduplicates
    # identical uploads (re-uploading the same bytes reuses the existing file).
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
    # Priority order matters: a dataset could satisfy more than one shape's column
    # requirements (e.g. it might have both "messages" and "text" columns), and this
    # order is the single source of truth for which canonical format wins. Keep this
    # in sync with training._normalize_dataset, which assumes the same priority.
    if {"prompt", "chosen", "rejected"}.issubset(columns):
        detected = "preference"
    elif "messages" in columns:
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
