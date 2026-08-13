from pathlib import Path

import pytest
from datasets import Dataset

from lora_finetune_studio.sources import (
    inspect_dataset,
    parse_hf_repo,
    save_upload,
    validate_upload,
)


@pytest.mark.parametrize(
    ("value", "repo_type", "expected"),
    [
        ("Qwen/Qwen3-0.6B", "model", "Qwen/Qwen3-0.6B"),
        ("https://huggingface.co/Qwen/Qwen3-0.6B", "model", "Qwen/Qwen3-0.6B"),
        (
            "https://huggingface.co/datasets/trl-lib/Capybara",
            "dataset",
            "trl-lib/Capybara",
        ),
    ],
)
def test_parse_hf_repo(value: str, repo_type: str, expected: str) -> None:
    assert parse_hf_repo(value, repo_type=repo_type) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://huggingface.co/owner/repo",
        "https://example.com/owner/repo",
        "https://huggingface.co/owner/repo/tree/main",
        "owner/repo/extra",
    ],
)
def test_parse_hf_repo_rejects_unsafe_locations(value: str) -> None:
    with pytest.raises(ValueError):
        parse_hf_repo(value, repo_type="model")


def test_inspect_dataset_detects_messages() -> None:
    dataset = Dataset.from_dict({"messages": [[{"role": "user", "content": "Hi"}]]})

    result = inspect_dataset(dataset)

    assert result.format == "messages"
    assert result.rows == 1


def test_inspect_dataset_detects_preference_before_prompt_completion() -> None:
    dataset = Dataset.from_dict(
        {
            "prompt": ["Why?"],
            "chosen": ["Because."],
            "rejected": ["No reason."],
            "completion": ["Unused"],
        }
    )

    result = inspect_dataset(dataset)

    assert result.format == "preference"


def test_upload_validation_and_content_addressing(tmp_path: Path) -> None:
    content = b'{"text":"hello"}\n'

    path = save_upload("sample.jsonl", content, tmp_path)

    assert path.read_bytes() == content
    assert validate_upload("sample.jsonl", len(content)) == ".jsonl"


def test_upload_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="CSV, JSON, or JSONL"):
        validate_upload("dataset.py", 10)
