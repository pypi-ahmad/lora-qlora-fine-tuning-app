from dataclasses import replace
from typing import ClassVar

import pytest
from datasets import Dataset

from lora_finetune_studio import quality
from lora_finetune_studio.models import DatasetSpec, TrainingConfig


def config():
    return TrainingConfig(
        model_id="test/model",
        datasets=[
            DatasetSpec(
                source="hub",
                repo_id="test/data",
                format="prompt_completion",
                prompt_column="prompt",
                completion_column="completion",
            )
        ],
    )


def test_cleanup_is_explicit_and_records_removed_rows(monkeypatch):
    data = Dataset.from_list(
        [
            {"prompt": "one", "completion": "yes"},
            {"prompt": "one", "completion": "yes"},
            {"prompt": "bad", "completion": None},
        ]
    )
    monkeypatch.setattr(quality, "load_training_dataset", lambda **_: data)
    original = quality.prepare_data(config())
    assert original.report["errors"] and original.report["duplicates"] == 1
    cleaned = quality.prepare_data(
        replace(config(), cleanup_invalid=True, cleanup_duplicates=True)
    )
    assert len(cleaned.train) == 1
    assert not cleaned.report["errors"]
    assert len(cleaned.report["removed"]) == 2
    assert len(data) == 3


def test_grouped_split_is_repeatable_and_keeps_prompt_variants_together(monkeypatch):
    data = Dataset.from_list(
        [
            {"prompt": str(i // 2), "completion": str(i), "group": str(i // 4)}
            for i in range(24)
        ]
    )
    monkeypatch.setattr(quality, "load_training_dataset", lambda **_: data)
    settings = config()
    settings.datasets[0].group_column = "group"
    first = quality.prepare_data(settings)
    second = quality.prepare_data(settings)
    assert first.membership == second.membership
    assert first.validation is not None
    for key in ("group", "prompt_hash", "hash"):
        assert not {r[key] for r in first.membership["train"]} & {
            r[key] for r in first.membership["validation"]
        }


def test_explicit_validation_overlap_blocks_training(monkeypatch):
    data = Dataset.from_list([{"prompt": "same", "completion": "yes"}])
    monkeypatch.setattr(quality, "load_training_dataset", lambda **_: data)
    settings = config()
    settings.validation_datasets = [replace(settings.datasets[0], split="test")]
    prepared = quality.prepare_data(settings)
    assert any("overlap" in error for error in prepared.report["errors"])


def test_fingerprint_changes_with_inputs_and_settings(monkeypatch):
    data = Dataset.from_list([{"prompt": "one", "completion": "yes"}])
    monkeypatch.setattr(quality, "load_training_dataset", lambda **_: data)
    assert (
        quality.prepare_data(config()).report["fingerprint"]
        != quality.prepare_data(replace(config(), max_length=256)).report["fingerprint"]
    )


def test_invalid_messages_and_identical_preferences_are_rejected():
    with pytest.raises(ValueError, match="messages"):
        quality.normalize_row(
            {"messages": [{"role": "user", "content": "hi"}]},
            DatasetSpec(source="upload", format="messages"),
        )
    with pytest.raises(ValueError, match="identical"):
        quality.normalize_row(
            {"prompt": "p", "chosen": "x", "rejected": "x"},
            DatasetSpec(
                source="upload",
                format="preference",
                prompt_column="prompt",
                chosen_column="chosen",
                rejected_column="rejected",
            ),
        )


def test_assistant_masks_detect_truncation_and_missing_support():
    class Tokenizer:
        chat_template = "template"
        mask: ClassVar[list[int]] = [0, 0, 1]

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2, 3], "assistant_masks": self.mask}

        def convert_ids_to_tokens(self, ids):
            return list(map(str, ids))

    tokenizer = Tokenizer()
    settings = replace(config(), max_length=2, loss_scope="assistant")
    data = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
    )
    report = quality.token_report(data, tokenizer, settings)
    assert report["zero_supervised_rows"] == 1
    assert report["examples"][0]["supervised"] == [0, 0]
    tokenizer.mask = [0, 0, 0]
    with pytest.raises(ValueError, match="assistant masks"):
        quality.token_report(data, tokenizer, settings)


def test_prompt_identity_matches_single_user_chat_and_plain_prompt():
    assert quality.prompt_identity({"text": "hi"}) == quality.prompt_identity(
        {"prompt": "hi"}
    )
    assert quality.prompt_identity({"prompt": "hi"}) == quality.prompt_identity(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    )


def test_missing_assistant_mask_key_is_rejected():
    class Tokenizer:
        chat_template = "unsupported"

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2]}

    data = Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": "answer"}]}]
    )
    with pytest.raises(ValueError, match="assistant masks"):
        quality.token_report(
            data, Tokenizer(), replace(config(), loss_scope="assistant")
        )
