from datasets import Dataset

from lora_finetune_studio.models import DatasetSpec, TrainingConfig
from lora_finetune_studio.training import _normalize_dataset, _split_dataset


def test_normalize_prompt_completion_columns() -> None:
    dataset = Dataset.from_dict({"question": ["Why?"], "answer": ["Because."]})
    config = TrainingConfig(
        model_id="owner/model",
        dataset=DatasetSpec(
            source="upload",
            local_path="sample.jsonl",
            format="prompt_completion",
            prompt_column="question",
            completion_column="answer",
        ),
    )

    normalized = _normalize_dataset(dataset, config)

    assert normalized.column_names == ["prompt", "completion"]
    assert normalized[0] == {"prompt": "Why?", "completion": "Because."}


def test_small_dataset_skips_eval_split() -> None:
    dataset = Dataset.from_dict({"text": ["a"] * 9})
    config = TrainingConfig(model_id="owner/model")

    train, evaluation = _split_dataset(dataset, config)

    assert len(train) == 9
    assert evaluation is None
