import json
from types import SimpleNamespace

import pytest
from datasets import Dataset

from lora_finetune_studio import provenance
from lora_finetune_studio.models import TrainingConfig
from lora_finetune_studio.quality import PreparedData


def test_manifest_preserves_cleaned_data_and_rejects_resume_drift(
    tmp_path, monkeypatch
):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "fixture GPU")
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="fixture"),
    )
    config = TrainingConfig(
        model_id="test/model",
        output_dir=str(tmp_path / "output"),
        input_fingerprint="original",
    )
    prepared = PreparedData(
        Dataset.from_list([{"text": "kept"}]),
        None,
        {"sources": [], "removed": [{"row_id": "train:0:1", "reason": "invalid"}]},
        {"train": [{"id": "train:0:0"}], "validation": []},
    )
    tokenizer = SimpleNamespace(chat_template="template")
    model = SimpleNamespace(peft_config={})
    provenance.save_manifest(config, prepared, tokenizer, model)
    assert json.loads(
        (tmp_path / "cleaned_train.jsonl").read_text(encoding="utf-8")
    ) == {"text": "kept"}
    provenance.save_manifest(config, prepared, tokenizer, model)
    tokenizer.chat_template = "changed"
    with pytest.raises(ValueError, match="template_hash"):
        provenance.save_manifest(config, prepared, tokenizer, model)
    tokenizer.chat_template = "template"
    prepared.membership["train"] = [{"id": "different"}]
    with pytest.raises(ValueError, match="split membership"):
        provenance.save_manifest(config, prepared, tokenizer, model)
