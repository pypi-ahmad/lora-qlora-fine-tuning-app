import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, OFTConfig

from lora_finetune_studio import training
from lora_finetune_studio.models import (
    ComputeType,
    DatasetSpec,
    PeftMode,
    TrainingApproach,
    TrainingConfig,
)
from lora_finetune_studio.training import (
    _format_conversations,
    _load_and_combine_datasets,
    _load_model_and_tokenizer,
    _normalize_dataset,
    _patch_qoft_peft_compatibility,
    _peft_config,
    _resolve_compute_dtype,
    _split_dataset,
    _trainer_components,
    _trainer_config,
)


def test_normalize_prompt_completion_columns() -> None:
    dataset = Dataset.from_dict({"question": ["Why?"], "answer": ["Because."]})
    spec = DatasetSpec(
        source="upload",
        local_path="sample.jsonl",
        format="prompt_completion",
        prompt_column="question",
        completion_column="answer",
    )

    normalized = _normalize_dataset(dataset, spec)

    assert normalized.column_names == ["prompt", "completion"]
    assert normalized[0] == {"prompt": "Why?", "completion": "Because."}


def test_normalize_preference_columns_preserves_conversations() -> None:
    prompt = [{"role": "user", "content": "Why?"}]
    chosen = [{"role": "assistant", "content": "Because."}]
    rejected = [{"role": "assistant", "content": "No reason."}]
    dataset = Dataset.from_dict(
        {"question": [prompt], "good": [chosen], "bad": [rejected]}
    )
    spec = DatasetSpec(
        source="upload",
        local_path="preference.jsonl",
        format="preference",
        prompt_column="question",
        chosen_column="good",
        rejected_column="bad",
    )

    normalized = _normalize_dataset(dataset, spec)

    assert normalized.column_names == ["prompt", "chosen", "rejected"]
    assert normalized[0] == {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def test_small_dataset_skips_eval_split() -> None:
    dataset = Dataset.from_dict({"text": ["a"] * 9})
    config = TrainingConfig(model_id="owner/model")

    train, evaluation = _split_dataset(dataset, config)

    assert len(train) == 9
    assert evaluation is None


def test_combine_datasets_preserves_all_rows_and_shuffles(monkeypatch) -> None:
    sources = {
        "owner/first": Dataset.from_dict({"text": ["a1", "a2"]}),
        "owner/second": Dataset.from_dict({"text": ["b1", "b2", "b3"]}),
    }
    monkeypatch.setattr(
        training,
        "load_training_dataset",
        lambda *, repo_id, **_kwargs: sources[repo_id],
    )
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(source="hub", repo_id=name, format="text") for name in sources
        ],
        seed=7,
    )

    combined = _load_and_combine_datasets(config, None)

    assert len(combined) == 5
    assert set(combined["text"]) == {"a1", "a2", "b1", "b2", "b3"}
    assert combined["text"] != ["a1", "a2", "b1", "b2", "b3"]


def test_combined_max_samples_is_global_and_deterministic(monkeypatch) -> None:
    sources = {
        "owner/first": Dataset.from_dict({"text": ["a1", "a2"]}),
        "owner/second": Dataset.from_dict({"text": ["b1", "b2", "b3"]}),
    }
    monkeypatch.setattr(
        training,
        "load_training_dataset",
        lambda *, repo_id, **_kwargs: sources[repo_id],
    )
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(source="hub", repo_id=name, format="text") for name in sources
        ],
        max_samples=3,
        seed=9,
    )

    first = _load_and_combine_datasets(config, None)
    second = _load_and_combine_datasets(config, None)

    assert len(first) == 3
    assert first["text"] == second["text"]


def test_combine_datasets_reports_incompatible_normalized_schemas(monkeypatch) -> None:
    sources = {
        "owner/first": Dataset.from_list(
            [{"messages": [{"role": "user", "content": "First"}]}]
        ),
        "owner/second": Dataset.from_list(
            [{"messages": [{"role": "user", "content": "Second", "name": "customer"}]}]
        ),
    }
    monkeypatch.setattr(
        training,
        "load_training_dataset",
        lambda *, repo_id, **_kwargs: sources[repo_id],
    )
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(source="hub", repo_id=name, format="messages")
            for name in sources
        ],
    )

    with pytest.raises(ValueError, match="incompatible schemas"):
        _load_and_combine_datasets(config, None)


def test_combine_datasets_identifies_source_load_failure(monkeypatch) -> None:
    def fail_second(*, repo_id, **_kwargs):
        if repo_id == "owner/second":
            raise OSError("download failed")
        return Dataset.from_dict({"text": ["first"]})

    monkeypatch.setattr(training, "load_training_dataset", fail_second)
    config = TrainingConfig(
        model_id="owner/model",
        datasets=[
            DatasetSpec(source="hub", repo_id="owner/first", format="text"),
            DatasetSpec(source="hub", repo_id="owner/second", format="text"),
        ],
    )

    with pytest.raises(RuntimeError, match=r"Dataset 2 \(owner/second\)"):
        _load_and_combine_datasets(config, None)


def test_unsloth_loader_uses_optimized_qlora_settings(monkeypatch) -> None:
    calls: dict = {}
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
    base_model = object()
    peft_model = object()

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return base_model, tokenizer

        @staticmethod
        def get_peft_model(model, **kwargs):
            assert model is base_model
            calls["peft"] = kwargs
            return peft_model

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = FakeFastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    config = TrainingConfig(
        model_id="owner/model",
        peft_mode=PeftMode.QLORA,
        use_unsloth=True,
        max_length=512,
        seed=7,
    )

    model, loaded_tokenizer, peft_config = _load_model_and_tokenizer(
        config, "token", torch.bfloat16
    )

    assert model is peft_model
    assert loaded_tokenizer is tokenizer
    assert peft_config is None
    assert calls["load"]["dtype"] is torch.bfloat16
    assert calls["load"]["load_in_4bit"] is True
    assert calls["load"]["load_in_16bit"] is False
    assert calls["load"]["revision"] is None
    assert calls["load"]["use_exact_model_name"] is False
    assert calls["peft"]["lora_dropout"] == 0
    assert calls["peft"]["use_gradient_checkpointing"] == "unsloth"


def test_unsloth_reward_loader_keeps_score_head(monkeypatch) -> None:
    calls: dict = {}
    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return object(), tokenizer

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft"] = kwargs
            return model

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = FakeFastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setattr(
        training, "get_peft_model", lambda model, config: (model, config)
    )
    config = TrainingConfig(
        model_id="owner/model",
        approach=TrainingApproach.REWARD,
        peft_mode=PeftMode.LORA,
        use_unsloth=True,
    )

    _, _, peft_config = _load_model_and_tokenizer(config, None, torch.bfloat16)

    assert calls["load"]["num_labels"] == 1
    assert "peft" not in calls
    assert peft_config is None


@pytest.mark.parametrize(
    ("method", "expected_type", "quantized"),
    [
        (PeftMode.LORA, LoraConfig, False),
        (PeftMode.QLORA, LoraConfig, True),
        (PeftMode.OFT, OFTConfig, False),
        (PeftMode.QOFT, OFTConfig, True),
    ],
)
def test_peft_methods_build_expected_adapter_and_quantization(
    method: PeftMode, expected_type: type, quantized: bool
) -> None:
    config = TrainingConfig(model_id="owner/model", peft_mode=method)

    peft_config = _peft_config(config)
    quantization = training._quantization_config(config, torch.bfloat16)

    assert isinstance(peft_config, expected_type)
    assert (quantization is not None) is quantized


def test_reward_peft_config_preserves_score_head() -> None:
    config = TrainingConfig(
        model_id="owner/model",
        approach=TrainingApproach.REWARD,
        peft_mode=PeftMode.OFT,
    )

    peft_config = _peft_config(config)

    assert isinstance(peft_config, OFTConfig)
    assert peft_config.modules_to_save == ["score"]


def test_qoft_compatibility_passes_dispatcher_config(monkeypatch) -> None:
    from peft.tuners.oft.bnb import Linear4bit

    calls: dict = {}

    def original(self, base_layer, adapter_name, config, r=8, **kwargs):
        calls.update(
            self=self,
            base_layer=base_layer,
            adapter_name=adapter_name,
            config=config,
            r=r,
            kwargs=kwargs,
        )

    monkeypatch.setattr(Linear4bit, "__init__", original)
    monkeypatch.setattr(training, "_QOFT_PEFT_PATCHED", False)
    config = OFTConfig(task_type="CAUSAL_LM")

    _patch_qoft_peft_compatibility()
    Linear4bit.__init__(object(), "base", "adapter", oft_config=config, r=16)

    assert calls["config"] is config
    assert calls["r"] == 16


@pytest.mark.parametrize("approach", list(TrainingApproach))
def test_every_approach_resolves_installed_trainer(approach: TrainingApproach) -> None:
    config_class, trainer_class = _trainer_components(approach)

    assert config_class.__name__.endswith("Config")
    assert trainer_class.__name__.endswith("Trainer")


@pytest.mark.parametrize(
    "approach",
    [TrainingApproach.DPO, TrainingApproach.KTO, TrainingApproach.ORPO],
)
def test_alignment_trainer_config_includes_beta(approach: TrainingApproach) -> None:
    config = TrainingConfig(
        model_id="owner/model",
        approach=approach,
        beta=0.25,
        max_grad_norm=0.75,
    )

    options = _trainer_config(config, torch.bfloat16, has_evaluation=False)

    assert options["beta"] == 0.25
    assert options["max_grad_norm"] == 0.75
    assert options["eval_strategy"] == "no"


@pytest.mark.parametrize(
    ("compute_type", "bf16_supported", "expected"),
    [
        (ComputeType.AUTO, True, torch.bfloat16),
        (ComputeType.AUTO, False, torch.float16),
        (ComputeType.BF16, False, torch.float16),
        (ComputeType.FP16, True, torch.float16),
        (ComputeType.FP32, True, torch.float32),
    ],
)
def test_resolve_compute_dtype(
    compute_type: ComputeType, bf16_supported: bool, expected: torch.dtype
) -> None:
    assert (
        _resolve_compute_dtype(compute_type, bf16_supported=bf16_supported) is expected
    )


def test_fp32_trainer_config_disables_mixed_precision_flags() -> None:
    config = TrainingConfig(model_id="owner/model")

    options = _trainer_config(config, torch.float32, has_evaluation=False)

    assert options["bf16"] is False
    assert options["fp16"] is False


def test_format_conversations_applies_chat_template_to_single_and_batch() -> None:
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_kwargs: messages[-1]["content"]
    )
    first = [{"role": "user", "content": "First"}]
    second = [{"role": "assistant", "content": "Second"}]

    assert _format_conversations({"messages": first}, tokenizer) == ["First"]
    assert _format_conversations({"messages": [first, second]}, tokenizer) == [
        "First",
        "Second",
    ]
