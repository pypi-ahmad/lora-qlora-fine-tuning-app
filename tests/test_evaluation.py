import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lora_finetune_studio.evaluation import (
    local_metrics,
    validate_rows,
    validate_schema,
)
from lora_finetune_studio.inference import format_inputs
from lora_finetune_studio.judge import judge_pair
from lora_finetune_studio.models import EvaluationConfig, TrainingConfig


def test_chat_uses_generation_prompt_and_text_stays_raw():
    calls = []

    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "chat"

        def __call__(self, prompt, **kwargs):
            return prompt

    tokenizer = Tokenizer()
    assert format_inputs(tokenizer, {"prompt": "hello"}, "chat") == "chat"
    assert calls[0][1]["add_generation_prompt"] is True
    assert calls[0][1]["tokenize"] is True
    assert format_inputs(tokenizer, {"prompt": "hello"}, "text") == "hello"
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="template"):
        format_inputs(tokenizer, {"prompt": "hello"}, "chat")


def test_metrics_preserve_missing_references_and_invalid_json():
    assert "exact_match" not in local_metrics({}, "hello", None)
    assert local_metrics({"reference": "YES"}, " yes ", None)["exact_match"] == 0
    assert local_metrics({"reference": "yes"}, " yes ", None)["exact_match"] == 1
    assert local_metrics({}, "bad", {"type": "object"})["schema_valid"] == 0
    assert local_metrics({}, '{"a":1}', {"type": "object"})["schema_valid"] == 1


def test_external_schema_references_and_malformed_rows_are_rejected():
    with pytest.raises(ValueError, match="local references"):
        validate_schema({"$ref": "https://example.com/schema"})
    with pytest.raises(ValueError, match="invalid inputs"):
        validate_rows([{"messages": "bad"}], "chat")
    with pytest.raises(ValueError, match="invalid inputs"):
        validate_rows(
            [{"prompt": "hi", "messages": [{"role": "user", "content": "different"}]}],
            "chat",
        )
    with pytest.raises(ValueError, match="invalid inputs"):
        validate_rows(
            [
                {
                    "prompt": "hi",
                    "chosen": [{"role": "assistant", "content": "yes"}],
                    "rejected": "no",
                }
            ],
            "reward",
        )


def test_evaluation_round_trip_and_legacy_defaults():
    config = TrainingConfig(
        model_id="a/b",
        job_kind="evaluation",
        parent_run_id="abc",
        evaluation=EvaluationConfig(rows=[{"prompt": "hi"}]),
    )
    assert TrainingConfig.from_dict(config.to_dict()) == config
    legacy = TrainingConfig(model_id="a/b").to_dict()
    legacy.pop("schema_version")
    legacy.pop("select_best_checkpoint")
    restored = TrainingConfig.from_dict(legacy)
    assert restored.schema_version == 1
    assert not restored.select_best_checkpoint
    assert replace(config, parent_run_id=None).validate()


def test_judge_contract_and_anonymized_mapping(monkeypatch):
    import openai

    captured = {}
    score = {"instruction_following": 5, "correctness": 5, "clarity": 5}
    response = SimpleNamespace(
        status="completed",
        output_text=json.dumps(
            {"A": score, "B": score, "winner": "tie", "justification": "Equal"}
        ),
        usage=None,
        id="test",
    )

    class Client:
        def __init__(self, **kwargs):
            self.responses = self
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            captured.update(kwargs)
            return response

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **kwargs: None)
    result = judge_pair({"prompt": "hi"}, "a", "b", seed=42, rubric="Be accurate")
    assert result["status"] == "completed"
    assert captured["model"] == "gpt-6-luna"
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["store"] is False and "tools" not in captured
    assert "fixture-secret" not in json.dumps(result)
    response.status = "incomplete"
    assert (
        judge_pair({"prompt": "hi"}, "a", "b", seed=42, rubric="test")["status"]
        == "error"
    )
    response.status = "completed"
    response.output_text = '{"winner":"A"}'
    assert (
        judge_pair({"prompt": "hi"}, "a", "b", seed=42, rubric="test")["status"]
        == "error"
    )


def test_missing_judge_credentials_are_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        judge_pair({"prompt": "hi"}, "a", "b", seed=1, rubric="test")["error"]
        == "Missing OPENAI_API_KEY"
    )


def test_agnes_uses_its_own_credentials_and_validates_output(monkeypatch):
    import openai

    captured = {}
    scores = {"instruction_following": 5, "correctness": 5, "clarity": 5}
    message = SimpleNamespace(
        content=json.dumps(
            {"A": scores, "B": scores, "winner": "tie", "justification": "Equal"}
        ),
        refusal=None,
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
        id="fixture",
    )

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            captured["request"] = kwargs
            return response

    monkeypatch.setenv("AGNESAI_API_KEY", "agnes-fixture-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **kwargs: None)
    result = judge_pair(
        {"prompt": "hi"}, "a", "b", seed=42, rubric="test", model="agnes-3.0-flash"
    )
    assert result["status"] == "completed"
    assert result["model"] == captured["request"]["model"] == "agnes-3.0-flash"
    assert captured["client"]["api_key"] == "agnes-fixture-secret"
    assert captured["client"]["base_url"] == "https://apihub.agnes-ai.com/v1"
    assert "agnes-fixture-secret" not in json.dumps(result)
    message.content = "not JSON"
    assert (
        judge_pair(
            {"prompt": "hi"}, "a", "b", seed=42, rubric="test", model="agnes-3.0-flash"
        )["status"]
        == "error"
    )
    monkeypatch.delenv("AGNESAI_API_KEY")
    assert (
        judge_pair({}, "a", "b", seed=1, rubric="test", model="agnes-3.0-flash")[
            "error"
        ]
        == "Missing AGNESAI_API_KEY"
    )


def test_judge_model_config_defaults_and_rejects_unknown_models():
    assert EvaluationConfig(rows=[{"prompt": "hi"}]).judge_model == "gpt-6-luna"
    assert EvaluationConfig(rows=[{"prompt": "hi"}], judge_model="unknown").validate()
