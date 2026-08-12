import pytest

from lora_finetune_studio import inference


def test_generate_text_releases_cuda_memory_after_loading_failure(monkeypatch) -> None:
    cleanup_calls: list[bool] = []
    monkeypatch.setattr(inference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference.torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(inference, "BitsAndBytesConfig", lambda **_kwargs: object())
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("load failed")),
    )
    monkeypatch.setattr(
        inference,
        "release_unused_cuda_memory",
        lambda: cleanup_calls.append(True),
    )

    with pytest.raises(ValueError, match="load failed"):
        inference.generate_text("owner/model", "Hello", token=None)

    assert cleanup_calls == [True]
