"""Small Ollama HTTP client; no additional dependency required."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://localhost:11434"


def _request(
    path: str, payload: dict[str, object] | None = None, timeout: int = 5
) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{DEFAULT_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def list_models() -> list[str]:
    response = _request("/api/tags")
    return [str(item["name"]) for item in response.get("models", [])]


def generate(model: str, prompt: str) -> str:
    response = _request(
        "/api/generate", {"model": model, "prompt": prompt, "stream": False}, 120
    )
    return str(response.get("response", ""))
