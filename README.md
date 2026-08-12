# LoRA Fine-tune Studio

[![CI](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/actions/workflows/ci.yml/badge.svg)](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pypi-ahmad/lora-qlora-fine-tuning-app)](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-2457D6.svg)](LICENSE)

A local Streamlit application for supervised fine-tuning of causal language models with LoRA or
QLoRA. It guides a user from hardware inspection and dataset validation through training,
checkpoint recovery, adapter evaluation, and optional Hugging Face Hub upload.

This is a local, single-user educational tool. GitHub hosts its source; training runs on the
Windows machine that launches the app.

## Features

- Detects CUDA, GPU VRAM, RAM, disk space, and BF16 support.
- Accepts Hugging Face model and dataset repositories or CSV, JSON, and JSONL uploads.
- Validates and previews conversational, plain-text, and prompt/completion datasets.
- Trains PEFT adapters with LoRA or four-bit QLoRA using TRL supervised fine-tuning.
- Runs one isolated worker with live progress, logs, cancellation, and checkpoint resume.
- Saves adapters, tokenizer files, metrics, and the exact training configuration locally.
- Compares base and adapter responses after training.
- Includes a separate playground for models already installed in Ollama.

## Requirements

- Windows 11 x86-64
- NVIDIA GPU with a current driver; at least 6 GB VRAM is recommended
- Internet access for first setup and Hugging Face downloads
- Hugging Face token for gated/private repositories or Hub uploads
- Ollama only when using the optional playground

Local training requires CUDA. CPU-only computers can open the interface but cannot start a run.

## Quick start

1. Download or clone this repository.
2. Set a persistent Hugging Face user environment variable:

   ```powershell
   [Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token", "User")
   ```

3. Double-click **`Launch LoRA Studio.cmd`**.

The launcher installs `uv` when missing, prepares Python 3.12 and the locked dependencies, starts
Streamlit in the background, and opens `http://localhost:8501`. First setup can take several
minutes. Startup logs are written to `.runs/streamlit.out.log` and
`.runs/streamlit.err.log`.

Public Hugging Face repositories work without a token. Use a read token for downloads and grant
write access only when pushing an adapter.

### Manual launch

```powershell
uv sync --group dev
uv run streamlit run streamlit_app.py
```

## First training run

1. Enter `Qwen/Qwen3-0.6B` as the model.
2. Upload `examples/sft_sample.jsonl`.
3. Select **QLoRA** and **Smoke test**.
4. Inspect the detected dataset format and column mapping.
5. Start training and follow the live status.
6. After completion, compare the base model with the adapter.
7. Find the adapter under `.runs/<run-id>/output/adapter`.

The smoke preset proves that the pipeline works; it does not prove production model quality.

## Dataset formats

Conversational JSONL:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Plain-text datasets use a `text` column. Prompt/completion datasets use `prompt` and `completion`,
or map equivalent columns in the interface. Uploads are limited to CSV, JSON, or JSONL and 200 MB.

## Documentation

| Document | Purpose |
| --- | --- |
| [Zero-to-hero handbook](docs/index.html) | Interactive tutorial for users, engineers, and business readers |
| [Usage guide](USAGE.md) | Complete operating manual and troubleshooting |
| [Technical reference](TECHNICAL.md) | Architecture, contracts, lifecycle, storage, and extension points |
| [Contributing](CONTRIBUTING.md) | Development workflow and pull-request requirements |
| [Security policy](SECURITY.md) | Supported versions, hardening, and private vulnerability reporting |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Community expectations and enforcement |
| [Changelog](CHANGELOG.md) | User-visible release history |

## Project boundaries

- The current trainer implements SFT, not DPO, RLHF, or RLAIF.
- Only one local training job can run at a time.
- Training and adapter comparison require an NVIDIA CUDA GPU.
- The app has no authentication and must not be exposed to an untrusted network.
- The Ollama playground does not convert or import the trained adapter.
- GitHub Pages and Streamlit Community Cloud cannot access a user's local GPU or Ollama service.

## Development

```powershell
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Security

Do not place tokens in source code or commit `.streamlit/secrets.toml`. Review model licenses,
dataset licenses, and sensitive training data before use. Report vulnerabilities privately as
described in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Ahmad Mujtaba.
