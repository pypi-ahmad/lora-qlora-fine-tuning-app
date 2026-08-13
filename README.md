# LoRA Fine-tune Studio

[![CI](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/actions/workflows/ci.yml/badge.svg)](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pypi-ahmad/lora-qlora-fine-tuning-app)](https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-2457D6.svg)](LICENSE)

A local Streamlit application for supervised fine-tuning, reward modeling, and preference
optimization with LoRA, QLoRA, OFT, or QOFT. It guides a user from hardware inspection and dataset validation through training,
checkpoint recovery, adapter evaluation, and optional Hugging Face Hub upload.

This is a local, single-user educational tool. GitHub hosts its source; training runs on the
Windows or Linux machine that launches the app.

## Features

- Detects CUDA, GPU VRAM, RAM, disk space, and BF16 support.
- Runs a read-only readiness scan for OS, CPU threads, available RAM, runtimes, and integrations.
- Shows live CUDA memory usage and can release unused PyTorch VRAM safely.
- Provides a confirmed shutdown control for the app and its active training worker.
- Organizes the workflow into eight focused pages in the left navigation.
- Combines multiple Hugging Face or uploaded CSV, JSON, and JSONL datasets in one training run.
- Validates conversational, plain-text, prompt/completion, and paired preference datasets.
- Trains SFT, Reward, DPO, KTO, and ORPO recipes with LoRA, QLoRA, OFT, or QOFT.
- Optionally accelerates LoRA and QLoRA on Windows with Unsloth Core.
- Runs one isolated worker with live progress, logs, cancellation, and checkpoint resume.
- Saves adapters, tokenizer files, metrics, and the exact training configuration locally.
- Compares base and adapter responses after training.
- Includes a separate playground for models already installed in Ollama.

## Requirements

- Windows 11 or x86-64 Linux, running natively
- NVIDIA GPU with a CUDA 13-compatible driver; at least 6 GB VRAM is recommended
- Git for cloning the repository
- Internet access for first setup and Hugging Face downloads
- Hugging Face token for gated/private repositories or Hub uploads
- Ollama only when using the optional playground

Local training requires CUDA. CPU-only computers can open the interface but cannot start a run.

## Install from GitHub

Repository: <https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app>

### 1. Clone the repository

Open PowerShell on Windows or a terminal on Linux:

```bash
git clone https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app.git
cd lora-qlora-fine-tuning-app
```

If Git is unavailable, download **Code → Download ZIP** from the repository page and extract it.

### 2. Configure Hugging Face access

Public models and datasets do not require a token. For gated/private repositories or Hub uploads,
create a token in your Hugging Face settings and configure `HF_TOKEN`.

Windows PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token", "User")
```

Linux shell:

```bash
echo 'export HF_TOKEN="hf_your_token"' >> ~/.bashrc
source ~/.bashrc
```

Replace `hf_your_token` with your token. Never save a real token in this repository.

### 3. Install and launch

Windows: double-click **`Launch LoRA Studio.cmd`** in File Explorer, or run:

```powershell
& ".\Launch LoRA Studio.cmd"
```

Linux:

```bash
bash "Launch LoRA Studio.sh"
```

The platform launcher installs `uv` when missing, prepares Python 3.14 and the locked CUDA 13 dependencies
inside the project `.venv`, and on Windows prepares Unsloth in `.venv-unsloth` with Python 3.13.
It then starts Streamlit in the background and opens
`http://localhost:8504`. First setup can take several minutes. Startup logs are written to
`.runs/streamlit.out.log` and
`.runs/streamlit.err.log`.

Launching again stops the previous LoRA Studio server and starts a fresh Streamlit session. An
active training worker is separate and continues until it completes or is cancelled from Monitor.

To shut down from the UI, select **Stop LoRA Studio** in the sidebar and then **Confirm stop**. This
cancels the active training worker before stopping Streamlit. Ollama and unrelated applications
are never terminated.

Public Hugging Face repositories work without a token. Use a read token for downloads and grant
write access only when pushing an adapter.

### Manual setup and launch

If `uv` is already installed, prepare the project environment directly:

```bash
uv sync --locked --no-dev --python 3.14
uv run streamlit run streamlit_app.py --server.port=8504
```

## First training run

1. Confirm CUDA and Hugging Face access on **System**.
2. Upload `examples/sft_sample.jsonl` on **Dataset**, inspect it, and select **Add dataset**.
3. Inspect `Qwen/Qwen3-0.6B` on **Model**.
4. Select **Supervised Fine-Tuning**, **QLoRA**, **Smoke test**, and whether to use Unsloth on
   **Training**, then save the settings.
5. Validate the summary and start from **Review & run**.
6. Follow the job and compare the completed adapter on **Monitor**.
7. Find the adapter under `.runs/<run-id>/output/adapter`.

The smoke preset proves that the pipeline works; it does not prove production model quality.

## Dataset formats

Conversational JSONL:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Plain-text datasets use a `text` column. Prompt/completion datasets use `prompt` and `completion`,
Preference datasets for Reward, DPO, KTO, and ORPO use `prompt`, `chosen`, and `rejected`, or map
equivalent columns in the interface. Uploads are limited to CSV, JSON, or JSONL and 200 MB.
Every dataset in one run must use the same detected or mapped format. The worker concatenates all
rows, shuffles them with the configured seed, and applies `max_samples` as a global cap.

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

- PPO, full tuning, freeze tuning, and distributed training are not implemented.
- KTO uses a minimum per-device batch size of two.
- OFT and QOFT use standard PEFT/TRL; Unsloth acceleration supports LoRA and QLoRA.
- Native Unsloth integration currently targets Windows; Linux continues to use the standard backend.
- Only one local training job can run at a time.
- Training and adapter comparison require an NVIDIA CUDA GPU.
- The VRAM cleanup control cannot free live models or memory owned by Ollama or other processes.
- The app has no authentication and must not be exposed to an untrusted network.
- The Ollama playground does not convert or import the trained adapter.
- GitHub Pages and Streamlit Community Cloud cannot access a user's local GPU or Ollama service.

## Development

```bash
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
