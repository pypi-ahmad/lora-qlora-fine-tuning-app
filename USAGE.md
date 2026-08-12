# Usage Guide

This guide explains how to install and operate LoRA Fine-tune Studio. For concepts and business
context, open the [zero-to-hero handbook](docs/index.html). For implementation details, see
[TECHNICAL.md](TECHNICAL.md).

## 1. Prepare the computer

Required:

- Windows 11 x86-64
- NVIDIA GPU with a current driver
- Internet access for installation and Hugging Face downloads
- enough disk space for dependencies, model cache, checkpoints, and adapters

At least 6 GB of VRAM is recommended. Larger models, longer sequences, and larger batches require
more memory. Ollama is optional and does not participate in training.

## 2. Configure Hugging Face access

Public repositories work without authentication. A token is required for gated/private
repositories and Hub uploads.

Create a token in Hugging Face settings, then store it as a persistent Windows user variable:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token", "User")
```

Use a read token for downloads. Grant write access only when you intentionally push an adapter.
Close an already running app before changing the variable, then launch it again.

The app also supports `HF_TOKEN` in an ignored local `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token"
```

Never commit that file or paste a real token into an issue, screenshot, log, or documentation.

## 3. Launch the application

### One-click launch

Double-click `Launch LoRA Studio.cmd` in the repository root.

The launcher:

1. checks for the project manifest, lockfile, and Streamlit entry point;
2. installs `uv` for the Windows user when missing;
3. synchronizes Python 3.12 and locked runtime dependencies;
4. starts Streamlit as a hidden background process;
5. waits for the health endpoint; and
6. opens `http://localhost:8501`.

If setup or startup fails, the console remains visible. Read:

- `.runs/streamlit.out.log`
- `.runs/streamlit.err.log`

### Manual launch

```powershell
uv sync --group dev
uv run streamlit run streamlit_app.py
```

The installed console entry point is also available after synchronization:

```powershell
uv run lora-finetune-studio
```

## 4. Read the hardware panel

The sidebar reports:

- CUDA availability;
- GPU model and total VRAM;
- system RAM;
- free disk space;
- BF16 support; and
- a conservative maximum model size.

| VRAM | Suggested maximum | Mode |
| --- | ---: | --- |
| Below 6 GB | 1B parameters | QLoRA |
| 6–9.9 GB | 3B parameters | QLoRA |
| 10–15.9 GB | 7B parameters | QLoRA |
| 16 GB or more | 13B parameters | QLoRA |

These warnings are estimates, not guarantees. Other GPU applications, sequence length, batch size,
architecture, and driver overhead affect the real limit.

## 5. Select a model and dataset

### Model

Enter either:

- a Hugging Face repository ID such as `Qwen/Qwen3-0.6B`; or
- its repository-root URL, such as `https://huggingface.co/Qwen/Qwen3-0.6B`.

Optionally change the revision from `main`. File URLs, tree URLs, arbitrary hosts, HTTP URLs, query
strings, and fragments are rejected.

### Hugging Face dataset

Select **Hugging Face**, then enter a dataset repository ID or root URL. Add a configuration name
only when the dataset has multiple configurations, and select the required split.

### Uploaded dataset

Select **Upload** and choose a CSV, JSON, or JSONL file no larger than 200 MB. The app stores it
under `.uploads` using a content hash.

### Inspect before continuing

Select the inspection action and verify:

- total rows and columns;
- previewed content;
- detected format; and
- model parameter count when metadata is available.

Supported dataset shapes:

Conversational JSONL:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Plain text:

```json
{"text":"One complete training sequence."}
```

Prompt/completion:

```json
{"prompt":"Write a reply:","completion":"Thanks for contacting us."}
```

When column names differ, map the source columns in the UI. Verify that every sample demonstrates
the output expected in production.

## 6. Configure training

### Choose PEFT mode

- **LoRA** trains small low-rank adapter matrices while loading the base model at normal training
  precision. It requires more VRAM.
- **QLoRA** trains the same type of adapter while loading the base model in four-bit NF4. It is the
  default recommendation for local GPUs.

### Choose a preset

| Preset | Intended use |
| --- | --- |
| Smoke test | Validate the complete pipeline in 20 steps |
| Standard | Normal first experiment with evaluation |
| Quality | Longer sequences and more epochs after the pipeline is proven |

Use **Show advanced controls** only when an experiment has a stated reason. Important controls:

- maximum sequence length: context per training sample;
- epochs: passes over selected samples;
- maximum steps/samples: hard limits for experiments;
- learning rate: adapter update size;
- batch size and gradient accumulation: effective batch versus VRAM;
- gradient checkpointing: lower VRAM in exchange for recomputation;
- evaluation ratio: held-out portion when the dataset has at least ten rows; and
- seed: repeatable sampling and splitting.

The app validates sequence length from 128 through 8192 and epochs above zero through 20.

### Optional Hub upload

Enable **Push adapter to Hugging Face Hub**, enter a destination repository, and use a token with
write access. Verify repository visibility before starting. The upload contains the adapter and
tokenizer, not a merged full model.

## 7. Start and monitor a run

Review warnings, acknowledge an above-recommendation model when applicable, then start training.
Only one worker may be active.

The monitor refreshes every two seconds and shows:

- queued, running, completed, failed, or cancelled state;
- step progress;
- loss, evaluation loss, learning rate, and epoch when reported;
- a terminal error when present; and
- the tail of `training.log`.

Closing the browser tab does not necessarily stop the worker. Reopen the app to continue reading
durable status files.

### Cancel

Use **Cancel training**. The job manager requests termination, waits up to ten seconds, then forces
the process to stop if required. Existing checkpoints remain.

### Resume

For a failed or cancelled run, select **Resume latest checkpoint**. Resume requires at least one
`checkpoint-*` directory. The highest numeric checkpoint is selected.

## 8. Find the results

Each run is stored under `.runs/<run-id>`:

```text
config.json
status.json
training.log
output/
├── checkpoint-*/
├── adapter/
├── metrics.json
└── training_config.json
```

Keep the base-model ID and revision with the adapter. A PEFT adapter does not contain the complete
base model.

## 9. Compare base and adapter

After a completed run, enter a representative prompt. The app generates a deterministic response
from the quantized base model, then from the same base model with the adapter attached.

This is a spot check. Use a held-out dataset and task-specific rubric before claiming an
improvement. Evaluate accuracy, formatting, hallucination, safety, latency, and regressions.

## 10. Use the Ollama playground

Start Ollama separately and install at least one Ollama model. The app lists models from
`http://localhost:11434/api/tags` and sends prompts to `/api/generate`.

The playground does not import, convert, merge, or test the adapter created by this training run.

## Troubleshooting

### The launcher cannot install `uv`

Check internet, proxy, antivirus, and PowerShell policy. Organizations that block remote installer
scripts should install `uv` using an approved method, then launch again.

### The browser does not open

Read `.runs/streamlit.err.log`. Confirm no other process owns port `8501`, then relaunch.

### Hugging Face returns 401 or 403

Verify that the token is available, has the required scope, and that any gated-model terms were
accepted on Hugging Face.

### Dataset format is not detected

Use CSV, JSON, or JSONL. Ensure rows consistently contain `messages`, `text`, or a pair of columns
that can be mapped to prompt and completion.

### CUDA is not detected

Confirm an NVIDIA GPU and current driver are installed:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

### CUDA runs out of memory

Use QLoRA, choose a smaller model, reduce maximum sequence length, keep batch size at one, and close
other GPU applications. Restart the app after an out-of-memory failure if GPU memory remains held.

### Resume says no checkpoint is available

The run stopped before the trainer's first checkpoint. Start a new run, or use a smaller smoke-test
configuration to validate the environment.

### Ollama shows no models

Start Ollama and install a model. This does not affect Hugging Face training.

## Safe use

- Do not expose the unauthenticated Streamlit server to an untrusted network.
- Do not train secrets or restricted personal data into model weights.
- Review model and dataset licenses before training or redistribution.
- Protect `.uploads`, `.runs`, model caches, checkpoints, adapters, and logs.
- Report security problems through [SECURITY.md](SECURITY.md).
