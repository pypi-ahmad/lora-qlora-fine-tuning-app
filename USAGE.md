# Usage Guide

This guide explains how to install and operate LoRA Fine-tune Studio. For concepts and business
context, open the [zero-to-hero handbook](docs/index.html). For implementation details, see
[TECHNICAL.md](TECHNICAL.md).

## 1. Prepare the computer

Required:

- Native Windows 11 or x86-64 Linux
- NVIDIA GPU with a CUDA 13-compatible driver
- Internet access for installation and Hugging Face downloads
- enough disk space for dependencies, model cache, checkpoints, and adapters

At least 6 GB of VRAM is recommended. Larger models, longer sequences, and larger batches require
more memory. Ollama is optional and does not participate in training.

## 2. Configure Hugging Face access

Public repositories work without authentication. A token is required for gated/private
repositories and Hub uploads.

Create a token in Hugging Face settings, then store it in your user environment.

Windows PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token", "User")
```

Linux shell (add this to your shell profile to persist it):

```bash
export HF_TOKEN="hf_your_token"
```

Use a read token for downloads. Grant write access only when you intentionally push an adapter.
Close an already running app before changing the variable, then launch it again.

The app also supports `HF_TOKEN` in an ignored local `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_your_token"
```

Never commit that file or paste a real token into an issue, screenshot, log, or documentation.

## 3. Launch the application

### Platform launch

On Windows, double-click `Launch LoRA Studio.cmd` in the repository root. On Linux, run:

```bash
bash "Launch LoRA Studio.sh"
```

The launcher:

1. checks for the project manifest, lockfile, and Streamlit entry point;
2. installs `uv` for the current user when missing;
3. synchronizes Python 3.14 and locked CUDA 13 runtime dependencies into `.venv`;
4. on Windows, synchronizes Python 3.13 and the locked native Unsloth stack into `.venv-unsloth`;
5. starts Streamlit as a hidden background process;
6. waits for the health endpoint; and
7. opens `http://localhost:8504`.

Launching again replaces the previous LoRA Studio server, so browser session state starts fresh.
An isolated training worker continues running and can be monitored after the new server starts.

If setup or startup fails, the console remains visible. Read:

- `.runs/streamlit.out.log`
- `.runs/streamlit.err.log`

### Manual launch

```bash
uv sync --group dev
uv run streamlit run streamlit_app.py --server.port=8504
```

The installed console entry point is also available after synchronization:

```bash
uv run lora-finetune-studio
```

## 4. Read the System and GPU memory pages

The **System** page reports:

- operating system release and build;
- logical CPU threads and currently available RAM;
- CUDA availability and accelerator details;
- GPU model and total VRAM;
- free disk space;
- BF16 support; and
- a conservative maximum model size;
- native Windows or Linux and uv `.venv` runtime status; and
- installed Python, uv, PyTorch, CUDA, bitsandbytes, Transformers, PEFT, TRL, Ollama, and
  Hugging Face integration status.

The scan is read-only. It never installs drivers or runtimes and never displays token values. If
current free VRAM is below 3.5 GB, the page warns that even the smallest supported QLoRA jobs need
attention before training.

The separate **GPU memory** page shows free VRAM plus memory currently allocated and reserved by
the Streamlit process. Select **Clear unused VRAM** to run Python garbage collection and release
unused PyTorch CUDA cache blocks. The button is disabled while a training worker is active; use
**Cancel training** on **Monitor** when you intend to stop that job and release its memory.

This control cannot unload a live model or release memory owned by the training worker, Ollama, or
another GPU application. Those processes must release or exit themselves.

| VRAM | Suggested maximum | Mode |
| --- | ---: | --- |
| Below 6 GB | 1B parameters | QLoRA |
| 6–9.9 GB | 3B parameters | QLoRA |
| 10–15.9 GB | 7B parameters | QLoRA |
| 16 GB or more | 13B parameters | QLoRA |

These warnings are estimates, not guarantees. Other GPU applications, sequence length, batch size,
architecture, and driver overhead affect the real limit.

## 5. Select a dataset and model

### Model

Open **Model**, then enter either:

- a Hugging Face repository ID such as `Qwen/Qwen3-0.6B`; or
- its repository-root URL, such as `https://huggingface.co/Qwen/Qwen3-0.6B`.

Optionally change the revision from `main`. File URLs, tree URLs, arbitrary hosts, HTTP URLs, query
strings, and fragments are rejected.

### Hugging Face dataset

Open **Dataset**, select **Hugging Face**, then enter a dataset repository ID or root URL. Add a
configuration name only when the dataset has multiple configurations, and select the required
split. Inspect the source, then select **Add dataset**.

### Uploaded dataset

On **Dataset**, select **Upload** and choose a CSV, JSON, or JSONL file no larger than 200 MB. The
app stores it under `.uploads` using a content hash.
Inspect and add each file separately.

### Inspect before continuing

Inspect each source on its page and verify:

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

Paired preference data for Reward, DPO, KTO, and ORPO:

```json
{"prompt":"Write a reply:","chosen":"Thanks for contacting us.","rejected":"No."}
```

When column names differ, map the source columns in the UI. Verify that every sample demonstrates
the output expected in production. Selected datasets appear together with row counts and can be
removed or remapped before training. SFT accepts `messages`, `text`, or `prompt_completion`;
preference approaches require `prompt`, `chosen`, and `rejected`.

The worker uses every row once per epoch and shuffles the combined dataset deterministically.
Larger datasets therefore contribute proportionally more examples. A preset's `max_samples` value
caps the combined dataset, not each source individually.

## 6. Configure training

Open **Training**, choose the settings below, then select **Save training settings**. Starting a
worker is intentionally reserved for **Review & run**.

### Choose Approach and Method

The support table, Approach dropdown, and linked Method dropdown share one compatibility registry.
Available approaches are Supervised Fine-Tuning, Reward Modeling, DPO, KTO, and ORPO. PPO is not
included because it requires additional policy, reference, reward, and value models.

- **LoRA** trains small low-rank adapter matrices while loading the base model at normal training
  precision. It requires more VRAM.
- **QLoRA** trains the same type of adapter while loading the base model in four-bit NF4. It is the
  default recommendation for local GPUs.
- **OFT** trains orthogonal adapters at normal training precision.
- **QOFT** combines OFT with four-bit NF4 loading.

### Choose the training backend

**Use Unsloth acceleration** defaults on when the native Windows runtime is ready. Disable it to
use the original Transformers/TRL backend. The saved choice is also used for checkpoint resume;
the app never silently switches backends. If Unsloth is unavailable, the toggle is disabled and
the page explains how to prepare it.
Selecting OFT or QOFT turns Unsloth off because those methods use the standard PEFT/TRL backend.

### Choose a preset

| Preset | Intended use |
| --- | --- |
| Smoke test | Validate the complete pipeline in 20 steps |
| Standard | Normal first experiment with evaluation |
| Quality | Longer sequences and more epochs after the pipeline is proven |

Use **Show advanced controls** only when an experiment has a stated reason. Important controls:

- maximum sequence length: context per training sample;
- maximum steps: the preset's hard training-step limit;
- beta: preference strength for DPO, KTO, and ORPO;
- batch size and gradient accumulation: effective batch versus VRAM;
- gradient checkpointing: lower VRAM in exchange for recomputation;
- evaluation ratio: held-out portion when the dataset has at least ten rows; and
- seed: repeatable sampling and splitting.

**Learning rate** defaults to the recommended value for the selected Approach. Choose **Custom** to
enter a value from `1e-7` through `1e-2`. Changing Approach restores its recommended default.

**Epochs** and **Maximum samples** default to the preset values. Custom epochs accept `0.1` through
`20`; custom sample limits accept positive integers and cap the combined shuffled dataset. Changing
Preset restores both defaults.

**Maximum gradient norm** defaults to `1.0`. Choose **Custom** to set any non-negative finite value;
`0` disables gradient clipping. **Compute type** defaults to Auto, which uses BF16 when supported
and FP16 otherwise. Explicit BF16 falls back to FP16 when necessary. FP32 is available for systems
with sufficient VRAM, is never selected automatically, and uses the standard Transformers/TRL
backend because Unsloth's optimized kernels require FP16 or BF16.

The app validates these values before a job can start.
KTO requires a per-device batch size of at least two.

### Optional Hub upload

Enable **Push adapter to Hugging Face Hub**, enter a destination repository, and use a token with
write access. Verify repository visibility before starting. The upload contains the adapter and
tokenizer, not a merged full model.

## 7. Start and monitor a run

Open **Review & run**, review warnings, acknowledge an above-recommendation model when applicable,
then select **Start training** when idle or **Add to queue** while another run is active. The app
switches to **Monitor** automatically. One worker remains active while additional jobs wait in
first-in-first-out order.

The monitor refreshes every two seconds and shows:

- queue position, model, approach, method, and preset for every waiting job;
- queued, running, completed, failed, or cancelled state;
- a step progress bar with a whole-number percentage;
- loss, evaluation loss, learning rate, and epoch when reported;
- a terminal error when present; and
- the tail of `training.log`.

Closing the browser tab does not necessarily stop the worker. Reopen the app to continue reading
durable status files.

### Cancel

Use **Cancel training** for the active worker or **Remove from queue** for a waiting run. Cancelling
the worker requests termination, waits up to ten seconds, then forces it to stop if required. The
next waiting job starts automatically and existing checkpoints remain.

### Stop LoRA Studio

Select **Stop LoRA Studio** at the bottom of the sidebar, review the warning, then select
**Confirm stop**. The app verifies and cancels its active training worker before stopping the
Streamlit server. If process ownership cannot be verified, shutdown is refused and the app remains
available. Waiting jobs remain queued and do not start during shutdown. Ollama, the browser, and
unrelated processes are not stopped.

The browser tab disconnects after shutdown and can be closed manually. Launch the app again to
start a fresh Streamlit session; saved runs, checkpoints, and adapters remain on disk, and the
first waiting job starts automatically.

### Resume

For a failed or cancelled run, select **Queue latest checkpoint**. Resume requires at least one
`checkpoint-*` directory. The highest numeric checkpoint is selected and appended to the queue.

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

On **Monitor**, after a completed run, enter a representative prompt. The app generates a
deterministic response from the quantized base model, then from the same base model with the
adapter attached.

This is a spot check. Use a held-out dataset and task-specific rubric before claiming an
improvement. Evaluate accuracy, formatting, hallucination, safety, latency, and regressions.

## 10. Use the Ollama playground

Start Ollama separately and install at least one Ollama model. The app lists models from
`http://localhost:11434/api/tags` and sends prompts to `/api/generate`.

The playground does not import, convert, merge, or test the adapter created by this training run.

## Troubleshooting

### The launcher cannot install `uv`

Check internet, proxy, antivirus, PowerShell policy, and the availability of `curl` or `wget` on
Linux. Organizations that block remote installer scripts should install `uv` using an approved
method, then launch again.

### The browser does not open

Read `.runs/streamlit.err.log`. Confirm no other process owns port `8504`, then relaunch.

### Hugging Face returns 401 or 403

Verify that the token is available, has the required scope, and that any gated-model terms were
accepted on Hugging Face.

### Dataset format is not detected

Use CSV, JSON, or JSONL. Ensure rows consistently contain `messages`, `text`, or a pair of columns
that can be mapped to prompt and completion.

### CUDA is not detected

Confirm an NVIDIA GPU and current driver are installed:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

### CUDA runs out of memory

Use QLoRA, choose a smaller model, reduce maximum sequence length, keep batch size at one, and close
other GPU applications. After a failed comparison, select **Clear unused VRAM** on **GPU memory**.
Cancel an active training job or stop the owning external application when the memory is not owned
by Streamlit. Restart the app only if its process still holds memory after cleanup.

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
