# Setup Guide

This guide explains how to clone, install, verify, run, update, and troubleshoot LoRA Fine-tune
Studio on native Windows 11 and x86-64 Linux. It covers the normal application runtime, the
optional Windows Unsloth runtime, Hugging Face access, development tools, and every direct project
dependency. To inspect the guided workflow without a GPU or the CUDA stack, use the
[read-only showcase](#41-read-only-showcase-no-cuda) after cloning.

Repository: <https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app>

## 1. Understand the two runtimes

The repository deliberately keeps the application and Unsloth in separate environments:

| Runtime | Platforms | Python | Directory | Purpose |
| --- | --- | --- | --- | --- |
| Main application | Windows 11 and x86-64 Linux | 3.14 | `.venv` | Streamlit UI, standard PEFT/TRL training, tests, and utilities |
| Native Unsloth | Windows 11 only in this app | 3.13.13 | `.venv-unsloth` | Optional accelerated LoRA and QLoRA training worker |

The Windows launcher prepares both environments. The Linux launcher prepares only the main
environment because the app's native Unsloth integration is currently Windows-only. Linux users
can still train with the standard backend.

Do not install Unsloth into `.venv`, and do not replace the locked environments with a shared
system Python environment. The separation prevents incompatible Python, PyTorch, and CUDA package
requirements from colliding.

## 2. System requirements

### Supported operating systems

- Windows 11 x86-64, running natively without WSL
- x86-64 Linux with Bash and glibc 2.24 or newer

macOS, Windows on ARM, Linux on ARM, WSL, Docker, AMD ROCm, Intel XPU, and CPU training are not
supported by this application configuration.

### GPU and driver

Local training requires:

- an NVIDIA CUDA-capable GPU;
- an NVIDIA driver from branch 580 or newer, because the locked PyTorch wheels use CUDA 13.0; and
- enough VRAM for the selected model, sequence length, batch size, and training method.

At least 6 GB VRAM is recommended. The current CUDA 13.0 `bitsandbytes` wheels target NVIDIA
compute capability 7.5 and newer, so Turing/RTX 20-series or newer is the practical baseline for
QLoRA. A CPU-only computer can install and open the interface, but the app will reject a training
run.

Check the GPU before setup:

```text
nvidia-smi
```

The command must show the NVIDIA GPU and driver version. Install or update the driver from the
[official NVIDIA driver page](https://www.nvidia.com/Download/index.aspx) when it is missing or
older than branch 580. NVIDIA documents CUDA 13.x driver compatibility in the
[CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The normal locked-wheel installation does not require a separately installed CUDA Toolkit. A
local toolkit and compiler are needed only when rebuilding CUDA packages from source, which is not
part of the supported setup path.

### Memory, storage, and network

Practical recommendations:

- 16 GB system RAM or more;
- at least 20 GB free disk space for the two Windows environments, package caches, and initial
  setup, plus additional space for models, datasets, checkpoints, and adapters; and
- a stable internet connection during initial setup and model downloads.

The first installation contacts GitHub, Astral's Python/`uv` distribution services, PyPI,
`download.pytorch.org`, and Hugging Face. Windows also downloads the locked Unsloth packages.
Configure an organizational proxy or firewall to allow those services when required.

### Required system tools

Windows needs:

- Git;
- Windows PowerShell, which is included with Windows 11;
- Command Prompt support for the `.cmd` launcher; and
- a web browser.

Linux needs:

- Git;
- Bash;
- either `curl` or `wget` so the launcher can install `uv`;
- `nohup` and a normal `/proc` filesystem; and
- optionally `xdg-open` to open the browser automatically.

Python, `pip`, Conda, Visual Studio, and the CUDA Toolkit do not need to be installed manually for
the standard setup. `uv` downloads the required Python versions.

## 3. Install Git

### Windows

Install Git for Windows from the [official Git download page](https://git-scm.com/install/windows),
or use Windows Package Manager:

```powershell
winget install --id Git.Git -e --source winget
```

Close and reopen PowerShell after installation, then verify:

```powershell
git --version
```

### Linux

Use the distribution package manager. Examples from the
[official Git Linux installation page](https://git-scm.com/install/linux) are:

```bash
# Debian or Ubuntu
sudo apt update
sudo apt install git curl

# Fedora
sudo dnf install git curl

# Arch Linux
sudo pacman -S git curl
```

Then verify:

```bash
git --version
```

## 4. Clone the application

Open PowerShell on Windows or a terminal on Linux, change to the parent directory where the project
should live, and run:

```bash
git clone https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app.git
cd lora-qlora-fine-tuning-app
```

GitHub's official explanation is available in
[Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

If Git cannot be installed, use **Code → Download ZIP** on GitHub and extract the archive. A ZIP
copy can run the app, but it cannot use `git pull` for updates.

Confirm that these files exist in the project root before continuing:

```text
pyproject.toml
uv.lock
streamlit_app.py
demo/streamlit_app.py
demo/requirements.txt
Launch LoRA Studio.cmd
Launch LoRA Studio.sh
unsloth-runtime/pyproject.toml
unsloth-runtime/uv.lock
```

### 4.1 Read-only showcase, no CUDA

After the clone, reviewers can run the isolated showcase without creating `.venv` or installing
PyTorch:

```powershell
uvx --from streamlit==1.61.1 streamlit run demo/streamlit_app.py
```

That process uses only Streamlit and `demo/fixtures/showcase.json`. It does not read `HF_TOKEN`,
download models, start a worker, or write `.runs`. Continue with the rest of this guide only when
you need the production training application.

## 5. Configure Hugging Face access

Public Hugging Face models and datasets work without a token. A token is required for gated or
private repositories and for uploading an adapter to the Hub.

Create a token from [Hugging Face access-token settings](https://huggingface.co/settings/tokens).
Use a read or narrowly scoped token for downloads. Use write permission only when the app must push
an adapter. See the official [token security documentation](https://huggingface.co/docs/hub/security-tokens).

### Windows PowerShell

Set a token for only the current terminal:

```powershell
$env:HF_TOKEN = "hf_replace_with_your_token"
```

Or save it to the current Windows user environment:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_replace_with_your_token", "User")
```

Open a new terminal after setting a persistent user variable. The Windows launcher also reads the
user-level variable when it starts the hidden Streamlit process.

### Linux Bash

Set a token for the current terminal:

```bash
export HF_TOKEN="hf_replace_with_your_token"
```

To persist it, add that export to the appropriate private shell profile and restrict access to the
profile. Launch the app from a terminal that has loaded the variable.

### Streamlit secrets alternative

The app also reads an ignored local file at `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_replace_with_your_token"
```

Never commit this file, paste a real token into documentation, or share it in logs or screenshots.
If a token is exposed, revoke it immediately in Hugging Face settings. The official environment
variable behavior is documented in the
[`huggingface_hub` environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

## 6. Recommended automatic installation

The launchers install `uv` for the current user when it is missing. `uv` then downloads the exact
Python version, creates the local virtual environment, and installs packages from the committed
lockfile. See the official [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/),
[Python management guide](https://docs.astral.sh/uv/guides/install-python/), and
[locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/).

The launchers execute Astral's official `uv` installer when necessary. Users who prefer to review
and install tools separately can install `uv` first using the official guide, then run the same
launcher.

### Windows

Double-click `Launch LoRA Studio.cmd` in File Explorer, or run it from PowerShell:

```powershell
& ".\Launch LoRA Studio.cmd"
```

The launcher:

1. confirms that the project manifest, lockfile, and Streamlit entry point exist;
2. locates `uv.exe` or installs `uv` for the current Windows user;
3. downloads Python 3.14 when needed;
4. creates `.venv` and installs the locked main CUDA 13.0 application stack;
5. downloads Python 3.13.13 when needed;
6. creates `.venv-unsloth` and installs the locked native Unsloth stack;
7. creates `.runs` when needed;
8. checks port `8504` and replaces only an earlier process running this app;
9. starts Streamlit as a hidden background process;
10. waits up to 90 seconds for the health endpoint; and
11. opens <http://localhost:8504> in the default browser.

First setup may take several minutes and download multiple gigabytes. Keep the launcher window open
until it reports success or an error.

### Linux

Run the script through Bash; executable permission is not required:

```bash
bash "Launch LoRA Studio.sh"
```

The launcher:

1. confirms that the required project files exist;
2. locates `uv`, or installs it with `curl` or `wget`;
3. downloads Python 3.14 when needed;
4. creates `.venv` and installs the locked main CUDA 13.0 application stack;
5. replaces the previous app process recorded in `.runs/streamlit.pid`;
6. starts Streamlit with `nohup` on port `8504`;
7. waits up to 90 seconds for the health endpoint; and
8. opens the browser with `xdg-open` when available.

The Linux launcher does not install `.venv-unsloth`. Leave **Use Unsloth** disabled in the app.

## 7. Manual installation and launch

Use this path when diagnosing a launcher failure or when automatic installer execution is not
permitted.

### Install `uv` manually

Windows Package Manager:

```powershell
winget install --id astral-sh.uv -e
```

Linux with `curl`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal if `uv` was added to the user path, then verify:

```text
uv --version
```

### Prepare the main runtime on Windows or Linux

From the repository root:

```text
uv sync --locked --no-dev --python 3.14
```

`--locked` prevents dependency resolution from silently changing `uv.lock`. `--no-dev` excludes
testing and linting tools from an end-user installation.

Launch from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.headless=true --server.port=8504
```

Launch from Linux:

```bash
.venv/bin/python -m streamlit run streamlit_app.py --server.headless=true --server.port=8504
```

Streamlit documents this command in
[Run your Streamlit app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app).
Keep the terminal open while using a manually launched app. Press `Ctrl+C` to stop it.

### Prepare native Unsloth manually on Windows

Run these commands in PowerShell from the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $PWD ".venv-unsloth"
uv sync --project unsloth-runtime --locked --no-dev --python 3.13.13
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

Verify the isolated runtime:

```powershell
.\.venv-unsloth\Scripts\python.exe -c "from importlib.metadata import version; print(version('unsloth'))"
```

The expected locked Unsloth version is `2026.8.15`. The app discovers this exact interpreter and
uses it only for a training run with **Use Unsloth** enabled. Unsloth's native Windows background
and alternatives are described in the official
[Windows installation guide](https://unsloth.ai/docs/get-started/install/windows-installation).

Do not substitute Unsloth Desktop or Unsloth Studio for this environment. They are separate user
interfaces and are not the Python runtime used by this app.

## 8. Dependency inventory

`pyproject.toml` declares direct dependencies and compatible ranges. `uv.lock` records the exact
main environment, including all transitive packages, hashes, platforms, and CUDA wheel sources.
The versions below are the versions currently committed in the lockfile.

### Main runtime

| Dependency | Declared range | Locked version | Purpose |
| --- | --- | --- | --- |
| Python | `>=3.14,<3.15` | 3.14 | Main application interpreter |
| `accelerate` | `>=1.12,<2` | 1.14.0 | Device placement and training acceleration |
| `bitsandbytes` | `>=0.49,<1` | 0.50.0 | 4-bit quantization and memory-efficient optimizers |
| `datasets` | `>=4,<5` | 4.8.5 | Dataset loading, mapping, merging, and splitting |
| `huggingface-hub` | `>=1.3,<2` | 1.27.0 | Hub authentication, metadata, downloads, and uploads |
| `peft` | `>=0.18,<1` | 0.20.0 | LoRA, QLoRA, OFT, and QOFT adapters |
| `psutil` | `>=7,<8` | 7.2.2 | Safe worker-process inspection and cancellation |
| `streamlit` | `>=1.57,<2` | 1.61.1 | Local multipage web interface |
| `torch` | `>=2.4` | 2.13.0+cu130 | CUDA tensor and model runtime |
| `transformers` | `>=4.57,<6` | 5.15.0 | Models, tokenizers, generation, and training integration |
| `trl` | `>=0.27,<1` | 0.29.1 | SFT, Reward, DPO, KTO, and ORPO trainers |

PyTorch comes from the official CUDA 13.0 wheel index at
<https://download.pytorch.org/whl/cu130>. General platform guidance is available from
[PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/). Current
`bitsandbytes` GPU and platform requirements are documented in its
[official installation guide](https://huggingface.co/docs/bitsandbytes/en/installation).

### Windows Unsloth runtime

The separate `unsloth-runtime/pyproject.toml` pins:

| Dependency | Locked requirement | Purpose |
| --- | --- | --- |
| Python | `>=3.13,<3.14` | Unsloth-compatible worker interpreter; launcher selects 3.13.13 |
| `torch` | 2.10.0 | Unsloth CUDA runtime |
| `torchvision` | 0.25.0 | Matching PyTorch vision package required by the stack |
| `torchao` | 0.16.0 | PyTorch quantization support |
| `unsloth[cu130-torch2100]` | 2026.8.15 | CUDA 13.0, PyTorch 2.10 Unsloth Core build |

The complete transitive set is locked in `unsloth-runtime/uv.lock`. Do not run an additional
`pip install unsloth`, `pip install torch`, or `uv pip install` inside either project environment;
an exact `uv sync` removes undeclared packages and restores the lockfile state. See the official
[Unsloth installation overview](https://unsloth.ai/docs/get-started/install-and-update).

### Development dependencies

These are installed only for contributors:

| Dependency | Declared range | Locked version | Purpose |
| --- | --- | --- | --- |
| `pytest` | `>=9,<10` | 9.1.1 | Automated tests |
| `ruff` | `>=0.15,<1` | 0.16.2 | Formatting and linting |
| `ty` | `>=0.0.19,<1` | 0.0.70 | Static type checking |

### Documentation dependencies

Installed only with `--group docs` when regenerating or checking the handbook:

| Dependency | Declared range | Purpose |
| --- | --- | --- |
| `beautifulsoup4` | `>=4.13,<5` | HTML rewrite and link checks for generated chapters |
| `markdown` | `>=3.8,<4` | `TUTORIAL.md` to HTML |
| `pypdf` | `>=6,<7` | Portable PDF content comparison |
| `reportlab` | `>=4.4,<5` | Deterministic handbook PDF generation |

## 9. Verify the installation

### Main interpreter and packages

Windows:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import streamlit, torch; print('Streamlit', streamlit.__version__); print('PyTorch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('CUDA runtime', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Linux:

```bash
.venv/bin/python --version
.venv/bin/python -c "import streamlit, torch; print('Streamlit', streamlit.__version__); print('PyTorch', torch.__version__); print('CUDA available', torch.cuda.is_available()); print('CUDA runtime', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

For a training-capable installation, `CUDA available` must be `True` and the expected GPU must be
shown.

### Application health

After launch, open:

- App: <http://localhost:8504>
- Health check: <http://localhost:8504/_stcore/health>

The health endpoint should return `ok`. Open **System** in the app and confirm the GPU, driver,
CUDA, disk, and Hugging Face checks.

### Windows Unsloth

Open **System** and confirm that the native Unsloth runtime is marked ready. Command-line check:

```powershell
.\.venv-unsloth\Scripts\python.exe -c "import torch, unsloth; print('PyTorch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0))"
```

Unsloth prints startup information during import; that is normal.

## 10. Run a small end-to-end smoke test

Use the repository's small local fixture so the test does not depend on a large dataset download:

1. Open **Dataset**.
2. Upload `examples/sft_sample.jsonl`.
3. Inspect it and select **Add dataset**.
4. Open **Model** and inspect `Qwen/Qwen3-0.6B`.
5. On **Training**, select **Supervised Fine-Tuning**, **QLoRA**, and **Smoke test**.
6. On Windows, optionally enable **Use Unsloth** after the System page reports it ready. Leave it
   disabled on Linux.
7. Keep compute type, learning rate, epochs, maximum gradient norm, and maximum samples at their
   preset/default values for the first test.
8. Save the settings, validate **Review & run**, and start training.
9. Follow the run on **Monitor** until it completes or reports a concrete error.

The first use of a model downloads it into the Hugging Face cache, so it is slower than later runs.
A successful smoke test writes an adapter under `.runs/<run-id>/output/adapter`.

## 11. Files, caches, logs, and ports

| Path or value | Meaning |
| --- | --- |
| `.venv` | Main Python 3.14 environment |
| `.venv-unsloth` | Optional Windows Python 3.13.13 Unsloth environment |
| `.runs/streamlit.out.log` | Background Streamlit standard output |
| `.runs/streamlit.err.log` | Background Streamlit errors |
| `.runs/streamlit.pid` | Linux background server PID |
| `.runs/<run-id>` | Saved run configuration, status, logs, checkpoints, and output |
| `.uploads` | App-managed uploaded datasets |
| `demo/` | Isolated read-only showcase, fixture, and Community Cloud requirements |
| `docs/` | Generated Zero-to-Mastery website and published PDF |
| `output/pdf/` | Canonical handbook PDF used by the tutorial check |
| `unsloth_compiled_cache` | Generated Unsloth compilation cache |
| `8504` | Local Streamlit port used by both launchers |

Hugging Face stores downloaded repositories under `$HF_HOME`, defaulting to
`~/.cache/huggingface`, with Hub repositories under `~/.cache/huggingface/hub`. On Windows, `~`
means the current user's profile directory. Set `HF_HOME` before launching if the cache must live on
another drive. See the official
[`HF_HOME` and `HF_HUB_CACHE` documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

Show the shared `uv` cache location with:

```text
uv cache dir
```

Models, package caches, uploads, runs, and environments are intentionally not committed to Git.

## 12. Stop, restart, and update

### Stop safely

Use **Stop LoRA Studio** in the sidebar, then confirm. This cancels the active owned training worker
before stopping Streamlit. Waiting jobs remain queued without starting. It does not stop Ollama or
unrelated processes.

For a manually launched foreground server, press `Ctrl+C` in its terminal.

Launching the platform launcher again replaces the previous LoRA Studio web server. A separately
running training worker continues unless it is cancelled from **Monitor** or through the confirmed
shutdown control.

When LoRA Studio starts again, it automatically launches the first waiting job in the durable FIFO
queue. Only one GPU worker runs at a time.

### Update a Git clone

Stop the app, open a terminal in the repository, and run:

```text
git pull --ff-only
```

Then run the platform launcher again. Its locked sync updates the environments when the committed
manifests or lockfiles changed. `--ff-only` refuses to overwrite or merge local work.

### Re-create a broken environment

This removes downloaded project environments, not models, uploads, or completed runs. Stop the app
first and verify that the terminal is in the repository root.

Windows PowerShell:

```powershell
Remove-Item -LiteralPath ".venv" -Recurse -Force
Remove-Item -LiteralPath ".venv-unsloth" -Recurse -Force
& ".\Launch LoRA Studio.cmd"
```

Linux:

```bash
rm -rf -- .venv
bash "Launch LoRA Studio.sh"
```

Do not delete `.runs` or `.uploads` unless their contents are no longer needed.

## 13. Optional Ollama playground

Ollama is not required for installation, training, evaluation, or adapter saving. Install it only
to use the app's optional Ollama playground. Use the official
[Ollama download page](https://ollama.com/download) and confirm that the Ollama service is running.

The playground talks to Ollama's existing local models. It does not convert, merge, or import the
adapter trained by this app.

## 14. Troubleshooting

### `git` is not recognized

Install Git, open a new terminal, and run `git --version`. On Windows, confirm that Git was added to
the user or system `PATH`.

### `uv` could not be installed

Confirm internet and proxy access to `https://astral.sh` and GitHub. Install `uv` manually from the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/), open a new
terminal, run `uv --version`, and retry the launcher.

### Locked synchronization fails

Do not delete or regenerate either `uv.lock` file. Confirm that the clone is complete, enough disk
space is available, and `download.pytorch.org` and PyPI are reachable. Run the matching manual
`uv sync --locked` command to see the full error. Re-create only the affected environment after
correcting the cause.

### `nvidia-smi` is missing or reports an old driver

Install an NVIDIA branch 580 or newer driver and restart the computer. The CUDA Toolkit alone does
not provide the required display/compute driver.

### PyTorch reports `CUDA available False`

Check `nvidia-smi`, restart after a driver update, and verify that the command is using the
repository's `.venv` interpreter. Do not replace the CUDA wheel with a CPU-only PyTorch build.

### The GPU is detected but QLoRA fails

Confirm that the GPU meets the current CUDA 13.0 `bitsandbytes` wheel target, that sufficient VRAM
is free, and that no unrelated application is occupying most GPU memory. Try the Smoke test preset,
the 0.6B model, a shorter sequence length, and batch size one.

### Windows Unsloth is unavailable

Run `Launch LoRA Studio.cmd` again and read its console error. Verify
`.venv-unsloth\Scripts\python.exe`, then run the Unsloth version check from this guide. Do not point
the app at Unsloth Desktop, a Conda environment, WSL, or the main `.venv`.

### Unsloth is unavailable on Linux

This is the current application boundary, not a failed installation. Disable **Use Unsloth** and
use standard LoRA, QLoRA, OFT, or QOFT training.

### Port 8504 is already in use

The Windows launcher refuses to terminate an unrelated process on that port. Stop that application
or run Streamlit manually on another port. On Linux, the launcher replaces only the process
recorded in `.runs/streamlit.pid`.

### The browser does not open

Open <http://localhost:8504> manually. On headless Linux, browser auto-open is optional. Check the
health endpoint and the two Streamlit log files.

### Hugging Face returns 401 or 403

Confirm the token is visible in the environment before the app starts, accept any gated model or
dataset license on its Hub page, and use a token whose scope permits the requested operation. A
write token is required only for Hub uploads.

### Setup or downloads run out of disk space

Check the project drive, the Hugging Face cache, and the `uv cache dir` location. Remove only
unneeded runs, models, or caches after confirming their exact paths. Moving `HF_HOME` to a larger
drive affects future Hugging Face downloads.

### Where to read the actual failure

- Web server: `.runs/streamlit.err.log`
- Web server output: `.runs/streamlit.out.log`
- Training worker: `.runs/<run-id>/training.log`
- Saved state: `.runs/<run-id>/status.json`

Never post logs publicly without checking them for private model IDs, dataset content, local paths,
and credentials.

## 15. Developer setup

Contributors install the main environment plus the development group:

```text
uv sync --locked --group dev --python 3.14
```

Run the application during development:

```text
uv run --locked streamlit run streamlit_app.py --server.port=8504
```

Run all required checks before submitting a change:

```text
uv sync --locked --group dev --group docs --python 3.14
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
uv run --group docs python scripts/build_tutorial.py --check
```

Unit, showcase, and Streamlit startup tests do not require a GPU. The tutorial `--check` compares
generated HTML after newline normalization and PDF metadata, page size, and text so Windows and
Linux checkouts stay in sync. Changes to real training or inference should also receive an
appropriate CUDA smoke test. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## 16. Security notes

- The app has no authentication. Keep it local and do not expose or port-forward it to an untrusted
  network.
- Use the least-privileged Hugging Face token and never commit `.streamlit/secrets.toml`.
- The code loads models with `trust_remote_code=False` and safetensors-only model loading where
  supported; do not weaken those boundaries during setup.
- Review remote installation scripts if organizational policy requires it, or install `uv` through
  an approved package manager before running the launcher.
- Model and dataset files are third-party content. Review their licenses and cards before use.

See [SECURITY.md](SECURITY.md) for the complete project security policy.

## Official references

- [Git for Windows](https://git-scm.com/install/windows)
- [Git for Linux](https://git-scm.com/install/linux)
- [GitHub: Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
- [`uv` installation](https://docs.astral.sh/uv/getting-started/installation/)
- [`uv` Python management](https://docs.astral.sh/uv/guides/install-python/)
- [`uv` locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [PyTorch: Start Locally](https://docs.pytorch.org/get-started/locally/)
- [`bitsandbytes` installation](https://huggingface.co/docs/bitsandbytes/en/installation)
- [Streamlit installation](https://docs.streamlit.io/get-started/installation)
- [Streamlit: Run your app](https://docs.streamlit.io/develop/concepts/architecture/run-your-app)
- [Hugging Face user access tokens](https://huggingface.co/docs/hub/security-tokens)
- [Hugging Face environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- [Unsloth installation](https://unsloth.ai/docs/get-started/install-and-update)
- [Unsloth native Windows installation](https://unsloth.ai/docs/get-started/install/windows-installation)
- [Ollama downloads](https://ollama.com/download)
