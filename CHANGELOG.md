# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/), and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-12

### Added

- A dedicated GPU memory page with metrics and safe Streamlit-process VRAM cleanup.
- A read-only System readiness scan covering OS, CPU threads, available RAM, disk, accelerators,
  QLoRA free-VRAM readiness, uv `.venv`, and installed software integrations.
- Native left-sidebar navigation for System, Dataset, Model, GPU memory, Training, Review & run,
  Monitor, and Ollama playground.
- Native Linux launcher and Linux CI coverage.
- A confirmed sidebar shutdown control that safely cancels the owned training worker before
  stopping Streamlit.

### Changed

- Platform launchers now replace the previous LoRA Studio server to create a fresh Streamlit
  session on every launch.
- Adapter comparison now releases model references and unused CUDA cache after both successful and
  failed inference attempts.
- Model and dataset inspection, training configuration, launch review, monitoring, and Ollama now
  run only on their focused pages while sharing session state.
- The supported runtime is now native Windows or Linux with Python 3.14 in a uv-managed `.venv`
  and the PyTorch CUDA 13.0 wheel index.

## [0.1.0] - 2026-08-12

### Added

- Local Streamlit workflow for supervised fine-tuning with LoRA or four-bit QLoRA.
- CUDA, VRAM, RAM, disk, and BF16 hardware inspection with conservative model-size guidance.
- Hugging Face model and dataset repository support with token verification and optional adapter
  upload.
- CSV, JSON, and JSONL uploads with validation, content-addressed storage, preview, and column
  mapping.
- Smoke, Standard, and Quality training presets plus advanced configuration controls.
- Isolated single-job worker with durable status, logs, cancellation, checkpoints, and resume.
- Local adapter, tokenizer, metrics, and training-configuration artifacts.
- Base-versus-adapter response comparison and an independent Ollama playground.
- One-click Windows launcher that prepares Python and dependencies before starting the app.
- Interactive zero-to-hero handbook, user guide, technical reference, contribution guide, security
  policy, and community code of conduct.
- Windows and Linux CI for formatting, linting, type checking, and tests.

### Security

- Restricted remote inputs to Hugging Face repository-root HTTPS URLs.
- Disabled remote model code and required safetensors model loading.
- Kept Hugging Face credentials out of saved run configuration and logs.
- Limited dataset uploads to approved formats and 200 MB.
- Added path validation and atomic run-status updates.

[Unreleased]: https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/releases/tag/v0.1.0
