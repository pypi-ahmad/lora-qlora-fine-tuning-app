# Contributing

Thank you for improving LoRA Fine-tune Studio. Keep changes small, explain their user impact, and
preserve the project's local single-user scope unless a proposal explicitly changes it.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report suspected
vulnerabilities through the private process in [SECURITY.md](SECURITY.md), not a public issue.

## Before starting

- Search existing issues and pull requests for related work.
- Open an issue before a large feature, new training method, dependency change, or architecture
  change.
- Do not include Hugging Face tokens, private datasets, model artifacts, run logs, or personal data
  in an issue or pull request.

## Development setup

Requirements:

- Windows 11 or x86-64 Linux
- Git and authenticated GitHub CLI
- `uv`
- Python 3.14, installed automatically by `uv` into the project `.venv`
- NVIDIA CUDA GPU only for real training or adapter-comparison tests

Prepare the development environment:

```bash
git clone https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app.git
cd lora-qlora-fine-tuning-app
uv sync --group dev
```

Work from `main`, not the `v0.5.1` release tag. Users who only want the published tree should follow
[SETUP.md](SETUP.md).

The unit and Streamlit startup tests do not require Ollama or a training-capable GPU.

## Workflow

1. Fork the repository and create a short-lived branch from `main`.
2. Use a focused name such as `feature/dataset-preview` or `fix/checkpoint-resume`.
3. Make the smallest complete change and add tests for observable behavior.
4. Update user or technical documentation when behavior, configuration, or architecture changes.
5. Add a human-readable entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for notable
   user-facing changes.
6. Run the required checks and inspect the complete diff.
7. Open a pull request explaining the problem, solution, user impact, and verification.

Use Conventional Commit-style messages:

```text
feat: add dataset column mapping
fix: select the newest checkpoint numerically
docs: explain Hugging Face token scopes
test: cover invalid repository URLs
chore: update development tooling
```

Keep refactors separate from behavior changes. Never rewrite shared branch history or force-push a
branch other people use.

## Engineering expectations

- Match existing Python and Streamlit patterns.
- Prefer standard-library or existing dependencies over new packages.
- Keep shared UI/worker data serializable through the contracts in `models.py`.
- Treat model IDs, dataset IDs, uploads, paths, and process IDs as untrusted input.
- Never serialize or log `HF_TOKEN`.
- Keep `trust_remote_code=False` and safetensors-only model loading unless a reviewed proposal
  justifies changing that security boundary.
- Avoid unrelated cleanup, generated abstractions, or speculative options.

## Required checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
```

When changing `TUTORIAL.md`, `scripts/build_tutorial.py`, or generated `docs/` output, also run:

```powershell
uv sync --group docs
uv run --group docs python scripts/build_tutorial.py --check
```

If the handbook is stale, rebuild with `uv run --group docs python scripts/build_tutorial.py` and
commit the generated website, manifest, and PDF copies. Do not edit generated HTML by hand.

Showcase changes belong under `demo/` and `tests/test_demo.py`. Keep that entry point free of CUDA,
Hugging Face, and worker imports.

Changes to training or inference should also receive a proportionate CUDA smoke test when suitable
hardware is available. State clearly in the pull request when hardware verification was not run.

Do not commit `.venv`, `.runs`, `.uploads`, `.streamlit/secrets.toml`, caches, downloaded models,
datasets, adapters, checkpoints, or analysis output.

## Pull-request checklist

- [ ] The change solves one clearly described problem.
- [ ] Tests cover new or corrected behavior.
- [ ] Ruff, ty, and pytest pass.
- [ ] Documentation and `CHANGELOG.md` are updated when needed.
- [ ] No credentials, private data, generated artifacts, or unrelated changes are included.
- [ ] GPU/Ollama verification status is stated when relevant.
- [ ] The pull request describes user impact and known limitations.

Maintainers may request changes to keep the project secure, understandable, and within scope.

## No financial contributions

This project does not want or accept donations, sponsorships, or any other financial support, and
never will. LoRA Fine-tune Studio is free and community-driven. The most valuable way to give back
is a well-written bug report, a focused pull request, or a documentation fix — see
[SUPPORT.md](SUPPORT.md).
