# Contributor onboarding

Use this guide for a first contribution to LoRA Fine-tune Studio. The exercises use
CPU-safe tests and do not require an NVIDIA GPU. If you only want to install the app,
start with [SETUP.md](SETUP.md).

## Prepare a checkout

Install Git and `uv`, clone the repository, and work from a branch based on `main`.
The main project uses Python 3.14 and a locked CUDA PyTorch wheel, even when you run
only CPU-safe tests. The first `uv sync` downloads the locked environment.

```powershell
git clone https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app.git
cd lora-qlora-fine-tuning-app
git switch -c docs/first-contribution
uv sync --group dev --group docs
```

On Linux, the same commands work in a shell. Use your own branch name for actual work.
The optional Windows Unsloth interpreter is a separate environment; the ordinary test
suite does not require it.

## Confirm the starting point

```powershell
uv run python --version
uv run pytest tests/test_models.py tests/test_quality.py
uv run ruff check src tests
```

The first command should report Python 3.14. The tests exercise serialized run
contracts and dataset review without running a model. If setup fails, use the
installation and driver troubleshooting in [SETUP.md](SETUP.md).

## Trace one user action

Follow the quality review before changing it:

1. `app_pages/review.py` collects the user's review and cleanup choices.
2. `models.py` defines the JSON-safe `TrainingConfig` passed to the worker.
3. `quality.py` checks every row, applies explicit cleanup, and creates split
   membership and a fingerprint.
4. `provenance.py` pins Hub revisions and rejects an input fingerprint that changed
   after review.
5. `training.py` prepares the run again inside the worker before fitting.

The focused tests are `tests/test_quality.py`, `tests/test_provenance.py`, and
`tests/test_training.py`. Read one test for a rule you want to change, then find the
source branch that implements it. The [developer guide](DEVELOPER_GUIDE.md) maps the
other workflows.

## Make a first change

Choose a small documentation fix or a narrow behavior change with a focused test.
For Python behavior, keep serialized defaults and legacy run loading in mind.
For Markdown, verify local links. `TUTORIAL.md` is the handbook source; the HTML and
PDF files under `docs/` are generated from it.

```powershell
uv run --group docs python scripts/build_tutorial.py
uv run --group docs python scripts/build_tutorial.py --check
```

Run the generator only when the tutorial source changes. Follow the
[contributor runbook](CONTRIBUTOR_RUNBOOK.md) for the full pre-pull-request checks and
the [contributing guide](CONTRIBUTING.md) for project expectations.
