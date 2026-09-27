# Contributor runbook

Use this sequence when preparing a code or documentation change. It complements the
[contributing guide](CONTRIBUTING.md), which covers project policy, and the
[developer guide](DEVELOPER_GUIDE.md), which explains ownership.

## 1. Establish a baseline

```powershell
git status --short
uv sync --group dev --group docs
uv run pytest
```

Record existing uncommitted files before editing. Keep unrelated work out of the
change. The suite does not require a training GPU. A first-time `uv sync` downloads
the locked environment and may take longer than later runs.

## 2. Locate the behavior and its test

Start at the action a user takes, then follow the matching boundary:

| User action | First source | Focused tests |
| --- | --- | --- |
| Inspect a Hub or uploaded dataset | `sources.py`, `app_pages/dataset.py` | `test_sources.py`, `test_app.py` |
| Review rows or split data | `quality.py`, `app_pages/review.py` | `test_quality.py`, `test_provenance.py` |
| Launch, cancel, or resume a job | `jobs.py`, `worker.py` | `test_jobs.py`, `test_worker.py`, `test_gpu_jobs.py` |
| Train an adapter | `training.py`, `models.py` | `test_training.py`, `test_models.py` |
| Evaluate or judge | `evaluation.py`, `inference.py`, `judge.py` | `test_evaluation.py`, `test_inference.py` |
| Change the handbook | `TUTORIAL.md`, `scripts/build_tutorial.py` | `test_tutorial.py` |

Test files live under `tests/`; the named implementation modules live under
`src/lora_finetune_studio/`. UI files live under `app_pages/`.

## 3. Make and verify the change

Keep persisted run formats compatible or document an intentional migration. Validate
untrusted rows and paths at their entry points. A config created in Streamlit must
still validate in the worker, because saved JSON can be edited or originate from an
older version. Add a focused test for behavior that changes.

Run the closest test module during editing. Before submitting, run the same gates as CI:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
uv run --group docs python scripts/build_tutorial.py --check
git diff --check
```

If `TUTORIAL.md` changed, regenerate the website and both PDF copies first:

```powershell
uv run --group docs python scripts/build_tutorial.py
uv run --group docs python scripts/build_tutorial.py --check
```

Inspect the generated diff. Do not edit `docs/*.html` or the PDFs by hand. If a GPU
path changed, use the bounded smoke command in [IMPROVEMENTS.md](IMPROVEMENTS.md) on
suitable hardware; report any case that did not run or needed a retry. A passing CPU
suite does not establish live GPU compatibility.

## 4. Review the result

Read `git status --short` and the complete diff. Check that documentation names the
actual UI controls, commands, file locations, credential requirements, and measured
limits. Keep secrets and local `.runs/` evidence out of the contribution. Add a short
`Unreleased` entry in [CHANGELOG.md](CHANGELOG.md) for a notable user-facing change.

In the pull request, describe the behavior, the tests you ran, hardware verification
if relevant, and any remaining limit. Follow [SECURITY.md](SECURITY.md) for private
vulnerability reports.
