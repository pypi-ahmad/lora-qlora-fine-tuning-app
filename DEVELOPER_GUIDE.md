# Developer guide

Use the runtime map below to find the code responsible for a change and follow data
across process boundaries. [TECHNICAL.md](TECHNICAL.md) has the field tables and full
architecture reference. New contributors can start with [ONBOARDING.md](ONBOARDING.md).

## Runtime map

| Boundary | Owner | What crosses it |
| --- | --- | --- |
| Browser to Streamlit | `streamlit_app.py`, `app_pages/` | Session draft, reviewed settings, UI actions |
| UI to job store | `models.py`, `jobs.py` | JSON `TrainingConfig`, status, FIFO run IDs |
| Job store to worker | `worker.py`, `queue_dispatcher.py` | Config path, process ownership, GPU lock, handoff |
| Worker to model stack | `quality.py`, `provenance.py`, `training.py`, `inference.py`, `evaluation.py` | Prepared rows, pinned revisions, adapters, metrics |
| Worker to outside services | `sources.py`, `judge.py`, `ollama.py` | Hub inputs, optional judge requests, separate local Ollama traffic |

The UI reruns after interactions. GPU work runs in a child process so it can continue
when the browser disconnects. `config.json` and `status.json` are the process contract;
`models.py` must stay importable in both the main and optional Unsloth interpreters.
Training, fit checks, and evaluations share one persistent FIFO queue and one GPU lock.

## Follow a run

1. Dataset and model pages inspect sources. The Training page saves settings. Review
   checks row quality and token coverage; the user applies the reviewed choices.
2. `jobs.enqueue_run` writes a run directory and appends its ID to `.runs/queue.json`.
   `jobs.dispatch_next_run` starts a worker when no owned worker is active.
3. For a new-format training job, the worker pins Hub commits, repeats data review,
   checks the fingerprint, loads the model, and records provenance. TRL then writes
   checkpoints and an adapter. A fit check stops after two steps without an adapter.
4. A completed training run can parent an evaluation job. The evaluator checks
   uploaded prompts against saved train/validation membership, loads base and adapter
   sequentially, saves local metrics, and can request an optional judge comparison.
5. On terminal status, a handoff process waits for the GPU worker to exit before
   starting the next job. The monitor reads durable files; it does not own the job.

The [technical handbook](TECHNICAL.md) describes run files and legacy configuration
handling. `tests/test_jobs.py`, `tests/test_gpu_jobs.py`, and `tests/test_worker.py`
cover queue and process behavior without a live GPU.

## Change a contract safely

`TrainingConfig` and `JobStatus` are serialized to disk. When adding a field, choose a
default that can load older `config.json` files, validate it at the shared boundary,
and pass it through the correct worker path. Update both the UI and the relevant tests.
Do not store credentials in either contract. `HF_TOKEN`, `OPENAI_API_KEY`, and
`AGNESAI_API_KEY` stay in the process environment.

A new dataset format needs inspection in `sources.py`, canonical validation in
`quality.py`, compatible recipe validation in `models.py`, and a trainer path in
`training.py`. A new training objective also needs its TRL mapping, adapter and model
choice, evaluation behavior, and tests. A recipe needs all of these paths to work.

An evaluation change belongs in `evaluation.py` for row checks, metrics, persistence,
and retry; in `inference.py` for GPU generation or reward scoring; and in `judge.py`
for provider requests and response validation. Keep an optional judge failure explicit
so local results remain inspectable.

## Choose verification

| Changed area | Focused checks |
| --- | --- |
| Data contracts or review | `uv run pytest tests/test_models.py tests/test_quality.py tests/test_provenance.py` |
| Worker, queue, cancellation | `uv run pytest tests/test_jobs.py tests/test_worker.py tests/test_gpu_jobs.py tests/test_queue_dispatcher.py` |
| Evaluation or judging | `uv run pytest tests/test_evaluation.py tests/test_inference.py tests/test_queue_ui.py` |
| Streamlit UI | `uv run pytest tests/test_app.py tests/test_demo.py tests/test_queue_ui.py` |
| Handbook source or builder | `uv run --group docs python scripts/build_tutorial.py --check` |

CI also runs Ruff formatting and lint, `ty check src`, the full pytest suite, and
the handbook check on Windows and Ubuntu. These tests mock GPU and network boundaries.
For a change to CUDA loading, quantization, a trainer, or adapter inference, run a
bounded hardware check on a suitable machine and report its exact outcome. The
[compatibility record](COMPATIBILITY.md) describes the short checks already run and
their limits.
