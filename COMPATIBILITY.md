# Compatibility evidence

Validation date: **2026-09-27**. Tested the current local implementation on Windows 11 with an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), CUDA 13.0, driver 617.14.

## GPU matrix

**30/30 supported combinations passed after targeted retries.** The validation phase took 50.36 minutes within the agreed 60-minute cap, including model preparation, optional runtime installation, and retries.

| Objective | Standard LoRA | Standard QLoRA | Standard OFT | Standard QOFT | Unsloth LoRA | Unsloth QLoRA |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | Pass | Pass | Pass | Pass | Pass after fix | Pass after fix |
| Reward modeling | Pass | Pass | Pass | Pass | Pass | Pass |
| DPO | Pass | Pass | Pass | Pass | Pass | Pass |
| KTO | Pass | Pass | Pass | Pass | Pass | Pass |
| ORPO | Pass | Pass | Pass | Pass | Pass on retry | Pass |

Each case used `Qwen/Qwen3-0.6B`, 12 synthetic source examples, 128-token training sequences, a validation split, two optimizer steps, saved adapter reload, and one evaluation example. Generative evaluations compared base and adapter with 16 new tokens; reward evaluations scored a chosen/rejected pair. The standard SFT QLoRA case used four steps to exercise checkpoint interruption and resume.

These short checks establish execution for this fixture and machine. They do not measure model quality, larger-model memory requirements, long-run stability, or other architectures. Unsloth OFT/QOFT remain unsupported.

### Runtime versions

| Component | Standard | Unsloth |
| --- | --- | --- |
| Python | 3.14.6 | 3.13.13 |
| PyTorch | 2.13.0+cu130 | 2.10.0+cu130 |
| Transformers | 5.15.0 | 5.5.0 |
| TRL | 0.29.1 | 0.24.0 |
| PEFT | 0.20.0 | 0.20.0 |
| Datasets | 4.8.5 | 4.3.0 |
| Unsloth | — | 2026.8.15 |

### Failures and corrections retained in the evidence

- The first two Unsloth SFT attempts required an explicit formatter for message datasets. The formatter was restored only for that backend and format; both cases then passed.
- One initial Unsloth ORPO LoRA worker crashed in `torch_cpu.dll` with Windows exception `0xc0000005`. A retry passed. The cause of that native crash is unresolved; one successful retry does not establish runtime stability.
- Worker logs now flush immediately and enable Python fault diagnostics. The smoke harness reconciles vanished workers instead of waiting until its deadline, and returns a nonzero exit code for failed cases.

## Other checks

- **147 automated tests passed**, including Streamlit queue/evaluation flows, loss defaults, explicit cleanup, grouped splits, provenance drift, judgment retries, GPU lock exclusion, and forced-cancellation handoff.
- Ruff, formatting, ty, `uv lock --check`, and tutorial synchronization passed. Shared worker source also parsed with Python 3.13 syntax rules.
- A separate two-step fit check completed without saving an adapter: 3.17 seconds of trainer runtime, 0.94 GiB peak PyTorch allocated memory, and 1.06 GiB peak reserved memory. These allocator measurements exclude other processes.
- Three live `gpt-6-luna` medium-effort judge checks completed: correct versus incorrect answer, identical answers, and an answer containing instructions to the evaluator. Results were respectively the correct answer, a tie, and the answer that addressed the task. This is a small contract check, not an accuracy benchmark.
- The real Qwen tokenizer selected 10 supervised tokens for full loss and 3 for completion-only loss on the same fixture. Its template lacks assistant-mask support, and assistant-only mode was correctly rejected.

## Feature limits

- Assistant-only loss requires a template returning usable masks and the standard backend. The pinned Unsloth preprocessing does not preserve those masks.
- Packing remains off by default and is rejected without FlashAttention 2/3. Packing was not GPU verified on this machine.
- Checkpoint resume was GPU verified for standard SFT QLoRA only. Early stopping is implemented through the trainer; an actual early-stop event was not exercised by these short runs.
- Python 3.14 tests emit upstream PyTorch JIT deprecation warnings. The suite passed despite those warnings.

## Local evidence

### Agnes judge addition

`agnes-3.0-flash` was added as an optional judge using its [documented Chat Completions endpoint](https://wiki.agnes-ai.com/en/docs/agnes-30-flash) and `AGNESAI_API_KEY`. Fifteen focused tests passed across judge routing/validation, evaluation retry, Streamlit model selection, and the demo. Ruff and ty passed. A live synthetic comparison returned a schema-valid judgment selecting the correct answer. An earlier malformed response was rejected explicitly; Agnes uses local schema validation rather than an assumed server-side JSON guarantee. Evidence is saved in `.runs/agnes-judge-smoke.json`. No additional GPU training was required for this API-only addition.

- `.runs/compatibility.json`: final matrix, per-case run IDs, metrics, manifests, and initial failed attempts.
- `.runs/compatibility-first-pass.json` and `.runs/compatibility-retries.json`: original outcomes and retry evidence.
- `.runs/smoke-20260927104338/`: configurations, logs, adapters, manifests, split membership, and evaluation results.
- `.runs/fit-smoke.json`, `.runs/loss-smoke.json`, and `.runs/judge-smoke.json`: additional checks.
- `.runs/orpo-native-crash.txt`: matching Windows crash event.

Run a fresh bounded matrix with `uv run python scripts/smoke_gpu.py --minutes 60`. Local evidence files are ignored by Git; this document records the observed outcomes and limits.
