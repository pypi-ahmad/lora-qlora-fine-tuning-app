# Data quality, evaluation, and reproducibility

## Prepare a run

1. Inspect and map your datasets on **Dataset**. Use **Validation sources and grouping** to reserve an inspected source for validation or select a grouping column.
2. Save your training settings. Advanced settings include the SFT loss scope, LoRA rank, alpha, dropout, target modules, packing, best-checkpoint selection, and early stopping.
3. On **Review & run**, select **Check quality / preview cleanup**. Review malformed rows, duplicates, source counts, token lengths, truncation, and supervised-token previews.
4. Select **Apply reviewed data settings**. Cleanup creates a derived dataset from recorded row selections; original sources remain unchanged. A run records every removed row and reason and saves cleaned JSONL files when rows were removed.
5. Start training or queue a two-step GPU fit check. A fit check measures the actual recipe and does not save or publish an adapter.

Malformed data and overlap between training and validation block new runs. Exact duplicates within a split are warnings unless you explicitly choose to remove them. Without a separate validation source, the app groups matching prompts and uses a deterministic validation split. An optional group column keeps related conversations together too. Small datasets or a single independent group can leave validation unavailable; the quality report explains this.

### Loss and packing

The `auto` loss scope preserves the existing defaults: full-conversation learning for messages and completion-only learning for prompt/completion data. Assistant-only loss requires a chat template that returns assistant masks. A template that lacks masks, or truncation that removes every supervised token, blocks the run. Full-text datasets cannot use assistant-only loss.

Packing is off by default. The worker rejects packing when the model does not use FlashAttention 2 or 3. Enabling a flag does not establish that packing works on a particular GPU or backend; check the compatibility evidence before using it.

The pinned Unsloth SFT preprocessing does not preserve assistant masks, so assistant-only loss requires the standard backend. Unsloth message datasets use an explicit chat-template formatter for full-conversation loss; prompt/completion datasets retain their structured completion masks.

Best-checkpoint selection minimizes validation loss. Early stopping is optional and requires an actual validation set. Learning curves, throughput, and peak PyTorch allocator memory are recorded locally. A successful two-step fit check is evidence for those steps, not a guarantee that every later batch will fit.

## Evaluate a completed adapter

Select a completed training run on **Monitor**, then queue an evaluation. Chat mode applies the saved chat template and generation prompt. Text completion passes raw text. The worker loads the base and adapter sequentially with matching generation settings. Inputs that exceed the model's context budget produce an error rather than silent truncation.

For a saved test set, upload JSONL:

```json
{"prompt":"What is two plus two?","reference":"4"}
{"messages":[{"role":"user","content":"Explain LoRA in one sentence."}]}
```

Use a single input format matching the selected mode. References are optional. Reward evaluation uses `prompt`, `chosen`, and `rejected`. It scores the trained reward adapter; it does not compare against a randomly initialized base scoring head.

The default sample count is 20, selected deterministically. Uploaded test prompts are checked against saved training and validation membership. A manually entered comparison prompt may reuse a training prompt, but that comparison is not a held-out test. Legacy runs without membership records display the limitation.

Local metrics include trimmed, case-sensitive exact match when a reference exists; JSON validity; optional JSON Schema validity; reward preference accuracy; latency; and generated token counts. Schema references must remain local. Missing references do not count as incorrect answers.

Download the complete JSON result or add human preferences and notes. Retrying unfinished evaluation retains saved generations and successful judgments.

### Optional AI judge

Choose **AI judge model**, then enable sending to the selected provider. The choices are
`gpt-6-luna` (the default, with medium reasoning effort) and `agnes-3.0-flash` (with the
provider's default reasoning setting). GPT requires `OPENAI_API_KEY` and `OPENAI_BASE_URL`;
Agnes requires `AGNESAI_API_KEY` and uses `https://apihub.agnes-ai.com/v1`.
Credentials stay in the server process and are absent from run configurations. Existing
saved evaluations default to GPT. Create a new evaluation to change the judge.

The judge sees anonymous A/B responses in seeded randomized order. It scores instruction following, correctness, and clarity, and returns a winner, tie, or abstention with a brief justification. Correctness is unavailable when the supplied evidence is insufficient. These are model judgments, not ground truth.

GPT requests use the Responses API, strict structured output, and `store=False`. Agnes uses its documented [Chat Completions API](https://wiki.agnes-ai.com/en/docs/agnes-30-flash), requests JSON in the prompt, and validates the complete response against the same schema locally; server-enforced structured output is not assumed. Both use no tools, a 4,096-output-token ceiling, a 120-second timeout, and at most two retries for transient API failures. Refusals, incomplete output, malformed JSON, invalid scores, and API failures remain explicit errors. Provider data-handling policies still apply. Local evaluation results remain available.

## Run artifacts and compatibility

New runs record `manifest.json`, `split_membership.json`, `quality_report.json`, and `metrics_history.jsonl` alongside their configuration and status. Evaluation writes `output/evaluation.json`; human ratings use `output/human_ratings.json`.

The manifest includes immutable Hub revisions, source hashes, effective settings, template hash, runtime versions, GPU information, and the application revision. Resume rejects changed input fingerprints, templates, or package versions. Old runs remain readable but cannot acquire historical provenance retroactively.

All application GPU jobs share the persistent FIFO queue and a worker-held OS lock. Closing the browser does not stop a worker. Cancellation verifies process ownership; the next job waits for the previous process to exit.

Run the bounded compatibility harness from the project root:

```powershell
uv run python scripts/smoke_gpu.py --minutes 60
```

The harness uses small local fixtures and `Qwen/Qwen3-0.6B`. It attempts training, adapter reload, generation or reward scoring, and checkpoint recovery. It writes `.runs/compatibility.json` with individual outcomes. Remaining cases are marked not run when the time budget expires. It may prepare the repository's locked Unsloth environment. It never uploads adapters.

See [COMPATIBILITY.md](COMPATIBILITY.md) for checked-in verification evidence. CPU tests alone do not establish GPU compatibility.
