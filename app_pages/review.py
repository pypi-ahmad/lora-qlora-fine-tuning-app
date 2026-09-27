"""Training review and launch page.

Re-validates the saved TrainingConfig against current runtime state (CUDA
availability, HF token, Unsloth readiness, queue occupancy) since those can
change between saving Training settings and pressing start here.

Read next: lora_finetune_studio/jobs.py for enqueue_run/active_run/
queued_runs, or app_pages/monitor.py for the page this hands off to.
"""

from dataclasses import replace

import streamlit as st

from lora_finetune_studio.hardware import model_size_warning
from lora_finetune_studio.jobs import active_run, enqueue_run, queued_runs
from lora_finetune_studio.models import resolve_compute_type
from lora_finetune_studio.sources import get_hf_token
from lora_finetune_studio.unsloth_runtime import inspect_unsloth_runtime

st.caption(
    "Review the exact configuration, resolve blockers, and start or queue training."
)

config = st.session_state.training_config
if config is None:
    st.info("Save training settings on the Training page before starting a run.")
    st.stop()

profile = st.session_state.hardware_profile
# What was requested (config.compute_type, e.g. Auto) can differ from what
# will actually run (e.g. FP16 when BF16 isn't supported) — both are shown
# below so a reviewer isn't surprised by the effective dtype.
effective_compute_type = resolve_compute_type(
    config.compute_type, bf16_supported=profile.bf16_supported
)

with st.container(border=True):
    st.subheader("Model")
    st.write(f"`{config.model_id}` at revision `{config.model_revision}`")
    st.subheader("Datasets")
    st.dataframe(
        [
            {
                "Source": dataset.repo_id or dataset.local_path or "Not configured",
                "Configuration": dataset.config_name or "Default",
                "Split": dataset.split,
                "Format": dataset.format,
            }
            for dataset in config.datasets
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("Training")
    st.json(
        {
            "backend": "Unsloth" if config.use_unsloth else "Transformers/TRL",
            "approach": config.approach,
            "method": config.peft_mode,
            "compute_type_requested": config.compute_type,
            "compute_type_effective": effective_compute_type,
            "preset": config.preset,
            "max_length": config.max_length,
            "epochs": config.epochs,
            "max_steps": config.max_steps,
            "max_samples": config.max_samples,
            "learning_rate": config.learning_rate,
            "beta": config.beta
            if config.approach.name in {"DPO", "KTO", "ORPO"}
            else None,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_grad_norm": config.max_grad_norm,
            "gradient_checkpointing": config.gradient_checkpointing,
            "evaluation": config.eval_enabled,
            "push_to_hub": config.push_to_hub,
            "hub_model_id": config.hub_model_id,
        }
    )

warning = model_size_warning(st.session_state.model_parameters, profile)
if warning:
    st.warning(warning, icon=":material/warning:")
    acknowledge_large_model = st.toggle(
        "I understand this run may exhaust GPU memory",
        key="acknowledge_large_model",
        persist_state="session",
    )
else:
    acknowledge_large_model = True

st.subheader("Dataset quality")
with st.expander(
    "Inspect data and preview cleanup", expanded=not bool(config.input_fingerprint)
):
    remove_invalid = st.checkbox(
        "Remove malformed rows in a derived dataset", value=config.cleanup_invalid
    )
    remove_duplicates = st.checkbox(
        "Remove exact duplicates in a derived dataset", value=config.cleanup_duplicates
    )
    if st.button("Check quality / preview cleanup"):
        try:
            from transformers import AutoTokenizer

            from lora_finetune_studio.provenance import pin_revisions
            from lora_finetune_studio.quality import prepare_data, token_report
            from lora_finetune_studio.sources import get_hf_token

            draft = replace(
                config,
                cleanup_invalid=remove_invalid,
                cleanup_duplicates=remove_duplicates,
                input_fingerprint=None,
            )
            pin_revisions(draft, get_hf_token())
            report_data = prepare_data(draft, get_hf_token())
            tokenizer = AutoTokenizer.from_pretrained(
                draft.model_id,
                revision=draft.model_revision,
                token=get_hf_token(),
                trust_remote_code=False,
            )
            for role, data in (
                ("train", report_data.train),
                ("validation", report_data.validation),
            ):
                if data is not None and len(data):
                    report_data.report[role + "_tokens"] = token_report(
                        data, tokenizer, draft
                    )
                    if report_data.report[role + "_tokens"]["zero_supervised_rows"]:
                        report_data.report["errors"].append(
                            "Some rows have no supervised tokens after truncation."
                        )
            st.session_state.quality_report = report_data.report
            draft.input_fingerprint = report_data.report["fingerprint"]
            st.session_state.quality_draft = draft
        except Exception as error:  # noqa: BLE001
            st.error(f"Quality check failed: {error}")
    report = st.session_state.get("quality_report")
    if report:
        st.json(report)
        if st.button("Apply reviewed data settings", disabled=bool(report["errors"])):
            st.session_state.training_config = st.session_state.quality_draft
            st.rerun()

errors = config.validate()
if config.schema_version >= 2 and not config.input_fingerprint:
    errors.append(
        "Run the dataset quality check and apply the reviewed settings before starting."
    )
st.json(
    {
        "loss_scope": config.loss_scope,
        "rank": config.lora_rank,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout
        if config.lora_dropout is not None
        else (0 if config.use_unsloth else 0.05),
        "targets": config.target_modules
        or ("projection layers" if config.use_unsloth else "all-linear"),
        "packing": config.packing,
        "best_checkpoint": config.select_best_checkpoint,
        "early_stopping_patience": config.early_stopping_patience,
    }
)
if config.use_unsloth:
    unsloth_runtime = inspect_unsloth_runtime()
    if not unsloth_runtime.available:
        errors.append(unsloth_runtime.detail)
if not profile.cuda_available:
    errors.append("CUDA GPU is required.")
if warning and not acknowledge_large_model:
    errors.append("Acknowledge the model-size warning before training.")
if config.push_to_hub and not get_hf_token():
    errors.append("HF_TOKEN is required to upload the adapter.")
running_job = active_run()
try:
    waiting_jobs = queued_runs()
except (OSError, ValueError) as error:
    waiting_jobs = []
    errors.append(f"Training queue could not be read: {error}")

if running_job or waiting_jobs:
    position = len(waiting_jobs) + 1
    st.info(
        f"One training worker is active. This configuration will wait at queue "
        f"position {position}."
    )

if errors:
    for error in errors:
        st.error(error)

if st.button("Queue two-step GPU fit check", disabled=bool(errors)):
    try:
        fit = replace(config, job_kind="fit_check", push_to_hub=False)
        st.session_state.run_id = enqueue_run(fit)
        st.switch_page("app_pages/monitor.py")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not queue fit check: {error}")

with st.container(horizontal=True):
    start_run = st.button(
        "Add to queue" if running_job or waiting_jobs else "Start training",
        type="primary",
        icon=":material/playlist_add:"
        if running_job or waiting_jobs
        else ":material/play_arrow:",
        disabled=bool(errors),
    )
    open_monitor = st.button(
        "Open monitor",
        icon=":material/monitoring:",
        disabled=not bool(st.session_state.run_id or running_job or waiting_jobs),
    )

if open_monitor:
    if not st.session_state.run_id and running_job:
        st.session_state.run_id = running_job
    elif not st.session_state.run_id and waiting_jobs:
        st.session_state.run_id = waiting_jobs[0]
    st.switch_page("app_pages/monitor.py")

if start_run:
    try:
        if config.schema_version >= 2:
            from lora_finetune_studio.provenance import prepare_run
            from lora_finetune_studio.sources import get_hf_token

            prepare_run(config, get_hf_token())
        run_id = enqueue_run(config)
        st.session_state.run_id = run_id
        st.switch_page("app_pages/monitor.py")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not start training: {error}")
