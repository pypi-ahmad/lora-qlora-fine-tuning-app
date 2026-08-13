"""Training review and launch page."""

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

errors = config.validate()
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
        run_id = enqueue_run(config)
        st.session_state.run_id = run_id
        st.switch_page("app_pages/monitor.py")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not start training: {error}")
