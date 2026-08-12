"""Training review and launch page."""

import streamlit as st

from lora_finetune_studio.hardware import model_size_warning
from lora_finetune_studio.jobs import active_run, create_run, launch_run
from lora_finetune_studio.sources import get_hf_token

st.caption(
    "Review the exact configuration, resolve blockers, and launch one local job."
)

config = st.session_state.training_config
if config is None:
    st.info("Save training settings on the Training page before starting a run.")
    st.stop()

with st.container(border=True):
    st.subheader("Model")
    st.write(f"`{config.model_id}` at revision `{config.model_revision}`")
    st.subheader("Dataset")
    source = config.dataset.repo_id or config.dataset.local_path or "Not configured"
    st.write(f"`{source}` · format `{config.dataset.format}`")
    st.subheader("Training")
    st.json(
        {
            "peft_mode": config.peft_mode,
            "preset": config.preset,
            "max_length": config.max_length,
            "epochs": config.epochs,
            "max_steps": config.max_steps,
            "max_samples": config.max_samples,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_checkpointing": config.gradient_checkpointing,
            "evaluation": config.eval_enabled,
            "push_to_hub": config.push_to_hub,
            "hub_model_id": config.hub_model_id,
        }
    )

profile = st.session_state.hardware_profile
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
if not profile.cuda_available:
    errors.append("CUDA GPU is required.")
if warning and not acknowledge_large_model:
    errors.append("Acknowledge the model-size warning before training.")
if config.push_to_hub and not get_hf_token():
    errors.append("HF_TOKEN is required to upload the adapter.")
running_job = active_run()
if running_job:
    errors.append("Another training job is already active.")

if errors:
    for error in errors:
        st.error(error)

with st.container(horizontal=True):
    start_run = st.button(
        "Start training",
        type="primary",
        icon=":material/play_arrow:",
        disabled=bool(errors),
    )
    open_monitor = st.button(
        "Open monitor",
        icon=":material/monitoring:",
        disabled=not bool(st.session_state.run_id or running_job),
    )

if open_monitor:
    if not st.session_state.run_id and running_job:
        st.session_state.run_id = running_job
    st.switch_page("app_pages/monitor.py")

if start_run:
    try:
        run_id = create_run(config)
        launch_run(run_id)
        st.session_state.run_id = run_id
        st.switch_page("app_pages/monitor.py")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not start training: {error}")
