"""Training monitor and completed-adapter evaluation page."""

from pathlib import Path

import streamlit as st

from lora_finetune_studio.inference import generate_text
from lora_finetune_studio.jobs import (
    active_run,
    cancel_run,
    read_config,
    read_log,
    read_status,
    resume_run,
)
from lora_finetune_studio.models import JobState
from lora_finetune_studio.sources import get_hf_token

st.caption(
    "Follow the active job, inspect logs, recover checkpoints, and test its adapter."
)

run_id = st.session_state.run_id or active_run()
if not run_id:
    st.info("No training run is selected. Start one from Review & run.")
    st.stop()
st.session_state.run_id = run_id


@st.fragment(run_every="2s")
def training_monitor(selected_run_id: str) -> None:
    try:
        status = read_status(selected_run_id)
    except (OSError, ValueError) as error:
        st.error(f"Cannot read job status: {error}")
        return
    with st.container(border=True):
        st.badge(
            status.state.value,
            color="green" if status.state is JobState.COMPLETED else "blue",
        )
        st.write(status.message)
        st.progress(status.progress)
        if status.metrics:
            st.json(status.metrics)
        if status.error:
            st.error(status.error)
        if status.state in {JobState.QUEUED, JobState.RUNNING} and st.button(
            "Cancel training", icon=":material/stop_circle:"
        ):
            cancel_run(selected_run_id)
            st.rerun(scope="fragment")
        if status.state in {JobState.CANCELLED, JobState.FAILED} and st.button(
            "Resume latest checkpoint", icon=":material/play_arrow:"
        ):
            try:
                resume_run(selected_run_id)
                st.rerun(scope="fragment")
            except (FileNotFoundError, RuntimeError) as error:
                st.error(str(error))
        log_expander = st.expander("Training log", on_change="rerun")
        if log_expander.open:
            with log_expander:
                st.code(
                    read_log(selected_run_id) or "Waiting for worker output...",
                    language="text",
                )


training_monitor(run_id)

try:
    status = read_status(run_id)
except OSError, ValueError:
    status = None
if status and status.state is JobState.COMPLETED and status.artifact_dir:
    st.subheader("Evaluate adapter")
    prompt = st.text_area(
        "Comparison prompt", placeholder="Write a concise explanation of LoRA."
    )
    if st.button("Compare base and adapter", disabled=not bool(prompt)):
        run_config = read_config(run_id)
        adapter_path = str(Path(status.artifact_dir) / "adapter")
        with st.spinner("Generating base response..."):
            base_response = generate_text(
                run_config.model_id,
                prompt,
                token=get_hf_token(),
                revision=run_config.model_revision,
            )
        with st.spinner("Generating adapter response..."):
            adapter_response = generate_text(
                run_config.model_id,
                prompt,
                token=get_hf_token(),
                revision=run_config.model_revision,
                adapter_path=adapter_path,
            )
        left, right = st.columns(2)
        left.subheader("Base model")
        left.write(base_response)
        right.subheader("Fine-tuned adapter")
        right.write(adapter_response)
