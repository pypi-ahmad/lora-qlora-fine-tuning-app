"""Training monitor and completed-adapter evaluation page."""

from pathlib import Path

import streamlit as st

from lora_finetune_studio.inference import generate_text
from lora_finetune_studio.jobs import (
    active_run,
    cancel_run,
    dispatch_next_run,
    list_runs,
    queued_runs,
    read_config,
    read_log,
    read_status,
    resume_run,
)
from lora_finetune_studio.models import JobState, TrainingApproach
from lora_finetune_studio.sources import get_hf_token

st.caption(
    "Follow training, inspect the FIFO queue, recover checkpoints, and test adapters."
)

try:
    dispatch_next_run()
except (OSError, RuntimeError, ValueError) as error:
    st.error(f"Cannot advance the training queue: {error}")

try:
    run_ids = list_runs()
except (OSError, ValueError) as error:
    st.error(f"Cannot list training runs: {error}")
    st.stop()
if not run_ids:
    st.info("No training run is selected. Start one from Review & run.")
    st.stop()

run_labels: dict[str, str] = {}
for candidate_id in run_ids:
    try:
        candidate_status = read_status(candidate_id)
        candidate_config = read_config(candidate_id)
        run_labels[candidate_id] = (
            f"{candidate_id} · {candidate_status.state.value} · "
            f"{candidate_config.model_id}"
        )
    except OSError, ValueError:
        run_labels[candidate_id] = candidate_id

preferred_run = st.session_state.run_id
if preferred_run not in run_ids:
    preferred_run = active_run() or run_ids[0]
if st.session_state.get("monitor_selected_run") not in run_ids:
    st.session_state.monitor_selected_run = preferred_run
run_id = st.selectbox(
    "Selected run",
    run_ids,
    format_func=run_labels.__getitem__,
    key="monitor_selected_run",
)
st.session_state.run_id = run_id


@st.fragment(run_every="2s")
def training_monitor(selected_run_id: str) -> None:
    try:
        dispatch_next_run()
        waiting_ids = queued_runs()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Cannot read the training queue: {error}")
        waiting_ids = []

    st.subheader("Training queue")
    if waiting_ids:
        queue_rows = []
        for position, waiting_id in enumerate(waiting_ids, start=1):
            try:
                waiting_config = read_config(waiting_id)
            except OSError, ValueError:
                continue
            queue_rows.append(
                {
                    "Position": position,
                    "Run": waiting_id,
                    "Model": waiting_config.model_id,
                    "Approach": waiting_config.approach.value,
                    "Method": waiting_config.peft_mode.value,
                    "Preset": waiting_config.preset.value,
                }
            )
        st.dataframe(queue_rows, hide_index=True, width="stretch")
    else:
        st.caption("No training jobs are waiting.")

    st.subheader("Run details")
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
        st.progress(status.progress, text=f"Progress: {status.progress:.0%}")
        if status.metrics:
            st.json(status.metrics)
        if status.error:
            st.error(status.error)
        cancel_label = (
            "Remove from queue"
            if status.state is JobState.QUEUED
            else "Cancel training"
        )
        if status.state in {JobState.QUEUED, JobState.RUNNING} and st.button(
            cancel_label,
            icon=":material/playlist_remove:"
            if status.state is JobState.QUEUED
            else ":material/stop_circle:",
        ):
            cancel_run(selected_run_id)
            st.rerun(scope="fragment")
        if status.state in {JobState.CANCELLED, JobState.FAILED} and st.button(
            "Queue latest checkpoint", icon=":material/playlist_add:"
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
    run_config = read_config(run_id)
    if run_config.approach is TrainingApproach.REWARD:
        st.info(
            "Reward-model adapters produce preference scores rather than text, "
            "so generative comparison is unavailable."
        )
        st.stop()
    st.subheader("Evaluate adapter")
    prompt = st.text_area(
        "Comparison prompt", placeholder="Write a concise explanation of LoRA."
    )
    if st.button("Compare base and adapter", disabled=not bool(prompt)):
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
