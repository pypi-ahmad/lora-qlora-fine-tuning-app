"""Training monitor and completed-adapter evaluation page.

The training_monitor() fragment below reruns on its own 2-second timer
(st.fragment(run_every="2s")) independently of full-page reruns, so its
status/queue/log reads must stay side-effect-free apart from the explicit
cancel/resume button actions. dispatch_next_run() is also called here (as it
is in streamlit_app.py) so the FIFO queue keeps advancing while a user is
simply watching this page.

Read next: lora_finetune_studio/jobs.py for the run lifecycle this page
polls, or lora_finetune_studio/inference.py for the base-vs-adapter
comparison below.
"""

import json
from dataclasses import replace
from pathlib import Path

import streamlit as st

from lora_finetune_studio.jobs import (
    active_run,
    cancel_run,
    dispatch_next_run,
    enqueue_run,
    list_runs,
    queued_runs,
    read_config,
    read_log,
    read_status,
    resume_run,
    retry_evaluation,
    write_json_atomic,
)
from lora_finetune_studio.models import EvaluationConfig, JobState, TrainingApproach

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
            f"{candidate_id} · {candidate_config.job_kind} · {candidate_status.state.value} · "
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
                    "Job": waiting_config.job_kind,
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
    state_key = f"monitor_state_{selected_run_id}"
    previous_state = st.session_state.get(state_key)
    st.session_state[state_key] = status.state.value
    if previous_state in {
        JobState.RUNNING.value,
        JobState.QUEUED.value,
    } and status.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
        st.rerun()
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
        if (
            read_config(selected_run_id).job_kind == "training"
            and status.state in {JobState.CANCELLED, JobState.FAILED}
            and st.button("Queue latest checkpoint", icon=":material/playlist_add:")
        ):
            try:
                resume_run(selected_run_id)
                st.rerun(scope="fragment")
            except (FileNotFoundError, RuntimeError) as error:
                st.error(str(error))
        # Log contents are only read when the expander is actually open, so
        # this 2-second fragment doesn't re-read (potentially large) log
        # files on every tick while the section is collapsed.
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
if status and status.artifact_dir:
    run_config = read_config(run_id)
    if run_config.schema_version < 2:
        st.caption("Legacy run: input and runtime provenance were not recorded.")
    root = Path(status.artifact_dir).parent
    history_path = root / "metrics_history.jsonl"
    if history_path.exists():
        history = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except ValueError:
                continue
        if history:
            st.subheader("Learning curves")
            series = [
                key
                for key in (
                    "loss",
                    "eval_loss",
                    "learning_rate",
                    "tokens_per_second",
                    "peak_allocated_gb",
                )
                if any(key in row for row in history)
            ]
            selected_series = st.selectbox("Metric history", series) if series else None
            if selected_series:
                st.line_chart(
                    [
                        {"step": row["step"], selected_series: row[selected_series]}
                        for row in history
                        if selected_series in row
                    ],
                    x="step",
                    y=selected_series,
                )
    result_path = Path(status.artifact_dir) / "evaluation.json"
    if run_config.job_kind == "evaluation" and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        st.subheader("Evaluation results")
        st.caption(result.get("provenance", ""))
        st.json(result.get("metrics", {}))
        st.download_button(
            "Download evaluation JSON",
            result_path.read_bytes(),
            file_name=f"{run_id}-evaluation.json",
            mime="application/json",
        )
        st.dataframe(result["results"])
        rating_path = Path(status.artifact_dir) / "human_ratings.json"
        saved_ratings = (
            json.loads(rating_path.read_text(encoding="utf-8")).get("ratings", [])
            if rating_path.exists()
            else []
        )
        if not saved_ratings:
            saved_ratings = [
                {"example": i + 1, "preference": "unrated", "note": ""}
                for i in range(len(result["results"]))
            ]
        ratings = st.data_editor(
            saved_ratings,
            key=f"human_ratings_{run_id}",
            disabled=["example"],
            column_config={
                "preference": st.column_config.SelectboxColumn(
                    options=["unrated", "base", "adapter", "tie", "abstain"]
                )
            },
        )
        if st.button("Save human ratings"):
            write_json_atomic(rating_path, {"ratings": ratings})
            st.success("Ratings saved locally.")
    if run_config.job_kind == "evaluation" and st.button(
        "Retry unfinished evaluation",
        disabled=status.state in {JobState.RUNNING, JobState.QUEUED},
    ):
        retry_evaluation(run_id)
        st.rerun()
    if status.state is JobState.COMPLETED and run_config.job_kind == "training":
        st.subheader("Evaluate adapter")
        reward = run_config.approach is TrainingApproach.REWARD
        with st.form("evaluation_form"):
            mode = (
                "reward"
                if reward
                else st.selectbox(
                    "Input mode",
                    ["chat", "text"],
                    index=0
                    if any(d.format == "messages" for d in run_config.datasets)
                    else 1,
                    format_func=lambda value: {
                        "chat": "Chat",
                        "text": "Text completion",
                    }[value],
                )
            )
            prompt = st.text_area("Comparison prompt")
            chosen = st.text_area("Chosen response") if reward else ""
            rejected = st.text_area("Rejected response") if reward else ""
            uploaded = st.file_uploader(
                "Saved test set (JSONL)",
                type=["jsonl"],
                help="Use prompt or messages, with optional reference. Reward rows use prompt/chosen/rejected.",
            )
            count = st.number_input("Evaluation sample count", 1, value=20)
            max_tokens = st.number_input("Maximum new tokens", 1, 4096, 128)
            schema_text = st.text_area("Expected JSON Schema (optional)")
            judge_model = st.selectbox(
                "AI judge model", ["gpt-6-luna", "agnes-3.0-flash"], disabled=reward
            )
            use_judge = st.checkbox(
                "Send selected examples and both responses to the selected AI judge",
                disabled=reward,
            )
            st.caption(
                "GPT uses medium effort and your configured endpoint. Agnes uses its own endpoint and default reasoning settings. Selected prompts, references, and answers are sent to the selected provider."
            )
            rubric = st.text_area(
                "Judge rubric",
                value="Evaluate instruction following, correctness, and clarity.",
            )
            submitted = st.form_submit_button("Queue evaluation")
        if submitted:
            try:
                from lora_finetune_studio.evaluation import (
                    validate_rows,
                    validate_schema,
                )

                if uploaded:
                    rows = [
                        json.loads(line)
                        for line in uploaded.getvalue().decode("utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    rows = (
                        [{"prompt": prompt, "chosen": chosen, "rejected": rejected}]
                        if reward
                        else [{"prompt": prompt}]
                    )
                schema = json.loads(schema_text) if schema_text.strip() else None
                validate_rows(rows, mode)
                validate_schema(schema)
                options = EvaluationConfig(
                    rows=rows,
                    mode=mode,
                    sample_limit=int(count),
                    max_new_tokens=int(max_tokens),
                    judge_enabled=use_judge,
                    judge_model=judge_model,
                    rubric=rubric,
                    expected_schema=schema,
                    held_out=uploaded is not None,
                )
                evaluation = replace(
                    run_config,
                    job_kind="evaluation",
                    parent_run_id=run_id,
                    evaluation=options,
                    resume_from_checkpoint=None,
                    push_to_hub=False,
                    use_unsloth=False,
                )
                errors = evaluation.validate()
                if errors:
                    raise ValueError(" ".join(errors))
                st.session_state.run_id = enqueue_run(evaluation)
                st.session_state.monitor_selected_run = st.session_state.run_id
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"Could not queue evaluation: {error}")
