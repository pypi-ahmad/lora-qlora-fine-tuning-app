"""Streamlit entry point for LoRA Fine-tune Studio."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import URLError

import streamlit as st

from lora_finetune_studio.hardware import detect_hardware, model_size_warning
from lora_finetune_studio.inference import generate_text
from lora_finetune_studio.jobs import (
    cancel_run,
    create_run,
    launch_run,
    read_log,
    read_status,
    resume_run,
)
from lora_finetune_studio.models import (
    PRESETS,
    DatasetSpec,
    JobState,
    PeftMode,
    Preset,
    TrainingConfig,
)
from lora_finetune_studio.ollama import generate as ollama_generate
from lora_finetune_studio.ollama import list_models
from lora_finetune_studio.sources import (
    get_hf_token,
    inspect_dataset,
    load_training_dataset,
    model_parameter_count,
    parse_hf_repo,
    save_upload,
    token_identity,
)

st.set_page_config(
    page_title="LoRA Fine-tune Studio",
    page_icon=":material/model_training:",
    layout="wide",
)

st.session_state.setdefault("inspection", None)
st.session_state.setdefault("dataset_spec", None)
st.session_state.setdefault("model_id", "Qwen/Qwen3-0.6B")
st.session_state.setdefault("model_revision", "main")
st.session_state.setdefault("model_parameters", None)
st.session_state.setdefault("run_id", None)
st.session_state.setdefault("ollama_messages", [])


@st.cache_data(ttl="5m", max_entries=8, show_spinner=False)
def hardware_profile():
    return detect_hardware()


profile = hardware_profile()
token = get_hf_token()
if not token:
    try:
        token = str(st.secrets["HF_TOKEN"])
        os.environ["HF_TOKEN"] = token
    except (FileNotFoundError, KeyError):
        token = None

st.title("LoRA Fine-tune Studio")
st.caption("Train LoRA or QLoRA adapters locally with supervised fine-tuning.")

with st.container(border=True):
    metric_columns = st.columns(4)
    metric_columns[0].metric("GPU", profile.gpu_name or "Not detected")
    metric_columns[1].metric("VRAM", f"{profile.vram_gb:g} GB")
    metric_columns[2].metric("System RAM", f"{profile.ram_gb:g} GB")
    metric_columns[3].metric("Free disk", f"{profile.free_disk_gb:g} GB")
    if profile.warning:
        st.error(profile.warning, icon=":material/error:")
    else:
        st.caption(
            f"Recommended: {profile.recommended_mode}, models up to approximately "
            f"{profile.recommended_max_billions:g}B parameters."
        )
    if token:
        st.badge("HF_TOKEN found", icon=":material/check:", color="green")
        if st.button("Verify Hugging Face token", icon=":material/verified_user:"):
            try:
                st.success(f"Authenticated as {token_identity(token)}")
            except Exception as error:  # noqa: BLE001
                st.error(f"Token verification failed: {error}")
    else:
        st.warning(
            "HF_TOKEN is not set. Public repositories work; gated and private ones do not."
        )

st.header("1. Select model and dataset")
source_mode = (
    st.segmented_control(
        "Dataset source",
        ["Hugging Face", "Upload"],
        default="Hugging Face",
        key="dataset_source_mode",
    )
    or "Hugging Face"
)

with st.form("source_form"):
    model_value = st.text_input(
        "Model repository",
        value=st.session_state.model_id,
        placeholder="Qwen/Qwen3-0.6B or https://huggingface.co/Qwen/Qwen3-0.6B",
    )
    revision = st.text_input("Model revision", value=st.session_state.model_revision)
    dataset_value = ""
    dataset_config = ""
    dataset_split = "train"
    uploaded = None
    if source_mode == "Hugging Face":
        dataset_value = st.text_input(
            "Dataset repository",
            placeholder="trl-lib/Capybara or https://huggingface.co/datasets/trl-lib/Capybara",
        )
        dataset_config = st.text_input(
            "Dataset configuration", help="Leave blank for default."
        )
        dataset_split = st.text_input("Dataset split", value="train")
    else:
        uploaded = st.file_uploader("Dataset file", type=["json", "jsonl", "csv"])
    inspect_submitted = st.form_submit_button(
        "Inspect sources", type="primary", icon=":material/search:"
    )

if inspect_submitted:
    try:
        model_id = parse_hf_repo(model_value, repo_type="model")
        if source_mode == "Hugging Face":
            dataset_spec = DatasetSpec(
                source="hub",
                repo_id=parse_hf_repo(dataset_value, repo_type="dataset"),
                config_name=dataset_config or None,
                split=dataset_split,
            )
        else:
            if uploaded is None:
                raise ValueError("Choose a dataset file.")
            upload_path = save_upload(uploaded.name, uploaded.getvalue())
            dataset_spec = DatasetSpec(
                source="upload", local_path=str(upload_path), split="train"
            )
        with st.spinner("Reading metadata and dataset preview..."):
            dataset = load_training_dataset(
                repo_id=dataset_spec.repo_id,
                local_path=dataset_spec.local_path,
                config_name=dataset_spec.config_name,
                split=dataset_spec.split,
                token=token,
            )
            inspection = inspect_dataset(dataset)
            parameters = model_parameter_count(model_id, revision, token)
        st.session_state.inspection = inspection
        st.session_state.dataset_spec = dataset_spec
        st.session_state.model_id = model_id
        st.session_state.model_revision = revision
        st.session_state.model_parameters = parameters
    except Exception as error:  # noqa: BLE001
        st.error(f"Source inspection failed: {error}")

inspection = st.session_state.inspection
dataset_spec = st.session_state.dataset_spec
if inspection and dataset_spec:
    with st.container(border=True):
        st.success(
            f"Dataset ready: {inspection.rows:,} rows, format `{inspection.format}`"
        )
        st.dataframe(inspection.preview, width="stretch")
        if inspection.format == "needs_mapping":
            st.warning("Map columns before training.")
            mapping_mode = st.segmented_control(
                "Training format", ["Text", "Prompt and completion"], default="Text"
            )
            if mapping_mode == "Text":
                dataset_spec.format = "text"
                dataset_spec.text_column = st.selectbox(
                    "Text column", inspection.columns
                )
            else:
                dataset_spec.format = "prompt_completion"
                dataset_spec.prompt_column = st.selectbox(
                    "Prompt column", inspection.columns
                )
                dataset_spec.completion_column = st.selectbox(
                    "Completion column", inspection.columns
                )
        else:
            dataset_spec.format = inspection.format
            if inspection.format == "text":
                dataset_spec.text_column = "text"
            elif inspection.format == "prompt_completion":
                dataset_spec.prompt_column = "prompt"
                dataset_spec.completion_column = "completion"

    parameter_count = st.session_state.model_parameters
    if parameter_count:
        st.caption(
            f"Model size: approximately {parameter_count / 1_000_000_000:.2f}B parameters"
        )
    warning = model_size_warning(parameter_count, profile)
    acknowledge_large_model = False
    if warning:
        st.warning(warning, icon=":material/warning:")
        acknowledge_large_model = st.toggle(
            "I understand this run may exhaust GPU memory"
        )

    st.header("2. Configure training")
    preset = (
        st.segmented_control("Preset", list(Preset), default=Preset.STANDARD)
        or Preset.STANDARD
    )
    peft_mode = (
        st.segmented_control(
            "PEFT method",
            list(PeftMode),
            default=profile.recommended_mode or PeftMode.QLORA,
        )
        or PeftMode.QLORA
    )
    show_advanced = st.toggle("Show advanced controls")
    push_to_hub = st.toggle("Push adapter to Hugging Face Hub")
    defaults = PRESETS[preset]

    with st.form("training_form"):
        hub_model_id = None
        if push_to_hub:
            hub_model_id = st.text_input(
                "Output Hub repository", placeholder="username/my-adapter"
            )
        if show_advanced:
            max_length = st.number_input(
                "Maximum sequence length",
                128,
                8192,
                int(defaults["max_length"]),
                step=128,
            )
            epochs = st.number_input(
                "Epochs", 0.1, 20.0, float(defaults["epochs"]), step=0.5
            )
            learning_rate = st.number_input(
                "Learning rate", 1e-6, 1e-2, 2e-4, format="%.6f"
            )
            batch_size = st.number_input("Per-device batch size", 1, 16, 1)
            accumulation = st.number_input(
                "Gradient accumulation steps",
                1,
                128,
                int(defaults["gradient_accumulation_steps"]),
            )
            gradient_checkpointing = st.checkbox("Gradient checkpointing", value=True)
        else:
            max_length = int(defaults["max_length"])
            epochs = float(defaults["epochs"])
            learning_rate = 2e-4
            batch_size = 1
            accumulation = int(defaults["gradient_accumulation_steps"])
            gradient_checkpointing = True
        start_submitted = st.form_submit_button(
            "Start training", type="primary", icon=":material/play_arrow:"
        )

    if start_submitted:
        config = TrainingConfig(
            model_id=st.session_state.model_id,
            model_revision=st.session_state.model_revision,
            dataset=dataset_spec,
            peft_mode=peft_mode,
            preset=preset,
            max_length=int(max_length),
            epochs=float(epochs),
            max_steps=int(defaults["max_steps"]),
            max_samples=defaults["max_samples"],
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            gradient_accumulation_steps=int(accumulation),
            gradient_checkpointing=gradient_checkpointing,
            eval_enabled=bool(defaults["eval_enabled"]),
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
        errors = config.validate()
        if not profile.cuda_available:
            errors.append("CUDA GPU is required.")
        if warning and not acknowledge_large_model:
            errors.append("Acknowledge the model-size warning before training.")
        if push_to_hub and not token:
            errors.append("HF_TOKEN is required to upload the adapter.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                run_id = create_run(config)
                launch_run(run_id)
                st.session_state.run_id = run_id
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"Could not start training: {error}")


@st.fragment(run_every="2s")
def training_monitor() -> None:
    run_id = st.session_state.get("run_id")
    if not run_id:
        return
    st.header("3. Monitor training")
    try:
        status = read_status(run_id)
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
            cancel_run(run_id)
            st.rerun(scope="fragment")
        if status.state in {JobState.CANCELLED, JobState.FAILED} and st.button(
            "Resume latest checkpoint", icon=":material/play_arrow:"
        ):
            try:
                resume_run(run_id)
                st.rerun(scope="fragment")
            except (FileNotFoundError, RuntimeError) as error:
                st.error(str(error))
        log_expander = st.expander("Training log", on_change="rerun")
        if log_expander.open:
            with log_expander:
                st.code(
                    read_log(run_id) or "Waiting for worker output...", language="text"
                )


training_monitor()

run_id = st.session_state.get("run_id")
if run_id:
    status = read_status(run_id)
    if status.state is JobState.COMPLETED and status.artifact_dir:
        st.header("4. Evaluate adapter")
        prompt = st.text_area(
            "Comparison prompt", placeholder="Write a concise explanation of LoRA."
        )
        if st.button("Compare base and adapter", disabled=not bool(prompt)):
            adapter_path = str(Path(status.artifact_dir) / "adapter")
            with st.spinner("Generating base response..."):
                base_response = generate_text(
                    st.session_state.model_id,
                    prompt,
                    token=token,
                    revision=st.session_state.model_revision,
                )
            with st.spinner("Generating adapter response..."):
                adapter_response = generate_text(
                    st.session_state.model_id,
                    prompt,
                    token=token,
                    revision=st.session_state.model_revision,
                    adapter_path=adapter_path,
                )
            left, right = st.columns(2)
            left.subheader("Base model")
            left.write(base_response)
            right.subheader("Fine-tuned adapter")
            right.write(adapter_response)

st.header("Ollama playground")
st.caption(
    "Tests models already installed in Ollama. It does not import the trained adapter."
)
try:
    ollama_models = list_models()
except (OSError, URLError, TimeoutError):
    ollama_models = []
if not ollama_models:
    st.info(
        "Ollama is unavailable or has no models. Start Ollama and pull a model first."
    )
else:
    ollama_model = st.selectbox("Ollama model", ollama_models)
    for message in st.session_state.ollama_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if ollama_prompt := st.chat_input("Message Ollama", submit_mode="disable"):
        st.session_state.ollama_messages.append(
            {"role": "user", "content": ollama_prompt}
        )
        with st.chat_message("user"):
            st.write(ollama_prompt)
        try:
            response = ollama_generate(ollama_model, ollama_prompt)
        except Exception as error:  # noqa: BLE001
            response = f"Ollama request failed: {error}"
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.ollama_messages.append(
            {"role": "assistant", "content": response}
        )
