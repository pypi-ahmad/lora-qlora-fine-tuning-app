"""Training configuration page."""

import streamlit as st

from lora_finetune_studio.models import PRESETS, PeftMode, Preset, TrainingConfig

st.caption("Configure PEFT and trainer settings, then save them for final review.")

dataset_spec = st.session_state.dataset_spec
missing: list[str] = []
if not st.session_state.model_ready:
    missing.append("Inspect a model on the Model page.")
if dataset_spec is None:
    missing.append("Inspect a dataset on the Dataset page.")
elif dataset_spec.format == "needs_mapping":
    missing.append("Save the dataset column mapping on the Dataset page.")
if missing:
    for message in missing:
        st.info(message)
    st.stop()

st.session_state.setdefault("training_preset", Preset.STANDARD)
st.session_state.setdefault("training_show_advanced", False)
st.session_state.setdefault("training_push_to_hub", False)
initial_defaults = PRESETS[st.session_state.training_preset]
st.session_state.setdefault("training_max_length", int(initial_defaults["max_length"]))
st.session_state.setdefault("training_epochs", float(initial_defaults["epochs"]))
st.session_state.setdefault("training_learning_rate", 2e-4)
st.session_state.setdefault("training_batch_size", 1)
st.session_state.setdefault(
    "training_accumulation", int(initial_defaults["gradient_accumulation_steps"])
)
st.session_state.setdefault("training_gradient_checkpointing", True)


def apply_preset_defaults() -> None:
    defaults = PRESETS[st.session_state.training_preset]
    st.session_state.training_max_length = int(defaults["max_length"])
    st.session_state.training_epochs = float(defaults["epochs"])
    st.session_state.training_accumulation = int(
        defaults["gradient_accumulation_steps"]
    )


preset = (
    st.segmented_control(
        "Preset",
        list(Preset),
        key="training_preset",
        on_change=apply_preset_defaults,
        persist_state="session",
    )
    or Preset.STANDARD
)
peft_mode = (
    st.segmented_control(
        "PEFT method",
        list(PeftMode),
        key="training_peft_mode",
        persist_state="session",
    )
    or PeftMode.QLORA
)
show_advanced = st.toggle(
    "Show advanced controls",
    key="training_show_advanced",
    persist_state="session",
)
push_to_hub = st.toggle(
    "Push adapter to Hugging Face Hub",
    key="training_push_to_hub",
    persist_state="session",
)
defaults = PRESETS[preset]

with st.form("training_settings_form"):
    hub_model_id = None
    if push_to_hub:
        hub_model_id = st.text_input(
            "Output Hub repository",
            placeholder="username/my-adapter",
            key="training_hub_model_id",
            persist_state="session",
        )
    if show_advanced:
        max_length = st.number_input(
            "Maximum sequence length",
            128,
            8192,
            step=128,
            key="training_max_length",
            persist_state="session",
        )
        epochs = st.number_input(
            "Epochs",
            0.1,
            20.0,
            step=0.5,
            key="training_epochs",
            persist_state="session",
        )
        learning_rate = st.number_input(
            "Learning rate",
            1e-6,
            1e-2,
            format="%.6f",
            key="training_learning_rate",
            persist_state="session",
        )
        batch_size = st.number_input(
            "Per-device batch size",
            1,
            16,
            key="training_batch_size",
            persist_state="session",
        )
        accumulation = st.number_input(
            "Gradient accumulation steps",
            1,
            128,
            key="training_accumulation",
            persist_state="session",
        )
        gradient_checkpointing = st.checkbox(
            "Gradient checkpointing",
            key="training_gradient_checkpointing",
            persist_state="session",
        )
    else:
        max_length = int(defaults["max_length"])
        epochs = float(defaults["epochs"])
        learning_rate = 2e-4
        batch_size = 1
        accumulation = int(defaults["gradient_accumulation_steps"])
        gradient_checkpointing = True
    save_submitted = st.form_submit_button(
        "Save training settings", type="primary", icon=":material/save:"
    )

if save_submitted:
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
    if errors:
        for error in errors:
            st.error(error)
    else:
        st.session_state.training_config = config
        st.success("Training settings saved. Continue to Review & run.")

if st.session_state.training_config:
    st.info("A saved training draft is ready for review.")
