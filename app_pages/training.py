"""Training configuration page."""

import streamlit as st

from lora_finetune_studio.models import (
    PRESETS,
    TRAINING_RECIPES,
    ComputeType,
    PeftMode,
    Preset,
    TrainingApproach,
    TrainingConfig,
)
from lora_finetune_studio.unsloth_runtime import inspect_unsloth_runtime

st.caption("Choose a supported recipe, then configure and save its trainer settings.")

st.subheader("Supported training approaches")
st.dataframe(
    [
        {
            "Approach": approach.value,
            **{
                method.value: method in recipe.methods
                for method in (
                    PeftMode.LORA,
                    PeftMode.QLORA,
                    PeftMode.OFT,
                    PeftMode.QOFT,
                )
            },
        }
        for approach, recipe in TRAINING_RECIPES.items()
    ],
    hide_index=True,
    width="stretch",
)


def approach_changed() -> None:
    selected = TrainingApproach(st.session_state.training_approach)
    recipe = TRAINING_RECIPES[selected]
    st.session_state.training_learning_rate_mode = "Default"
    st.session_state.training_learning_rate = recipe.learning_rate
    st.session_state.training_batch_size = max(
        st.session_state.get("training_batch_size", 1), recipe.minimum_batch_size
    )
    if st.session_state.get("training_peft_mode") not in recipe.methods:
        st.session_state.training_peft_mode = recipe.methods[0]
    st.session_state.training_config = None


def method_changed() -> None:
    selected = PeftMode(st.session_state.training_peft_mode)
    if selected in {PeftMode.OFT, PeftMode.QOFT}:
        st.session_state.training_use_unsloth = False
    st.session_state.training_config = None


def compute_type_changed() -> None:
    selected = ComputeType(st.session_state.training_compute_type)
    if selected is ComputeType.FP32:
        st.session_state.training_use_unsloth = False
    st.session_state.training_config = None


approach = st.selectbox(
    "Approach",
    list(TrainingApproach),
    key="training_approach",
    on_change=approach_changed,
    persist_state="session",
)
recipe = TRAINING_RECIPES[approach]
if st.session_state.get("training_peft_mode") not in recipe.methods:
    st.session_state.training_peft_mode = recipe.methods[0]
peft_mode = st.selectbox(
    "Method",
    list(recipe.methods),
    key="training_peft_mode",
    on_change=method_changed,
    persist_state="session",
)

dataset_specs = st.session_state.dataset_specs
missing: list[str] = []
if not st.session_state.model_ready:
    missing.append("Inspect a model on the Model page.")
if not dataset_specs:
    missing.append("Add at least one dataset on the Dataset page.")
elif any(spec.format not in recipe.dataset_formats for spec in dataset_specs):
    missing.append(
        "Remap or remove datasets that are incompatible with the selected Approach."
    )
if missing:
    for message in missing:
        st.info(message)
    st.stop()

st.session_state.setdefault("training_preset", Preset.STANDARD)
st.session_state.setdefault("training_show_advanced", False)
st.session_state.setdefault("training_push_to_hub", False)
unsloth_runtime = inspect_unsloth_runtime()
selected_compute_type = ComputeType(st.session_state.training_compute_type)
unsloth_supported = (
    peft_mode in {PeftMode.LORA, PeftMode.QLORA}
    and selected_compute_type is not ComputeType.FP32
)
st.session_state.setdefault("training_use_unsloth", unsloth_runtime.available)
if not unsloth_runtime.available or not unsloth_supported:
    st.session_state.training_use_unsloth = False
initial_defaults = PRESETS[st.session_state.training_preset]
st.session_state.setdefault("training_max_length", int(initial_defaults["max_length"]))
st.session_state.setdefault("training_epochs", float(initial_defaults["epochs"]))
st.session_state.setdefault("training_batch_size", 1)
st.session_state.training_batch_size = max(
    st.session_state.training_batch_size, recipe.minimum_batch_size
)
st.session_state.setdefault(
    "training_accumulation", int(initial_defaults["gradient_accumulation_steps"])
)
st.session_state.setdefault("training_gradient_checkpointing", True)


def apply_preset_defaults() -> None:
    defaults = PRESETS[st.session_state.training_preset]
    st.session_state.training_max_length = int(defaults["max_length"])
    st.session_state.training_epochs_mode = "Default"
    st.session_state.training_epochs = float(defaults["epochs"])
    st.session_state.training_max_samples_mode = "Default"
    st.session_state.training_max_samples = int(defaults["max_samples"] or 100)
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
learning_rate_mode = (
    st.segmented_control(
        "Learning rate",
        ("Default", "Custom"),
        key="training_learning_rate_mode",
        persist_state="session",
    )
    or "Default"
)
epochs_mode = (
    st.segmented_control(
        "Epochs",
        ("Default", "Custom"),
        key="training_epochs_mode",
        persist_state="session",
    )
    or "Default"
)
max_samples_mode = (
    st.segmented_control(
        "Maximum samples",
        ("Default", "Custom"),
        key="training_max_samples_mode",
        persist_state="session",
    )
    or "Default"
)
max_grad_norm_mode = (
    st.segmented_control(
        "Maximum gradient norm",
        ("Default", "Custom"),
        key="training_max_grad_norm_mode",
        persist_state="session",
    )
    or "Default"
)
compute_type = st.selectbox(
    "Compute type",
    list(ComputeType),
    key="training_compute_type",
    on_change=compute_type_changed,
    help="Auto uses BF16 when supported and otherwise uses FP16.",
    persist_state="session",
)
if compute_type is ComputeType.FP32:
    st.warning(
        "FP32 greatly increases VRAM use and training time. It uses the standard "
        "Transformers/TRL backend because Unsloth's optimized kernels require "
        "FP16 or BF16."
    )
use_unsloth = st.toggle(
    "Use Unsloth acceleration",
    key="training_use_unsloth",
    disabled=not unsloth_runtime.available or not unsloth_supported,
    help="Uses the repository-local native Windows Unsloth runtime.",
    persist_state="session",
)
if not unsloth_runtime.available:
    st.info(unsloth_runtime.detail)
elif not unsloth_supported:
    st.info("OFT, QOFT, and FP32 compute use the standard PEFT/TRL backend.")
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
    if learning_rate_mode == "Custom":
        learning_rate = st.number_input(
            "Custom learning rate",
            1e-7,
            1e-2,
            format="%.7f",
            key="training_learning_rate",
            persist_state="session",
        )
    else:
        learning_rate = recipe.learning_rate
        st.caption(
            f"Using the recommended default for {approach.value}: `{learning_rate:g}`"
        )
    if epochs_mode == "Custom":
        epochs = st.number_input(
            "Custom epochs",
            0.1,
            20.0,
            step=0.5,
            key="training_epochs",
            persist_state="session",
        )
    else:
        epochs = float(defaults["epochs"])
        st.caption(f"Using the {preset.value} default: `{epochs:g}` epochs")
    if max_samples_mode == "Custom":
        max_samples = st.number_input(
            "Custom maximum samples",
            min_value=1,
            step=1,
            key="training_max_samples",
            persist_state="session",
        )
    else:
        max_samples = defaults["max_samples"]
        sample_default = f"{max_samples:,}" if max_samples is not None else "all"
        st.caption(f"Using the {preset.value} default: `{sample_default}` samples")
    if max_grad_norm_mode == "Custom":
        max_grad_norm = st.number_input(
            "Custom maximum gradient norm",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key="training_max_grad_norm",
            help="Set to 0 to disable gradient clipping.",
            persist_state="session",
        )
    else:
        max_grad_norm = 1.0
        st.caption("Using the trainer default maximum gradient norm: `1.0`")
    if show_advanced:
        max_length = st.number_input(
            "Maximum sequence length",
            128,
            8192,
            step=128,
            key="training_max_length",
            persist_state="session",
        )
        beta = (
            st.number_input(
                "Beta",
                min_value=0.001,
                max_value=10.0,
                step=0.05,
                key="training_beta",
                persist_state="session",
                help="Controls preference strength relative to the reference model.",
            )
            if recipe.uses_beta
            else 0.1
        )
        batch_size = st.number_input(
            "Per-device batch size",
            recipe.minimum_batch_size,
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
        beta = 0.1
        batch_size = recipe.minimum_batch_size
        accumulation = int(defaults["gradient_accumulation_steps"])
        gradient_checkpointing = True
    save_submitted = st.form_submit_button(
        "Save training settings", type="primary", icon=":material/save:"
    )

if save_submitted:
    config = TrainingConfig(
        model_id=st.session_state.model_id,
        model_revision=st.session_state.model_revision,
        datasets=list(dataset_specs),
        approach=approach,
        peft_mode=peft_mode,
        use_unsloth=use_unsloth,
        compute_type=compute_type,
        preset=preset,
        max_length=int(max_length),
        epochs=float(epochs),
        max_steps=int(defaults["max_steps"]),
        max_samples=int(max_samples) if max_samples is not None else None,
        learning_rate=float(learning_rate),
        beta=float(beta),
        batch_size=int(batch_size),
        gradient_accumulation_steps=int(accumulation),
        max_grad_norm=float(max_grad_norm),
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
