"""Streamlit entry point for LoRA Fine-tune Studio."""

from __future__ import annotations

import os

import streamlit as st

from lora_finetune_studio.hardware import detect_hardware
from lora_finetune_studio.jobs import cancel_active_run, dispatch_next_run
from lora_finetune_studio.lifecycle import schedule_application_exit
from lora_finetune_studio.models import (
    TRAINING_RECIPES,
    ComputeType,
    PeftMode,
    TrainingApproach,
)
from lora_finetune_studio.sources import get_hf_token

st.set_page_config(
    page_title="LoRA Fine-tune Studio",
    page_icon=":material/model_training:",
    layout="wide",
)

st.session_state.setdefault("inspection", None)
if (
    "dataset_specs" not in st.session_state
    or "dataset_inspections" not in st.session_state
):
    legacy_dataset = st.session_state.get("dataset_spec")
    legacy_inspection = st.session_state.get("inspection")
    if legacy_dataset and legacy_inspection:
        st.session_state.dataset_specs = [legacy_dataset]
        st.session_state.dataset_inspections = [legacy_inspection]
    else:
        st.session_state.dataset_specs = []
        st.session_state.dataset_inspections = []
st.session_state.setdefault("pending_dataset_spec", None)
st.session_state.setdefault("pending_dataset_inspection", None)
st.session_state.setdefault("model_id", "Qwen/Qwen3-0.6B")
st.session_state.setdefault("model_revision", "main")
st.session_state.setdefault("model_parameters", None)
st.session_state.setdefault("model_ready", False)
st.session_state.setdefault("training_config", None)
st.session_state.setdefault("training_approach", TrainingApproach.SFT)
st.session_state.setdefault(
    "training_learning_rate", TRAINING_RECIPES[TrainingApproach.SFT].learning_rate
)
st.session_state.setdefault("training_learning_rate_mode", "Default")
st.session_state.setdefault("training_epochs_mode", "Default")
st.session_state.setdefault("training_max_samples_mode", "Default")
st.session_state.setdefault("training_max_samples", 100)
st.session_state.setdefault("training_max_grad_norm_mode", "Default")
st.session_state.setdefault("training_max_grad_norm", 1.0)
st.session_state.setdefault("training_compute_type", ComputeType.AUTO)
st.session_state.setdefault("training_beta", 0.1)
st.session_state.setdefault("run_id", None)
st.session_state.setdefault("ollama_messages", [])
st.session_state.setdefault("acknowledge_large_model", False)
st.session_state.setdefault("confirm_shutdown", False)
if "hardware_profile" not in st.session_state:
    st.session_state.hardware_profile = detect_hardware()
if "training_peft_mode" not in st.session_state:
    st.session_state.training_peft_mode = (
        st.session_state.hardware_profile.recommended_mode or PeftMode.QLORA
    )

token = get_hf_token()
if not token:
    try:
        token = str(st.secrets["HF_TOKEN"])
        os.environ["HF_TOKEN"] = token
    except FileNotFoundError, KeyError:
        pass

try:
    dispatch_next_run()
except (OSError, RuntimeError, ValueError) as error:
    st.session_state.queue_start_error = str(error)
else:
    st.session_state.pop("queue_start_error", None)

pages = [
    st.Page(
        "app_pages/system.py",
        title="System",
        icon=":material/computer:",
        url_path="system",
        default=True,
    ),
    st.Page(
        "app_pages/dataset.py",
        title="Dataset",
        icon=":material/database:",
        url_path="dataset",
    ),
    st.Page(
        "app_pages/model.py",
        title="Model",
        icon=":material/model_training:",
        url_path="model",
    ),
    st.Page(
        "app_pages/gpu_memory.py",
        title="GPU memory",
        icon=":material/memory:",
        url_path="gpu_memory",
    ),
    st.Page(
        "app_pages/training.py",
        title="Training",
        icon=":material/tune:",
        url_path="training",
    ),
    st.Page(
        "app_pages/review.py",
        title="Review & run",
        icon=":material/checklist:",
        url_path="review",
    ),
    st.Page(
        "app_pages/monitor.py",
        title="Monitor",
        icon=":material/monitoring:",
        url_path="monitor",
    ),
    st.Page(
        "app_pages/ollama.py",
        title="Ollama playground",
        icon=":material/chat:",
        url_path="ollama",
    ),
]

page = st.navigation(pages, position="sidebar", expanded=True)

with st.sidebar:
    if st.session_state.get("queue_start_error"):
        st.warning(
            f"Training queue could not advance: {st.session_state.queue_start_error}"
        )
    st.divider()
    if not st.session_state.confirm_shutdown:
        if st.button(
            "Stop LoRA Studio",
            icon=":material/power_settings_new:",
            width="stretch",
        ):
            st.session_state.confirm_shutdown = True
            st.rerun()
    else:
        st.warning(
            "This stops LoRA Studio and cancels its active training worker. "
            "Ollama and unrelated applications are not affected."
        )
        with st.container(horizontal=True):
            if st.button("Keep running", icon=":material/close:"):
                st.session_state.confirm_shutdown = False
                st.rerun()
            if st.button(
                "Confirm stop",
                icon=":material/power_settings_new:",
                type="primary",
            ):
                try:
                    cancelled_run = cancel_active_run(dispatch_next=False)
                except (OSError, RuntimeError, ValueError) as error:
                    st.error(f"LoRA Studio was not stopped: {error}")
                else:
                    message = "Stopping LoRA Studio"
                    if cancelled_run:
                        message += f" after cancelling training run {cancelled_run}"
                    st.info(f"{message}. You may close this browser tab.")
                    schedule_application_exit()
                    st.stop()

st.title(page.title)
page.run()
