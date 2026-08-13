"""Read-only public showcase for LoRA Fine-tune Studio."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "showcase.json"
SHOWCASE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CONFIG = SHOWCASE["training_config"]

st.set_page_config(
    page_title="LoRA Fine-tune Studio showcase",
    page_icon=":material/model_training:",
    layout="wide",
)

st.title("LoRA Fine-tune Studio")
st.caption(
    "A guided local workflow for dataset validation, PEFT configuration, "
    "training review, and adapter monitoring."
)
st.info(
    "Synthetic read-only demonstration. It performs no training, model downloads, "
    "uploads, network requests, or persistence.",
    icon=":material/shield:",
)

dataset_tab, configure_tab, review_tab, monitor_tab = st.tabs(
    ["1. Dataset", "2. Configure", "3. Review", "4. Monitor"]
)

with dataset_tab:
    st.subheader("Inspect before training")
    st.caption(
        "The production app validates the uploaded schema and column mapping before "
        "a dataset can enter a run."
    )
    with st.container(border=True):
        st.markdown("**Source:** `examples/sft_sample.jsonl`")
        st.markdown("**Detected format:** conversational messages")
        st.markdown("**Rows:** 10 synthetic examples")
    st.dataframe(SHOWCASE["dataset_preview"], hide_index=True)

with configure_tab:
    st.subheader("Choose a supported recipe")
    model, method, preset = st.columns(3)
    model.metric("Base model", CONFIG["model_id"])
    method.metric(
        "Approach and method", f"{CONFIG['approach']} · {CONFIG['peft_mode']}"
    )
    preset.metric("Preset", CONFIG["preset"])
    with st.container(border=True):
        st.markdown("**Maximum steps:** 20")
        st.markdown("**Maximum sequence length:** 512 tokens")
        st.markdown("**Gradient accumulation:** 4 steps")
        st.markdown("**Hub upload:** disabled")
    st.caption("The smoke preset checks pipeline wiring; it is not a quality claim.")

with review_tab:
    st.subheader("Validate the complete run contract")
    st.success(
        "The synthetic configuration satisfies the production validation contract.",
        icon=":material/check_circle:",
    )
    st.json(CONFIG)
    st.button(
        "Start training",
        icon=":material/play_arrow:",
        type="primary",
        disabled=True,
        help="Execution is disabled in the public showcase.",
    )

with monitor_tab:
    st.subheader("Inspect a run")
    status, progress, artifacts = st.columns(3)
    status.metric("Fixture state", "Completed preview")
    progress.metric("Configured steps", "20")
    artifacts.metric("Artifact type", "PEFT adapter")
    st.line_chart(
        {"Relative synthetic loss": SHOWCASE["relative_loss"]},
        x_label="Illustrative checkpoint",
        y_label="Relative value",
    )
    st.caption(
        "Illustrative fixture data only. The chart is not a benchmark or a measured "
        "training result."
    )

with st.container(horizontal=True):
    st.link_button(
        "Code",
        "https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app",
        icon=":material/code:",
    )
    st.link_button(
        "Setup",
        "https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/blob/main/SETUP.md",
        icon=":material/build:",
    )
    st.link_button(
        "Architecture",
        "https://github.com/pypi-ahmad/lora-qlora-fine-tuning-app/blob/main/TECHNICAL.md",
        icon=":material/account_tree:",
    )
