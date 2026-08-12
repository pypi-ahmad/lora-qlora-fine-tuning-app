"""Model selection and inspection page."""

import streamlit as st

from lora_finetune_studio.hardware import model_size_warning
from lora_finetune_studio.sources import (
    get_hf_token,
    model_parameter_count,
    parse_hf_repo,
)

st.caption("Select the Hugging Face base model and verify its metadata.")

with st.form("model_source_form"):
    model_value = st.text_input(
        "Model repository",
        value=st.session_state.model_id,
        placeholder="Qwen/Qwen3-0.6B or https://huggingface.co/Qwen/Qwen3-0.6B",
        key="model_repo_input",
        persist_state="session",
    )
    revision = st.text_input(
        "Model revision",
        value=st.session_state.model_revision,
        key="model_revision_input",
        persist_state="session",
    )
    inspect_submitted = st.form_submit_button(
        "Inspect model", type="primary", icon=":material/search:"
    )

if inspect_submitted:
    try:
        model_id = parse_hf_repo(model_value, repo_type="model")
        with st.spinner("Reading model metadata..."):
            parameters = model_parameter_count(
                model_id, revision or "main", get_hf_token()
            )
        st.session_state.model_id = model_id
        st.session_state.model_revision = revision or "main"
        st.session_state.model_parameters = parameters
        st.session_state.model_ready = True
        st.session_state.training_config = None
        st.session_state.acknowledge_large_model = False
    except Exception as error:  # noqa: BLE001
        st.error(f"Model inspection failed: {error}")

if st.session_state.model_ready:
    parameters = st.session_state.model_parameters
    with st.container(border=True):
        st.success(f"Model ready: `{st.session_state.model_id}`")
        st.write(f"Revision: `{st.session_state.model_revision}`")
        if parameters:
            st.metric("Parameters", f"{parameters / 1_000_000_000:.2f}B")
        else:
            st.caption("Parameter count is not available from safetensors metadata.")
        warning = model_size_warning(parameters, st.session_state.hardware_profile)
        if warning:
            st.warning(warning, icon=":material/warning:")
