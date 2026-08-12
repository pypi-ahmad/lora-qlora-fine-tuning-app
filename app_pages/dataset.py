"""Dataset selection and inspection page."""

import streamlit as st

from lora_finetune_studio.models import DatasetSpec
from lora_finetune_studio.sources import (
    get_hf_token,
    inspect_dataset,
    load_training_dataset,
    parse_hf_repo,
    save_upload,
)

st.caption("Choose, validate, preview, and map the supervised fine-tuning dataset.")

source_mode = (
    st.segmented_control(
        "Dataset source",
        ["Hugging Face", "Upload"],
        default="Hugging Face",
        key="dataset_source_mode",
        persist_state="session",
    )
    or "Hugging Face"
)

with st.form("dataset_source_form"):
    dataset_value = ""
    dataset_config = ""
    dataset_split = "train"
    uploaded = None
    if source_mode == "Hugging Face":
        dataset_value = st.text_input(
            "Dataset repository",
            placeholder="trl-lib/Capybara or https://huggingface.co/datasets/trl-lib/Capybara",
            key="dataset_repo_input",
            persist_state="session",
        )
        dataset_config = st.text_input(
            "Dataset configuration",
            help="Leave blank for default.",
            key="dataset_config_input",
            persist_state="session",
        )
        dataset_split = st.text_input(
            "Dataset split",
            value="train",
            key="dataset_split_input",
            persist_state="session",
        )
    else:
        uploaded = st.file_uploader("Dataset file", type=["json", "jsonl", "csv"])
    inspect_submitted = st.form_submit_button(
        "Inspect dataset", type="primary", icon=":material/search:"
    )

if inspect_submitted:
    try:
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
        with st.spinner("Reading dataset preview..."):
            dataset = load_training_dataset(
                repo_id=dataset_spec.repo_id,
                local_path=dataset_spec.local_path,
                config_name=dataset_spec.config_name,
                split=dataset_spec.split,
                token=get_hf_token(),
            )
            inspection = inspect_dataset(dataset)
        dataset_spec.format = inspection.format
        if inspection.format == "text":
            dataset_spec.text_column = "text"
        elif inspection.format == "prompt_completion":
            dataset_spec.prompt_column = "prompt"
            dataset_spec.completion_column = "completion"
        st.session_state.inspection = inspection
        st.session_state.dataset_spec = dataset_spec
        st.session_state.training_config = None
        for key in (
            "dataset_mapping_mode",
            "dataset_text_column",
            "dataset_prompt_column",
            "dataset_completion_column",
        ):
            st.session_state.pop(key, None)
    except Exception as error:  # noqa: BLE001
        st.error(f"Dataset inspection failed: {error}")

inspection = st.session_state.inspection
dataset_spec = st.session_state.dataset_spec
if inspection and dataset_spec:
    with st.container(border=True):
        st.subheader("Dataset preview")
        st.write(f"{inspection.rows:,} rows · detected format: `{inspection.format}`")
        st.dataframe(inspection.preview, width="stretch")
        if dataset_spec.format == "needs_mapping":
            st.warning("Map columns before configuring training.")
            mapping_mode = (
                st.segmented_control(
                    "Training format",
                    ["Text", "Prompt and completion"],
                    default="Text",
                    key="dataset_mapping_mode",
                    persist_state="session",
                )
                or "Text"
            )
            if mapping_mode == "Text":
                text_column = st.selectbox(
                    "Text column",
                    inspection.columns,
                    key="dataset_text_column",
                    persist_state="session",
                )
                prompt_column = completion_column = None
            else:
                text_column = None
                prompt_column = st.selectbox(
                    "Prompt column",
                    inspection.columns,
                    key="dataset_prompt_column",
                    persist_state="session",
                )
                completion_column = st.selectbox(
                    "Completion column",
                    inspection.columns,
                    key="dataset_completion_column",
                    persist_state="session",
                )
            if st.button("Save column mapping", icon=":material/save:"):
                if mapping_mode == "Text":
                    dataset_spec.format = "text"
                    dataset_spec.text_column = text_column
                    dataset_spec.prompt_column = None
                    dataset_spec.completion_column = None
                else:
                    dataset_spec.format = "prompt_completion"
                    dataset_spec.text_column = None
                    dataset_spec.prompt_column = prompt_column
                    dataset_spec.completion_column = completion_column
                st.session_state.training_config = None
                st.success("Column mapping saved.")
        else:
            st.success("Dataset is ready for training.")
