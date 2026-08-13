"""Dataset collection, inspection, and mapping page."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import streamlit as st

from lora_finetune_studio.models import (
    TRAINING_RECIPES,
    DatasetSpec,
    TrainingApproach,
)
from lora_finetune_studio.sources import (
    DatasetInspection,
    get_hf_token,
    inspect_dataset,
    load_training_dataset,
    parse_hf_repo,
    save_upload,
)

approach = TrainingApproach(st.session_state.training_approach)
recipe = TRAINING_RECIPES[approach]
st.caption(
    f"Inspect and add one or more compatible datasets for {approach}. "
    f"Required format: {', '.join(recipe.dataset_formats)}."
)
st.session_state.setdefault("pending_dataset_replace_index", None)


def source_label(spec: DatasetSpec) -> str:
    source = spec.repo_id or spec.local_path or "Not configured"
    return Path(source).name if spec.local_path else source


def source_identity(
    spec: DatasetSpec,
) -> tuple[str, str | None, str | None, str | None, str]:
    return (
        spec.source,
        spec.repo_id,
        spec.local_path,
        spec.config_name,
        spec.split,
    )


def source_key(spec: DatasetSpec) -> str:
    return hashlib.sha256(repr(source_identity(spec)).encode()).hexdigest()[:12]


def is_compatible(spec: DatasetSpec) -> bool:
    return spec.format in recipe.dataset_formats


def clear_pending() -> None:
    st.session_state.pending_dataset_spec = None
    st.session_state.pending_dataset_inspection = None
    st.session_state.pending_dataset_replace_index = None


def save_pending_dataset(spec: DatasetSpec, inspection: DatasetInspection) -> None:
    selected: list[DatasetSpec] = st.session_state.dataset_specs
    replace_index = st.session_state.pending_dataset_replace_index
    other_specs = [
        item for index, item in enumerate(selected) if index != replace_index
    ]
    if source_identity(spec) in {source_identity(item) for item in other_specs}:
        raise ValueError("This dataset source is already selected.")
    if spec.format not in recipe.dataset_formats:
        raise ValueError(f"{approach} does not support `{spec.format}` datasets.")
    compatible_formats = {item.format for item in other_specs if is_compatible(item)}
    if compatible_formats and spec.format not in compatible_formats:
        expected = next(iter(compatible_formats))
        raise ValueError(
            f"Selected datasets use `{expected}` format; this dataset uses `{spec.format}`."
        )
    if replace_index is None:
        st.session_state.dataset_specs.append(spec)
        st.session_state.dataset_inspections.append(inspection)
    else:
        st.session_state.dataset_specs[replace_index] = spec
        st.session_state.dataset_inspections[replace_index] = inspection
    clear_pending()
    st.session_state.training_config = None


selected_specs: list[DatasetSpec] = st.session_state.dataset_specs
selected_inspections: list[DatasetInspection] = st.session_state.dataset_inspections

if selected_specs:
    st.subheader("Selected datasets")
    st.dataframe(
        [
            {
                "Source": source_label(spec),
                "Configuration": spec.config_name or "Default",
                "Split": spec.split,
                "Format": spec.format,
                "Rows": inspection.rows,
                "Status": "Ready" if is_compatible(spec) else "Needs remapping",
            }
            for spec, inspection in zip(
                selected_specs, selected_inspections, strict=True
            )
        ],
        hide_index=True,
        width="stretch",
    )
    for index, spec in enumerate(list(selected_specs)):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.write(f"`{source_label(spec)}`")
            if not is_compatible(spec) and st.button(
                "Remap",
                icon=":material/conversion_path:",
                key=f"remap_dataset_{source_key(spec)}",
            ):
                st.session_state.pending_dataset_spec = replace(
                    spec,
                    format="needs_mapping",
                    text_column=None,
                    prompt_column=None,
                    completion_column=None,
                    chosen_column=None,
                    rejected_column=None,
                )
                st.session_state.pending_dataset_inspection = selected_inspections[
                    index
                ]
                st.session_state.pending_dataset_replace_index = index
                st.rerun()
            if st.button(
                "Remove",
                icon=":material/delete:",
                key=f"remove_dataset_{source_key(spec)}",
            ):
                st.session_state.dataset_specs.pop(index)
                st.session_state.dataset_inspections.pop(index)
                st.session_state.training_config = None
                clear_pending()
                st.rerun()
else:
    st.info("No datasets selected yet.")

st.subheader("Add a dataset")
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
            placeholder="trl-lib/Capybara or owner/preference-dataset",
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
            pending_spec = DatasetSpec(
                source="hub",
                repo_id=parse_hf_repo(dataset_value, repo_type="dataset"),
                config_name=dataset_config or None,
                split=dataset_split,
            )
        else:
            if uploaded is None:
                raise ValueError("Choose a dataset file.")
            upload_path = save_upload(uploaded.name, uploaded.getvalue())
            pending_spec = DatasetSpec(
                source="upload", local_path=str(upload_path), split="train"
            )
        with st.spinner("Reading dataset preview..."):
            dataset = load_training_dataset(
                repo_id=pending_spec.repo_id,
                local_path=pending_spec.local_path,
                config_name=pending_spec.config_name,
                split=pending_spec.split,
                token=get_hf_token(),
            )
            pending_inspection = inspect_dataset(dataset)
        pending_spec.format = pending_inspection.format
        if pending_inspection.format == "text":
            pending_spec.text_column = "text"
        elif pending_inspection.format == "prompt_completion":
            pending_spec.prompt_column = "prompt"
            pending_spec.completion_column = "completion"
        elif pending_inspection.format == "preference":
            pending_spec.prompt_column = "prompt"
            pending_spec.chosen_column = "chosen"
            pending_spec.rejected_column = "rejected"
        st.session_state.pending_dataset_spec = pending_spec
        st.session_state.pending_dataset_inspection = pending_inspection
        st.session_state.pending_dataset_replace_index = None
        for key in (
            "dataset_mapping_mode",
            "dataset_text_column",
            "dataset_prompt_column",
            "dataset_completion_column",
            "dataset_chosen_column",
            "dataset_rejected_column",
        ):
            st.session_state.pop(key, None)
    except Exception as error:  # noqa: BLE001
        st.error(f"Dataset inspection failed: {error}")

pending_spec = st.session_state.pending_dataset_spec
pending_inspection = st.session_state.pending_dataset_inspection
if pending_spec and pending_inspection:
    with st.container(border=True):
        action = (
            "Remap dataset"
            if st.session_state.pending_dataset_replace_index is not None
            else "Add dataset"
        )
        st.subheader("Inspected dataset")
        st.write(
            f"`{source_label(pending_spec)}` · {pending_inspection.rows:,} rows · "
            f"detected format: `{pending_inspection.format}`"
        )
        st.dataframe(pending_inspection.preview, width="stretch")

        mapping_required = pending_spec.format not in recipe.dataset_formats
        mapping_mode = None
        if recipe.dataset_formats == ("preference",):
            mapping_mode = "Preference" if mapping_required else None
        elif mapping_required:
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

        text_column = pending_spec.text_column
        prompt_column = pending_spec.prompt_column
        completion_column = pending_spec.completion_column
        chosen_column = pending_spec.chosen_column
        rejected_column = pending_spec.rejected_column
        if mapping_mode == "Text":
            text_column = st.selectbox(
                "Text column",
                pending_inspection.columns,
                key="dataset_text_column",
                persist_state="session",
            )
        elif mapping_mode == "Prompt and completion":
            prompt_column = st.selectbox(
                "Prompt column",
                pending_inspection.columns,
                key="dataset_prompt_column",
                persist_state="session",
            )
            completion_column = st.selectbox(
                "Completion column",
                pending_inspection.columns,
                key="dataset_completion_column",
                persist_state="session",
            )
        elif mapping_mode == "Preference":
            prompt_column = st.selectbox(
                "Prompt column",
                pending_inspection.columns,
                key="dataset_prompt_column",
                persist_state="session",
            )
            chosen_column = st.selectbox(
                "Chosen column",
                pending_inspection.columns,
                key="dataset_chosen_column",
                persist_state="session",
            )
            rejected_column = st.selectbox(
                "Rejected column",
                pending_inspection.columns,
                key="dataset_rejected_column",
                persist_state="session",
            )

        if st.button(action, type="primary", icon=":material/add:"):
            try:
                if mapping_mode == "Text":
                    pending_spec = replace(
                        pending_spec,
                        format="text",
                        text_column=text_column,
                        prompt_column=None,
                        completion_column=None,
                        chosen_column=None,
                        rejected_column=None,
                    )
                elif mapping_mode == "Prompt and completion":
                    if prompt_column == completion_column:
                        raise ValueError(
                            "Prompt and completion must use different columns."
                        )
                    pending_spec = replace(
                        pending_spec,
                        format="prompt_completion",
                        text_column=None,
                        prompt_column=prompt_column,
                        completion_column=completion_column,
                        chosen_column=None,
                        rejected_column=None,
                    )
                elif mapping_mode == "Preference":
                    if len({prompt_column, chosen_column, rejected_column}) != 3:
                        raise ValueError(
                            "Prompt, chosen, and rejected must use different columns."
                        )
                    pending_spec = replace(
                        pending_spec,
                        format="preference",
                        text_column=None,
                        prompt_column=prompt_column,
                        completion_column=None,
                        chosen_column=chosen_column,
                        rejected_column=rejected_column,
                    )
                save_pending_dataset(pending_spec, pending_inspection)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
