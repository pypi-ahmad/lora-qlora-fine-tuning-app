"""Live GPU memory page."""

import streamlit as st

from lora_finetune_studio.hardware import (
    cuda_memory_stats,
    release_unused_cuda_memory,
)
from lora_finetune_studio.jobs import active_run

profile = st.session_state.hardware_profile
st.caption(
    "Inspect CUDA memory and release unused cache owned by this Streamlit process."
)

memory_before = None
if profile.cuda_available:
    try:
        memory_before = cuda_memory_stats()
        metrics = st.columns(4)
        metrics[0].metric("Free VRAM", f"{memory_before.free_gb:.2f} GB")
        metrics[1].metric("Total VRAM", f"{memory_before.total_gb:.2f} GB")
        metrics[2].metric("App allocated", f"{memory_before.allocated_gb:.2f} GB")
        metrics[3].metric("App reserved", f"{memory_before.reserved_gb:.2f} GB")
    except RuntimeError as error:
        st.warning(f"Cannot read GPU memory: {error}")
else:
    st.warning("CUDA GPU not detected.")

running_job = active_run()
clear_memory = st.button(
    "Clear unused VRAM",
    icon=":material/memory:",
    disabled=not profile.cuda_available or running_job is not None,
)
if running_job:
    st.info("Cancel the active training job on Monitor to release its VRAM.")
elif clear_memory:
    try:
        release_unused_cuda_memory()
        memory_after = cuda_memory_stats()
        released_gb = max(
            0.0,
            memory_after.free_gb
            - (memory_before.free_gb if memory_before else memory_after.free_gb),
        )
        if released_gb >= 0.01:
            st.success(f"Released approximately {released_gb:.2f} GB of VRAM.")
        else:
            st.info("No unused PyTorch VRAM was available to release.")
    except RuntimeError as error:
        st.error(f"Could not clear GPU memory: {error}")

st.caption(
    "Live models, the training worker, Ollama, and other GPU processes keep their own VRAM."
)
