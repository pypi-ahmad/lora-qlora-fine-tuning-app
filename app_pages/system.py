"""Read-only system readiness page."""

import streamlit as st

from lora_finetune_studio.hardware import (
    MIN_QLORA_FREE_VRAM_GB,
    REQUIRED_CUDA_VERSION,
    SUPPORTED_OPERATING_SYSTEMS,
    SUPPORTED_PYTHON,
    cuda_memory_stats,
    scan_system,
)
from lora_finetune_studio.sources import get_hf_token, token_identity
from lora_finetune_studio.unsloth_runtime import inspect_unsloth_runtime

profile = st.session_state.hardware_profile
scan = scan_system()
token = get_hf_token()
unsloth_runtime = inspect_unsloth_runtime()

st.caption(
    "Read-only readiness scan. It never installs drivers, changes runtimes, or displays secret values."
)

runtime_issues: list[str] = []
if scan.os_name not in SUPPORTED_OPERATING_SYSTEMS:
    runtime_issues.append("Native Windows or Linux is required.")
if tuple(map(int, scan.python_version.split(".")[:2])) != SUPPORTED_PYTHON:
    runtime_issues.append(
        f"Python {SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]} is required."
    )
if scan.cuda_version != REQUIRED_CUDA_VERSION:
    runtime_issues.append(f"PyTorch CUDA {REQUIRED_CUDA_VERSION} is required.")
if not scan.uv_venv_active:
    runtime_issues.append("Run the application from the uv-managed project .venv.")
if runtime_issues:
    for issue in runtime_issues:
        st.warning(issue)
else:
    st.success(
        f"Runtime ready: {scan.native_runtime} · Python 3.14 · CUDA 13.0 · uv .venv"
    )

free_vram_gb = None
if profile.cuda_available:
    try:
        free_vram_gb = cuda_memory_stats().free_gb
    except RuntimeError:
        pass
if free_vram_gb is None:
    st.error("QLoRA training is unavailable because a CUDA GPU was not detected.")
elif free_vram_gb < MIN_QLORA_FREE_VRAM_GB:
    st.warning(
        "QLoRA training needs attention. "
        f"At least {MIN_QLORA_FREE_VRAM_GB:g} GB of currently free VRAM is required "
        "for the smallest supported QLoRA jobs."
    )
else:
    st.success(f"QLoRA readiness: {free_vram_gb:.2f} GB free VRAM is available.")

st.subheader("Operating system")
with st.container(horizontal=True):
    st.metric("Operating system", scan.os_name, border=True)
    st.metric("Release", scan.os_release, border=True)
    st.metric("CPU threads", scan.cpu_threads, border=True)
    st.metric("Available RAM", f"{scan.available_ram_gb:.2f} GB", border=True)
    st.metric("Free disk", f"{scan.free_disk_gb:.1f} GB", border=True)
st.caption(f"Build: {scan.os_version}")

st.subheader("Accelerators")
with st.container(border=True):
    if profile.cuda_available:
        with st.container(horizontal=True):
            st.metric("GPU", profile.gpu_name or "Detected")
            st.metric("CUDA runtime", scan.cuda_version or "Unavailable")
            st.metric("Total VRAM", f"{profile.vram_gb:g} GB")
            st.metric("Free VRAM", f"{free_vram_gb:.2f} GB")
            st.metric("BF16", "Supported" if profile.bf16_supported else "Unavailable")
        st.caption(
            f"Recommended: {profile.recommended_mode}, models up to approximately "
            f"{profile.recommended_max_billions:g}B parameters."
        )
    else:
        st.error(profile.warning or "CUDA GPU not detected.")

st.subheader("Training runtime")
st.caption(
    "This project supports native Windows and Linux with CUDA 13.0 and a uv-managed .venv."
)
with st.container(border=True):
    st.markdown("**Native runtime**")
    st.badge(scan.native_runtime, icon=":material/check:", color="green")
    if scan.uv_venv_active:
        st.badge("uv .venv active", icon=":material/check:", color="green")
    else:
        st.badge("uv .venv inactive", icon=":material/warning:", color="orange")
    st.caption("The platform launcher creates and uses .venv automatically.")

st.subheader("Software and integrations")
integration_rows = [
    {
        "Software": item.name,
        "Status": "Available" if item.available else "Missing",
        "Version or detail": item.detail,
    }
    for item in scan.software
]
integration_rows.append(
    {
        "Software": "Unsloth Core",
        "Status": "Available" if unsloth_runtime.available else "Missing",
        "Version or detail": unsloth_runtime.version or unsloth_runtime.detail,
    }
)
integration_rows.append(
    {
        "Software": "Hugging Face token",
        "Status": "Configured" if token else "Not configured",
        "Version or detail": "Value hidden",
    }
)
st.dataframe(integration_rows, hide_index=True, width="stretch")

with st.container(border=True):
    st.subheader("Hugging Face access")
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
