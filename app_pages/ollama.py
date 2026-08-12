"""Independent local Ollama playground page."""

from urllib.error import URLError

import streamlit as st

from lora_finetune_studio.ollama import generate as ollama_generate
from lora_finetune_studio.ollama import list_models

st.caption(
    "Test models already installed in Ollama. This does not import the trained adapter."
)

try:
    ollama_models = list_models()
except OSError, URLError, TimeoutError:
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
