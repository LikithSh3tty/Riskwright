"""Riskwright UI.

Sections are added over the following sessions: EDA, risk prediction,
explainability, derived rules, chatbot. For now this renders the platform
status, which doubles as a smoke test that the UI container can reach the API
across the compose network.
"""

from __future__ import annotations

import streamlit as st

from ui.api_client import ApiError, base_url, health

st.set_page_config(
    page_title="Riskwright",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Riskwright")
st.caption("Credit risk scoring, explanation, and plain-English data queries")

st.subheader("Platform status")

try:
    status = health()
except ApiError as exc:
    st.error(str(exc))
    st.info(f"The UI is configured to call {base_url()}")
    st.stop()

left, middle, right = st.columns(3)

database = status.get("database", {})
with left:
    st.metric("Database", "connected" if database.get("connected") else "unavailable")
    if database.get("error"):
        st.caption(database["error"])

with middle:
    st.metric("Model", "loaded" if status.get("model_loaded") else "not trained yet")

with right:
    st.metric("LLM", status.get("llm_model") or "not configured")

st.divider()
st.write(
    "Sections for EDA, risk prediction, explainability, derived rules, and the "
    "chatbot are wired in as each module lands."
)
