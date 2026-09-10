"""
app.py

Phase 3 - Streamlit conversational interface.

Run with:
    streamlit run scripts/app.py

Flow:
  1. User sends a free-text message ("Biodiversity is declining on my land")
  2. conversation.py extracts any values already present, checks what's missing
  3. If required slots are missing, the assistant asks a clarifying question
     (multi-turn memory -- previous answers are retained across turns)
  4. Once all required slots are filled, reasoning.py retrieves evidence and
     generates a schema-compliant, source-cited recommendation set
  5. The user can keep chatting -- e.g. "what about pollution nearby?" -- to
     add optional variables and get a refined answer
"""

import sys
import os
import json
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import conversation
import reasoning

st.set_page_config(page_title="Darukaa.Earth Biodiversity Assistant", page_icon="🌱", layout="centered")

st.title("🌱 Darukaa.Earth — Biodiversity Intelligence Assistant")
st.caption(
    "Describe your land conditions in your own words, or provide exact figures. "
    "I'll ask for anything I'm missing, then give evidence-backed recommendations."
)

# --- Session state setup -----------------------------------------------
if "conv_state" not in st.session_state:
    st.session_state.conv_state = conversation.ConversationState()
if "messages" not in st.session_state:
    st.session_state.messages = []   # for rendering the chat UI

with st.sidebar:
    st.subheader("Collected conditions")
    collected = st.session_state.conv_state.collected
    if collected:
        st.json(collected)
    else:
        st.write("_Nothing collected yet — start chatting._")

    st.subheader("Or enter structured input directly")
    with st.form("structured_input_form"):
        soc = st.number_input("Soil organic carbon (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm/year)", min_value=0, max_value=5000, value=0, step=10)
        land_use = st.selectbox(
            "Land use type",
            ["", "monoculture", "intercropping", "agroforestry", "natural_forest", "grassland", "urban", "fallow"],
        )
        submitted = st.form_submit_button("Submit structured input")
        if submitted:
            values = {}
            if soc > 0:
                values["soil_organic_carbon"] = soc
            if rainfall > 0:
                values["rainfall"] = rainfall
            if land_use:
                values["land_use_type"] = land_use
            st.session_state.conv_state.merge_extracted_values(values)
            st.session_state.messages.append({
                "role": "user",
                "content": f"[Structured input submitted: {json.dumps(values)}]",
            })
            st.rerun()

    if st.button("Reset conversation"):
        st.session_state.conv_state = conversation.ConversationState()
        st.session_state.messages = []
        st.rerun()

# --- Render existing chat history ---------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle new user input -----------------------------------------------
user_input = st.chat_input("Describe your land conditions, or ask a follow-up question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    state = st.session_state.conv_state
    state.add_user_message(user_input)

    with st.spinner("Reading your message..."):
        extracted = conversation.extract_slots_from_message(user_input)
        state.merge_extracted_values(extracted)

    missing = state.missing_required_slots()

    if missing:
        question = conversation.generate_clarifying_question(missing)
        state.add_assistant_message(question)
        st.session_state.messages.append({"role": "assistant", "content": question})
        with st.chat_message("assistant"):
            st.markdown(question)
    else:
        with st.spinner("Retrieving evidence and generating recommendations..."):
            result = reasoning.generate_recommendations(state.collected)
            answer = reasoning.format_recommendations_markdown(result)

        state.add_assistant_message(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.info(
            "You can keep chatting to refine this — e.g. mention pollution levels, "
            "soil pH, or habitat fragmentation for a more detailed answer."
        )