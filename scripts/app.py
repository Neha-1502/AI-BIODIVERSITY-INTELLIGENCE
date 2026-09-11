"""
app.py

Darukaa.Earth — Biodiversity Intelligence chatbot.

Run with:
    streamlit run scripts/app.py
"""

import html
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

import conversation
import llm_client
import query_kb
import reasoning

WELCOME = """I am **Darukaa Intelligence** — an environmental scientist with a retrievable knowledge layer.

Tell me what is happening on your land. I will:
1. Ask for missing conditions (soil, climate, land use)
2. Retrieve matching interventions from a structured knowledge base and indexed reports
3. Return evidence-backed recommendations that connect **at least three** environmental variables

Start with: *Biodiversity is declining on my land.*"""

JSON_EXAMPLE = {
    "soil_organic_carbon": 0.3,
    "rainfall": 250,
    "land_use_type": "monoculture",
    "soil_ph": 5.2,
    "soil_moisture": 8,
    "temperature": 32,
    "habitat_fragmentation_index": 0.6,
    "habitat_diversity": 0.25,
    "species_richness": 30,
    "pollution_index": 45,
    "deforestation_rate": 2.5,
    "region": "semi-arid",
}

COVERAGE = {
    "Soil health": ["soil_organic_carbon", "soil_ph", "soil_moisture"],
    "Land use": ["land_use_type"],
    "Biodiversity": ["species_richness", "habitat_diversity", "habitat_fragmentation_index"],
    "Climate": ["rainfall", "temperature"],
    "Human impact": ["pollution_index", "deforestation_rate"],
}

STARTERS = [
    "Biodiversity is declining on my land.",
    "SOC is 0.3%, rainfall is low, and the crop is monoculture wheat in a semi-arid region.",
    "Soil is acidic and fragmented; species richness looks low.",
]

CSS = """
<style>
    @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap");

    html, body, .stApp, [data-testid="stMarkdownContainer"], [data-testid="stChatMessage"] {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    /* Streamlit toolbar/sidebar controls are Material ligatures; do not restyle their font. */
    span[data-testid="stIconMaterial"],
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: 400 !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap;
        -webkit-font-feature-settings: "liga";
        font-feature-settings: "liga";
    }
    .stApp {
        background:
            radial-gradient(1000px 480px at 8% -8%, rgba(61, 220, 151, 0.16), transparent 55%),
            radial-gradient(800px 400px at 100% 0%, rgba(47, 128, 237, 0.08), transparent 50%),
            #0B1210;
    }
    .intel-shell {
        border: 1px solid rgba(61, 220, 151, 0.22);
        background: linear-gradient(180deg, rgba(16, 28, 24, 0.92), rgba(11, 18, 16, 0.75));
        border-radius: 18px;
        padding: 1.05rem 1.2rem 0.85rem 1.2rem;
        margin-bottom: 0.9rem;
    }
    .intel-kicker {
        font-family: "IBM Plex Mono", monospace;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.68rem;
        color: #7EE0B0;
        margin: 0 0 0.35rem 0;
    }
    .intel-title {
        font-size: 1.7rem;
        font-weight: 650;
        color: #F4FBF7;
        margin: 0 0 0.3rem 0;
        line-height: 1.2;
    }
    .intel-sub { color: #9BB5A8; font-size: 0.93rem; margin: 0 0 0.75rem 0; }
    .intel-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .intel-pill {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.7rem;
        color: #D7F6E6;
        border: 1px solid rgba(61, 220, 151, 0.28);
        background: rgba(61, 220, 151, 0.08);
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(61, 220, 151, 0.12);
        border-radius: 16px;
        background: rgba(21, 33, 28, 0.72);
        padding: 0.15rem 0.2rem;
        overflow: visible;
    }
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarUser"] { display: none !important; }
    .stChatMessage p { line-height: 1.55; }
    section[data-testid="stSidebar"] {
        background: #101A16;
        border-right: 1px solid rgba(61, 220, 151, 0.15);
    }
    .rec-card {
        border: 1px solid rgba(61, 220, 151, 0.2);
        background: rgba(12, 22, 18, 0.9);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin: 0.55rem 0;
    }
    .rec-card h4 {
        margin: 0 0 0.45rem 0;
        color: #F4FBF7;
        font-size: 1.02rem;
    }
    .rec-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.55rem 0 0.35rem 0;
    }
    .rec-chip {
        font-family: "IBM Plex Mono", monospace;
        font-size: 0.68rem;
        color: #B7E6CF;
        background: rgba(61, 220, 151, 0.1);
        border-radius: 6px;
        padding: 0.16rem 0.42rem;
    }
    .rec-label { color: #7EE0B0; font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .rec-body { color: #D5E6DD; font-size: 0.92rem; margin: 0.2rem 0 0.45rem 0; }
    .starter-note { color: #9BB5A8; font-size: 0.82rem; margin: 0.2rem 0 0.4rem 0; }
</style>
"""


def _init_state():
    if "conv_state" not in st.session_state:
        st.session_state.conv_state = conversation.ConversationState()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME, "kind": "welcome"}
        ]
    if "pending_turn" not in st.session_state:
        st.session_state.pending_turn = None
    if "site_profile" not in st.session_state:
        st.session_state.site_profile = {}


def _coverage_status(collected: dict) -> str:
    bits = []
    for label, keys in COVERAGE.items():
        have = sum(1 for k in keys if k in collected)
        mark = "●" if have else "○"
        bits.append(f"{mark} {label} ({have}/{len(keys)})")
    return "  \n".join(bits)


def _parse_structured_json(raw: str) -> dict:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object of metric: value pairs.")
    cleaned = {}
    for key, value in data.items():
        if key in conversation.ALL_SLOTS and value is not None and value != "":
            cleaned[key] = value
    if not cleaned:
        raise ValueError("No recognized metric keys in the JSON object.")
    return cleaned


def _render_retrieval(trace):
    if not trace:
        return
    interventions = trace.get("interventions") if isinstance(trace, dict) else trace
    chains = trace.get("cross_metric_chains") if isinstance(trace, dict) else []
    condition_evidence = trace.get("condition_evidence") if isinstance(trace, dict) else []
    variables = trace.get("variables") if isinstance(trace, dict) else []

    with st.expander("How this answer was retrieved", expanded=True):
        if variables:
            st.caption("Variables used together")
            st.write(", ".join(variables))
        if chains:
            st.caption("Multi-metric relationship chains")
            for chain in chains[:8]:
                st.markdown(f"- {chain}")
        if condition_evidence:
            st.caption("Vector search on site conditions")
            for ev in condition_evidence:
                rel = ev.get("relevance")
                rel_txt = f", relevance {rel}" if rel is not None else ""
                st.markdown(f"- [{html.escape(str(ev.get('filename')))}{rel_txt}] {html.escape(str(ev.get('excerpt')))}…")
        if not interventions:
            st.write("No structured intervention matched these conditions.")
            return
        st.caption("Structured intervention matches")
        for item in interventions:
            st.markdown(f"**{item.get('id', '')} — {item.get('action', '')}**")
            if item.get("triggered_by"):
                st.markdown("Triggered by: " + "; ".join(item["triggered_by"]))
            if item.get("relationship_chains"):
                for chain in item["relationship_chains"]:
                    st.markdown(f"- {chain}")
            for ev in item.get("evidence") or []:
                rel = ev.get("relevance")
                rel_txt = f", relevance {rel}" if rel is not None else ""
                st.markdown(f"- RAG [{ev.get('filename')}{rel_txt}]: {ev.get('excerpt')}…")


def _rec_card(rec: dict, index: int) -> str:
    action = html.escape(str(rec.get("action", "")))
    why = html.escape(str(rec.get("reasoning", "")))
    expected = html.escape(str(rec.get("expected_improvement", "n/a")))
    horizon = html.escape(str(rec.get("time_horizon", "")).replace("_", " "))
    confidence = html.escape(str(rec.get("confidence", "n/a")))
    metrics = rec.get("impacted_metrics") or []
    sources = rec.get("sources") or []
    chips = "".join(f'<span class="rec-chip">{html.escape(str(m))}</span>' for m in metrics)
    source_txt = html.escape(", ".join(str(s) for s in sources))
    return f"""
    <div class="rec-card">
      <div class="rec-label">Recommendation {index}</div>
      <h4>{action}</h4>
      <div class="rec-label">Why it works</div>
      <p class="rec-body">{why}</p>
      <div class="rec-label">Expected improvement</div>
      <p class="rec-body">{expected}</p>
      <div class="rec-meta">
        <span class="rec-chip">horizon · {horizon}</span>
        <span class="rec-chip">confidence · {confidence}</span>
        {chips}
      </div>
      <div class="rec-label">Evidence</div>
      <p class="rec-body">{source_txt or "No source attached"}</p>
    </div>
    """


def _generate_answer(state: conversation.ConversationState, user_focus: str | None):
    result = reasoning.generate_recommendations(
        state.collected,
        conversation_history=state.history,
        user_focus=user_focus,
    )
    recs = result.get("recommendations") or []
    used = result.get("_variables_used") or list(state.collected.keys())
    parts = []
    if result.get("summary"):
        parts.append(result["summary"])
    if len(used) >= 3:
        parts.append(
            f"_Multi-metric reasoning used {len(used)} variables together: "
            + ", ".join(used)
            + "._"
        )
    else:
        parts.append(
            "_Add pollution, pH, fragmentation, or habitat diversity "
            "so I can connect more than the minimum three variables._"
        )
    return "\n\n".join(parts), recs, result.get("_retrieval"), used


def process_user_text(user_text: str, extracted_override: dict | None = None):
    state = st.session_state.conv_state
    state.add_user_message(user_text)
    state.last_focus = user_text

    try:
        extracted = extracted_override if extracted_override is not None else (
            conversation.extract_slots_from_message(user_text, history=state.history)
        )
        state.merge_extracted_values(extracted)
        st.session_state.site_profile = dict(state.collected)
    except llm_client.LLMUnavailableError:
        fail = (
            "The language model is temporarily unavailable. "
            "Paste JSON in the intel panel to continue without extraction."
        )
        state.add_assistant_message(fail)
        st.session_state.messages.append({"role": "assistant", "content": fail, "kind": "error"})
        return

    missing = state.missing_required_slots()
    if missing:
        question = conversation.generate_clarifying_question(missing, state.collected)
        state.add_assistant_message(question)
        st.session_state.messages.append({"role": "assistant", "content": question, "kind": "clarify"})
        return

    implied = conversation.implied_optional_slots(
        user_text, state.collected, state.asked_optional
    )
    if implied and extracted_override is None:
        for slot in implied:
            state.asked_optional.add(slot)
        question = conversation.generate_clarifying_question(implied, state.collected)
        state.add_assistant_message(question)
        st.session_state.messages.append({"role": "assistant", "content": question, "kind": "clarify"})
        return

    try:
        answer, recs, retrieval, used = _generate_answer(state, user_focus=user_text)
    except Exception as e:
        fail = "Recommendation generation failed. Send the message again, or submit JSON from the intel panel."
        print(f"[error] generate_recommendations failed: {e}")
        st.session_state.messages.append({"role": "assistant", "content": fail, "kind": "error"})
        return

    state.add_assistant_message(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "kind": "recommendation",
        "recommendations": recs,
        "retrieval": retrieval,
        "variables": used,
    })


def _queue_user_text(text: str, extracted: dict | None = None):
    st.session_state.pending_turn = {"text": text, "extracted": extracted}


st.set_page_config(
    page_title="Darukaa Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _warm_knowledge_layer():
    query_kb.ensure_index()
    return True


with st.spinner("Loading knowledge layer…"):
    _warm_knowledge_layer()

_init_state()
st.markdown(CSS, unsafe_allow_html=True)

pending = st.session_state.pending_turn
if pending:
    st.session_state.pending_turn = None
    st.session_state.messages.append({"role": "user", "content": pending["text"]})
    process_user_text(pending["text"], extracted_override=pending.get("extracted"))

st.markdown(
    """
    <div class="intel-shell">
      <p class="intel-kicker">Darukaa.Earth · live knowledge layer</p>
      <p class="intel-title">Biodiversity Intelligence</p>
      <p class="intel-sub">Conversational scientist. Structured retrieval. Evidence-backed recommendations — not generic advice.</p>
      <div class="intel-pills">
        <span class="intel-pill">text + JSON input</span>
        <span class="intel-pill">multi-turn memory</span>
        <span class="intel-pill">RAG + structured KB</span>
        <span class="intel-pill">≥3 variables together</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Intel panel**")
    collected = st.session_state.get("site_profile") or st.session_state.conv_state.collected
    st.caption("Site profile in this conversation")
    if collected:
        st.json(collected)
    else:
        st.caption("Nothing collected yet.")

    st.markdown("**Knowledge coverage**")
    st.markdown(_coverage_status(collected))

    st.divider()
    st.markdown("**Structured input (JSON)**")
    st.caption("Mandatory input path besides chat. Keys must match known metrics.")
    json_raw = st.text_area(
        "Paste JSON",
        value="",
        height=170,
        placeholder=json.dumps(JSON_EXAMPLE, indent=2),
        label_visibility="collapsed",
    )
    if st.button("Ingest JSON", use_container_width=True):
        try:
            values = _parse_structured_json(json_raw)
        except Exception as e:
            st.error(str(e))
        else:
            _queue_user_text(f"[Structured JSON] {json.dumps(values)}", extracted=values)
            st.rerun()

    if st.button("Load brief example JSON", use_container_width=True):
        _queue_user_text(
            f"[Structured JSON] {json.dumps(JSON_EXAMPLE)}",
            extracted=_parse_structured_json(json.dumps(JSON_EXAMPLE)),
        )
        st.rerun()

    if st.button("Reset session", use_container_width=True):
        st.session_state.conv_state = conversation.ConversationState()
        st.session_state.messages = [{"role": "assistant", "content": WELCOME, "kind": "welcome"}]
        st.session_state.pending_turn = None
        st.session_state.site_profile = {}
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("kind") == "recommendation" and msg.get("recommendations"):
            cards = "".join(
                _rec_card(rec, i) for i, rec in enumerate(msg["recommendations"], 1)
            )
            st.markdown(cards, unsafe_allow_html=True)
        if msg.get("retrieval") is not None or msg.get("variables"):
            _render_retrieval(msg.get("retrieval"))

if len(st.session_state.messages) <= 1:
    st.markdown('<p class="starter-note">Try a starting query</p>', unsafe_allow_html=True)
    cols = st.columns(len(STARTERS))
    for col, starter in zip(cols, STARTERS):
        if col.button(starter, use_container_width=True):
            _queue_user_text(starter)
            st.rerun()

user_input = st.chat_input("Ask as you would an environmental scientist…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.status("Intelligence pipeline", expanded=True) as status:
        st.write("Extracting environmental conditions from conversation")
        st.write("Retrieving structured interventions and report evidence")
        st.write("Reasoning across at least three variables")
        process_user_text(user_input)
        status.update(label="Grounded response ready", state="complete")
    st.rerun()
