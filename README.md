# Darukaa Intelligence

AI biodiversity chatbot for the Darukaa.Earth hackathon. It behaves as an environmental scientist with a **retrievable knowledge layer**, not a generic LLM chat.

The system:

- Asks for missing land conditions (multi-turn memory)
- Matches interventions in a structured knowledge base
- Pulls supporting passages from indexed reports (RAG)
- Returns evidence-backed recommendations that connect **at least three** environmental variables

## What it covers (brief requirements)

| Requirement | How it is implemented |
|---|---|
| Knowledge system | Structured JSON + Chroma vector store over corpus reports |
| Conversational intelligence | Slot filling, clarifying questions, session memory |
| Evidence-backed recommendations | Action, scientific why, impacted metrics, time horizon, confidence, sources |
| Multi-metric reasoning | Relationship graph + all collected variables used together |
| Input | Chat text **and** structured JSON |
| Output | Recommendation cards + expandable retrieval trace |

Knowledge domains: soil health, land use, biodiversity, climate, human impact.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_vector_db.py
streamlit run scripts/app.py
```

Open [http://localhost:8501](http://localhost:8501).

Copy `.env.example` to `.env` and set an API key. Supported providers: `gemini`, `groq`, `openai`, `anthropic`. Optional: `FALLBACK_LLM_PROVIDER` if the primary key is rate-limited.

If the LLM is unavailable, the app still returns **raw structured matches** from the knowledge base instead of inventing advice.

### Demo path (matches the brief)

1. Chat: `Biodiversity is declining on my land.`
2. The system asks for soil organic carbon %, rainfall pattern, and land use type.
3. Or skip extraction: **Load brief example JSON** in the intel panel (SOC 0.3%, low rainfall, monoculture, plus the other metrics).
4. Expand **How this answer was retrieved** to show structured matches, relationship chains, and RAG excerpts.

## Architecture

```
User (chat or JSON)
    → conversation.py     extract metrics, ask if incomplete
    → query_kb.py         structured intervention match
                          + relationship chains
                          + Chroma semantic search
    → reasoning.py        LLM synthesizes ONLY retrieved evidence
    → app.py              Streamlit intelligence chatbot UI
```

```
darukaa-kb/
├── knowledge_base/
│   ├── structured_knowledge.json   # metrics, relationship graph, 17 interventions
│   └── corpus/                     # FAO / IPCC / IPBES extracts + supporting notes
├── chroma_store/                   # local vector DB (from build_vector_db.py)
├── scripts/
│   ├── app.py                      # Streamlit UI
│   ├── conversation.py             # multi-turn slots + clarifying questions
│   ├── reasoning.py                # grounded recommendation schema
│   ├── query_kb.py                 # structured + RAG retrieval
│   ├── llm_client.py               # Gemini / Groq / OpenAI / Anthropic
│   ├── embedder.py                 # TF-IDF embeddings (offline)
│   ├── build_vector_db.py          # chunk + index corpus
│   ├── extract_corpus.py           # PDF → corpus text
│   └── test_conversation_flow.py   # clarifying-question sanity check
├── .streamlit/config.toml
├── .env.example
└── requirements.txt
```

### Retrieval

**Structured path** — `structured_knowledge.json` stores metric thresholds, a directed relationship graph (for example `soil_organic_carbon → microbial_diversity → pollinator_support`), and interventions with `applicable_conditions`, mechanism, impacted metrics, expected effect, time horizon, confidence, and sources.

Given `{"soil_organic_carbon": 0.3, "rainfall": 250, "land_use_type": "monoculture"}`, matching is deterministic: every listed condition must be present and satisfied.

**Vector path** — corpus files are chunked (~220 words, 40-word overlap), embedded, and stored in Chroma. Retrieval uses the site conditions and each matched intervention to pull cited passages.

`query_kb.retrieve_full_context(user_inputs)` returns both paths for the reasoning layer.

## Share / public URL

The app is a local Streamlit process. To share it from this machine:

```bash
npx --yes cloudflared tunnel --url http://127.0.0.1:8501
```

That prints a `https://….trycloudflare.com` link. It works from other devices **only while this computer and the tunnel stay running**.

To run on another machine, copy the repo (and `.env`), install dependencies, rebuild the vector store if `chroma_store/` is missing, then `streamlit run scripts/app.py`.

## Notes

- Embeddings are **TF-IDF** (fully offline). Swap `embedder.py` for sentence-transformers if you want true semantic retrieval; `fit_and_save` / `load` / `embed` stay the same.
- Rebuild the index after changing `knowledge_base/corpus/`: `python scripts/build_vector_db.py`.
- Geo / spatial context is a **bonus** in the brief and is not required for the current demo.
