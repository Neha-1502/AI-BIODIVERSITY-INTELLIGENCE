"""
query_kb.py

Phase 2 - Knowledge System pipeline (step 2 of 2).

Combines two retrieval paths, which together satisfy the "retrievable knowledge
layer, not just prompts" requirement:

  A) STRUCTURED lookup: given user metric values, find matching intervention
     records + relationship-graph edges from knowledge_base/structured_knowledge.json
     (deterministic, no embeddings -- this is your "structured dataset" evidence)

  B) VECTOR retrieval: semantic search over the embedded corpus chunks in Chroma
     to pull supporting narrative evidence/citations for those interventions
     (this is your "RAG" evidence)

This script is meant to be imported by your reasoning/chat layer (Phase 3), but
can also be run directly to sanity-check retrieval quality:

    python scripts/query_kb.py
"""

import os
import json
import chromadb
import embedder

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
STRUCTURED_KB_PATH = os.path.join(BASE_DIR, "knowledge_base", "structured_knowledge.json")
DB_DIR = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "biodiversity_knowledge"
_INDEX_READY = False


def ensure_index():
    """Build the Chroma index on first use if it is not already on disk (Streamlit Cloud)."""
    global _INDEX_READY
    if _INDEX_READY:
        return
    vectorizer_path = os.path.join(DB_DIR, "tfidf_vectorizer.pkl")
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        client.get_collection(name=COLLECTION_NAME)
        if os.path.exists(vectorizer_path):
            _INDEX_READY = True
            return
    except Exception:
        pass
    import build_vector_db

    build_vector_db.build()
    _INDEX_READY = True


def load_structured_kb():
    with open(STRUCTURED_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _condition_met(user_value, condition_str):
    """Evaluates threshold conditions, list membership, or exact equality."""
    if isinstance(condition_str, list):
        return user_value in condition_str
    condition_str = str(condition_str).strip()
    try:
        if condition_str.startswith("<="):
            return float(user_value) <= float(condition_str[2:].strip())
        if condition_str.startswith(">="):
            return float(user_value) >= float(condition_str[2:].strip())
        if condition_str.startswith("<"):
            return float(user_value) < float(condition_str[1:].strip())
        if condition_str.startswith(">"):
            return float(user_value) > float(condition_str[1:].strip())
    except (ValueError, TypeError):
        return False
    return str(user_value).strip().lower() == condition_str.lower()


def find_matching_interventions(user_inputs: dict, kb: dict):
    """
    user_inputs example:
        {"soil_organic_carbon": 0.3, "rainfall": 250, "land_use_type": "monoculture"}
    Returns interventions whose applicable_conditions are ALL present in
    user_inputs and satisfied. A missing metric does not match -- otherwise
    a rainfall-gated intervention would fire from land use alone.

    Each returned intervention is a shallow copy of the KB record with an
    added "_matched_on" dict recording exactly which user-supplied metric(s)
    satisfied which condition(s) -- e.g. {"habitat_fragmentation_index":
    {"value": 0.6, "condition": "> 0.5"}}. Without this, an intervention that
    only needs an *optional* metric (like habitat_fragmentation_index) can
    fire silently once that value has been collected in an earlier
    conversation turn, with nothing in the output tying it back to that
    specific number -- this makes the trigger explicit and auditable instead
    of a black box.
    """
    matches = []
    for intervention in kb["interventions"]:
        conditions = intervention["applicable_conditions"]
        satisfied = True
        matched_on = {}
        for metric, condition in conditions.items():
            if metric not in user_inputs:
                satisfied = False
                break
            if _condition_met(user_inputs[metric], condition):
                matched_on[metric] = {"value": user_inputs[metric], "condition": condition}
            else:
                satisfied = False
                break
        if satisfied and matched_on:
            enriched = dict(intervention)
            enriched["_matched_on"] = matched_on
            matches.append(enriched)
    return matches


def get_relationship_chain(metric_name: str, kb: dict):
    """Returns relationship_graph edges starting from a given metric -- used to build
    the multi-metric reasoning chain (e.g. soil_organic_carbon -> microbial_diversity
    -> pollinator_support)."""
    chain = []
    seen_edges = set()
    frontier = [metric_name]
    visited = set()
    for _ in range(5):  # cap traversal depth
        next_frontier = []
        for m in frontier:
            if m in visited:
                continue
            visited.add(m)
            for edge in kb["relationship_graph"]:
                if edge["from"] == m:
                    edge_key = (edge["from"], edge["to"])
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        chain.append(edge)
                    next_frontier.append(edge["to"])
        frontier = next_frontier
        if not frontier:
            break
    return chain


def semantic_search(query: str, n_results: int = 3):
    """Queries the Chroma vector store for supporting evidence chunks."""
    ensure_index()
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    vectorizer = embedder.load()
    query_embedding = embedder.embed(vectorizer, [query])

    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "filename": meta.get("filename", "unknown"),
            "relevance_score": round(1 - dist, 3),
        })
    return hits


def _unique_chains_for_metrics(metric_names: list, kb: dict) -> list:
    chains = []
    seen = set()
    for metric in metric_names:
        for edge in get_relationship_chain(metric, kb):
            key = (edge["from"], edge["to"])
            if key not in seen:
                seen.add(key)
                chains.append(edge)
    return chains


def retrieve_full_context(user_inputs: dict):
    """
    Main entry point for the reasoning layer. Given user metric inputs, returns:
      - matched structured interventions
      - relationship chains spanning the user's variables (multi-metric)
      - semantic search hits for each intervention and for the site conditions
    """
    kb = load_structured_kb()
    interventions = find_matching_interventions(user_inputs, kb)
    cross_metric_chains = _unique_chains_for_metrics(list(user_inputs.keys()), kb)

    condition_query = " ".join(f"{k} {v}" for k, v in user_inputs.items())
    try:
        condition_evidence = semantic_search(
            condition_query or "soil organic carbon rainfall land use biodiversity",
            n_results=3,
        )
    except Exception as e:
        condition_evidence = []
        print(f"[warning] semantic search unavailable ({e}). Run build_vector_db.py first.")

    enriched = []
    for intervention in interventions:
        relationship_chains = _unique_chains_for_metrics(
            intervention["impacted_metrics"], kb
        )
        evidence_query = intervention["action"] + " " + " ".join(intervention["impacted_metrics"])
        try:
            evidence_chunks = semantic_search(evidence_query, n_results=2)
        except Exception as e:
            evidence_chunks = []
            print(f"[warning] semantic search unavailable ({e}). Run build_vector_db.py first.")

        enriched.append({
            "intervention": intervention,
            "relationship_chains": relationship_chains,
            "evidence_chunks": evidence_chunks,
        })

    return {
        "matched_interventions": enriched,
        "cross_metric_chains": cross_metric_chains,
        "condition_evidence": condition_evidence,
        "variables": list(user_inputs.keys()),
    }


if __name__ == "__main__":
    # Sanity check using the hackathon brief's example use case
    example_inputs = {
        "soil_organic_carbon": 0.3,
        "rainfall": 250,
        "land_use_type": "monoculture",
    }

    print("=" * 70)
    print("TEST QUERY: SOC=0.3%, rainfall=250mm (semi-arid), land_use=monoculture")
    print("=" * 70)

    payload = retrieve_full_context(example_inputs)
    results = payload["matched_interventions"]

    print("Variables used together:", ", ".join(payload["variables"]))
    if payload["cross_metric_chains"]:
        print("Cross-metric chains:")
        for edge in payload["cross_metric_chains"][:8]:
            print(f"  {edge['from']} -> {edge['to']} ({edge['relationship']})")

    if not results:
        print("No matching interventions found -- check structured_knowledge.json conditions.")

    for r in results:
        iv = r["intervention"]
        print(f"\n--- Intervention: {iv['action']} ---")
        if iv.get("_matched_on"):
            trigger = ", ".join(f"{m}={d['value']} ({d['condition']})" for m, d in iv["_matched_on"].items())
            print(f"Triggered by: {trigger}")
        print(f"Mechanism: {iv['mechanism']}")
        print(f"Impacted metrics: {', '.join(iv['impacted_metrics'])}")
        print(f"Expected effect: {iv['expected_effect']}")
        print(f"Time horizon: {iv['time_horizon']} | Confidence: {iv['confidence']}")
        print(f"Structured sources: {', '.join(iv['sources'])}")

        if r["relationship_chains"]:
            print("Relationship chain:")
            for edge in r["relationship_chains"]:
                print(f"  {edge['from']} -> {edge['to']} ({edge['relationship']}): {edge['note']}")

        if r["evidence_chunks"]:
            print("Supporting evidence (retrieved via vector search):")
            for chunk in r["evidence_chunks"]:
                print(f"  [{chunk['filename']}, relevance={chunk['relevance_score']}] {chunk['text'][:150]}...")

    print("\n" + "=" * 70)
    print("TEST QUERY 2: habitat fragmentation / species richness")
    print("=" * 70)
    hits = semantic_search("habitat fragmentation species richness biodiversity loss", n_results=3)
    for h in hits:
        print(f"[{h['filename']}, relevance={h['relevance_score']}] {h['text'][:150]}...")