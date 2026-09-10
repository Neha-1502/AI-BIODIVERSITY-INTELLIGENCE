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


def load_structured_kb():
    with open(STRUCTURED_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _condition_met(user_value, condition_str):
    """Evaluates simple threshold conditions like '< 1.0' or '> 0.5' or a list membership."""
    if isinstance(condition_str, list):
        return user_value in condition_str
    condition_str = condition_str.strip()
    try:
        if condition_str.startswith("<"):
            return float(user_value) < float(condition_str[1:].strip())
        if condition_str.startswith(">"):
            return float(user_value) > float(condition_str[1:].strip())
    except (ValueError, TypeError):
        return False
    return False


def find_matching_interventions(user_inputs: dict, kb: dict):
    """
    user_inputs example:
        {"soil_organic_carbon": 0.3, "rainfall": 250, "land_use_type": "monoculture"}
    Returns interventions whose applicable_conditions are satisfied by user_inputs.
    """
    matches = []
    for intervention in kb["interventions"]:
        conditions = intervention["applicable_conditions"]
        satisfied = True
        checked_any = False
        for metric, condition in conditions.items():
            if metric in user_inputs:
                checked_any = True
                if not _condition_met(user_inputs[metric], condition):
                    satisfied = False
                    break
        if satisfied and checked_any:
            matches.append(intervention)
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


def retrieve_full_context(user_inputs: dict):
    """
    Main entry point for the reasoning layer. Given user metric inputs, returns:
      - matched structured interventions
      - relationship chains for each impacted metric
      - semantic search hits providing narrative evidence for each intervention
    """
    kb = load_structured_kb()
    interventions = find_matching_interventions(user_inputs, kb)

    enriched = []
    for intervention in interventions:
        relationship_chains = []
        seen = set()
        for metric in intervention["impacted_metrics"]:
            for edge in get_relationship_chain(metric, kb):
                key = (edge["from"], edge["to"])
                if key not in seen:
                    seen.add(key)
                    relationship_chains.append(edge)

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

    return enriched


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

    results = retrieve_full_context(example_inputs)

    if not results:
        print("No matching interventions found -- check structured_knowledge.json conditions.")

    for r in results:
        iv = r["intervention"]
        print(f"\n--- Intervention: {iv['action']} ---")
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