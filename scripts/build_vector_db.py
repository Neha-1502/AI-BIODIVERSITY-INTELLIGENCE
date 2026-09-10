"""
build_vector_db.py

Phase 2 - Knowledge System pipeline (step 1 of 2).

What this does:
  1. Reads every .txt file in knowledge_base/corpus/
  2. Splits each document into overlapping chunks (~350 tokens, ~50 token overlap)
  3. Embeds each chunk with a sentence-transformers model
  4. Stores chunks + embeddings + metadata (source file, topic tags) in a local
     persistent Chroma vector database

Run this once (and again whenever you add new documents to knowledge_base/corpus/):
    python scripts/build_vector_db.py

This gives you a real, inspectable retrieval pipeline -- exactly what the hackathon
brief asks you to demonstrate ("Clearly show how knowledge is retrieved and used").
"""

import os
import re
import glob
import chromadb
import embedder

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "corpus")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "biodiversity_knowledge"

CHUNK_SIZE_WORDS = 220      # approx ~300-350 tokens
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    """Simple word-based sliding window chunker with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def extract_metadata(raw_text: str):
    """Pulls SOURCE and TOPIC lines from the top of each corpus file, if present."""
    source_match = re.search(r"SOURCE:\s*(.+)", raw_text)
    topic_match = re.search(r"TOPIC:\s*(.+)", raw_text)
    source = source_match.group(1).strip() if source_match else "unknown"
    topics = topic_match.group(1).strip() if topic_match else ""
    return source, topics


def build():
    os.makedirs(DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    # Reset collection each run so re-indexing is idempotent during development
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    files = glob.glob(os.path.join(CORPUS_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {CORPUS_DIR}. Add source documents first.")
        return

    all_ids, all_docs, all_metadatas = [], [], []

    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        source, topics = extract_metadata(raw_text)
        # strip header lines before chunking so they don't pollute embeddings
        body = re.sub(r"SOURCE:.*\n|TOPIC:.*\n", "", raw_text).strip()

        chunks = chunk_text(body)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk_{i}"
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metadatas.append({
                "filename": filename,
                "source": source,
                "topics": topics,
            })

    # Fit the embedder on the full chunk corpus, then embed every chunk
    vectorizer = embedder.fit_and_save(all_docs)
    all_embeddings = embedder.embed(vectorizer, all_docs)

    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metadatas,
        embeddings=all_embeddings,
    )

    print(f"Indexed {len(all_ids)} chunks from {len(files)} document(s) into '{COLLECTION_NAME}'.")
    print(f"Vector store persisted at: {DB_DIR}")


if __name__ == "__main__":
    build()
