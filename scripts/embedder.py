"""
embedder.py

Pluggable embedding backend for the RAG pipeline.

Default: TF-IDF (scikit-learn) -- works fully offline, no model download needed.
Good enough to demonstrate a real retrieval pipeline for the hackathon.

Upgrade path (recommended once you have internet access, e.g. at the venue):
  - sentence-transformers ("all-MiniLM-L6-v2") for real semantic embeddings, or
  - OpenAI / Anthropic-compatible embedding API for production quality

To upgrade: implement the same fit_and_save() / load() / embed() interface
below with your model of choice. Nothing else in build_vector_db.py or
query_kb.py needs to change.
"""

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_store", "tfidf_vectorizer.pkl")


def fit_and_save(corpus_texts):
    """Fits a TF-IDF vectorizer on the full corpus and persists it to disk.
    Call this once during indexing (build_vector_db.py)."""
    vectorizer = TfidfVectorizer(
        max_features=2048,
        stop_words="english",
        ngram_range=(1, 2),
    )
    vectorizer.fit(corpus_texts)

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer


def load():
    """Loads the previously fitted vectorizer. Raises if build_vector_db.py hasn't run yet."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "No fitted vectorizer found. Run `python scripts/build_vector_db.py` first."
        )
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def embed(vectorizer, texts):
    """Transforms a list of texts into dense embedding vectors (list of list[float])."""
    matrix = vectorizer.transform(texts)
    return matrix.toarray().tolist()
