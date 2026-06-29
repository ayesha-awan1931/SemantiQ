"""
similarity.py
=============
Handles model loading, embedding generation, and cosine similarity calculations.
Uses sentence-transformers/all-MiniLM-L6-v2 (free HuggingFace model).
NO preprocessing, NO tokenization, NO stemming, NO lemmatization.
"""

import numpy as np
import time
import streamlit as st
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIM    = 384          # embedding dimension for MiniLM-L6-v2
MODEL_INFO   = {
    "name":        "all-MiniLM-L6-v2",
    "provider":    "sentence-transformers / HuggingFace",
    "dimension":   MODEL_DIM,
    "max_tokens":  256,
    "license":     "Apache 2.0",
    "size":        "~80 MB",
    "description": (
        "A lightweight sentence-transformer that maps text to dense "
        "384-dimensional vectors. Trained with contrastive learning on "
        "large text corpora; excellent for semantic similarity tasks."
    ),
}


# ──────────────────────────────────────────────────────────────
#  Model Loading (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    Returns the loaded model instance.
    """
    return SentenceTransformer(MODEL_NAME)


# ──────────────────────────────────────────────────────────────
#  Embedding Generation
# ──────────────────────────────────────────────────────────────
def get_embeddings(texts: list[str]) -> tuple[np.ndarray, float]:
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        Raw input texts (words, sentences, or paragraphs).
        No preprocessing is applied.

    Returns
    -------
    embeddings : np.ndarray  shape (n, 384)
    elapsed_ms : float       inference time in milliseconds
    """
    model = load_model()
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise for cosine via dot product
        show_progress_bar=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return embeddings, round(elapsed_ms, 2)


# ──────────────────────────────────────────────────────────────
#  Cosine Similarity (single pair)
# ──────────────────────────────────────────────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised vectors.
    Since embeddings are pre-normalised, this equals the dot product.

    Returns a float in [0.0, 1.0] (clamped).
    """
    score = float(np.dot(vec_a, vec_b))
    return round(max(0.0, min(1.0, score)), 6)


# ──────────────────────────────────────────────────────────────
#  Pairwise Similarity Matrix
# ──────────────────────────────────────────────────────────────
def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute an N×N cosine similarity matrix for all embedding pairs.

    Parameters
    ----------
    embeddings : np.ndarray  shape (N, D), L2-normalised

    Returns
    -------
    matrix : np.ndarray  shape (N, N), values in [0, 1]
    """
    matrix = np.dot(embeddings, embeddings.T)
    matrix = np.clip(matrix, 0.0, 1.0)
    return np.round(matrix, 6)


# ──────────────────────────────────────────────────────────────
#  Top-K Similar Results
# ──────────────────────────────────────────────────────────────
def get_top_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_texts: list[str],
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[dict]:
    """
    Retrieve top-K most similar texts to a query embedding.

    Parameters
    ----------
    query_embedding  : np.ndarray  shape (D,)
    corpus_embeddings: np.ndarray  shape (N, D)
    corpus_texts     : list[str]   corresponding text labels
    top_k            : int
    exclude_self     : bool        skip score ≥ 0.9999 (identical)

    Returns
    -------
    list of dicts: [{"text": str, "score": float, "rank": int}, ...]
    """
    scores = np.dot(corpus_embeddings, query_embedding)

    results = []
    for idx, score in enumerate(scores):
        if exclude_self and score >= 0.9999:
            continue
        results.append({
            "text":  corpus_texts[idx],
            "score": round(float(score), 6),
            "rank":  0,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    for rank, item in enumerate(results[:top_k], start=1):
        item["rank"] = rank

    return results[:top_k]


# ──────────────────────────────────────────────────────────────
#  Similarity Label Utility
# ──────────────────────────────────────────────────────────────
def similarity_label(score: float) -> tuple[str, str]:
    """
    Map a similarity score to a descriptive label and colour hex.

    Returns (label, hex_colour)
    """
    if score >= 0.90:
        return ("Very High",   "#43E97B")
    elif score >= 0.75:
        return ("High",        "#38F9D7")
    elif score >= 0.55:
        return ("Moderate",    "#6C63FF")
    elif score >= 0.35:
        return ("Low",         "#FF9A3C")
    else:
        return ("Very Low",    "#FF6584")
