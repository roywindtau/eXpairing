"""
user_vector.py
--------------
Builds a TF-IDF pantry vector for a given user's current pantry items.
This vector is used by serve_cb.py to compute cosine similarity between
the user's pantry profile and every recipe's ingredient embedding.

The vectorizer must be the SAME one used in train_cb.py — same vocabulary,
same IDF weights. We load it from the saved cb_vectorizer.pkl artifact.

Example
-------
    from backend.ml.user_vector import build_pantry_vector
    from backend.ml.serve_cb   import _vectorizer   # already loaded singleton

    pantry = ["eggs", "whole milk", "butter", "cheddar cheese"]
    vec    = build_pantry_vector(pantry, _vectorizer)
    # vec is a (1, vocab_size) sparse matrix ready for cosine_similarity()

Why a separate module?
----------------------
serve_cb.py calls sklearn's vectorizer directly inline for simplicity.
This module exists as a standalone utility for:
  - Debugging: inspect what the pantry vector looks like
  - Testing: unit-test pantry vectorization in isolation
  - Future: building user embedding profiles for a vector DB lookup
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import scipy.sparse as sp

CB_VECTORIZER_PATH = Path("models/cb_vectorizer.pkl")


def load_vectorizer():
    """Load the saved TF-IDF vectorizer. Returns None if not trained yet."""
    if not CB_VECTORIZER_PATH.exists():
        return None
    with open(CB_VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def build_pantry_vector(
    pantry_ingredients: list[str],
    vectorizer=None,
) -> Optional[sp.csr_matrix]:
    """
    Transform a list of pantry ingredient names into a TF-IDF vector.

    The pantry is treated as a single "document" — all ingredient names
    joined by spaces. The vectorizer maps this to the same feature space
    as the recipe embeddings in cb_matrix.npz.

    Args:
        pantry_ingredients: canonical ingredient names, e.g. ["eggs", "milk"]
        vectorizer:         fitted TfidfVectorizer. If None, loads from disk.

    Returns:
        sparse matrix of shape (1, vocab_size), or None if vectorizer unavailable.
    """
    vec = vectorizer or load_vectorizer()
    if vec is None:
        return None

    if not pantry_ingredients:
        # Return a zero vector of the right shape
        return sp.csr_matrix((1, len(vec.vocabulary_)))

    doc = " ".join(pantry_ingredients)
    return vec.transform([doc])


def pantry_vector_to_dict(
    pantry_ingredients: list[str],
    vectorizer=None,
    top_n: int = 20,
) -> dict[str, float]:
    """
    Return the top-N non-zero TF-IDF features for a pantry.
    Useful for debugging — shows which ingredient tokens carry the most weight.

    Example output for pantry ["eggs", "whole milk", "butter"]:
        {"eggs": 0.62, "whole milk": 0.58, "milk": 0.41, "butter": 0.38, ...}
    """
    vec = vectorizer or load_vectorizer()
    if vec is None:
        return {}

    matrix = build_pantry_vector(pantry_ingredients, vec)
    if matrix is None:
        return {}

    feature_names = vec.get_feature_names_out()
    scores        = matrix.toarray()[0]
    nonzero_idx   = np.nonzero(scores)[0]

    top = sorted(nonzero_idx, key=lambda i: scores[i], reverse=True)[:top_n]
    return {feature_names[i]: round(float(scores[i]), 4) for i in top}
