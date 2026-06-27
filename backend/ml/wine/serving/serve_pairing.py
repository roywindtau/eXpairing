"""
serve_pairing.py
----------------
MODULE 4 of the wine<->recipe pairing feature: the pairing scorer.

WHAT IT DOES
------------
Given a recipe (its ingredient list), rank wines by how well they pair with it.
Both sides live in the same 12-dim food-category space:

    recipe  --Module 3 (recipe_categories.recipe_vector)--> 12-dim vector
    wines   --Module 2 (build_wine_pairing_vectors)-------> 12-dim matrix (precomputed)

The pairing score is the COSINE SIMILARITY between the recipe vector and each
wine vector. Both are L2-normalized at rest, so cosine reduces to a dot product:
a wine scores high when its harmonize-derived categories overlap the recipe's
ingredient-derived categories (e.g. a Seafood/Creamy recipe matches a wine that
pairs with Seafood and Creamy dishes).

This is a SIMILARITY-based v1. Contrast-style sensory rules (acid cuts fat, etc.)
are intentionally out of scope here; they can be layered on top later without
changing the shared space.

WHY THIS IS PURE CONTENT-BASED
------------------------------
No user history is used. The score depends only on recipe content vs wine
content -- exactly the "pair me a wine for this dish" use case.

Public API
----------
    pairing_available() -> bool
    pair_wines(ingredients, top_n, style_filter=None) -> list[(wine_id, score)]

Artifacts loaded (built by data.pairing.build_wine_pairing_vectors):
    models/wine_pair_matrix.npz   (n_wines x 12) L2-normalized
    models/wine_pair_ids.npy      wine_id per row
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data.pairing.recipe_categories import recipe_vector

MODELS_DIR   = Path("models")
_PAIR_MATRIX = MODELS_DIR / "wine_pair_matrix.npz"
_PAIR_IDS    = MODELS_DIR / "wine_pair_ids.npy"

_state: dict | None = None


def _load() -> dict | None:
    """Lazy-load the wine pairing matrix once. None if not built yet."""
    global _state
    if _state is not None:
        return _state or None
    if not (_PAIR_MATRIX.exists() and _PAIR_IDS.exists()):
        _state = {}
        return None
    mat = sp.load_npz(_PAIR_MATRIX).tocsr().astype(np.float64)
    ids = np.load(_PAIR_IDS)
    _state = {"mat": mat, "ids": ids}
    return _state


def pairing_available() -> bool:
    return _load() is not None


def pair_wines(
    ingredients: list[str],
    top_n: int = 10,
    candidate_rows: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """
    Rank wines by pairing fit for a recipe's ingredients.

    ingredients    : recipe ingredient strings
    top_n          : how many wines to return
    candidate_rows : optional row-index subset to restrict scoring to (used when
                     the caller has already style-filtered the catalog).

    Returns [(wine_id, score)] sorted desc by score. Score is cosine in [0, 1]
    (vectors are non-negative). Empty if the matrix isn't built or the recipe
    produced a zero vector.
    """
    st = _load()
    if st is None:
        return []

    rvec = recipe_vector(ingredients)          # 12-dim, L2-normalized
    if not np.any(rvec):
        return []

    mat = st["mat"]
    ids = st["ids"]
    if candidate_rows is not None:
        mat = mat[candidate_rows]
        ids = ids[candidate_rows]

    # cosine == dot product: both sides already unit-normalized.
    scores = mat.dot(rvec)                       # (n_wines,)

    k = min(top_n, scores.shape[0])
    if k <= 0:
        return []
    # argpartition for the top-k, then sort just those.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(ids[i]), float(scores[i])) for i in top_idx if scores[i] > 0.0]
