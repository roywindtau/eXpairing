"""
serve_cb.py
-----------
Content-based filtering serving layer.

ROLE IN THE SYSTEM: CB BOOST
-----------------------------
CB is a secondary signal in the CF-first formula:

    final_score = γ·CF(user,recipe)       ← base preference prediction
                + δ·CB(pantry, recipe)    ← ingredient profile boost ✓
                + α·expiry_urgency        ← domain adjustment
                + β·match_ratio           ← domain adjustment

CB does NOT replace CF. It boosts recipes whose ingredient profile
naturally aligns with the user's pantry composition — capturing
cuisine affinity that CF may miss (especially for new items or users).

CONTENT-BASED FILTERING: BY THE BOOK
--------------------------------------
CB builds:
    Item profiles: TF-IDF vectors over ingredient tokens (unigrams + bigrams)
    User profile:  aggregated pantry vector (ingredients joined as a document)

Similarity: cosine similarity between user profile and item profile.

    cb_score(u, r) = cos(pantry_vector(u), recipe_vector(r))
                   = (pantry_vector · recipe_vector)
                     / (‖pantry_vector‖ · ‖recipe_vector‖)

This captures cuisine affinity: a pantry with miso, soy sauce, and
sesame oil will naturally cosine-match Japanese recipes more than
Italian ones, even without any explicit preference signal.

DIFFERENCE FROM CF
------------------
CF uses USER INTERACTION DATA (ratings, co-cooking patterns).
CB uses ITEM CONTENT (ingredient text). No user history needed.
Together they form the hybrid system.

SAVED ARTIFACTS
---------------
    models/cb_matrix.npz       sparse TF-IDF matrix (n_recipes × vocab)
    models/cb_recipe_ids.npy   recipe_id for each row
    models/cb_vectorizer.pkl   fitted TfidfVectorizer

All loaded once at module level (singleton pattern for performance).
"""

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

CB_MATRIX_PATH     = Path("models/cb_matrix.npz")
CB_RECIPE_IDS_PATH = Path("models/cb_recipe_ids.npy")
CB_VECTORIZER_PATH = Path("models/cb_vectorizer.pkl")

_matrix     = None
_recipe_ids = None
_vectorizer = None
_loaded     = False


def _load():
    global _matrix, _recipe_ids, _vectorizer, _loaded
    if _loaded:
        return
    if (CB_MATRIX_PATH.exists() and
            CB_RECIPE_IDS_PATH.exists() and
            CB_VECTORIZER_PATH.exists()):
        _matrix     = sp.load_npz(CB_MATRIX_PATH)
        _recipe_ids = np.load(CB_RECIPE_IDS_PATH)
        with open(CB_VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        print(f"[serve_cb] Loaded CB matrix {_matrix.shape}")
    else:
        print("[serve_cb] CB artifacts not found — CB scores will be 0.0")
    _loaded = True


def model_available() -> bool:
    _load()
    return _matrix is not None


def cb_similarity_batch(
    pantry_ingredients: list[str],
    recipe_ids: list[int],
) -> dict[int, float]:
    """
    Compute cosine similarity between the user's pantry profile and
    each candidate recipe's ingredient TF-IDF vector.

    User profile = pantry ingredients joined as a single document,
    vectorized with the same TfidfVectorizer used during training.

    Args:
        pantry_ingredients: canonical ingredient names the user has
        recipe_ids:         recipe IDs to score

    Returns:
        dict recipe_id -> cosine similarity [0,1]
        Returns empty dict if CB model is not available.
    """
    _load()
    if _matrix is None or _vectorizer is None:
        return {}

    # Build user profile vector from pantry
    pantry_doc    = " ".join(pantry_ingredients)
    pantry_vector = _vectorizer.transform([pantry_doc])   # (1, vocab_size)

    id_to_row = {int(rid): i for i, rid in enumerate(_recipe_ids)}

    rows, valid_ids = [], []
    for rid in recipe_ids:
        row = id_to_row.get(rid)
        if row is not None:
            rows.append(row)
            valid_ids.append(rid)

    if not rows:
        return {}

    recipe_submatrix = _matrix[rows]
    sims = cosine_similarity(pantry_vector, recipe_submatrix)[0]

    return {
        rid: round(float(sim), 6)
        for rid, sim in zip(valid_ids, sims)
    }


def cb_taste_profile_batch(
    rated_recipe_ids: list[int],
    ratings: list[float],
    candidate_recipe_ids: list[int],
) -> dict[int, float]:
    """
    CB scores based on the user's taste profile from rating history.

    For warm users (≥5 ratings) the user profile is built as a weighted
    average of rated recipe TF-IDF vectors, where weight = (rating - 3.0).
    This gives positive weight to liked recipes and negative weight to
    disliked ones, so the profile captures genuine taste rather than just
    "recipes the user has seen".

    Negative cosine similarities are clipped to 0 — we don't penalise
    recipes just because they're dissimilar; we only boost good matches.

    Args:
        rated_recipe_ids:    recipe IDs the user has rated
        ratings:             corresponding rating values (1-5)
        candidate_recipe_ids: recipe IDs to score

    Returns:
        dict recipe_id -> CB taste similarity [0, 1]
        Returns empty dict if CB model is not available or no rated recipes
        have non-zero weights.
    """
    _load()
    if _matrix is None or _vectorizer is None:
        return {}
    if not rated_recipe_ids:
        return {}

    id_to_row = {int(rid): i for i, rid in enumerate(_recipe_ids)}

    # Build weighted taste profile from rated recipes
    profile_vec = None
    total_weight = 0.0
    for rid, rating in zip(rated_recipe_ids, ratings):
        row = id_to_row.get(int(rid))
        if row is None:
            continue
        weight = float(rating) - 3.0   # -2 to +2 range
        vec = _matrix[row]             # (1, vocab_size) sparse row
        if profile_vec is None:
            profile_vec = weight * vec
        else:
            profile_vec = profile_vec + weight * vec
        total_weight += abs(weight)

    if profile_vec is None or total_weight == 0:
        return {}

    profile_vec = profile_vec / total_weight  # normalize

    # Score candidates
    rows, valid_ids = [], []
    for rid in candidate_recipe_ids:
        row = id_to_row.get(int(rid))
        if row is not None:
            rows.append(row)
            valid_ids.append(rid)

    if not rows:
        return {}

    recipe_submatrix = _matrix[rows]
    sims = cosine_similarity(profile_vec, recipe_submatrix)[0]

    return {
        rid: round(float(max(0.0, sim)), 6)
        for rid, sim in zip(valid_ids, sims)
    }
