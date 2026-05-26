"""
serve_drink_cb.py
-----------------
Content-based serving layer for drinks. Mirrors backend/ml/serve_cb.py.

ROLE IN THE SYSTEM
------------------
Used in two flows:

  Path A — Recipe pairing (in drink_scoring.score_drinks_for_recipe):
      cb_for_recipe(recipe) returns cosine(bridged_recipe_doc, every_drink)

  Path B — Standalone "Drinks For You" (in drink_scoring.score_drinks_for_user):
      cb_for_user(user_id, db) builds a weighted profile from the user's
      rated recipes (via flavor_bridge.bridge_recipe_doc) and cosines
      against every drink. This is the no-CF flow: even users with zero
      drink ratings can get personalized drink suggestions from their food
      history alone.

WHY BOTH WORK ON THE SAME MATRIX
--------------------------------
Both queries vectorize a "bridged recipe document" with the same
TfidfVectorizer that was fit on drink docs. Tokens not in the drink
vocab are silently dropped at transform-time (sklearn default). That
means only flavor-bridge-injected tokens (e.g. "seafood", "red",
"beef") actually contribute to the cosine — which is exactly what we
want.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from backend.ml.drinks.serving.flavor_bridge import bridge_recipe_doc

CB_MATRIX_PATH     = Path("models/drink_cb_matrix.npz")
CB_IDS_PATH        = Path("models/drink_cb_ids.npy")
CB_KINDS_PATH      = Path("models/drink_cb_kinds.npy")
CB_VECTORIZER_PATH = Path("models/drink_cb_vectorizer.pkl")

_matrix     = None
_drink_ids  = None
_kinds      = None
_vectorizer = None
_loaded     = False


def _load():
    global _matrix, _drink_ids, _kinds, _vectorizer, _loaded
    if _loaded:
        return
    if (CB_MATRIX_PATH.exists() and CB_IDS_PATH.exists() and
            CB_KINDS_PATH.exists() and CB_VECTORIZER_PATH.exists()):
        _matrix    = sp.load_npz(CB_MATRIX_PATH)
        _drink_ids = np.load(CB_IDS_PATH)
        _kinds     = np.load(CB_KINDS_PATH, allow_pickle=True)
        with open(CB_VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
        print(f"[serve_drink_cb] Loaded drink CB matrix {_matrix.shape}")
    else:
        print("[serve_drink_cb] Drink CB artifacts not found — cb scores will be empty.")
    _loaded = True


def model_available() -> bool:
    _load()
    return _matrix is not None


def _reset_for_tests():
    """Hook used by tests to clear the module-level singleton."""
    global _matrix, _drink_ids, _kinds, _vectorizer, _loaded
    _matrix = _drink_ids = _kinds = _vectorizer = None
    _loaded = False


def _kind_mask(kind_filter: Optional[str]) -> np.ndarray:
    """Boolean index over rows matching the requested kind ('beer'|'wine'|None)."""
    if kind_filter is None:
        return np.ones(len(_kinds), dtype=bool)
    return _kinds == kind_filter


def _scores_from_vector(query_vec, kind_filter: Optional[str]) -> dict[int, float]:
    """Cosine `query_vec` (1 x vocab) against the (filtered) drink matrix."""
    mask = _kind_mask(kind_filter)
    if not mask.any():
        return {}
    sub_matrix = _matrix[mask]
    sub_ids    = _drink_ids[mask]
    sims = cosine_similarity(query_vec, sub_matrix)[0]
    return {
        int(did): round(float(max(0.0, sim)), 6)
        for did, sim in zip(sub_ids, sims)
    }


def cb_for_recipe(recipe, kind_filter: Optional[str] = None) -> dict[int, float]:
    """
    Path A: cosine similarity between this recipe and every drink.

    Args:
        recipe:       any object with .ingredients_csv and .tags_csv attrs
                      (Recipe ORM row or SimpleNamespace in tests)
        kind_filter:  None | "beer" | "wine"
    Returns:
        dict[drink_id, cosine in [0,1]]   (empty if model not loaded)
    """
    _load()
    if _matrix is None or _vectorizer is None:
        return {}

    doc = bridge_recipe_doc(recipe)
    if not doc.strip():
        return {}

    query_vec = _vectorizer.transform([doc])
    return _scores_from_vector(query_vec, kind_filter)


def cb_for_user(
    user_id: int,
    db,
    kind_filter: Optional[str] = None,
    min_rating: float = 1.0,
) -> dict[int, float]:
    """
    Path B: build a weighted taste profile from the user's RECIPE rating history,
    then cosine it against every drink.

    Weighting scheme (matches serve_cb.cb_taste_profile_batch):
        weight = rating - 3.0   ∈ [-2, +2]
    A 5-star recipe pushes its bridge-tokens INTO the profile; a 1-star recipe
    pushes them OUT. Profile is L2-normalized by sum of |weights|.

    Args:
        user_id:      app-side user id
        db:           SQLAlchemy session (we query UserEvent + Recipe)
        kind_filter:  None | "beer" | "wine"
        min_rating:   floor on rating; events below are skipped entirely
                       (default 1.0 = use all events)

    Returns:
        dict[drink_id, score in [0,1]]
        Empty dict if model unavailable or user has no usable rated recipes.
    """
    _load()
    if _matrix is None or _vectorizer is None:
        return {}

    # Local import keeps the module light when only Path A is used.
    from backend.db.models import Recipe, UserEvent

    events = (
        db.query(UserEvent.recipe_id, UserEvent.rating)
        .filter(
            UserEvent.user_id == user_id,
            UserEvent.event_type == "rate",
            UserEvent.rating.isnot(None),
            UserEvent.rating >= min_rating,
        )
        .all()
    )
    if not events:
        return {}

    recipe_ids = [r_id for r_id, _ in events]
    recipes = (
        db.query(Recipe)
        .filter(Recipe.id.in_(recipe_ids))
        .all()
    )
    recipe_by_id = {r.id: r for r in recipes}

    profile_vec = None
    total_weight = 0.0
    for r_id, rating in events:
        recipe = recipe_by_id.get(r_id)
        if recipe is None:
            continue
        doc = bridge_recipe_doc(recipe)
        if not doc.strip():
            continue
        weight = float(rating) - 3.0
        if weight == 0:
            continue
        vec = _vectorizer.transform([doc])
        if profile_vec is None:
            profile_vec = weight * vec
        else:
            profile_vec = profile_vec + weight * vec
        total_weight += abs(weight)

    if profile_vec is None or total_weight == 0:
        return {}

    profile_vec = profile_vec / total_weight
    return _scores_from_vector(profile_vec, kind_filter)
