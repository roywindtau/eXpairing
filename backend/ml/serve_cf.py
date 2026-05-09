"""
serve_cf.py
-----------
CF serving layer. Selects strategy automatically based on rating count.

PREDICTION TARGET
-----------------
We predict: P(user will rate recipe highly | user, recipe)
Proxied by estimated rating [1,5], normalized to [0,1].

MODEL: BIASED MATRIX FACTORIZATION (Funk SVD)
---------------------------------------------
The warm CF model is Surprise's SVD class, which implements Funk SVD /
biased matrix factorization — NOT true truncated SVD of the rating matrix.

    predicted(u, r) = μ + b_u + b_r + p_u · q_r^T

Where:
    μ    = global mean rating
    b_u  = user bias (does this user rate higher/lower than average?)
    b_r  = recipe bias (is this recipe consistently rated higher/lower?)
    p_u  = user latent factor vector   (learned by SGD)
    q_r  = recipe latent factor vector (learned by SGD)

This is biased matrix factorization. True SVD would decompose R = UΣV^T
using all singular values of the full sparse rating matrix — expensive and
less suited to the sparsity level here (~99% missing entries).

IMPLICIT vs EXPLICIT SIGNALS
-----------------------------
  Explicit: star ratings 1-5 → train biased MF latent vectors
  Implicit: cook events + n_missing → beta_updater (revealed preference)
            skip events → 7-day feed exclusion

Explicit ratings drive CF predictions. Implicit signals drive the
domain-adjustment weights (β). Both are used — complementary signal types.

CF SCORING — SOFT BLEND
-----------------------

Rather than a hard switch at MIN_RATINGS_FOR_CF, scores are a
weighted blend of cold-start and biased MF proportional to rating count:

    alpha     = min(n_ratings / MIN_RATINGS_FOR_CF, 1.0)
    cf_score  = (1 - alpha) * cold_start_score
              + alpha       * mf_score

  n_ratings = 0  →  alpha = 0.0  →  pure cold start (MF not called)
  n_ratings = 1  →  alpha = 0.2  →  80% cold start, 20% biased MF
  n_ratings = 3  →  alpha = 0.6  →  40% cold start, 60% biased MF
  n_ratings ≥ 5  →  alpha = 1.0  →  pure biased MF (cold start not called)

This eliminates the cliff where ranking abruptly changes on the 5th
rating. Biased MF has some signal from the first rating — just weak signal.
The blend lets it contribute proportionally.

STRATEGIES
----------
  "item_based_cold_start"  n_ratings = 0 or model unavailable
  "blended"                0 < n_ratings < MIN_RATINGS_FOR_CF
  "biased_mf"              n_ratings >= MIN_RATINGS_FOR_CF

COLD START (seeds)
------------------
    Preference-seeded item-based CF (see cold_start.py).
    Item similarities come entirely from co-rating patterns — not content.
    Seeds are inferred from diet_tags + pantry (pseudo-interactions).
    Personalized: different tags/pantry → different seeds → different scores.

DATA SPARSITY
-------------
~99% of the user-item matrix is unknown. Biased MF handles this via
low-rank approximation (SGD on observed entries only).
Item-based CF handles cold edges via item neighborhoods.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from backend.ml.cold_start import personalized_cold_start

CF_MODEL_PATH   = Path("models/cf_model.pkl")
SIM_MATRIX_PATH = Path("models/item_sim_matrix.npz")
SIM_IDS_PATH    = Path("models/item_sim_recipe_ids.npy")

MIN_RATINGS_FOR_CF = 5  # threshold for biased MF activation

_cf_model   = None   # Surprise SVD instance (biased MF / Funk SVD)
_sim_matrix = None
_sim_ids    = None
_loaded     = False


def _load():
    global _cf_model, _sim_matrix, _sim_ids, _loaded
    if _loaded:
        return
    if CF_MODEL_PATH.exists():
        with open(CF_MODEL_PATH, "rb") as f:
            _cf_model = pickle.load(f)
        print("[serve_cf] Loaded biased MF model (Funk SVD)")
    else:
        print("[serve_cf] No biased MF model — warm CF unavailable")
    if SIM_MATRIX_PATH.exists() and SIM_IDS_PATH.exists():
        _sim_matrix = sp.load_npz(SIM_MATRIX_PATH)
        _sim_ids    = np.load(SIM_IDS_PATH)
        print(f"[serve_cf] Loaded item-sim matrix {_sim_matrix.shape}")
    else:
        print("[serve_cf] No item-sim matrix — cold-start CF unavailable")
    _loaded = True


def _norm(raw: float, lo: float = 1.0, hi: float = 5.0) -> float:
    """Normalize biased MF predicted rating [1,5] to [0,1]."""
    return round(max(0.0, min(1.0, (raw - lo) / (hi - lo))), 6)


def _mf_scores(user_id: int, recipe_ids: list[int]) -> dict[int, float]:
    """
    Warm CF: biased MF predicted ratings (μ + b_u + b_r + p_u·q_r^T),
    normalized to [0,1]. For unknown users/recipes, Surprise falls back
    to the global mean.
    """
    if _cf_model is None:
        return {}
    return {
        rid: _norm(_cf_model.predict(str(user_id), str(rid)).est)
        for rid in recipe_ids
    }


def similar_recipes(
    recipe_id: int,
    n: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Return N most similar recipes by item-based CF similarity.
    Used for 'you might also like' rail.
    """
    _load()
    if _sim_matrix is None or _sim_ids is None:
        return []
    id_to_row = {int(rid): i for i, rid in enumerate(_sim_ids)}
    row_idx   = id_to_row.get(recipe_id)
    if row_idx is None:
        return []
    sim_row = _sim_matrix[row_idx].toarray().flatten()
    exclude = (exclude_ids or set()) | {recipe_id}
    return sorted(
        [(int(_sim_ids[i]), float(sim_row[i]))
         for i in np.argsort(sim_row)[::-1]
         if int(_sim_ids[i]) not in exclude and sim_row[i] > 0],
        key=lambda x: x[1], reverse=True
    )[:n]


def _blend_alpha(n_ratings: int) -> float:
    """Fraction of biased-MF weight in the blended CF score [0.0, 1.0]."""
    return min(n_ratings / MIN_RATINGS_FOR_CF, 1.0)


def cf_strategy_name(n_ratings: int) -> str:
    """Human-readable strategy label for the API response."""
    _load()
    if _cf_model is None or n_ratings == 0:
        return "item_based_cold_start"
    if n_ratings >= MIN_RATINGS_FOR_CF:
        return "biased_mf"
    return "blended"


def get_cf_scores(
    user_id: int,
    recipe_ids: list[int],
    n_user_ratings: int = 0,
    user_diet_tags: list[str] | None = None,
    pantry_ingredients: list[str] | None = None,
    all_recipes: list[dict] | None = None,
) -> dict[int, float]:
    """
    Main CF scoring entry point. Blends cold-start and SVD proportionally.

    alpha = min(n_user_ratings / MIN_RATINGS_FOR_CF, 1.0)
    score = (1 - alpha) * cold_start + alpha * svd

    At 0 ratings SVD is not called. At MIN_RATINGS_FOR_CF cold start is
    not called. Between them both contribute.

    Args:
        user_id:            app user ID
        recipe_ids:         candidates to score
        n_user_ratings:     explicit ratings accumulated so far
        user_diet_tags:     preference seeds for cold start
        pantry_ingredients: preference seeds for cold start
        all_recipes:        corpus for seed selection (cold start)

    Returns:
        dict recipe_id -> CF score [0,1]
    """
    _load()

    alpha = _blend_alpha(n_user_ratings)

    # Pure cold start — biased MF has no signal yet or model unavailable
    if alpha == 0.0 or _cf_model is None:
        return personalized_cold_start(
            candidate_recipe_ids=recipe_ids,
            all_recipes=all_recipes or [],
            user_diet_tags=user_diet_tags or [],
            pantry_ingredients=pantry_ingredients or [],
            sim_matrix=_sim_matrix,
            sim_recipe_ids=_sim_ids,
        )

    # Pure biased MF — fully warm user
    if alpha == 1.0:
        return _mf_scores(user_id, recipe_ids)

    # Soft blend — both signals contribute proportionally to rating count
    cold = personalized_cold_start(
        candidate_recipe_ids=recipe_ids,
        all_recipes=all_recipes or [],
        user_diet_tags=user_diet_tags or [],
        pantry_ingredients=pantry_ingredients or [],
        sim_matrix=_sim_matrix,
        sim_recipe_ids=_sim_ids,
    )
    mf = _mf_scores(user_id, recipe_ids)
    return {
        rid: round((1 - alpha) * cold.get(rid, 0.0) + alpha * mf.get(rid, 0.0), 6)
        for rid in recipe_ids
    }


def is_warm_user(n: int) -> bool:
    return n >= MIN_RATINGS_FOR_CF

def cf_model_available() -> bool:
    """True when the biased MF (Funk SVD) model is loaded."""
    _load(); return _cf_model is not None

# Backward-compatible alias
svd_available = cf_model_available

def item_sim_available() -> bool:
    _load(); return _sim_matrix is not None
