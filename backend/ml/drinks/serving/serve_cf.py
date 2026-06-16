"""
serve_drink_cf.py
-----------------
CF serving layer for wine drinks.

STRATEGY MATRIX
---------------
                   |  wine candidate
    ───────────────┼────────────────────────────
    0 explicit     |  bayesian_popularity
    ≥ 1 explicit   |  item_sim_from_user_history

Wine is too sparse to train matrix factorization on, so warm users are
served via item-item similarity seeded from their rating history rather
than a latent-factor model.

The "user history" used to seed item-sim INCLUDES synthetic events
(from drink_synthesizer.py), so a user who has cooked lots of food but
never rated a drink still gets item-sim-driven wine suggestions that
reflect their food preferences.

PUBLIC API
----------
    item_sim_available(kind="wine") -> bool
    cf_strategy_name(n_explicit, kind) -> str
    get_cf_scores(user_id, drinks_with_kinds, db) -> dict[drink_id, score]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

from backend.ml.drinks.serving.cold_start import bayesian_popularity, item_sim_seed_scores

SIM_WINE_PATH    = Path("models/drink_sim_wine.npz")
SIM_WINE_IDS_PATH = Path("models/drink_sim_wine_ids.npy")

MIN_RATINGS_FOR_CF = 5

_sim_wine      = None
_sim_wine_ids  = None
_loaded        = False


def _load():
    global _sim_wine, _sim_wine_ids, _loaded
    if _loaded:
        return
    if SIM_WINE_PATH.exists() and SIM_WINE_IDS_PATH.exists():
        _sim_wine     = sp.load_npz(SIM_WINE_PATH)
        _sim_wine_ids = np.load(SIM_WINE_IDS_PATH)
        print(f"[serve_drink_cf] Loaded wine item-sim {_sim_wine.shape}")
    _loaded = True


def _reset_for_tests():
    """Test hook to clear module-level singleton between fixtures."""
    global _sim_wine, _sim_wine_ids, _loaded
    _sim_wine = _sim_wine_ids = None
    _loaded = False


# ── availability flags ──────────────────────────────────────────────────

def item_sim_available(kind: str) -> bool:
    _load()
    if kind == "wine":
        return _sim_wine is not None
    return False


# ── helpers ──────────────────────────────────────────────────────────────

def _user_seed_drinks(user_id: int, kind: str, db) -> tuple[list[int], list[float]]:
    """
    Pull the user's rating history for one kind (BOTH explicit + synthetic).
    Returns (drink_ids, weights) where weight = rating - 3.0.
    Used to seed item-sim scoring at serve time.
    """
    from backend.db.models import Drink, DrinkEvent
    rows = (
        db.query(DrinkEvent.drink_id, DrinkEvent.rating)
        .join(Drink, Drink.id == DrinkEvent.drink_id)
        .filter(DrinkEvent.user_id == user_id)
        .filter(DrinkEvent.event_type == "rate")
        .filter(DrinkEvent.rating.isnot(None))
        .filter(Drink.kind == kind)
        .all()
    )
    ids:     list[int]   = []
    weights: list[float] = []
    for did, rating in rows:
        w = float(rating) - 3.0
        if w == 0:
            continue
        ids.append(int(did))
        weights.append(w)
    return ids, weights


def _count_explicit_ratings(user_id: int, db) -> int:
    """How many real (non-synthetic) drink ratings has this user submitted?"""
    from backend.db.models import DrinkEvent
    return (
        db.query(DrinkEvent.id)
        .filter(DrinkEvent.user_id == user_id)
        .filter(DrinkEvent.event_type == "rate")
        .filter(DrinkEvent.rating.isnot(None))
        .filter(DrinkEvent.synthetic == False)  # noqa: E712
        .count()
    )


def cf_strategy_name(n_explicit: int, kind: str) -> str:
    """Human-readable strategy label (used in API responses for debugging)."""
    _load()
    if n_explicit == 0:
        return "popularity_cold_start"
    # Wine is too sparse to train matrix factorization — always item-sim from history.
    return "wine_item_sim"


# ── main entry point ────────────────────────────────────────────────────

def get_cf_scores(
    user_id: int,
    drinks_with_kinds: list[tuple[int, str]],
    db,
    drinks_by_id: Optional[dict[int, dict]] = None,
) -> dict[int, float]:
    """
    Compute CF scores for a list of (drink_id, kind) pairs.

    Args:
        user_id:           app-side user id
        drinks_with_kinds: candidates as [(drink_id, "wine"), ...]
        db:                SQLAlchemy session
        drinks_by_id:      precomputed dict[drink_id, {"avg_rating", "n_ratings"}]
                           If None, will be queried from DB on-demand.

    Returns:
        dict[drink_id, score in [0,1]]
    """
    _load()
    if not drinks_with_kinds:
        return {}

    wine_ids = [did for did, k in drinks_with_kinds if k == "wine"]

    # Lazy-load drink stats only if we need them for popularity scoring.
    def _ensure_drinks_by_id() -> dict[int, dict]:
        nonlocal drinks_by_id
        if drinks_by_id is not None:
            return drinks_by_id
        from backend.db.models import Drink
        rows = (
            db.query(Drink.id, Drink.avg_rating, Drink.n_ratings)
            .filter(Drink.id.in_(wine_ids))
            .all()
        )
        drinks_by_id = {
            int(r[0]): {"avg_rating": r[1], "n_ratings": r[2]} for r in rows
        }
        return drinks_by_id

    out: dict[int, float] = {}

    if wine_ids:
        seed_ids, seed_weights = _user_seed_drinks(user_id, "wine", db)
        if not seed_ids:
            out.update(bayesian_popularity(wine_ids, _ensure_drinks_by_id()))
        else:
            sim_scores = item_sim_seed_scores(
                wine_ids, seed_ids, seed_weights, _sim_wine, _sim_wine_ids
            )
            if sim_scores:
                out.update(sim_scores)
            else:
                out.update(bayesian_popularity(wine_ids, _ensure_drinks_by_id()))

    return out
