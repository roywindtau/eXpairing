"""
serve_drink_cf.py
-----------------
CF serving layer for drinks. Mirrors backend/ml/serve_cf.py but with the
per-kind asymmetry the drink datasets require.

STRATEGY MATRIX
---------------
                   |  beer candidate         |  wine candidate
    ───────────────┼─────────────────────────┼────────────────────────────
    0 explicit     |  bayesian_popularity    |  bayesian_popularity
    1-4 explicit   |  blend(item_sim, SVD)   |  item_sim_from_user_history
    ≥ 5 explicit   |  SVD                    |  item_sim_from_user_history

The "user history" used to seed item-sim INCLUDES synthetic events
(from drink_synthesizer.py), so a user who has cooked lots of beef but
never rated a drink still gets item-sim-driven beer/wine suggestions
that reflect their food preferences. Synthetic events are filtered out
of SVD training (see train_drink_cf.py) so they don't pollute the latent
factors, but they're fair game as preference signals at serve time.

PUBLIC API
----------
    cf_model_available()  -> bool
    item_sim_available(kind="beer"|"wine") -> bool
    cf_strategy_name(n_explicit, kind) -> str
    get_cf_scores(user_id, drinks_with_kinds, db) -> dict[drink_id, score]
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp

from backend.ml.drink_cold_start import bayesian_popularity, item_sim_seed_scores

CF_MODEL_PATH    = Path("models/drink_cf_model.pkl")
SIM_BEER_PATH    = Path("models/drink_sim_beer.npz")
SIM_BEER_IDS_PATH = Path("models/drink_sim_beer_ids.npy")
SIM_WINE_PATH    = Path("models/drink_sim_wine.npz")
SIM_WINE_IDS_PATH = Path("models/drink_sim_wine_ids.npy")

MIN_RATINGS_FOR_CF = 5

_cf_model      = None
_sim_beer      = None
_sim_beer_ids  = None
_sim_wine      = None
_sim_wine_ids  = None
_loaded        = False


def _load():
    global _cf_model, _sim_beer, _sim_beer_ids, _sim_wine, _sim_wine_ids, _loaded
    if _loaded:
        return
    if CF_MODEL_PATH.exists():
        with open(CF_MODEL_PATH, "rb") as f:
            _cf_model = pickle.load(f)
        print("[serve_drink_cf] Loaded drink SVD model (beer)")
    else:
        print("[serve_drink_cf] No SVD model — beer warm CF unavailable")

    if SIM_BEER_PATH.exists() and SIM_BEER_IDS_PATH.exists():
        _sim_beer     = sp.load_npz(SIM_BEER_PATH)
        _sim_beer_ids = np.load(SIM_BEER_IDS_PATH)
        print(f"[serve_drink_cf] Loaded beer item-sim {_sim_beer.shape}")
    if SIM_WINE_PATH.exists() and SIM_WINE_IDS_PATH.exists():
        _sim_wine     = sp.load_npz(SIM_WINE_PATH)
        _sim_wine_ids = np.load(SIM_WINE_IDS_PATH)
        print(f"[serve_drink_cf] Loaded wine item-sim {_sim_wine.shape}")
    _loaded = True


def _reset_for_tests():
    """Test hook to clear module-level singleton between fixtures."""
    global _cf_model, _sim_beer, _sim_beer_ids, _sim_wine, _sim_wine_ids, _loaded
    _cf_model = _sim_beer = _sim_beer_ids = _sim_wine = _sim_wine_ids = None
    _loaded = False


# ── availability flags ──────────────────────────────────────────────────

def cf_model_available() -> bool:
    _load()
    return _cf_model is not None


def item_sim_available(kind: str) -> bool:
    _load()
    if kind == "beer":
        return _sim_beer is not None
    if kind == "wine":
        return _sim_wine is not None
    return False


# ── helpers ──────────────────────────────────────────────────────────────

def _norm_rating(raw: float, lo: float = 0.0, hi: float = 5.0) -> float:
    """Normalize an SVD predicted rating to [0,1]. Beer ratings live in [0,5]."""
    return round(max(0.0, min(1.0, (raw - lo) / (hi - lo))), 6)


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


def _blend_alpha(n_explicit: int) -> float:
    return min(n_explicit / MIN_RATINGS_FOR_CF, 1.0)


def cf_strategy_name(n_explicit: int, kind: str) -> str:
    """Human-readable strategy label (used in API responses for debugging)."""
    _load()
    if n_explicit == 0:
        return "popularity_cold_start"
    # Wine never uses SVD — too sparse to train. Always item-sim from history.
    if kind == "wine":
        return "wine_item_sim"
    if n_explicit >= MIN_RATINGS_FOR_CF and _cf_model is not None:
        return "biased_mf"
    if _cf_model is not None:
        return "blended"
    return "beer_item_sim"


# ── SVD scoring (beer-warm only) ────────────────────────────────────────

def _svd_scores(user_id: int, beer_ids: Iterable[int]) -> dict[int, float]:
    """Surprise SVD predictions normalized to [0,1]; empty if no model."""
    if _cf_model is None:
        return {}
    return {
        int(bid): _norm_rating(_cf_model.predict(str(user_id), str(bid)).est)
        for bid in beer_ids
    }


# ── main entry point ────────────────────────────────────────────────────

def get_cf_scores(
    user_id: int,
    drinks_with_kinds: list[tuple[int, str]],
    db,
    drinks_by_id: Optional[dict[int, dict]] = None,
) -> dict[int, float]:
    """
    Compute CF scores for a mixed list of (drink_id, kind) pairs.

    Args:
        user_id:           app-side user id
        drinks_with_kinds: candidates as [(drink_id, "beer"|"wine"), ...]
        db:                SQLAlchemy session
        drinks_by_id:      precomputed dict[drink_id, {"avg_rating", "n_ratings"}]
                           If None, will be queried from DB on-demand.

    Returns:
        dict[drink_id, score in [0,1]]
    """
    _load()
    if not drinks_with_kinds:
        return {}

    beer_ids = [did for did, k in drinks_with_kinds if k == "beer"]
    wine_ids = [did for did, k in drinks_with_kinds if k == "wine"]

    n_explicit = _count_explicit_ratings(user_id, db)

    # Lazy-load drink stats only if we need them for popularity scoring.
    def _ensure_drinks_by_id() -> dict[int, dict]:
        nonlocal drinks_by_id
        if drinks_by_id is not None:
            return drinks_by_id
        from backend.db.models import Drink
        all_ids = beer_ids + wine_ids
        rows = (
            db.query(Drink.id, Drink.avg_rating, Drink.n_ratings)
            .filter(Drink.id.in_(all_ids))
            .all()
        )
        drinks_by_id = {
            int(r[0]): {"avg_rating": r[1], "n_ratings": r[2]} for r in rows
        }
        return drinks_by_id

    out: dict[int, float] = {}

    # ── beer ─────────────────────────────────────────────────────────────
    if beer_ids:
        if n_explicit == 0:
            out.update(bayesian_popularity(beer_ids, _ensure_drinks_by_id()))
        else:
            alpha = _blend_alpha(n_explicit)
            seed_ids, seed_weights = _user_seed_drinks(user_id, "beer", db)
            item_sim = item_sim_seed_scores(
                beer_ids, seed_ids, seed_weights, _sim_beer, _sim_beer_ids
            )
            if alpha >= 1.0 and _cf_model is not None:
                out.update(_svd_scores(user_id, beer_ids))
            elif _cf_model is None:
                out.update(item_sim or bayesian_popularity(beer_ids, _ensure_drinks_by_id()))
            else:
                svd = _svd_scores(user_id, beer_ids)
                for bid in beer_ids:
                    out[bid] = round(
                        (1 - alpha) * item_sim.get(bid, 0.0)
                        + alpha     * svd.get(bid, 0.0),
                        6,
                    )

    # ── wine (never SVD) ────────────────────────────────────────────────
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
