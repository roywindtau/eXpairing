"""
drink_cold_start.py
-------------------
Cold-start helpers for the drink CF layer (used by serve_drink_cf.py).

Two functions:
  1. bayesian_popularity()  — score = smoothed (avg_rating, n_ratings).
     Used when the user has no rating history or item-sim isn't available.

  2. item_sim_seed_scores() — weighted-sum cosine to user's "seed" drinks.
     Used in the blended-warmth band (1-4 ratings) and as a way to score
     drinks for users whose only signal is synthetic events from the
     recipe-side synthesizer (which still represent revealed preferences,
     just less reliable than explicit ratings).

This module is intentionally simpler than backend/ml/cold_start.py:
recipes have preference seeds (diet_tags + pantry); drinks don't, so
the only available seed when the user has no drink history is "global
popularity", and the only available seed when they DO have history is
"the drinks they've rated".
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp

GLOBAL_MEAN_RATING = 3.5   # neutral prior across drink datasets (rough midpoint of 1-5)
SMOOTHING_C        = 10    # Bayesian smoothing constant (higher = trust prior more)


def bayesian_popularity(
    drink_ids: Iterable[int],
    drinks_by_id: dict[int, dict],
    C: float = SMOOTHING_C,
    global_mean: float = GLOBAL_MEAN_RATING,
) -> dict[int, float]:
    """
    Bayesian-smoothed popularity score for each candidate drink.

        score(d) = (n_d * avg_d + C * global_mean) / (n_d + C)

    Smoothed so a drink with 1 rating of 5.0 doesn't trivially beat a
    drink with 500 ratings averaging 4.4. Final scores are linearly
    normalized to [0, 1] within the candidate pool so they slot into
    the same weight as any other CF score downstream.

    Args:
        drink_ids:    candidate drink_ids to score
        drinks_by_id: dict mapping drink_id -> {"avg_rating": float|None,
                                                "n_ratings": int}
        C:            smoothing strength (default 10)
        global_mean:  prior mean rating (default 3.5)
    """
    raw: dict[int, float] = {}
    for did in drink_ids:
        d = drinks_by_id.get(int(did))
        if d is None:
            raw[int(did)] = 0.0
            continue
        n   = float(d.get("n_ratings") or 0)
        avg = float(d.get("avg_rating") or global_mean)
        raw[int(did)] = (n * avg + C * global_mean) / (n + C)

    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    if hi - lo < 1e-9:
        # all equal — return mid-scale so they don't all collapse to 0
        return {d: 0.5 for d in raw}
    return {d: round((s - lo) / (hi - lo), 6) for d, s in raw.items()}


def item_sim_seed_scores(
    candidate_drink_ids: Iterable[int],
    seed_drink_ids: list[int],
    seed_weights: list[float],
    sim_matrix: Optional[sp.csr_matrix],
    sim_ids: Optional[np.ndarray],
) -> dict[int, float]:
    """
    Weighted item-based CF score: how similar is each candidate to the
    user's rated drinks, weighted by how much they liked each seed.

        score(c) = sum_i w_i * sim(c, seed_i) / sum_i |w_i|

    where w_i = (seed_rating_i - 3.0) so high-rated seeds pull candidates
    upward and low-rated seeds push them down. Returns empty dict if no
    sim matrix is available or no seeds overlap with the matrix.

    Output is clipped to [0,1] (cosine values are already in [-1,1] but
    after the +ve/-ve weighting we may go slightly negative; we floor at 0).
    """
    if sim_matrix is None or sim_ids is None or len(seed_drink_ids) == 0:
        return {}
    assert len(seed_drink_ids) == len(seed_weights), "seeds and weights must align"

    id_to_row = {int(sid): i for i, sid in enumerate(sim_ids)}
    seed_rows: list[int] = []
    weights:   list[float] = []
    for sid, w in zip(seed_drink_ids, seed_weights):
        row = id_to_row.get(int(sid))
        if row is None or w == 0:
            continue
        seed_rows.append(row)
        weights.append(float(w))

    total_w = sum(abs(w) for w in weights)
    if not seed_rows or total_w == 0:
        return {}

    out: dict[int, float] = {}
    for did in candidate_drink_ids:
        row = id_to_row.get(int(did))
        if row is None:
            out[int(did)] = 0.0
            continue
        score = sum(
            float(sim_matrix[row, sr]) * w
            for sr, w in zip(seed_rows, weights)
        ) / total_w
        out[int(did)] = round(max(0.0, min(1.0, score)), 6)
    return out
