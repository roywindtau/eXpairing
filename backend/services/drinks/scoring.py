"""
drink_scoring.py
----------------
Final ranking layer for drinks. Combines CB, CF, expert, and prior signals
into a single ranked list of DrinkScore objects.

Mirrors backend/services/scoring.py (recipe ranker) but drops the recipe-
specific machinery (pantry expiry, ingredient match ratio, MMR diversity)
since none of them apply to a drink catalog.

Two paths, two formulas
-----------------------
Path A — given a recipe, suggest drinks to pair with it:

    final_A = 0.45·cb_score
            + 0.25·cf_score
            + 0.20·expert_boost
            + 0.10·popularity_prior

Path B — "Drinks For You", driven by the user's recipe taste:

    final_B = 0.55·cb_score
            + 0.30·cf_score
            + 0.15·popularity_prior

The expert layer is Path-A-only: it needs a SPECIFIC recipe to pair
against. Path B has none, so the weight redistributes to CB.

Score calibration
-----------------
Each component is min-max normalized across the candidate pool before
the weighted blend. Same reasoning as recipe scoring.py: prevents one
dimension (e.g. popularity prior, which spans orders of magnitude
because of log(n_ratings)) from dominating the final order.

No DB queries
-------------
All signals are passed in as plain dicts. The router does the DB work.
This keeps drink_scoring pure-functional and trivial to unit test with
SimpleNamespace fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

# ── weights (design doc §"Scoring formulas") ────────────────────────────

WEIGHTS_PATH_A = {
    "cb":     0.45,
    "cf":     0.25,
    "expert": 0.20,
    "prior":  0.10,
}

WEIGHTS_PATH_B = {
    "cb":    0.55,
    "cf":    0.30,
    "prior": 0.15,
}


# ── data class ──────────────────────────────────────────────────────────

@dataclass
class DrinkScore:
    """
    Full scoring breakdown for one drink against a recipe (Path A) or
    against the user's food taste profile (Path B). Returned in the
    ranked list so the API can expose explainability.
    """
    drink_id:     int
    drink_name:   str
    kind:         str           # "wine"
    final_score:  float
    cb_score:     float
    cf_score:     float
    expert_boost: float         # 0.0 in Path B
    prior_score:  float
    cf_strategy:  str = ""      # e.g. "wine_item_sim" | "popularity_cold_start"
    # Useful for the UI:
    matched_harmonize: list[str] = field(default_factory=list)


# ── helpers ─────────────────────────────────────────────────────────────

def _popularity_prior(drink) -> float:
    """
    Bayesian-ish popularity prior, on a much wider scale than [0,1] —
    that's fine because we min-max calibrate before blending.

        prior(d) = avg_rating(d) * log(1 + n_ratings(d))

    The log(n) damp keeps a wine with 5000 ratings from outweighing a
    wine with 500 by 10×. After min-max calibration the prior contributes
    a small but stable tiebreaker among similarly-scored candidates.
    """
    n   = float(getattr(drink, "n_ratings", None)  or 0)
    avg = float(getattr(drink, "avg_rating", None) or 0.0)
    if n <= 0 or avg <= 0:
        return 0.0
    return avg * math.log1p(n)


def _calibrate(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0,1]; all-equal → 0.5 vector."""
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _blend(weights: dict[str, float], components: dict[str, float]) -> float:
    """Sum w_k * component_k for k in weights ∩ components."""
    return sum(weights[k] * components.get(k, 0.0) for k in weights)


# ── Path A — recipe pairing ─────────────────────────────────────────────

def rank_drinks_for_recipe(
    recipe,
    candidates: Iterable,
    cb_scores: dict[int, float],
    cf_scores: dict[int, float],
    expert_boosts: dict[int, float],
    cf_strategies: Optional[dict[int, str]] = None,
    top_n: int = 10,
) -> list[DrinkScore]:
    """
    Path A — given one recipe + a candidate drink pool + all pre-computed
    signal dicts, return the top-N drinks ranked by the Path-A formula.

    Args:
        recipe:        Recipe ORM row (used by callers for context; not directly
                       read here — all signals are pre-computed).
        candidates:    iterable of Drink ORM rows
        cb_scores:     drink_id -> CB cosine in [0,1]
        cf_scores:     drink_id -> CF score in [0,1]
        expert_boosts: drink_id -> rule-based boost in [0,0.25]
        cf_strategies: drink_id -> debug label ("biased_mf", "wine_item_sim"...)
        top_n:         max results (use 0 to return all sorted)

    Returns:
        list[DrinkScore] sorted by final_score descending.
    """
    candidates = list(candidates)
    if not candidates:
        return []

    scores: list[DrinkScore] = []
    for d in candidates:
        scores.append(DrinkScore(
            drink_id     = d.id,
            drink_name   = d.name,
            kind         = d.kind,
            final_score  = 0.0,                  # computed after calibration
            cb_score     = float(cb_scores.get(d.id, 0.0)),
            cf_score     = float(cf_scores.get(d.id, 0.0)),
            expert_boost = float(expert_boosts.get(d.id, 0.0)),
            prior_score  = _popularity_prior(d),
            cf_strategy  = (cf_strategies or {}).get(d.id, ""),
        ))

    cal_cb     = _calibrate([s.cb_score     for s in scores])
    cal_cf     = _calibrate([s.cf_score     for s in scores])
    cal_expert = _calibrate([s.expert_boost for s in scores])
    cal_prior  = _calibrate([s.prior_score  for s in scores])

    for i, s in enumerate(scores):
        s.final_score = round(_blend(WEIGHTS_PATH_A, {
            "cb":     cal_cb[i],
            "cf":     cal_cf[i],
            "expert": cal_expert[i],
            "prior":  cal_prior[i],
        }), 6)

    scores.sort(key=lambda s: -s.final_score)
    return scores if top_n <= 0 else scores[:top_n]


# ── Path B — "Drinks For You" ───────────────────────────────────────────

def rank_drinks_for_user(
    candidates: Iterable,
    cb_scores: dict[int, float],
    cf_scores: dict[int, float],
    cf_strategies: Optional[dict[int, str]] = None,
    top_n: int = 10,
) -> list[DrinkScore]:
    """
    Path B — no recipe, no expert boost. CB carries the most weight
    because the input signal (the user's food taste vector) is itself
    a content-based representation.

    Args:
        candidates:    iterable of Drink ORM rows
        cb_scores:     drink_id -> CB cosine in [0,1]
        cf_scores:     drink_id -> CF score in [0,1]
        cf_strategies: drink_id -> debug label
        top_n:         max results

    Returns:
        list[DrinkScore] sorted by final_score descending.
        expert_boost is always 0.0 in Path B output.
    """
    candidates = list(candidates)
    if not candidates:
        return []

    scores: list[DrinkScore] = []
    for d in candidates:
        scores.append(DrinkScore(
            drink_id     = d.id,
            drink_name   = d.name,
            kind         = d.kind,
            final_score  = 0.0,
            cb_score     = float(cb_scores.get(d.id, 0.0)),
            cf_score     = float(cf_scores.get(d.id, 0.0)),
            expert_boost = 0.0,
            prior_score  = _popularity_prior(d),
            cf_strategy  = (cf_strategies or {}).get(d.id, ""),
        ))

    cal_cb    = _calibrate([s.cb_score    for s in scores])
    cal_cf    = _calibrate([s.cf_score    for s in scores])
    cal_prior = _calibrate([s.prior_score for s in scores])

    for i, s in enumerate(scores):
        s.final_score = round(_blend(WEIGHTS_PATH_B, {
            "cb":    cal_cb[i],
            "cf":    cal_cf[i],
            "prior": cal_prior[i],
        }), 6)

    scores.sort(key=lambda s: -s.final_score)
    return scores if top_n <= 0 else scores[:top_n]
