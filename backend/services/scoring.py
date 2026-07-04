"""
scoring.py — core ranking engine for eXpairing.

PREDICTION TARGET
-----------------
We model recommendation as predicting:

    P(user will cook recipe | user, recipe, pantry state)

Signals:
  - Explicit: star ratings (1-5) → SVD latent vectors
  - Implicit: cook events, skips → beta_updater, cold start seeds

ARCHITECTURE: CF-FIRST
-----------------------
The system is structured as CF with domain-specific adjustments:

    base_score  = CF(user, recipe)          ← predicts user preference
    final_score = base_score
                + domain_adjustments        ← expiry urgency + availability
                + cb_boost                  ← ingredient profile affinity

This is intentional. CF is the core predictive model. Expiry urgency and
ingredient match are feasibility constraints — they adjust what's actually
cookable tonight, they don't replace the preference prediction.

Formally:

    final_score = γ · cf_score                    (CF base)
                + δ · cb_score                    (CB boost)
                + α · expiry_urgency_score        (domain: waste minimization)
                + β · match_ratio                 (domain: availability)

Where β is per-user and learned from revealed cooking behavior.

DEFAULT WEIGHTS
---------------
    γ = 0.35   CF base      — highest single weight, reflecting CF primacy
    δ = 0.10   CB boost
    α = 0.35   expiry urgency
    β = 0.20   ingredient match (overridden per user, learned from behavior)

When CF is unavailable (cold start path uses item-based CF scores, not zero):
    γ stays at 0.35 using cold-start item-CF scores.
When CB model is not trained:
    δ redistributes to α and γ so the formula is always valid.

SCORE CALIBRATION
-----------------
Each component (cf_score, cb_score, expiry_score, match_ratio) is
min-max normalized across all candidates before the weighted blend.
This prevents one component from dominating due to scale differences —
e.g., expiry urgency often spans a wider range than CF scores on sparse data.
Calibration is applied over the full candidate pool before MMR selection.

MMR DIVERSITY RERANKING
-----------------------
After scoring, the top (3 × top_n) candidates are passed through
Maximal Marginal Relevance (MMR) to reduce ingredient-monotony in the feed:

    MMR(r) = λ · final_score(r) − (1−λ) · max_sim(r, selected)

where similarity is ingredient Jaccard. λ=0.7 keeps relevance primary.
MMR selects greedily: always pick the candidate maximizing the above score.

DATA SPARSITY NOTE
------------------
The Food.com user-item matrix is ~99% empty. Two complementary strategies:
  - SVD matrix factorization: handles sparsity via low-rank approximation
  - Item-based CF: uses item-item similarities (more robust at sparse edges)
See item_similarity.py and serve_cf.py for implementation.

IMPLICIT vs EXPLICIT SIGNALS
-----------------------------
We use both signal types:
  Explicit: star ratings 1-5  → SVD training data
  Implicit: cook events       → beta updater (revealed preference)
            skip events       → future negative signal
Implicit signals are more frequent and reflect real behaviour. Explicit
ratings give calibrated preference strength.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.services.expiry import pantry_urgency_map
from backend.services.ingredient_match import match_ingredients, expiry_weighted_match

# ── default weights ────────────────────────────────────────────────────────
# γ (CF) intentionally highest single weight — CF is the core model.
# α and β together form the domain-adjustment layer.
DEFAULT_GAMMA: float = 0.35   # CF base score
DEFAULT_DELTA: float = 0.10   # CB ingredient profile boost
DEFAULT_ALPHA: float = 0.35   # expiry urgency (domain adjustment)
DEFAULT_BETA:  float = 0.20   # ingredient match (per-user, learned)


@dataclass
class RecipeScore:
    """
    Full scoring breakdown for one (user, recipe) pair.
    Returned in the ranked list so the UI can show explainability.
    """
    recipe_id:             int
    recipe_name:           str
    final_score:           float
    expiry_urgency:        float     # domain adjustment: expiry urgency
    match_ratio:           float     # domain adjustment: ingredient availability
    cf_score:              float     # CF base score (SVD or item-based cold start)
    cb_score:              float     # CB ingredient profile boost
    matched_ingredients:   list[str] = field(default_factory=list)
    missing_ingredients:   list[str] = field(default_factory=list)
    total_ingredients:     int = 0


@dataclass
class UserProfile:
    """Minimal user profile for scoring."""
    user_id: int   = 0
    beta:    float = DEFAULT_BETA   # availability weight: per-user, learned
    has_cf:  bool  = False
    has_cb:  bool  = False


def _resolve_weights(
    user: UserProfile,
    gamma: float,
    delta: float,
    alpha: float,
) -> tuple[float, float, float, float]:
    """
    Resolve final weights for (alpha, beta, gamma, delta).

    CF weight (gamma) is always active — cold start provides item-based
    CF scores so it is never truly zero. CB weight (delta) is zeroed
    when the CB model is not trained, and redistributed to alpha and gamma.
    """
    eff_delta = delta if user.has_cb else 0.0
    reclaim   = delta - eff_delta

    # Redistribute unused CB weight: half to CF, half to expiry urgency
    eff_gamma = gamma + reclaim / 2
    eff_alpha = alpha + reclaim / 2
    eff_beta  = user.beta

    total = eff_gamma + eff_delta + eff_alpha + eff_beta
    if total == 0:
        return (0.25, 0.25, 0.25, 0.25)

    return (
        eff_alpha / total,
        eff_beta  / total,
        eff_gamma / total,
        eff_delta / total,
    )


MMR_LAMBDA: float = 0.7   # relevance vs. diversity trade-off for MMR


def _calibrate(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0,1]."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _ingredient_jaccard(a: "RecipeScore", b: "RecipeScore") -> float:
    """Ingredient Jaccard similarity for MMR diversity penalty."""
    set_a = set(a.matched_ingredients) | set(a.missing_ingredients)
    set_b = set(b.matched_ingredients) | set(b.missing_ingredients)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _mmr_rerank(
    candidates: list["RecipeScore"], top_n: int, lambda_: float = MMR_LAMBDA
) -> list["RecipeScore"]:
    """
    Maximal Marginal Relevance reranking.
    Selects top_n from candidates balancing relevance and ingredient diversity.
    """
    if len(candidates) <= top_n:
        return candidates
    selected: list[RecipeScore] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda r: r.final_score)
        else:
            def mmr_score(r: RecipeScore, sel: list[RecipeScore] = selected) -> float:
                max_sim = max(_ingredient_jaccard(r, s) for s in sel)
                return lambda_ * r.final_score - (1.0 - lambda_) * max_sim
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def score_recipe(
    recipe: dict,
    urgency_map: dict[str, float],
    pantry_ingredients: list[str],
    user: UserProfile,
    cf_scores: Optional[dict[int, float]] = None,
    cb_scores: Optional[dict[int, float]] = None,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    delta: float = DEFAULT_DELTA,
) -> RecipeScore:
    """
    Score a single (user, recipe) pair.

    Architecture: CF predicts base preference; domain adjustments
    (expiry + availability) modify the final ranking.
    """
    recipe_id   = recipe["id"]
    recipe_name = recipe.get("name", f"Recipe {recipe_id}")
    ingredients = recipe.get("ingredients", [])

    # Domain adjustments
    match_result = match_ingredients(ingredients, pantry_ingredients)
    match_ratio  = match_result["match_ratio"]
    expiry_score = expiry_weighted_match(ingredients, urgency_map)

    # CF base score (item-based cold start or SVD — never truly missing)
    cf_score = (cf_scores or {}).get(recipe_id, 0.0)

    # CB ingredient profile boost
    cb_score = (cb_scores or {}).get(recipe_id, 0.0)

    # Resolve weights
    w_alpha, w_beta, w_gamma, w_delta = _resolve_weights(
        user, gamma, delta, alpha
    )

    # CF-first: gamma carries the base prediction weight
    final = (
        w_gamma * cf_score      # CF base
        + w_delta * cb_score    # CB boost
        + w_alpha * expiry_score  # domain: waste
        + w_beta  * match_ratio   # domain: availability
    )

    return RecipeScore(
        recipe_id=recipe_id,
        recipe_name=recipe_name,
        final_score=round(final, 6),
        expiry_urgency=round(expiry_score, 4),
        match_ratio=round(match_ratio, 4),
        cf_score=round(cf_score, 4),
        cb_score=round(cb_score, 4),
        matched_ingredients=match_result["matched"],
        missing_ingredients=match_result["missing"],
        total_ingredients=match_result["total"],
    )


def rank_recipes(
    pantry_items: list[dict],
    recipes: list[dict],
    user_profile: Optional[dict] = None,
    cf_scores: Optional[dict[int, float]] = None,
    cb_scores: Optional[dict[int, float]] = None,
    top_n: int = 20,
) -> list[RecipeScore]:
    """
    Rank recipes for a user given their current pantry.

    Called by the FastAPI /recipes/ranked endpoint.

    CF is always active (cold-start item-based CF when SVD unavailable).
    CB is optional — omitting it redistributes its weight to CF and expiry.

    Args:
        pantry_items:   [{"ingredient": str, "expiry_date": "YYYY-MM-DD"}]
        recipes:        [{"id": int, "name": str, "ingredients": [str]}]
        user_profile:   {"user_id", "beta", "has_cf", "has_cb"}
        cf_scores:      recipe_id -> CF score [0,1]
        cb_scores:      recipe_id -> CB similarity [0,1]
        top_n:          max results

    Returns:
        List of RecipeScore sorted by final_score descending
    """
    p = user_profile or {}
    user = UserProfile(
        user_id = p.get("user_id", 0),
        beta    = p.get("beta", DEFAULT_BETA),
        has_cf  = p.get("has_cf", False),
        has_cb  = p.get("has_cb", False),
    )

    urgency_map        = pantry_urgency_map(pantry_items)
    pantry_ingredients = [item["ingredient"] for item in pantry_items]

    scored = [
        score_recipe(r, urgency_map, pantry_ingredients, user,
                     cf_scores, cb_scores)
        for r in recipes
    ]

    # Score calibration: min-max normalize each component across candidates
    # so no single dimension dominates due to scale differences.
    if len(scored) > 1:
        cal_cf      = _calibrate([s.cf_score      for s in scored])
        cal_cb      = _calibrate([s.cb_score      for s in scored])
        cal_expiry  = _calibrate([s.expiry_urgency for s in scored])
        cal_match   = _calibrate([s.match_ratio   for s in scored])

        # Recompute final_score with calibrated components using the same weights
        w_alpha, w_beta, w_gamma, w_delta = _resolve_weights(
            user, DEFAULT_GAMMA, DEFAULT_DELTA, DEFAULT_ALPHA
        )
        for i, s in enumerate(scored):
            s.final_score = round(
                w_gamma * cal_cf[i]
                + w_delta * cal_cb[i]
                + w_alpha * cal_expiry[i]
                + w_beta  * cal_match[i],
                6,
            )

    scored.sort(key=lambda s: s.final_score, reverse=True)

    if top_n <= 0:
        return scored

    # MMR reranking over top 3×top_n candidates for ingredient diversity
    pool = scored[: top_n * 3]
    return _mmr_rerank(pool, top_n)
