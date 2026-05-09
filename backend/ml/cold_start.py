"""
cold_start.py
-------------
Personalized cold-start for users with no rating history.

THIS IS STILL COLLABORATIVE FILTERING
--------------------------------------
A common misunderstanding: cold start using content signals looks like
content-based filtering. It is not. Here is the distinction:

    Content-based (CB):
        Similarity based on item attributes (ingredients, tags).
        Does NOT use any user interaction data.

    This cold-start (CF):
        Similarity matrix derived entirely from USER INTERACTION DATA
        (Food.com ratings). sim(i,j) = cosine(R_T[i], R_T[j]) where
        R_T is the mean-centered user×recipe rating matrix.
        Seeds act as PSEUDO-INTERACTIONS — inferred preference anchors.
        No ingredient or tag semantics enter the similarity computation.

Therefore: this is Collaborative Filtering with inferred preferences
rather than explicit ratings. The community's co-rating patterns drive
recommendations; we simply substitute a preference-inferred anchor set
for the user's missing rating history.

PREDICTION TARGET
-----------------
Cold-start CF estimates:
    P(user will enjoy recipe | similar_users_enjoyed_similar_recipes,
                               user_diet_tags, user_pantry)

THE ALGORITHM
-------------
Step 1 — Seed selection (preference inference)
    Use diet_tags + pantry as a preference proxy to select N_SEEDS
    anchor recipes without relying on rating history:

        seed_score(r) = tag_weight · tag_match(r, user_tags)
                      + pantry_weight · pantry_overlap(r, pantry)

Step 2 — Seed diversification
    Cap each primary cuisine/tag at max_per_tag seeds.
    Prevents all 30 seeds being Italian when the user likes Italian —
    diversity in the seed set produces diversity in recommendations.
    (Inspired by MMR — Maximal Marginal Relevance)

Step 3 — Item-CF scoring
    For each candidate recipe:
        score(r) = mean cosine_similarity(r, seed_i) for seed_i in seeds

    This is item-based CF: sim(i,j) from the rating matrix, not content.

Step 4 — Automatic transition to SVD
    At MIN_RATINGS_FOR_CF ratings, serve_cf.py switches to SVD.
    The cold-start scaffold is temporary — it dissolves as ratings arrive.

DATA SPARSITY NOTE
------------------
The Food.com matrix is ~99% empty. Item-based CF handles this better
than user-based CF at cold-start edges because item similarity vectors
are denser (recipes have more ratings than most new users).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import Optional
from backend.services.ingredient_match import ingredient_matches as _ingredient_matches

N_SEEDS            = 30
MIN_TAG_MATCH      = 0.5
PANTRY_SEED_WEIGHT = 0.4


def _tag_match_score(recipe_tags: set[str], user_tags: set[str]) -> float:
    if not user_tags:
        return 1.0
    matched = len(recipe_tags & user_tags)
    return matched / len(user_tags)


def _pantry_match_score(
    recipe_ingredients: list[str],
    pantry_ingredients: set[str],
) -> float:
    """
    Fraction of recipe ingredients covered by the user's pantry.

    Uses the same ingredient_matches() function as the main scoring pipeline
    (three-pass: exact, head-noun word-boundary, token_set_ratio) — consistent
    matching across cold-start seed selection and live ranking.

    Replaces the previous naive substring check (any(p in ing or ing in p))
    which caused false positives like "corn" matching "peppercorns".
    """
    if not pantry_ingredients or not recipe_ingredients:
        return 0.0
    matched = sum(
        1 for ing in recipe_ingredients
        if any(_ingredient_matches(p, ing) for p in pantry_ingredients)
    )
    return matched / len(recipe_ingredients)


def select_seeds(
    all_recipes: list[dict],
    user_diet_tags: list[str],
    pantry_ingredients: list[str],
    n_seeds: int = N_SEEDS,
    pantry_weight: float = PANTRY_SEED_WEIGHT,
) -> list[int]:
    """
    Select seed recipe IDs representing this user's inferred preferences.

    Seeds replace explicit ratings as the CF anchor set.
    A vegetarian user with eggs and milk gets vegetarian recipes featuring
    those ingredients as seeds — different seeds than a vegan user or a
    meat-eater, producing different item-CF scores downstream.
    """
    user_tags  = {t.strip().lower() for t in user_diet_tags if t.strip()}
    pantry_set = {p.strip().lower() for p in pantry_ingredients if p.strip()}
    tag_weight = 1.0 - pantry_weight

    scored = []
    for recipe in all_recipes:
        r_tags = {t.strip().lower()
                  for t in (recipe.get("tags") or []) if t.strip()}
        r_ings = [i.lower() for i in (recipe.get("ingredients") or [])]

        tag_score    = _tag_match_score(r_tags, user_tags)
        pantry_score = _pantry_match_score(r_ings, pantry_set)

        if user_tags and tag_score == 0.0:
            continue  # hard filter: must satisfy at least one diet tag

        combined = tag_weight * tag_score + pantry_weight * pantry_score
        scored.append((recipe["id"], combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in scored[:n_seeds]]


def diversify_seeds(
    seed_ids: list[int],
    seed_recipes: list[dict],
    max_per_tag: int = 5,
) -> list[int]:
    """
    Cap seeds per primary cuisine/tag to prevent echo-chamber cold start.

    Without diversification, all 30 seeds might be Italian — producing
    a feed of exclusively Italian recommendations. Capping ensures the
    item-CF scores reflect the user's full taste profile.
    """
    id_to_recipe  = {r["id"]: r for r in seed_recipes}
    tag_counts: dict[str, int] = {}
    diversified   = []

    for sid in seed_ids:
        recipe  = id_to_recipe.get(sid)
        if recipe is None:
            diversified.append(sid)
            continue
        tags    = [t.strip().lower()
                   for t in (recipe.get("tags") or []) if t.strip()]
        primary = tags[0] if tags else "untagged"
        if tag_counts.get(primary, 0) < max_per_tag:
            tag_counts[primary] = tag_counts.get(primary, 0) + 1
            diversified.append(sid)

    return diversified


def cold_start_cf_scores(
    candidate_recipe_ids: list[int],
    seed_recipe_ids: list[int],
    sim_matrix: sp.csr_matrix,
    sim_recipe_ids: np.ndarray,
) -> dict[int, float]:
    """
    Score candidates using item-based CF anchored on the seed set.

        score(candidate) = mean cosine_similarity(candidate, seed_i)
                           for seed_i in seed_recipe_ids

    The similarity matrix was built from USER RATINGS (not content).
    This is item-based CF — the same formula from the course:
        sim(i,j) = cos(R_T[i], R_T[j]) = (R_T[i]·R_T[j]) / (‖R_T[i]‖·‖R_T[j]‖)
    """
    if sim_matrix is None or len(seed_recipe_ids) == 0:
        return {rid: 0.0 for rid in candidate_recipe_ids}

    id_to_row = {int(rid): i for i, rid in enumerate(sim_recipe_ids)}
    seed_rows = [id_to_row[sid] for sid in seed_recipe_ids if sid in id_to_row]

    if not seed_rows:
        return {rid: 0.0 for rid in candidate_recipe_ids}

    scores = {}
    for rid in candidate_recipe_ids:
        row_idx = id_to_row.get(rid)
        if row_idx is None:
            scores[rid] = 0.0
            continue
        sim_values = [float(sim_matrix[row_idx, sr]) for sr in seed_rows]
        scores[rid] = round(float(np.mean(sim_values)), 6)

    max_s = max(scores.values()) if scores else 0.0
    if max_s > 0:
        scores = {rid: round(s / max_s, 6) for rid, s in scores.items()}

    return scores


def _fallback_preference_scores(
    candidate_recipe_ids: list[int],
    all_recipes: list[dict],
    user_diet_tags: list[str],
    pantry_ingredients: list[str],
    pantry_weight: float = PANTRY_SEED_WEIGHT,
) -> dict[int, float]:
    """
    Tag+pantry preference scores when no CF sim_matrix is trained yet.
    Normalizes to [0,1] so scores slot into the same γ weight as trained CF.
    """
    user_tags  = {t.strip().lower() for t in user_diet_tags if t.strip()}
    pantry_set = {p.strip().lower() for p in pantry_ingredients if p.strip()}
    tag_weight = 1.0 - pantry_weight

    id_to_recipe = {r["id"]: r for r in all_recipes}
    scores: dict[int, float] = {}

    for rid in candidate_recipe_ids:
        recipe = id_to_recipe.get(rid)
        if recipe is None:
            scores[rid] = 0.0
            continue
        r_tags = {t.strip().lower() for t in (recipe.get("tags") or []) if t.strip()}
        r_ings = [i.lower() for i in (recipe.get("ingredients") or [])]
        tag_score    = _tag_match_score(r_tags, user_tags)
        pantry_score = _pantry_match_score(r_ings, pantry_set)
        scores[rid]  = tag_weight * tag_score + pantry_weight * pantry_score

    # No normalization here — raw scores are already in [0,1] by construction
    # (tag_weight * tag_fraction + pantry_weight * pantry_fraction).
    # Normalizing to [0,1] would inflate all values to near 1.0, making CF
    # dominate the final ranking and drowning out expiry urgency + match_ratio.
    return {rid: round(s, 6) for rid, s in scores.items()}


def personalized_cold_start(
    candidate_recipe_ids: list[int],
    all_recipes: list[dict],
    user_diet_tags: list[str],
    pantry_ingredients: list[str],
    sim_matrix: Optional[sp.csr_matrix],
    sim_recipe_ids: Optional[np.ndarray],
    n_seeds: int = N_SEEDS,
) -> dict[int, float]:
    """
    Full cold-start CF pipeline. Entry point called by serve_cf.py.

    1. Select seeds from all_recipes using diet_tags + pantry
    2. Diversify seeds across cuisines (MMR-inspired)
    3. Score candidates by mean item-CF similarity to seeds

    Returns dict recipe_id -> CF score [0,1].
    Falls back to tag+pantry preference scores if sim_matrix unavailable.
    """
    if sim_matrix is None or sim_recipe_ids is None:
        return _fallback_preference_scores(
            candidate_recipe_ids, all_recipes, user_diet_tags, pantry_ingredients
        )

    seeds = select_seeds(
        all_recipes=all_recipes,
        user_diet_tags=user_diet_tags,
        pantry_ingredients=pantry_ingredients,
        n_seeds=n_seeds * 2,
    )
    seed_recipes = [r for r in all_recipes if r["id"] in set(seeds)]
    seeds        = diversify_seeds(seeds, seed_recipes)[:n_seeds]

    if not seeds:
        seeds = [int(sim_recipe_ids[i])
                 for i in range(min(n_seeds, len(sim_recipe_ids)))]

    return cold_start_cf_scores(
        candidate_recipe_ids=candidate_recipe_ids,
        seed_recipe_ids=seeds,
        sim_matrix=sim_matrix,
        sim_recipe_ids=sim_recipe_ids,
    )
