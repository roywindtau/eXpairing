"""
recipes.py
----------
Main recommendation endpoint.

CANDIDATE GENERATION (two-stage pipeline)
-----------------------------------------
Recommendation at scale follows a two-stage pattern:
    Stage 1 — Candidate generation: reduce 230k recipes to ~2000
    Stage 2 — Ranking: score and rank the 2000 candidates

Stage 1 filters applied in order:
    1. Diet tag filter   (hard constraint — user cannot eat these)
    2. Popularity cap    (limit to 2000 highest-rated candidates)

This is intentional system design — scoring 230k recipes per request
in real-time is infeasible. The candidate set captures the relevant
search space while keeping latency acceptable.

RANKING (CF-first)
------------------
    final_score = γ·CF(user,recipe)    ← base preference prediction
                + δ·CB(pantry,recipe)  ← ingredient profile boost
                + α·expiry_urgency     ← domain: waste minimization
                + β·match_ratio        ← domain: availability

CF strategy soft-blended by rating count:
    n_ratings = 0       →  item-based cold start CF (preference seeds)
    0 < n_ratings < 5   →  blend: (1-α)·cold_start + α·SVD, α = n/5
    n_ratings ≥ 5       →  SVD matrix factorization (fully personalized)

DIVERSITY
---------
To prevent repetitive recommendations (e.g. all pasta dishes):
    - Candidate set is pre-filtered by diet tags
    - Top-N cap limits the feed to 20 recipes
    - Future: MMR (Maximal Marginal Relevance) for active diversity
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User, PantryItem, Recipe, UserEvent
from backend.services.scoring import rank_recipes, RecipeScore
from backend.ml.serve_cf import (
    get_cf_scores, cf_strategy_name, is_warm_user,
    svd_available, item_sim_available, MIN_RATINGS_FOR_CF,
)
from backend.ml.serve_cb import (
    cb_similarity_batch, cb_taste_profile_batch, model_available as cb_available,
)

router = APIRouter(tags=["recipes"])

# ── tag helpers ────────────────────────────────────────────────────────────

# Food.com uses these as namespace / category headers, not as recipe attributes.
# They describe the *type* of a nearby tag, not the recipe itself.
_STRUCTURAL_TAGS = frozenset({
    'time-to-make', 'course', 'main-ingredient', 'cuisine',
    'preparation', 'occasion', 'equipment', 'dietary',
    'presentation', 'served-cold', 'served-hot',
    'number-of-servings', 'low-in-something', 'healthy-2',
})


def _clean_tags(tags_csv: str | None) -> list[str]:
    if not tags_csv:
        return []
    return [t.strip() for t in tags_csv.split(',')
            if t.strip() and t.strip() not in _STRUCTURAL_TAGS]


# ── response schemas ───────────────────────────────────────────────────────

class RecipeScoreOut(BaseModel):
    recipe_id:           int
    recipe_name:         str
    final_score:         float
    match_ratio:         float
    expiry_urgency:      float
    cf_score:            float
    cb_score:            float
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    total_ingredients:   int
    tags:                list[str]
    minutes:             int | None
    avg_rating:          float | None
    cf_strategy:         str   # "biased_mf" | "item_based_cold_start" | "blended" | "none"
    cb_model_available:  bool  # True when CB model is loaded and pantry is non-empty


class EventIn(BaseModel):
    user_id:    int
    recipe_id:  int
    event_type: str          # "cook" | "skip" | "rate"
    rating:     float | None = None
    n_missing:  int   | None = None


# ── helpers ────────────────────────────────────────────────────────────────

def _count_user_ratings(user_id: int, db: Session) -> int:
    return (
        db.query(UserEvent)
        .filter(UserEvent.user_id == user_id,
                UserEvent.event_type == "rate")
        .count()
    )


def _pantry_dicts(items: list[PantryItem]) -> list[dict]:
    return [{"ingredient": i.ingredient,
             "expiry_date": i.expiry_date.isoformat()} for i in items]


def _recipe_dicts(recipes: list[Recipe]) -> list[dict]:
    return [{"id": r.id, "name": r.name, "ingredients": r.ingredients}
            for r in recipes]


def _to_out(score: RecipeScore, recipe: Recipe,
            cf_strategy: str, cb_model_available: bool) -> RecipeScoreOut:
    return RecipeScoreOut(
        recipe_id=score.recipe_id,
        recipe_name=score.recipe_name,
        final_score=round(score.final_score, 4),
        match_ratio=round(score.match_ratio, 4),
        expiry_urgency=round(score.expiry_urgency, 4),
        cf_score=round(score.cf_score, 4),
        cb_score=round(score.cb_score, 4),
        matched_ingredients=score.matched_ingredients,
        missing_ingredients=score.missing_ingredients,
        total_ingredients=score.total_ingredients,
        tags=_clean_tags(recipe.tags_csv),
        minutes=recipe.minutes,
        avg_rating=recipe.avg_rating,
        cf_strategy=cf_strategy,
        cb_model_available=cb_model_available,
    )


# ── GET /recipes/ranked ────────────────────────────────────────────────────

@router.get("/recipes/ranked", response_model=list[RecipeScoreOut])
def get_ranked_recipes(
    user_id: int = Query(...),
    top_n:   int = Query(20),
    db: Session = Depends(get_db),
):
    """
    Two-stage recommendation pipeline:

    Stage 1 — Candidate generation:
        Filter by diet_tags → cap at 2000 recipes
        (reduces 230k → ~2000 candidates before scoring)

    Stage 2 — Ranking:
        Score each candidate with CF-first formula.
        CF strategy auto-selected: cold-start or SVD.
        Return top_n results with full score breakdown.
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Load pantry
    pantry_items = (
        db.query(PantryItem)
        .filter(PantryItem.user_id == user_id)
        .order_by(PantryItem.expiry_date)
        .all()
    )

    # ── Stage 1: Candidate generation ──────────────────────────────────────
    # Filter 1: diet tag hard constraints
    # Applied iteratively — if all tags together return zero results
    # (can happen on small dev corpus), we progressively relax until we
    # have candidates. This prevents an empty feed on restrictive tag combos.
    diet_tags = [t.strip() for t in (user.diet_tags or "").split(",") if t.strip()]

    def _query_with_tags(tags: list[str]) -> list:
        q = db.query(Recipe)
        for tag in tags:
            q = q.filter(Recipe.tags_csv.ilike(f"%{tag}%"))
        # Bayesian-smoothed rating: (avg * n + prior * weight) / (n + weight)
        # Prevents single 5★ reviews from dominating candidate selection.
        # Recipes with ≥5 ratings (= in item-sim matrix) naturally float to top.
        bayesian = (
            (func.coalesce(Recipe.avg_rating, 0) * Recipe.n_ratings + 3.5 * 5)
            / (Recipe.n_ratings + 5)
        )
        return (q.order_by(bayesian.desc())
                 .limit(2000)
                 .all())

    recipes = _query_with_tags(diet_tags)

    # Graceful fallback: relax tags one by one if corpus too small
    remaining_tags = list(diet_tags)
    while not recipes and remaining_tags:
        remaining_tags = remaining_tags[:-1]  # drop the most restrictive tag
        recipes = _query_with_tags(remaining_tags)

    # Final fallback: unfiltered top-2000 by Bayesian score
    if not recipes:
        bayesian = (
            (func.coalesce(Recipe.avg_rating, 0) * Recipe.n_ratings + 3.5 * 5)
            / (Recipe.n_ratings + 5)
        )
        recipes = (db.query(Recipe)
                   .order_by(bayesian.desc())
                   .limit(2000)
                   .all())

    if not recipes:
        return []

    # Skip exclusion: hide recipes the user dismissed in the last 7 days
    skip_cutoff = datetime.now() - timedelta(days=7)
    skipped_ids = {
        row[0]
        for row in db.query(UserEvent.recipe_id)
        .filter(
            UserEvent.user_id    == user_id,
            UserEvent.event_type == "skip",
            UserEvent.created_at >= skip_cutoff,
        )
        .all()
    }
    if skipped_ids:
        recipes = [r for r in recipes if r.id not in skipped_ids]

    if not recipes:
        return []

    recipe_ids              = [r.id for r in recipes]
    pantry_ingredient_names = [item.ingredient for item in pantry_items]

    # ── Stage 2: CF scoring — soft blend of cold start and SVD ────────────
    n_ratings = _count_user_ratings(user_id, db)

    user_diet_tags = [t.strip() for t in (user.diet_tags or "").split(",")
                      if t.strip()]
    all_recipes_for_seeds = [
        {
            "id":          r.id,
            "ingredients": r.ingredients,
            "tags":        _clean_tags(r.tags_csv),
        }
        for r in recipes
    ]

    # Blend weight alpha = min(n_ratings / threshold, 1.0)
    # 0 ratings → pure cold start; threshold+ ratings → pure SVD; in between → both
    cf_scores = get_cf_scores(
        user_id=user_id,
        recipe_ids=recipe_ids,
        n_user_ratings=n_ratings,
        user_diet_tags=user_diet_tags,
        pantry_ingredients=pantry_ingredient_names,
        all_recipes=all_recipes_for_seeds,
    )
    cf_strategy = cf_strategy_name(n_ratings)

    # CB scores — taste profile for warm users, pantry profile for cold start
    use_cb = cb_available()
    if use_cb and is_warm_user(n_ratings):
        # Warm user: build taste profile from rated recipes
        rated_rows = (
            db.query(UserEvent.recipe_id, UserEvent.rating)
            .filter(UserEvent.user_id    == user_id,
                    UserEvent.event_type == "rate",
                    UserEvent.rating.isnot(None))
            .all()
        )
        rated_ids  = [row[0] for row in rated_rows]
        rated_vals = [float(row[1]) for row in rated_rows]
        cb_scores = cb_taste_profile_batch(rated_ids, rated_vals, recipe_ids)
    elif use_cb and bool(pantry_ingredient_names):
        # Cold-start user: use pantry as content proxy
        cb_scores = cb_similarity_batch(pantry_ingredient_names, recipe_ids)
    else:
        cb_scores = None

    # Auto-promote user to warm CF when threshold crossed
    warm = is_warm_user(n_ratings) and svd_available()
    if warm and not user.has_cf:
        user.has_cf = True
        db.commit()

    user_profile = {
        "user_id": user.id,
        "beta":    user.beta,
        "has_cf":  warm,
        "has_cb":  use_cb,
    }

    ranked: list[RecipeScore] = rank_recipes(
        pantry_items=_pantry_dicts(pantry_items),
        recipes=_recipe_dicts(recipes),
        user_profile=user_profile,
        cf_scores=cf_scores,
        cb_scores=cb_scores,
        top_n=top_n,
    )

    recipe_map = {r.id: r for r in recipes}
    return [
        _to_out(score, recipe_map[score.recipe_id], cf_strategy, use_cb)
        for score in ranked
        if score.recipe_id in recipe_map
    ]


# ── GET /recipes/search ────────────────────────────────────────────────────

@router.get("/recipes/search")
def search_recipes(
    q:     str = Query("", description="Search term (name or ingredient)"),
    tag:   str = Query("", description="Filter by tag e.g. vegetarian"),
    limit: int = Query(40),
    db: Session = Depends(get_db),
):
    """Browse all recipes with text search and tag filter. Not personalized."""
    query = db.query(Recipe)
    if q.strip():
        term  = f"%{q.strip().lower()}%"
        query = query.filter(
            Recipe.name.ilike(term) | Recipe.ingredients_csv.ilike(term)
        )
    if tag.strip():
        query = query.filter(Recipe.tags_csv.ilike(f"%{tag.strip()}%"))

    recipes = (query
               .order_by(Recipe.avg_rating.desc().nullslast())
               .limit(limit)
               .all())
    return [
        {
            "id":          r.id,
            "name":        r.name,
            "ingredients": r.ingredients[:8],
            "tags":        _clean_tags(r.tags_csv)[:6],
            "minutes":     r.minutes,
            "avg_rating":  r.avg_rating,
            "n_ratings":   r.n_ratings,
        }
        for r in recipes
    ]


# ── GET /recipes/{recipe_id} ───────────────────────────────────────────────

@router.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {
        "id":          recipe.id,
        "name":        recipe.name,
        "ingredients": recipe.ingredients,
        "tags":        _clean_tags(recipe.tags_csv),
        "minutes":     recipe.minutes,
        "n_steps":     recipe.n_steps,
        "avg_rating":  recipe.avg_rating,
        "n_ratings":   recipe.n_ratings,
        "description": recipe.description,
        "steps":       recipe.steps,
    }


# ── POST /events ───────────────────────────────────────────────────────────

@router.post("/events", status_code=201)
def log_event(payload: EventIn, db: Session = Depends(get_db)):
    """
    Log a user interaction. Three event types:

    "cook"  — implicit signal: user cooked this recipe.
              n_missing logged for beta_updater (revealed preference).
    "skip"  — implicit signal: user dismissed recipe from feed.
    "rate"  — explicit signal: star rating 1-5.
              Feeds SVD training data for collaborative filtering.
    """
    if payload.event_type not in ("cook", "skip", "rate"):
        raise HTTPException(status_code=422,
                            detail="event_type must be cook, skip, or rate")
    if payload.event_type == "rate" and payload.rating is None:
        raise HTTPException(status_code=422,
                            detail="rating required when event_type is rate")

    event = UserEvent(
        user_id=payload.user_id,
        recipe_id=payload.recipe_id,
        event_type=payload.event_type,
        rating=payload.rating,
        n_missing=payload.n_missing,
    )
    db.add(event)
    db.commit()

    return {"status": "ok", "event_id": event.id}
