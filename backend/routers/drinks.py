"""
drinks.py
---------
HTTP API for wine recommendations + browsing + event logging.

(The module keeps its `drinks` route prefix as a stable feature name, but the
catalog is wine-only — there is no longer a `kind` discriminator.)

Architecture
------------
Mirrors the two-stage pipeline used in routers/recipes.py:

    Stage 1 — Candidate generation
        SQL order by Bayesian-smoothed popularity, cap at 2000 candidates.

    Stage 2 — Ranking
        Compute CB + CF + expert (Path A only) + popularity prior,
        min-max calibrate across the pool, blend per Path-A or Path-B
        weights, return top_n.

Routes
------
    GET  /drinks/ranked               Path B  ("Drinks For You")
    GET  /drinks/pairings/{recipe_id} Path A  (pair with a recipe)
    GET  /drinks/search               browse/search
    GET  /drinks/{drink_id}           detail
    POST /drink-events                rate a wine
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Wine, WineEvent, Recipe, User
from backend.ml.wine.serving.serve_cb import cb_for_recipe, cb_for_user, model_available as cb_available
from backend.ml.wine.serving.serve_cf import cf_strategy_name, get_cf_scores
from backend.services.wine.scoring import (
    WineScore,
    rank_wines_for_recipe,
    rank_wines_for_user,
)
from backend.services.wine.expert_pairing import expert_boost_batch

router = APIRouter(tags=["drinks"])

CANDIDATE_POOL_SIZE = 2000   # Stage-1 cap, mirrors recipes router


# ── pydantic schemas ────────────────────────────────────────────────────

class DrinkScoreOut(BaseModel):
    drink_id:     int
    drink_name:   str
    kind:         str = "wine"
    final_score:  float
    cb_score:     float
    cf_score:     float
    expert_boost: float
    prior_score:  float
    cf_strategy:  str
    # metadata
    avg_rating:   Optional[float] = None
    n_ratings:    int = 0
    abv:          Optional[float] = None
    producer:     Optional[str] = None
    # wine attributes
    style:         Optional[str] = None
    wine_type:     Optional[str] = None
    variety:       Optional[str] = None
    harmonize_csv: Optional[str] = None


class DrinkEventIn(BaseModel):
    user_id:    int
    drink_id:   int
    event_type: str   # v1: "rate"
    rating:     Optional[float] = None


# ── helpers ─────────────────────────────────────────────────────────────

def _candidates(db: Session, limit: int) -> list[Wine]:
    """Stage 1: top-N wines by Bayesian-smoothed popularity."""
    # bayesian = (n*avg + C*prior) / (n + C), with C=5, prior=3.5
    bayesian = (
        (Wine.avg_rating * Wine.n_ratings + 3.5 * 5)
        / (Wine.n_ratings + 5)
    )
    return (
        db.query(Wine)
          .order_by(bayesian.desc().nullslast())
          .limit(limit)
          .all()
    )


def _to_out(s: WineScore, wine: Wine) -> DrinkScoreOut:
    return DrinkScoreOut(
        drink_id=s.wine_id,
        drink_name=s.wine_name,
        final_score=round(s.final_score, 4),
        cb_score=round(s.cb_score, 4),
        cf_score=round(s.cf_score, 4),
        expert_boost=round(s.expert_boost, 4),
        prior_score=round(s.prior_score, 4),
        cf_strategy=s.cf_strategy,
        avg_rating=wine.avg_rating,
        n_ratings=wine.n_ratings or 0,
        abv=wine.abv,
        producer=wine.producer,
        style=wine.style,
        wine_type=wine.style,
        variety=wine.grapes_csv,
        harmonize_csv=wine.harmonize_csv,
    )


def _count_user_explicit_wine_ratings(db: Session, user_id: int) -> int:
    return (
        db.query(WineEvent)
        .filter(WineEvent.user_id    == user_id)
        .filter(WineEvent.event_type == "rate")
        .filter(WineEvent.rating.isnot(None))
        .filter(WineEvent.synthetic  == False)  # noqa: E712
        .count()
    )


# ── GET /drinks/ranked  (Path B) ────────────────────────────────────────

@router.get("/drinks/ranked", response_model=list[DrinkScoreOut])
def get_ranked_drinks(
    user_id: int = Query(...),
    top_n:   int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Path B — "Drinks For You". Ranks wines by:
      - CB score from the user's RECIPE rating history (via flavor_bridge)
      - CF score routed by wine rating count (popularity / item-sim)
      - Popularity prior tiebreaker

    No expert boost (no specific recipe to pair against).
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail=f"User {user_id} not found")

    candidates = _candidates(db, CANDIDATE_POOL_SIZE)
    if not candidates:
        return []

    # CB: from user's recipe history (aggregated bridged docs)
    cb_scores: dict[int, float] = {}
    if cb_available():
        cb_scores = cb_for_user(user_id, db)

    # CF: standard get_cf_scores routing
    cf_scores = get_cf_scores(user_id, [w.id for w in candidates], db)
    n_explicit = _count_user_explicit_wine_ratings(db, user_id)
    cf_strategies = {w.id: cf_strategy_name(n_explicit) for w in candidates}

    ranked = rank_wines_for_user(
        candidates=candidates,
        cb_scores=cb_scores,
        cf_scores=cf_scores,
        cf_strategies=cf_strategies,
        top_n=top_n,
    )

    wine_map = {w.id: w for w in candidates}
    return [_to_out(s, wine_map[s.wine_id]) for s in ranked]


# ── GET /drinks/pairings/{recipe_id}  (Path A) ──────────────────────────

@router.get("/drinks/pairings/{recipe_id}", response_model=list[DrinkScoreOut])
def get_drink_pairings(
    recipe_id: int,
    user_id:   int = Query(...),
    top_n:     int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Path A — given a specific recipe, suggest wines. Adds the expert-rules
    boost (Harmonize match) on top of CB + CF.
    """
    recipe = db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(404, detail=f"Recipe {recipe_id} not found")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail=f"User {user_id} not found")

    candidates = _candidates(db, CANDIDATE_POOL_SIZE)
    if not candidates:
        return []

    # CB: vs recipe
    cb_scores = cb_for_recipe(recipe) if cb_available() else {}

    # CF: standard routing
    cf_scores = get_cf_scores(user_id, [w.id for w in candidates], db)
    n_explicit = _count_user_explicit_wine_ratings(db, user_id)
    cf_strategies = {w.id: cf_strategy_name(n_explicit) for w in candidates}

    # Expert: Harmonize match
    expert = expert_boost_batch(recipe, candidates)

    ranked = rank_wines_for_recipe(
        recipe=recipe,
        candidates=candidates,
        cb_scores=cb_scores,
        cf_scores=cf_scores,
        expert_boosts=expert,
        cf_strategies=cf_strategies,
        top_n=top_n,
    )

    wine_map = {w.id: w for w in candidates}
    return [_to_out(s, wine_map[s.wine_id]) for s in ranked]


# ── GET /drinks/search ──────────────────────────────────────────────────

@router.get("/drinks/search")
def search_drinks(
    q:     str = Query("", description="Search term (name or style/variety)"),
    limit: int = Query(40, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Browse wines with text search. Not personalized."""
    query = db.query(Wine)
    if q.strip():
        term = f"%{q.strip().lower()}%"
        query = query.filter(
            Wine.name.ilike(term)
            | Wine.style.ilike(term)
            | Wine.grapes_csv.ilike(term)
        )
    rows = (
        query.order_by(Wine.avg_rating.desc().nullslast())
             .limit(limit)
             .all()
    )
    return [
        {
            "id":            w.id,
            "name":          w.name,
            "kind":          "wine",
            "style":         w.style,
            "harmonize_csv": w.harmonize_csv,
            "producer":      w.producer,
            "abv":           w.abv,
            "avg_rating":    w.avg_rating,
            "n_ratings":     w.n_ratings,
        }
        for w in rows
    ]


# ── GET /drinks/{drink_id} ──────────────────────────────────────────────

@router.get("/drinks/{drink_id}")
def get_drink_detail(drink_id: int, db: Session = Depends(get_db)):
    wine = db.get(Wine, drink_id)
    if not wine:
        raise HTTPException(404, detail="Wine not found")
    return {
        "id":              wine.id,
        "name":            wine.name,
        "kind":            "wine",
        "producer":        wine.producer,
        "country":         wine.country,
        "abv":             wine.abv,
        "avg_rating":      wine.avg_rating,
        "n_ratings":       wine.n_ratings,
        "style":           wine.style,
        # wine-specific
        "grapes_csv":      wine.grapes_csv,
        "region":          wine.region,
        "body":            wine.body,
        "acidity":         wine.acidity,
        "harmonize_csv":   wine.harmonize_csv,
    }


# ── POST /drink-events ──────────────────────────────────────────────────

@router.post("/drink-events", status_code=201)
def log_drink_event(payload: DrinkEventIn, db: Session = Depends(get_db)):
    """
    Record a wine rating. v1 supports only event_type='rate' with a rating.

    Deliberately NO synthesizer hook here — only RECIPE ratings trigger
    wine synthesis. Wine ratings are the user's explicit signal and feed
    the item-sim path directly.
    """
    if payload.event_type != "rate":
        raise HTTPException(422, detail="event_type must be 'rate' in v1")
    if payload.rating is None:
        raise HTTPException(422, detail="rating required when event_type is 'rate'")
    if not (0.0 <= payload.rating <= 5.0):
        raise HTTPException(422, detail="rating must be in [0, 5]")

    if not db.get(Wine, payload.drink_id):
        raise HTTPException(404, detail=f"Wine {payload.drink_id} not found")

    event = WineEvent(
        user_id=payload.user_id,
        wine_id=payload.drink_id,
        event_type="rate",
        rating=payload.rating,
        synthetic=False,
    )
    db.add(event)
    db.commit()
    return {"status": "ok", "event_id": event.id}
