"""
drinks.py
---------
HTTP API for drink recommendations + browsing + event logging.

Architecture
------------
Mirrors the two-stage pipeline used in routers/recipes.py:

    Stage 1 — Candidate generation
        SQL filter by kind + order by Bayesian-smoothed popularity,
        cap at 2000 candidates.

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
    POST /drink-events                rate a drink
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Drink, DrinkEvent, Recipe, User
from backend.ml.drinks.serving.serve_cb import cb_for_recipe, cb_for_user, model_available as cb_available
from backend.ml.drinks.serving.serve_cf import cf_strategy_name, get_cf_scores
from backend.services.drinks.scoring import (
    DrinkScore,
    rank_drinks_for_recipe,
    rank_drinks_for_user,
)
from backend.services.drinks.expert_pairing import expert_boost_batch

router = APIRouter(tags=["drinks"])

CANDIDATE_POOL_SIZE = 2000   # Stage-1 cap, mirrors recipes router


# ── pydantic schemas ────────────────────────────────────────────────────

class DrinkScoreOut(BaseModel):
    drink_id:     int
    drink_name:   str
    kind:         str
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
    # kind-specific
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

def _kind_to_filter(kind: Optional[str]) -> Optional[str]:
    """Normalize the kind query param. 'all'/'' → None (no filter)."""
    if kind is None:
        return None
    k = kind.strip().lower()
    if k in ("", "all", "any"):
        return None
    if k in ("beer", "wine"):
        return k
    raise HTTPException(
        status_code=422,
        detail=f"kind must be 'beer', 'wine', or 'all' (got '{kind}')",
    )


def _candidates(db: Session, kind_filter: Optional[str], limit: int) -> list[Drink]:
    """Stage 1: top-N drinks by Bayesian-smoothed popularity, optionally by kind."""
    # bayesian = (n*avg + C*prior) / (n + C), with C=5, prior=3.5
    bayesian = (
        (Drink.avg_rating * Drink.n_ratings + 3.5 * 5)
        / (Drink.n_ratings + 5)
    )
    q = db.query(Drink)
    if kind_filter is not None:
        q = q.filter(Drink.kind == kind_filter)
    return (
        q.order_by(bayesian.desc().nullslast())
         .limit(limit)
         .all()
    )


def _split_cb_by_kind(
    recipe, kind_filter: Optional[str],
) -> dict[int, float]:
    """Path A: call cb_for_recipe per kind so the kind_filter is honored."""
    if not cb_available():
        return {}
    if kind_filter is None:
        beer = cb_for_recipe(recipe, kind_filter="beer")
        wine = cb_for_recipe(recipe, kind_filter="wine")
        return {**beer, **wine}
    return cb_for_recipe(recipe, kind_filter=kind_filter)


def _to_out(s: DrinkScore, drink: Drink) -> DrinkScoreOut:
    return DrinkScoreOut(
        drink_id=s.drink_id,
        drink_name=s.drink_name,
        kind=s.kind,
        final_score=round(s.final_score, 4),
        cb_score=round(s.cb_score, 4),
        cf_score=round(s.cf_score, 4),
        expert_boost=round(s.expert_boost, 4),
        prior_score=round(s.prior_score, 4),
        cf_strategy=s.cf_strategy,
        avg_rating=drink.avg_rating,
        n_ratings=drink.n_ratings or 0,
        abv=drink.abv,
        producer=drink.producer,
        style=drink.style,
        wine_type=drink.wine_type,
        variety=drink.variety,
        harmonize_csv=drink.harmonize_csv,
    )


def _count_user_explicit_drink_ratings(db: Session, user_id: int) -> int:
    return (
        db.query(DrinkEvent)
        .filter(DrinkEvent.user_id    == user_id)
        .filter(DrinkEvent.event_type == "rate")
        .filter(DrinkEvent.rating.isnot(None))
        .filter(DrinkEvent.synthetic  == False)  # noqa: E712
        .count()
    )


# ── GET /drinks/ranked  (Path B) ────────────────────────────────────────

@router.get("/drinks/ranked", response_model=list[DrinkScoreOut])
def get_ranked_drinks(
    user_id: int = Query(...),
    kind:    Optional[str] = Query(None, description="'beer' | 'wine' | 'all'"),
    top_n:   int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Path B — "Drinks For You". Ranks drinks by:
      - CB score from the user's RECIPE rating history (via flavor_bridge)
      - CF score routed by drink rating count (popularity / item-sim / SVD)
      - Popularity prior tiebreaker

    No expert boost (no specific recipe to pair against).
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail=f"User {user_id} not found")

    kind_filter = _kind_to_filter(kind)
    candidates = _candidates(db, kind_filter, CANDIDATE_POOL_SIZE)
    if not candidates:
        return []

    # CB: from user's recipe history (aggregated bridged docs)
    cb_scores: dict[int, float] = {}
    if cb_available():
        if kind_filter is None:
            cb_scores.update(cb_for_user(user_id, db, kind_filter="beer"))
            cb_scores.update(cb_for_user(user_id, db, kind_filter="wine"))
        else:
            cb_scores = cb_for_user(user_id, db, kind_filter=kind_filter)

    # CF: standard get_cf_scores routing
    drinks_with_kinds = [(d.id, d.kind) for d in candidates]
    cf_scores = get_cf_scores(user_id, drinks_with_kinds, db)
    n_explicit = _count_user_explicit_drink_ratings(db, user_id)
    cf_strategies = {d.id: cf_strategy_name(n_explicit, d.kind) for d in candidates}

    ranked = rank_drinks_for_user(
        candidates=candidates,
        cb_scores=cb_scores,
        cf_scores=cf_scores,
        cf_strategies=cf_strategies,
        top_n=top_n,
    )

    drink_map = {d.id: d for d in candidates}
    return [_to_out(s, drink_map[s.drink_id]) for s in ranked]


# ── GET /drinks/pairings/{recipe_id}  (Path A) ──────────────────────────

@router.get("/drinks/pairings/{recipe_id}", response_model=list[DrinkScoreOut])
def get_drink_pairings(
    recipe_id: int,
    user_id:   int = Query(...),
    kind:      Optional[str] = Query(None, description="'beer' | 'wine' | 'all'"),
    top_n:     int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Path A — given a specific recipe, suggest drinks. Adds the expert-rules
    boost (Harmonize match + beer style heuristics) on top of CB + CF.
    """
    recipe = db.get(Recipe, recipe_id)
    if not recipe:
        raise HTTPException(404, detail=f"Recipe {recipe_id} not found")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, detail=f"User {user_id} not found")

    kind_filter = _kind_to_filter(kind)
    candidates = _candidates(db, kind_filter, CANDIDATE_POOL_SIZE)
    if not candidates:
        return []

    # CB: vs recipe (kind-filtered so we don't waste compute)
    cb_scores = _split_cb_by_kind(recipe, kind_filter)

    # CF: standard routing
    drinks_with_kinds = [(d.id, d.kind) for d in candidates]
    cf_scores = get_cf_scores(user_id, drinks_with_kinds, db)
    n_explicit = _count_user_explicit_drink_ratings(db, user_id)
    cf_strategies = {d.id: cf_strategy_name(n_explicit, d.kind) for d in candidates}

    # Expert: Harmonize match + beer style heuristics
    expert = expert_boost_batch(recipe, candidates)

    ranked = rank_drinks_for_recipe(
        recipe=recipe,
        candidates=candidates,
        cb_scores=cb_scores,
        cf_scores=cf_scores,
        expert_boosts=expert,
        cf_strategies=cf_strategies,
        top_n=top_n,
    )

    drink_map = {d.id: d for d in candidates}
    return [_to_out(s, drink_map[s.drink_id]) for s in ranked]


# ── GET /drinks/search ──────────────────────────────────────────────────

@router.get("/drinks/search")
def search_drinks(
    q:     str = Query("", description="Search term (name or style/variety)"),
    kind:  Optional[str] = Query(None),
    limit: int = Query(40, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Browse drinks with text search and kind filter. Not personalized."""
    kind_filter = _kind_to_filter(kind)
    query = db.query(Drink)
    if kind_filter is not None:
        query = query.filter(Drink.kind == kind_filter)
    if q.strip():
        term = f"%{q.strip().lower()}%"
        query = query.filter(
            Drink.name.ilike(term)
            | Drink.style.ilike(term)
            | Drink.variety.ilike(term)
        )
    rows = (
        query.order_by(Drink.avg_rating.desc().nullslast())
             .limit(limit)
             .all()
    )
    return [
        {
            "id":            d.id,
            "name":          d.name,
            "kind":          d.kind,
            "style":         d.style,
            "wine_type":     d.wine_type,
            "variety":       d.variety,
            "harmonize_csv": d.harmonize_csv,
            "producer":      d.producer,
            "abv":           d.abv,
            "avg_rating":    d.avg_rating,
            "n_ratings":     d.n_ratings,
        }
        for d in rows
    ]


# ── GET /drinks/{drink_id} ──────────────────────────────────────────────

@router.get("/drinks/{drink_id}")
def get_drink_detail(drink_id: int, db: Session = Depends(get_db)):
    drink = db.get(Drink, drink_id)
    if not drink:
        raise HTTPException(404, detail="Drink not found")
    return {
        "id":              drink.id,
        "name":            drink.name,
        "kind":            drink.kind,
        "producer":        drink.producer,
        "country":         drink.country,
        "abv":             drink.abv,
        "avg_rating":      drink.avg_rating,
        "n_ratings":       drink.n_ratings,
        # beer
        "style":           drink.style,
        "ibu":             drink.ibu,
        "avg_aroma":       drink.avg_aroma,
        "avg_taste":       drink.avg_taste,
        "avg_palate":      drink.avg_palate,
        "avg_appearance":  drink.avg_appearance,
        # wine
        "wine_type":       drink.wine_type,
        "variety":         drink.variety,
        "grapes_csv":      drink.grapes_csv,
        "region":          drink.region,
        "body":            drink.body,
        "acidity":         drink.acidity,
        "harmonize_csv":   drink.harmonize_csv,
    }


# ── POST /drink-events ──────────────────────────────────────────────────

@router.post("/drink-events", status_code=201)
def log_drink_event(payload: DrinkEventIn, db: Session = Depends(get_db)):
    """
    Record a drink rating. v1 supports only event_type='rate' with a rating.

    Deliberately NO synthesizer hook here — only RECIPE ratings trigger
    drink synthesis. Drink ratings are the user's explicit signal and feed
    the SVD (beer) / item-sim (wine) paths directly.
    """
    if payload.event_type != "rate":
        raise HTTPException(422, detail="event_type must be 'rate' in v1")
    if payload.rating is None:
        raise HTTPException(422, detail="rating required when event_type is 'rate'")
    if not (0.0 <= payload.rating <= 5.0):
        raise HTTPException(422, detail="rating must be in [0, 5]")

    if not db.get(Drink, payload.drink_id):
        raise HTTPException(404, detail=f"Drink {payload.drink_id} not found")

    event = DrinkEvent(
        user_id=payload.user_id,
        drink_id=payload.drink_id,
        event_type="rate",
        rating=payload.rating,
        synthetic=False,
    )
    db.add(event)
    db.commit()
    return {"status": "ok", "event_id": event.id}
