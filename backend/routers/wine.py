"""
wine.py
-------
HTTP API for wine recommendations + rating.

"Recommend me a wine" is personalized: a blend of collaborative filtering
(ALS, services/wine/scoring.py) and content-based scoring, with a fruit-seeded
cold start for brand-new users and a popularity floor throughout. "Pair me a
wine" ranks wines against a recipe via the content-based pairing model
(serve_pairing.py). All three model groups (CF, CB, pairing) live under
backend/ml/wine/ and the app degrades gracefully to popularity when an
artifact is missing.

Routes
------
    GET  /wine/ranked        top-N wines ("Suggest me a wine"), personalized
    POST /wine/pair          top-N wines that pair with a given recipe
    POST /wine-events        rate a wine
    POST /wine/preferences   set fruit-inferred taste prefs (cold-start onboarding)
    GET  /wine/preferences   read a user's stored taste prefs
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Recipe, User, Wine, WineEvent
from backend.routers.wine_schemas import (
    PairedWineOut,
    PairRequest,
    WineEventIn,
    WineOut,
    WinePreferencesIn,
    WinePreferencesOut,
)
from backend.services.wine.preference_profile import infer_details
from backend.services.wine.serializers import to_out as _to_out

router = APIRouter(tags=["wine"])


# ── GET /wine/ranked ────────────────────────────────────────────────────

@router.get("/wine/ranked", response_model=list[WineOut])
def get_ranked_wines(
    top_n: int = Query(10, ge=1, le=100),
    user_id: Optional[int] = Query(None),
    styles: Optional[list[str]] = Query(None),
    db: Session = Depends(get_db),
):
    """
    "Suggest me a wine".

    Without user_id: top-N wines by Bayesian-smoothed popularity (back-compat).
    With user_id: personalized — style-filtered, blended CF+CB for warm users,
    popularity cold start for new users (see services/wine/scoring.py).
    styles: optional explicit style filter (e.g. Red, White) — overrides the
    auto-derived "styles you drink".
    """
    style_set = {s for s in styles if s} if styles else None
    if user_id is not None:
        from backend.services.wine.scoring import rank_wines
        return [_to_out(w) for w in rank_wines(db, user_id, top_n, styles=style_set)]

    bayesian = (
        (Wine.avg_rating * Wine.n_ratings + 3.5 * 5)
        / (Wine.n_ratings + 5)
    )
    if style_set:
        # honor an explicit style filter even on the non-personalized path
        return [_to_out(w) for w in
                db.query(Wine).filter(Wine.style.in_(style_set))
                  .order_by(bayesian.desc().nullslast()).limit(top_n).all()]
    rows = (
        db.query(Wine)
          .order_by(bayesian.desc().nullslast())
          .limit(top_n)
          .all()
    )
    return [_to_out(w) for w in rows]


# ── POST /wine/pair ─────────────────────────────────────────────────────

@router.post("/wine/pair", response_model=list[PairedWineOut])
def pair_wine_with_recipe(payload: PairRequest, db: Session = Depends(get_db)):
    """
    "Pair me a wine for this recipe."

    Pure content-based: maps the recipe's ingredients to the 12 food categories
    (Module 3), then ranks wines by a blend of category cosine + empirical pairing
    rules (Modules 2 + 4). The top-scoring pool is MMR-reranked for light variety
    (so the 5 picks aren't five near-identical bottles). No user history is used.
    """
    from backend.ml.wine.serving.serve_pairing import pair_wines, pairing_available
    from backend.ml.wine.serving import serve_cb
    from backend.services.wine.helpers import mmr_rerank

    recipe = db.get(Recipe, payload.recipe_id)
    if recipe is None:
        raise HTTPException(404, detail=f"Recipe {payload.recipe_id} not found")
    if not pairing_available():
        raise HTTPException(
            503,
            detail="Pairing model not built. Run "
                   "`python -m data.pairing.build_wine_pairing_vectors`.",
        )

    top_n = max(1, min(payload.top_n, 100))
    # over-fetch a pool so MMR has room to diversify, then trim to top_n.
    ranked = pair_wines(recipe.ingredients, top_n=top_n * 4)
    if not ranked:
        # Weak/no sensory signal (e.g. a plain veg dish we can't read): rather than
        # a misleading wall, offer a versatile crowd-pleaser. Dry sparkling/rosé is
        # the classic "goes with anything" safe pick. score 0 -> UI flags it as a
        # general suggestion, not a precise match.
        bayesian = (Wine.avg_rating * Wine.n_ratings + 3.5 * 5) / (Wine.n_ratings + 5)
        safe = (db.query(Wine)
                  .filter(Wine.style.in_(["Sparkling", "Rosé"]))
                  .order_by(bayesian.desc().nullslast())
                  .limit(top_n).all())
        return [PairedWineOut(**_to_out(w).model_dump(), pairing_score=0.0)
                for w in safe]

    score_of = {wid: score for wid, score in ranked}
    pool = {w.id: w for w in
            db.query(Wine).filter(Wine.id.in_(list(score_of))).all()}
    candidates = [pool[wid] for wid, _ in ranked if wid in pool]

    # MMR rerank for light diversity (lambda high = stay close to relevance).
    cb_sim = (serve_cb.pairwise_similarity([w.id for w in candidates])
              if serve_cb.cb_available() else {})
    diversified = mmr_rerank(candidates, score_of, top_n, cb_sim=cb_sim, lambda_=0.8)

    return [
        PairedWineOut(**_to_out(w).model_dump(), pairing_score=score_of[w.id])
        for w in diversified
    ]


# ── POST /wine-events ───────────────────────────────────────────────────

@router.post("/wine-events", status_code=201)
def log_wine_event(payload: WineEventIn, db: Session = Depends(get_db)):
    """Record a wine rating. v1 supports only event_type='rate'."""
    if payload.event_type != "rate":
        raise HTTPException(422, detail="event_type must be 'rate' in v1")
    if payload.rating is None:
        raise HTTPException(422, detail="rating required when event_type is 'rate'")
    if not (0.0 <= payload.rating <= 5.0):
        raise HTTPException(422, detail="rating must be in [0, 5]")

    if not db.get(Wine, payload.wine_id):
        raise HTTPException(404, detail=f"Wine {payload.wine_id} not found")

    event = WineEvent(
        user_id=payload.user_id,
        wine_id=payload.wine_id,
        event_type="rate",
        rating=payload.rating,
        synthetic=False,
    )
    db.add(event)
    db.commit()
    return {"status": "ok", "event_id": event.id}


# ── wine preferences (cold-start onboarding) ─────────────────────────────

def _prefs_out(details: dict) -> WinePreferencesOut:
    return WinePreferencesOut(
        fruits=details.get("fruits", []),
        grapes=details.get("grapes", []),
        body=details.get("body"),
        acidity=details.get("acidity"),
        styles=details.get("styles", []),
    )


@router.post("/wine/preferences", response_model=WinePreferencesOut)
def set_wine_preferences(payload: WinePreferencesIn, db: Session = Depends(get_db)):
    """
    Cold-start onboarding: infer wine taste details from the user's fruit picks
    and persist them on the user. These seed the CB ranking for new users and
    decay automatically as the user rates wines (see services/wine/scoring.py).
    """
    user = db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(404, detail=f"User {payload.user_id} not found")

    details = infer_details(payload.fruits)
    user.wine_prefs = json.dumps(details) if details else None
    db.commit()
    return _prefs_out(details)


@router.get("/wine/preferences", response_model=WinePreferencesOut)
def get_wine_preferences(
    user_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Return the user's stored wine taste details (empty if none set)."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, detail=f"User {user_id} not found")
    if not user.wine_prefs:
        return _prefs_out({})
    try:
        details = json.loads(user.wine_prefs)
    except (ValueError, TypeError):
        details = {}
    return _prefs_out(details)
