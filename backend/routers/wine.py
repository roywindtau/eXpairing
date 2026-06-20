"""
wine.py
-------
HTTP API for wine recommendations + rating.

Current scope is deliberately minimal: "recommend me a wine" returns the
top-N most popular wines (Bayesian-smoothed). Per-user CF/CB ranking and
recipe pairing are future work (see the wine training scripts under
backend/ml/wine/training/), not wired here yet.

Routes
------
    GET  /wine/ranked    top-N popular wines ("Suggest me a wine")
    POST /wine-events    rate a wine
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Wine, WineEvent

router = APIRouter(tags=["wine"])


# ── pydantic schemas ────────────────────────────────────────────────────

class WineOut(BaseModel):
    wine_id:       int
    wine_name:     str
    avg_rating:    Optional[float] = None
    n_ratings:     int = 0
    abv:           Optional[float] = None
    producer:      Optional[str] = None
    style:         Optional[str] = None
    variety:       Optional[str] = None
    harmonize_csv: Optional[str] = None
    # structural attributes the CB ranking matches on ("why this wine")
    acidity:       Optional[str] = None
    body:          Optional[str] = None
    region:        Optional[str] = None


class WineEventIn(BaseModel):
    user_id:    int
    wine_id:    int
    event_type: str   # v1: "rate"
    rating:     Optional[float] = None


# ── helpers ─────────────────────────────────────────────────────────────

def _to_out(w: Wine) -> WineOut:
    return WineOut(
        wine_id=w.id,
        wine_name=w.name,
        avg_rating=w.avg_rating,
        n_ratings=w.n_ratings or 0,
        abv=w.abv,
        producer=w.producer,
        style=w.style,
        variety=w.grapes_csv,
        harmonize_csv=w.harmonize_csv,
        acidity=w.acidity,
        body=w.body,
        region=w.region,
    )


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
