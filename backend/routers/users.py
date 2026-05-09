"""
users.py
--------
User creation and profile management.

Endpoints:
    POST /users             -- create new user (called on first app open)
    GET  /users/{user_id}   -- get profile
    PUT  /users/{user_id}   -- update beta, diet tags
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User

router = APIRouter(prefix="/users", tags=["users"])


class UserIn(BaseModel):
    name:      str | None = None
    beta:      float      = Field(default=0.35, ge=0.0, le=1.0)
    diet_tags: str | None = None   # comma-separated e.g. "vegetarian,gluten-free"


class UserOut(BaseModel):
    id:        int
    name:      str | None
    beta:      float
    has_cf:    bool
    has_cb:    bool
    diet_tags: str | None

    model_config = {"from_attributes": True}


@router.post("", response_model=UserOut, status_code=201)
def create_user(payload: UserIn, db: Session = Depends(get_db)):
    """Create a new user. Called on first app launch (onboarding page)."""
    user = User(
        name=payload.name,
        beta=payload.beta,
        diet_tags=payload.diet_tags,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UserIn, db: Session = Depends(get_db)):
    """Update beta or dietary preferences."""
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.name      = payload.name
    user.beta      = payload.beta
    user.diet_tags = payload.diet_tags
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}/stats")
def get_user_stats(user_id: int, db: Session = Depends(get_db)):
    """
    Returns rating count, cook count, and CF readiness for the profile page.
    The frontend uses this to show progress toward warm CF (5 ratings needed).
    """
    from backend.db.models import UserEvent
    from backend.ml.serve_cf import MIN_RATINGS_FOR_CF

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    n_ratings = (db.query(UserEvent)
                   .filter(UserEvent.user_id == user_id,
                           UserEvent.event_type == "rate")
                   .count())
    n_cooked  = (db.query(UserEvent)
                   .filter(UserEvent.user_id == user_id,
                           UserEvent.event_type == "cook")
                   .count())
    n_skipped = (db.query(UserEvent)
                   .filter(UserEvent.user_id == user_id,
                           UserEvent.event_type == "skip")
                   .count())

    # Revealed beta from cooking behaviour (requires ≥3 cook events)
    revealed_beta = None
    avg_missing   = None
    if n_cooked >= 3:
        import pandas as pd
        from backend.services.beta_updater import _compute_revealed_beta
        cook_rows = (
            db.query(UserEvent.n_missing)
            .filter(UserEvent.user_id == user_id,
                    UserEvent.event_type == "cook")
            .all()
        )
        cook_df = pd.DataFrame(cook_rows, columns=["n_missing"])
        revealed_beta, avg_missing = _compute_revealed_beta(cook_df)

    return {
        "user_id":              user_id,
        "n_ratings":            n_ratings,
        "n_cooked":             n_cooked,
        "n_skipped":            n_skipped,
        "ratings_for_warm_cf":  MIN_RATINGS_FOR_CF,
        "warm_cf_progress_pct": min(100, round(n_ratings / MIN_RATINGS_FOR_CF * 100)),
        "is_warm":              n_ratings >= MIN_RATINGS_FOR_CF,
        "beta":                 round(user.beta, 3),
        "revealed_beta":        round(revealed_beta, 3) if revealed_beta is not None else None,
        "avg_missing":          round(avg_missing, 2)   if avg_missing   is not None else None,
    }
