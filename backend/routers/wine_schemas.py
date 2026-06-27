"""
wine_schemas.py
---------------
Pydantic request/response models for the wine API (backend/routers/wine.py).
Kept separate from the router so the route handlers stay focused on HTTP logic.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WineOut(BaseModel):
    wine_id:       int
    wine_name:     str
    avg_rating:    Optional[float] = None
    n_ratings:     int = 0
    abv:           Optional[float] = None
    producer:      Optional[str] = None
    country:       Optional[str] = None
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


class PairRequest(BaseModel):
    recipe_id: int
    top_n:     int = 5


class PairedWineOut(WineOut):
    # cosine pairing score in [0, 1] between the recipe and this wine
    pairing_score: float
