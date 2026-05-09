"""
pantry.py
---------
REST endpoints for managing a user's pantry.

Endpoints:
    GET  /pantry/{user_id}              -- list all pantry items for a user
    POST /pantry/{user_id}              -- add one item
    PUT  /pantry/{user_id}/{item_id}    -- update expiry date or quantity
    DELETE /pantry/{user_id}/{item_id}  -- remove one item
    DELETE /pantry/{user_id}            -- clear entire pantry
"""

from __future__ import annotations

from datetime import date
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import PantryItem, Recipe, User

router = APIRouter(prefix="/pantry", tags=["pantry"])


# ---------------------------------------------------------------------------
# Pydantic schemas (request / response shapes)
# ---------------------------------------------------------------------------

class PantryItemIn(BaseModel):
    ingredient:  str
    expiry_date: date
    raw_name:    str | None = None
    quantity:    str | None = None


class PantryItemOut(BaseModel):
    id:          int
    ingredient:  str
    expiry_date: date
    raw_name:    str | None
    quantity:    str | None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


def _get_item_or_404(item_id: int, user_id: int, db: Session) -> PantryItem:
    item = db.query(PantryItem).filter(
        PantryItem.id == item_id,
        PantryItem.user_id == user_id,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail=f"Pantry item {item_id} not found")
    return item


# ---------------------------------------------------------------------------
# Ingredient suggest
# ---------------------------------------------------------------------------

_ingredient_vocab: List[str] = []

def _ensure_vocab(db: Session) -> List[str]:
    """Lazy-build the sorted ingredient vocabulary from the recipe corpus."""
    global _ingredient_vocab
    if _ingredient_vocab:
        return _ingredient_vocab
    rows = db.query(Recipe.ingredients_csv).all()
    vocab: set[str] = set()
    for (csv,) in rows:
        for token in csv.split(","):
            word = token.strip().lower()
            if word and len(word) > 1:
                vocab.add(word)
    _ingredient_vocab = sorted(vocab)
    return _ingredient_vocab


# NOTE: must be declared before /{user_id} so FastAPI doesn't treat "suggest"
# as an integer user_id (type coercion would reject it, but explicit ordering
# is clearer).
@router.get("/suggest", response_model=List[str])
def suggest_ingredients(
    q: str = Query(..., min_length=1, description="Prefix/substring to match"),
    limit: int = Query(10, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Return canonical ingredient names that match the query.
    Prefix matches are returned first, then substring matches."""
    term = q.strip().lower()
    if not term:
        return []
    vocab = _ensure_vocab(db)
    prefix  = [v for v in vocab if v.startswith(term)]
    substr  = [v for v in vocab if term in v and not v.startswith(term)]
    return (prefix + substr)[:limit]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{user_id}", response_model=list[PantryItemOut])
def list_pantry(user_id: int, db: Session = Depends(get_db)):
    """Return all pantry items for a user, sorted by expiry date ascending."""
    _get_user_or_404(user_id, db)
    items = (
        db.query(PantryItem)
        .filter(PantryItem.user_id == user_id)
        .order_by(PantryItem.expiry_date)
        .all()
    )
    return items


@router.post("/{user_id}", response_model=PantryItemOut, status_code=201)
def add_pantry_item(user_id: int, payload: PantryItemIn, db: Session = Depends(get_db)):
    """Add a single ingredient to a user's pantry."""
    _get_user_or_404(user_id, db)
    item = PantryItem(
        user_id=user_id,
        ingredient=payload.ingredient.strip().lower(),
        expiry_date=payload.expiry_date,
        raw_name=payload.raw_name,
        quantity=payload.quantity,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.post("/{user_id}/bulk", response_model=list[PantryItemOut], status_code=201)
def add_pantry_items_bulk(
    user_id: int,
    payload: list[PantryItemIn],
    db: Session = Depends(get_db),
):
    """
    Add multiple pantry items at once.
    Used by the vision agent after a photo scan.
    """
    _get_user_or_404(user_id, db)
    items = [
        PantryItem(
            user_id=user_id,
            ingredient=p.ingredient.strip().lower(),
            expiry_date=p.expiry_date,
            raw_name=p.raw_name,
            quantity=p.quantity,
        )
        for p in payload
    ]
    db.add_all(items)
    db.commit()
    for item in items:
        db.refresh(item)
    return items


@router.put("/{user_id}/{item_id}", response_model=PantryItemOut)
def update_pantry_item(
    user_id: int,
    item_id: int,
    payload: PantryItemIn,
    db: Session = Depends(get_db),
):
    """Update expiry date or quantity of an existing pantry item."""
    item = _get_item_or_404(item_id, user_id, db)
    item.ingredient  = payload.ingredient.strip().lower()
    item.expiry_date = payload.expiry_date
    item.raw_name    = payload.raw_name
    item.quantity    = payload.quantity
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{user_id}/{item_id}", status_code=204)
def delete_pantry_item(user_id: int, item_id: int, db: Session = Depends(get_db)):
    """Remove a single pantry item."""
    item = _get_item_or_404(item_id, user_id, db)
    db.delete(item)
    db.commit()


@router.delete("/{user_id}", status_code=204)
def clear_pantry(user_id: int, db: Session = Depends(get_db)):
    """Delete all pantry items for a user (used after a fresh photo scan)."""
    _get_user_or_404(user_id, db)
    db.query(PantryItem).filter(PantryItem.user_id == user_id).delete()
    db.commit()
