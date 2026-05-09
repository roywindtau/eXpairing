"""
shopping.py
-----------
Buy-list / shopping list endpoints.

A user can add missing ingredients from any recipe to a persistent list,
check items off while shopping, and remove them when done.

Deduplication: adding an ingredient that already exists for this user is
silently skipped — the caller receives a `skipped` list so the UI can
report "2 added, 1 already in list".

No quantities are stored.  If two recipes both need "eggs", only one
"eggs" entry is kept — the source recipe shown is whichever was added first.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User, ShoppingListItem

router = APIRouter(prefix="/shopping", tags=["shopping"])


# ── schemas ────────────────────────────────────────────────────────────────

class ShoppingItemOut(BaseModel):
    id:                 int
    ingredient:         str
    source_recipe_id:   Optional[int]
    source_recipe_name: Optional[str]
    is_checked:         bool

    class Config:
        from_attributes = True


class AddItemsIn(BaseModel):
    ingredients: List[str]
    recipe_id:   Optional[int] = None
    recipe_name: Optional[str] = None


class AddItemsOut(BaseModel):
    added:   List[ShoppingItemOut]
    skipped: List[str]          # already in list


class ToggleIn(BaseModel):
    is_checked: bool


# ── helpers ────────────────────────────────────────────────────────────────

def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


def _item_out(item: ShoppingListItem) -> ShoppingItemOut:
    return ShoppingItemOut(
        id=item.id,
        ingredient=item.ingredient,
        source_recipe_id=item.source_recipe_id,
        source_recipe_name=item.source_recipe_name,
        is_checked=item.is_checked,
    )


# ── GET /shopping/{user_id} ────────────────────────────────────────────────

@router.get("/{user_id}", response_model=list[ShoppingItemOut])
def get_shopping_list(user_id: int, db: Session = Depends(get_db)):
    _get_user_or_404(user_id, db)
    items = (
        db.query(ShoppingListItem)
        .filter(ShoppingListItem.user_id == user_id)
        .order_by(ShoppingListItem.added_at)
        .all()
    )
    return [_item_out(i) for i in items]


# ── POST /shopping/{user_id} — batch add ──────────────────────────────────

@router.post("/{user_id}", response_model=AddItemsOut, status_code=201)
def add_to_shopping_list(
    user_id: int,
    payload: AddItemsIn,
    db: Session = Depends(get_db),
):
    _get_user_or_404(user_id, db)

    existing = {
        row.ingredient.lower()
        for row in db.query(ShoppingListItem.ingredient)
        .filter(ShoppingListItem.user_id == user_id)
        .all()
    }

    added, skipped = [], []
    for raw in payload.ingredients:
        name = raw.strip().lower()
        if not name:
            continue
        if name in existing:
            skipped.append(name)
            continue
        item = ShoppingListItem(
            user_id=user_id,
            ingredient=name,
            source_recipe_id=payload.recipe_id,
            source_recipe_name=payload.recipe_name,
            is_checked=False,
        )
        db.add(item)
        existing.add(name)
        added.append(item)

    db.commit()
    for item in added:
        db.refresh(item)

    return AddItemsOut(added=[_item_out(i) for i in added], skipped=skipped)


# ── PATCH /shopping/{user_id}/{item_id} — toggle checked ──────────────────

@router.patch("/{user_id}/{item_id}", response_model=ShoppingItemOut)
def toggle_item(
    user_id: int,
    item_id: int,
    payload: ToggleIn,
    db: Session = Depends(get_db),
):
    item = (
        db.query(ShoppingListItem)
        .filter(ShoppingListItem.id == item_id,
                ShoppingListItem.user_id == user_id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.is_checked = payload.is_checked
    db.commit()
    db.refresh(item)
    return _item_out(item)


# ── DELETE /shopping/{user_id}/{item_id} — remove one item ────────────────

@router.delete("/{user_id}/{item_id}", status_code=204)
def remove_item(user_id: int, item_id: int, db: Session = Depends(get_db)):
    item = (
        db.query(ShoppingListItem)
        .filter(ShoppingListItem.id == item_id,
                ShoppingListItem.user_id == user_id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(item)
    db.commit()


# ── DELETE /shopping/{user_id} — clear checked items (or all) ─────────────

@router.delete("/{user_id}", status_code=204)
def clear_shopping_list(
    user_id: int,
    checked_only: bool = Query(True, description="If true, remove only checked items"),
    db: Session = Depends(get_db),
):
    _get_user_or_404(user_id, db)
    q = db.query(ShoppingListItem).filter(ShoppingListItem.user_id == user_id)
    if checked_only:
        q = q.filter(ShoppingListItem.is_checked == True)
    q.delete(synchronize_session=False)
    db.commit()
