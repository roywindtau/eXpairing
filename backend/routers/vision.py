"""
vision.py
---------
POST /vision/scan
    Accepts a fridge/pantry photo, calls GPT-4o vision to identify
    products and read expiry dates, and returns a list of pantry items
    ready for the user to confirm before adding to their pantry.

    The frontend shows the detected items in a confirmation step —
    the user can edit expiry dates for any nulls before saving.

POST /vision/confirm/{user_id}
    Takes the confirmed list and bulk-inserts into the pantry.
    Separate from scan so the user has a chance to review/edit.

GET /vision/mock
    Returns a realistic fake scan result. Used in dev/demo when
    no OpenAI key is available. Lets you demo the full flow.
"""

from __future__ import annotations

import os
from datetime import date as _date
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import PantryItem, User
from backend.services.vision_agent import (
    scan_image,
    mock_scan,
    IngredientCanonicalizer,
)

router = APIRouter(prefix='/vision', tags=['vision'])

# Build canonicalizer once at module load (warm)
_canonicalizer: IngredientCanonicalizer | None = None

def _get_canon() -> IngredientCanonicalizer:
    global _canonicalizer
    if _canonicalizer is None:
        _canonicalizer = IngredientCanonicalizer.from_db()
    return _canonicalizer


# ── schemas ────────────────────────────────────────────────────────────────

class ScannedItem(BaseModel):
    ingredient:  str
    expiry_date: str | None   # null = user must fill in
    raw_name:    str
    quantity:    str | None


class ConfirmPayload(BaseModel):
    items: list[ScannedItem]


# ── GET /vision/mock ───────────────────────────────────────────────────────

@router.get('/mock', response_model=list[ScannedItem])
def get_mock_scan():
    """
    Returns a realistic fake scan result for dev/demo.
    Use this when no OPENAI_API_KEY is available.
    """
    return mock_scan()


# ── POST /vision/scan ──────────────────────────────────────────────────────

@router.post('/scan', response_model=list[ScannedItem])
async def scan_fridge(photo: UploadFile = File(...)):
    """
    Upload a JPEG/PNG fridge photo. Returns detected products.

    The list may contain items with expiry_date=null — the UI
    should prompt the user to fill these in before confirming.

    Requires OPENAI_API_KEY environment variable.
    Falls back to mock scan in dev mode (VISION_MOCK=true).
    """
    if os.environ.get('VISION_MOCK', '').lower() in ('1', 'true', 'yes'):
        return mock_scan()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                'OPENAI_API_KEY not set. '
                'Set the environment variable or use GET /vision/mock for demo mode.'
            ),
        )

    image_bytes = await photo.read()
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail='Image too large (max 10 MB)')

    try:
        items = scan_image(
            image_bytes=image_bytes,
            api_key=api_key,
            canonicalizer=_get_canon(),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return items


# ── POST /vision/confirm/{user_id} ────────────────────────────────────────

@router.post('/confirm/{user_id}', status_code=201)
def confirm_scan(
    user_id: int,
    payload: ConfirmPayload,
    db: Session = Depends(get_db),
):
    """
    Bulk-insert confirmed scanned items into the user's pantry.

    Called after the user has reviewed the scan results and filled in
    any missing expiry dates. Items with no expiry_date are rejected
    (the UI enforces this before calling confirm).
    """
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f'User {user_id} not found')

    invalid = [i.ingredient for i in payload.items if not i.expiry_date]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f'Missing expiry dates for: {", ".join(invalid)}',
        )

    new_items = [
        PantryItem(
            user_id=user_id,
            ingredient=item.ingredient.strip().lower(),
            expiry_date=_date.fromisoformat(item.expiry_date),
            raw_name=item.raw_name,
            quantity=item.quantity,
        )
        for item in payload.items
    ]

    db.add_all(new_items)
    db.commit()
    for i in new_items:
        db.refresh(i)

    return {
        'added': len(new_items),
        'items': [
            {'id': i.id, 'ingredient': i.ingredient, 'expiry_date': str(i.expiry_date)}
            for i in new_items
        ],
    }
