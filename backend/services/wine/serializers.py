"""
serializers.py
--------------
Mapping helpers from Wine ORM objects to wine API response models.
"""

from __future__ import annotations

from backend.db.models import Wine
from backend.routers.wine_schemas import WineOut


def to_out(w: Wine) -> WineOut:
    """Convert a Wine row into its API representation."""
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
