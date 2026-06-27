"""
helpers.py
----------
Low-level utilities for wine scoring: DB queries, normalization, pool sizing.
Imported by scoring.py; nothing here depends on ML serving modules.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.db.models import Wine, WineEvent

_POOL_BASE = 300
_POOL_STEP = 50
_POOL_MAX  = 5000


def candidate_pool_size(n_ratings: int) -> int:
    """Grow the candidate pool with experience so power users aren't starved."""
    return min(_POOL_BASE + n_ratings * _POOL_STEP, _POOL_MAX)


def bayesian_score(db: Session):
    return (Wine.avg_rating * Wine.n_ratings + 3.5 * 5) / (Wine.n_ratings + 5)


def popularity_top_n(db: Session, top_n: int, styles: set[str] | None = None) -> list[Wine]:
    q = db.query(Wine)
    if styles:
        q = q.filter(Wine.style.in_(styles))
    return q.order_by(bayesian_score(db).desc().nullslast()).limit(top_n).all()


def liked_wines(db: Session, user_id: int) -> list[tuple[int, float]]:
    """Real (non-synthetic) rating events for a user."""
    rows = (db.query(WineEvent.wine_id, WineEvent.rating)
              .filter(WineEvent.user_id == user_id,
                      WineEvent.event_type == "rate",
                      WineEvent.synthetic == False,           # noqa: E712
                      WineEvent.rating.isnot(None))
              .all())
    return [(int(w), float(r)) for w, r in rows]


def user_styles(db: Session, wine_ids: list[int]) -> set[str]:
    if not wine_ids:
        return set()
    rows = db.query(Wine.style).filter(Wine.id.in_(wine_ids)).distinct().all()
    return {s for (s,) in rows if s}


def minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}
