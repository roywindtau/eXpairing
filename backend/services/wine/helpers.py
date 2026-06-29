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
    """Real (non-synthetic) ratings for a user, one per wine (latest wins).

    A user may rate the same wine more than once; we keep only their most
    recent rating so re-rating can't inflate the warm threshold or double-count
    a wine in the taste profile.
    """
    rows = (db.query(WineEvent.wine_id, WineEvent.rating)
              .filter(WineEvent.user_id == user_id,
                      WineEvent.event_type == "rate",
                      WineEvent.synthetic == False,           # noqa: E712
                      WineEvent.rating.isnot(None))
              .order_by(WineEvent.created_at.asc(), WineEvent.id.asc())
              .all())
    latest: dict[int, float] = {}
    for w, r in rows:
        latest[int(w)] = float(r)          # later rows overwrite -> latest kept
    return list(latest.items())


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


MMR_LAMBDA = 0.7   # relevance vs. diversity trade-off (higher = more relevance)


def _wine_similarity(a: Wine, b: Wine, cb_sim: dict[tuple[int, int], float]) -> float:
    """Pairwise wine similarity for MMR. Uses precomputed CB cosines when available,
    falls back to style+grape overlap."""
    key = (min(a.id, b.id), max(a.id, b.id))
    if key in cb_sim:
        return cb_sim[key]
    # fallback: same style = 0.5, same primary grape = +0.5
    same_style = float(a.style == b.style and a.style is not None)
    grapes_a = set((a.grapes_csv or "").split(","))
    grapes_b = set((b.grapes_csv or "").split(","))
    overlap = len(grapes_a & grapes_b) / len(grapes_a | grapes_b) if grapes_a | grapes_b else 0.0
    return 0.5 * same_style + 0.5 * overlap


def mmr_rerank(
    candidates: list[Wine],
    scores: dict[int, float],
    top_n: int,
    cb_sim: dict[tuple[int, int], float] | None = None,
    lambda_: float = MMR_LAMBDA,
) -> list[Wine]:
    """
    Maximal Marginal Relevance reranking for diversity.
    MMR(w) = λ · score(w) − (1−λ) · max_sim(w, already_selected)
    Always picks the highest-scored wine first, then penalizes subsequent
    picks that are too similar to what's already selected.
    """
    if len(candidates) <= top_n:
        return candidates
    cb_sim = cb_sim or {}
    remaining = list(candidates)
    selected: list[Wine] = []
    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda w: scores.get(w.id, 0.0))
        else:
            def mmr_score(w: Wine, sel: list[Wine] = selected) -> float:
                max_sim = max(_wine_similarity(w, s, cb_sim) for s in sel)
                return lambda_ * scores.get(w.id, 0.0) - (1.0 - lambda_) * max_sim
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected
