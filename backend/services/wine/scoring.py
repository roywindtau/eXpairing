"""
scoring.py
----------
Personalized wine ranking for "recommend me a wine".

Pipeline:
    1. cold start  — user with 0 ratings gets top-N popularity (no regression)
    2. style FILTER — candidates restricted to styles the user actually drinks
    3. blend       — warm users: 0.5*CF + 0.5*CB (min-max normalized); CF is
                     noise for brand-new users, so popularity covers cold start.

WARM_THRESHOLD ratings is where CF becomes trustworthy. Below it (but ≥1) CB +
popularity carry; at/above it we use the 50/50 CF/CB blend.

Scores from CF (raw dot) and CB (cosine) live on different scales, so each is
min-max normalized across the candidate pool before blending (same calibration
discipline as the recipe scorer).
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import Wine, WineEvent
from backend.ml.wine.serving import serve_cb, serve_cf

WARM_THRESHOLD = 5          # ratings before CF is trusted
CANDIDATE_POOL = 300        # top-popular wines (within style) to score
CF_WEIGHT = 0.5
CB_WEIGHT = 0.5


# ── popularity (Bayesian-smoothed) ──────────────────────────────────────

def _bayesian(db: Session):
    return (Wine.avg_rating * Wine.n_ratings + 3.5 * 5) / (Wine.n_ratings + 5)


def popularity_top_n(db: Session, top_n: int, styles: set[str] | None = None):
    q = db.query(Wine)
    if styles:
        q = q.filter(Wine.style.in_(styles))
    return q.order_by(_bayesian(db).desc().nullslast()).limit(top_n).all()


# ── helpers ─────────────────────────────────────────────────────────────

def _liked(db: Session, user_id: int) -> list[tuple[int, float]]:
    """Real (non-synthetic) rating events for a user."""
    rows = (db.query(WineEvent.wine_id, WineEvent.rating)
              .filter(WineEvent.user_id == user_id,
                      WineEvent.event_type == "rate",
                      WineEvent.synthetic == False,           # noqa: E712
                      WineEvent.rating.isnot(None))
              .all())
    return [(int(w), float(r)) for w, r in rows]


def _user_styles(db: Session, wine_ids: list[int]) -> set[str]:
    if not wine_ids:
        return set()
    rows = db.query(Wine.style).filter(Wine.id.in_(wine_ids)).distinct().all()
    return {s for (s,) in rows if s}


def _minmax(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


# ── main entry point ────────────────────────────────────────────────────

def rank_wines(db: Session, user_id: int, top_n: int = 5) -> list[Wine]:
    liked = _liked(db, user_id)

    # 1. COLD START — no ratings → popularity, unchanged behavior
    if not liked:
        return popularity_top_n(db, top_n)

    # 2. STYLE FILTER — only styles the user drinks
    styles = _user_styles(db, [w for w, _ in liked])
    pool = popularity_top_n(db, CANDIDATE_POOL, styles=styles or None)
    liked_ids = {w for w, _ in liked}
    candidates = [w for w in pool if w.id not in liked_ids]   # don't re-recommend rated
    if not candidates:
        return popularity_top_n(db, top_n, styles=styles or None)
    cand_ids = [w.id for w in candidates]

    warm = len(liked) >= WARM_THRESHOLD and serve_cf.cf_available()

    cb = _minmax(serve_cb.cb_scores(liked, cand_ids)) if serve_cb.cb_available() else {}
    cf = _minmax(serve_cf.cf_scores(liked, cand_ids)) if warm else {}

    # 3. BLEND
    by_id = {w.id: w for w in candidates}
    pop = _minmax({w.id: ((w.avg_rating or 0) * (w.n_ratings or 0) + 17.5)
                          / ((w.n_ratings or 0) + 5) for w in candidates})

    scored = []
    for wid in cand_ids:
        if warm:
            s = CF_WEIGHT * cf.get(wid, 0.0) + CB_WEIGHT * cb.get(wid, 0.0)
        elif cb:
            # warming (1..4 ratings): CB carries, popularity breaks CB ties
            s = 0.7 * cb.get(wid, 0.0) + 0.3 * pop.get(wid, 0.0)
        else:
            s = pop.get(wid, 0.0)
        scored.append((s, wid))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [by_id[wid] for _, wid in scored[:top_n]]
