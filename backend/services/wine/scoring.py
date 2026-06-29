"""
scoring.py
----------
Personalized wine ranking for "recommend me a wine".

Pipeline:
    1. cold start  — 0 ratings → top popularity, lightly MMR-diversified
    2. style FILTER — candidates restricted to styles the user actually drinks
    3. blend       — warm users: 0.45*CF + 0.45*CB + 0.10*popularity (min-max
                     normalized); the popularity floor keeps results sensible
                     when CF/CB give no signal.

Scores from CF (raw dot) and CB (cosine) live on different scales, so each is
min-max normalized across the candidate pool before blending (same calibration
discipline as the recipe scorer).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.db.models import Wine
from backend.ml.wine.serving import serve_cb, serve_cf
from backend.services.wine.helpers import (
    candidate_pool_size,
    liked_wines,
    minmax,
    mmr_rerank,
    popularity_top_n,
    user_styles,
)

WARM_THRESHOLD = 5
# Warm blend weights. The small popularity term is a floor so results stay
# sensible even when CF (user absent from the ALS factors) or CB (flat taste
# profile) contribute no signal — without it the blend can collapse to 0.
CF_WEIGHT = 0.45
CB_WEIGHT = 0.45
POP_WEIGHT = 0.10


def _pop_prior(w: Wine) -> float:
    """Bayesian-smoothed popularity (prior mean 3.5 over 5 pseudo-ratings)."""
    n = w.n_ratings or 0
    return ((w.avg_rating or 0) * n + 17.5) / (n + 5)


def rank_wines(db: Session, user_id: int, top_n: int = 5,
               styles: set[str] | None = None) -> list[Wine]:
    """
    styles: if given, the user's explicit style choice — overrides the
    auto-derived "styles you drink". Empty/None falls back to auto.
    """
    liked = liked_wines(db, user_id)

    # 1. COLD START — no ratings → popularity, lightly diversified so the first
    #    impression isn't N near-identical bottles. Honors an explicit style pick.
    if not liked:
        pool = popularity_top_n(db, top_n * 4, styles=styles or None)
        if len(pool) <= top_n:
            return pool
        pop = minmax({w.id: _pop_prior(w) for w in pool})
        cb_sim = (serve_cb.pairwise_similarity([w.id for w in pool])
                  if serve_cb.cb_available() else {})
        return mmr_rerank(pool, pop, top_n, cb_sim=cb_sim)

    # 2. STYLE FILTER — explicit choice wins; else the styles the user drinks
    styles = styles or user_styles(db, [w for w, r in liked if r >= 3.0])
    pool = popularity_top_n(db, candidate_pool_size(len(liked)), styles=styles or None)
    liked_ids = {w for w, _ in liked}
    candidates = [w for w in pool if w.id not in liked_ids]
    if not candidates:
        return popularity_top_n(db, top_n, styles=styles or None)
    cand_ids = [w.id for w in candidates]

    warm = len(liked) >= WARM_THRESHOLD and serve_cf.cf_available()

    cb_raw = serve_cb.cb_scores(liked, cand_ids) if serve_cb.cb_available() else {}
    # Drop candidates anti-correlated with the taste profile (negative cosine).
    # Only apply if enough candidates survive — otherwise fall back to ranking all.
    if cb_raw:
        pos_ids = [wid for wid in cand_ids if cb_raw.get(wid, 0.0) >= 0]
        if len(pos_ids) >= top_n:
            cand_ids = pos_ids
            candidates = [w for w in candidates if w.id in set(cand_ids)]
    cb = minmax({wid: cb_raw[wid] for wid in cand_ids if wid in cb_raw})
    cf = minmax(serve_cf.cf_scores(liked, cand_ids)) if warm else {}

    # 3. BLEND
    by_id = {w.id: w for w in candidates}
    pop = minmax({w.id: _pop_prior(w) for w in candidates})

    scores: dict[int, float] = {}
    for wid in cand_ids:
        if warm:
            scores[wid] = (CF_WEIGHT * cf.get(wid, 0.0)
                           + CB_WEIGHT * cb.get(wid, 0.0)
                           + POP_WEIGHT * pop.get(wid, 0.0))
        elif cb:
            scores[wid] = 0.7 * cb.get(wid, 0.0) + 0.3 * pop.get(wid, 0.0)
        else:
            scores[wid] = pop.get(wid, 0.0)

    # 4. MMR rerank: take top 3×top_n by score, diversify via pairwise CB cosine
    mmr_pool_ids = sorted(cand_ids, key=lambda w: scores[w], reverse=True)[: top_n * 3]
    mmr_pool = [by_id[wid] for wid in mmr_pool_ids]
    cb_sim = serve_cb.pairwise_similarity(mmr_pool_ids) if serve_cb.cb_available() else {}
    return mmr_rerank(mmr_pool, scores, top_n, cb_sim=cb_sim)
