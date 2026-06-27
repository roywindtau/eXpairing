"""
serve_cf.py
-----------
Collaborative-filtering wine scoring from the trained ALS model.

Loads wine_als_model.npz (user_factors, item_factors, id maps). Our app users
are NOT in the training set (those user_ids are X-Wines ids), so we ALWAYS
fold in: given the wines an app user rated, solve for their latent vector from
the item factors, then score = user_vec · item_factors.

Fold-in (standard ALS user update for a single user):
    given items the user interacted with, with confidence c_i = 1 + alpha*rating,
    x_u = (Yᵀ Cu Y + reg·I)⁻¹ Yᵀ Cu p
    where Y = item factors of rated items, p = 1 (implicit positive), Cu = diag(c_i).

Public API
----------
    cf_available() -> bool
    cf_scores(liked: list[(wine_id, rating)], candidate_ids) -> dict[wine_id, score]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL = Path("models") / "wine_als_model.npz"

ALPHA = 5.0          # confidence: C = 1 + alpha*rating  (matches training)
REG = 0.05           # regularization (matches training)

_state: dict | None = None


def _load() -> dict | None:
    global _state
    if _state is not None:
        return _state or None
    if not _MODEL.exists():
        _state = {}
        return None
    d = np.load(_MODEL, allow_pickle=True)
    item_factors = d["item_factors"].astype(np.float64)
    item_ids = d["item_ids"]
    _state = {
        "item_factors": item_factors,
        "row_of": {int(w): i for i, w in enumerate(item_ids)},
        "f": item_factors.shape[1],
    }
    return _state


def cf_available() -> bool:
    return _load() is not None


def _fold_in(liked, st) -> np.ndarray | None:
    """Solve for the user's latent vector from positively-rated items.
    Ratings < 3 are excluded — they are negative signals and don't belong
    in an implicit-positive model (CB handles dislikes via signed coefficients)."""
    rows, conf = [], []
    for wine_id, rating in liked:
        if float(rating) < 3.0:
            continue
        i = st["row_of"].get(int(wine_id))
        if i is None:
            continue
        rows.append(i)
        conf.append(1.0 + ALPHA * float(rating))
    if not rows:
        return None
    Y = st["item_factors"][rows]                 # (k, f)
    c = np.asarray(conf)                          # (k,)
    f = st["f"]
    # YᵀCuY + reg·I
    A = (Y * c[:, None]).T @ Y + REG * np.eye(f)
    b = (Y * c[:, None]).T @ np.ones(len(rows))   # YᵀCu p, p=1
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def cf_scores(liked, candidate_ids) -> dict[int, float]:
    """
    CF score = folded-in user vector · candidate item factor.
    Returns {wine_id: score} (0.0 when CF unavailable or no usable signal).
    Scores are raw dot products; the blend min-max normalizes them.
    """
    st = _load()
    if st is None or not liked:
        return {int(c): 0.0 for c in candidate_ids}
    x_u = _fold_in(liked, st)
    if x_u is None:
        return {int(c): 0.0 for c in candidate_ids}

    out: dict[int, float] = {}
    for c in candidate_ids:
        i = st["row_of"].get(int(c))
        out[int(c)] = float(x_u @ st["item_factors"][i]) if i is not None else 0.0
    return out
