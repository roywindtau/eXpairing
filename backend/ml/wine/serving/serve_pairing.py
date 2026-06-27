"""
serve_pairing.py
----------------
MODULE 4 of the wine<->recipe pairing feature: the pairing scorer.

WHAT IT DOES
------------
Given a recipe (its ingredient list), rank wines by how well they pair with it.
Both sides live in the same 12-dim food-category space:

    recipe  --Module 3 (recipe_categories.recipe_vector)--> 12-dim vector
    wines   --Module 2 (build_wine_pairing_vectors)-------> 12-dim matrix (precomputed)

The pairing score BLENDS two signals:

  1. COSINE SIMILARITY between the recipe vector and each wine vector. Both are
     L2-normalized, so cosine reduces to a dot product: a wine scores high when
     its harmonize-derived categories overlap the recipe's ingredient-derived
     categories (a Seafood recipe matches a wine that pairs with Seafood).

  2. RULE QUALITY from the empirical (wine_category x food_category) table
     extracted from the labeled pairing dataset (data.pairing.extract_pairing_rules
     -> models/pairing_rules.json). This captures sommelier logic that harmonize
     overlap misses -- e.g. Sparkling x Seafood = 3.96 (best cell), Red x Red Meat
     beats White x Red Meat -- keyed off the wine's STYLE and the recipe's food
     category mix.

    final = ALPHA_COSINE * cosine + BETA_RULES * rule_quality

Both components are in [0, 1]. If the rule table isn't built, falls back to pure
cosine.

Public API
----------
    pairing_available() -> bool
    pair_wines(ingredients, top_n, style_filter=None) -> list[(wine_id, score)]

Artifacts loaded (built by data.pairing.build_wine_pairing_vectors):
    models/wine_pair_matrix.npz   (n_wines x 12) L2-normalized
    models/wine_pair_ids.npy      wine_id per row
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data.pairing.pairing_vocabulary import CATEGORIES, CATEGORY_INDEX
from data.pairing.recipe_categories import recipe_vector

MODELS_DIR   = Path("models")
_PAIR_MATRIX = MODELS_DIR / "wine_pair_matrix.npz"
_PAIR_IDS    = MODELS_DIR / "wine_pair_ids.npy"
_RULES       = MODELS_DIR / "pairing_rules.json"

# blend weights: cosine (harmonize overlap) vs the empirical rule table.
ALPHA_COSINE = 0.6
BETA_RULES   = 0.4

# wine.style -> the rule table's wine_category
_STYLE_TO_WINE_CAT = {
    "Red": "Red", "White": "White", "Rosé": "Rosé", "Rose": "Rosé",
    "Sparkling": "Sparkling", "Dessert": "Dessert",
    "Dessert/Port": "Fortified", "Port": "Fortified", "Fortified": "Fortified",
}

# Rule quality (1-5) for food categories the empirical CSV does NOT cover.
# Sourced from sommelier guidance rather than the data. Currently: "Nutty"
# (tahini/sesame/tree nuts) -> aromatic/nutty WHITES, lightly Sparkling, poor Red.
#   Sources: winefolly.com (wine & hummus), bonterra.com (white wine pairing guide)
_WEB_RULES: dict[str, dict[str, float]] = {
    "Nutty": {
        "White": 4.3, "Sparkling": 3.6, "Dessert": 3.4, "Fortified": 3.8,
        "Rosé": 3.0, "Red": 2.2,
    },
}

_state: dict | None = None


def _load() -> dict | None:
    """Lazy-load the wine pairing matrix + rule table once. None if not built."""
    global _state
    if _state is not None:
        return _state or None
    if not (_PAIR_MATRIX.exists() and _PAIR_IDS.exists()):
        _state = {}
        return None
    mat = sp.load_npz(_PAIR_MATRIX).tocsr().astype(np.float64)
    ids = np.load(_PAIR_IDS)

    st: dict = {"mat": mat, "ids": ids, "rule_mat": None}

    # Build a per-wine rule-quality matrix (n_wines x 12): for each wine, its
    # empirical quality against each food category, derived from its style.
    if _RULES.exists():
        rules = json.loads(_RULES.read_text(encoding="utf-8"))
        quality = rules["quality"]
        gmean = rules["global_mean"]
        lo, hi = rules["scale"]

        # wine_category quality row per food category, normalized to [0,1].
        # cat_row[wine_cat] = np.array over the food categories.
        # Start every cell at the global mean so unseen (wine_cat, food_cat) pairs
        # are neutral rather than zero.
        wine_cats = set(quality) | {wc for fmap in _WEB_RULES.values() for wc in fmap}
        cat_row: dict[str, np.ndarray] = {
            wc: np.full(len(CATEGORIES), gmean, dtype=np.float64) for wc in wine_cats
        }
        # 1) fill from the empirical CSV table
        for wc, fmap in quality.items():
            for fc, q in fmap.items():
                if fc in CATEGORY_INDEX:
                    cat_row[wc][CATEGORY_INDEX[fc]] = q
        # 2) overlay web-sourced rules for categories the CSV doesn't cover (Nutty)
        for fc, wmap in _WEB_RULES.items():
            if fc not in CATEGORY_INDEX:
                continue
            for wc, q in wmap.items():
                cat_row.setdefault(wc, np.full(len(CATEGORIES), gmean))[CATEGORY_INDEX[fc]] = q
        # normalize all rows to [0,1]
        for wc in cat_row:
            cat_row[wc] = (cat_row[wc] - lo) / (hi - lo)

        # map each wine row -> its normalized quality vector via style.
        from backend.db.database import SessionLocal
        from backend.db.models import Wine
        db = SessionLocal()
        try:
            style_of = {w_id: style for w_id, style in
                        db.query(Wine.id, Wine.style).all()}
        finally:
            db.close()

        default_row = np.full(len(CATEGORIES),
                              (gmean - lo) / (hi - lo), dtype=np.float64)
        rule_mat = np.empty((len(ids), len(CATEGORIES)), dtype=np.float64)
        for i, wid in enumerate(ids):
            wc = _STYLE_TO_WINE_CAT.get(style_of.get(int(wid)) or "", None)
            rule_mat[i] = cat_row.get(wc, default_row)
        st["rule_mat"] = rule_mat

    _state = st
    return _state


def pairing_available() -> bool:
    return _load() is not None


def pair_wines(
    ingredients: list[str],
    top_n: int = 10,
    candidate_rows: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """
    Rank wines by pairing fit for a recipe's ingredients.

    ingredients    : recipe ingredient strings
    top_n          : how many wines to return
    candidate_rows : optional row-index subset to restrict scoring to (used when
                     the caller has already style-filtered the catalog).

    Returns [(wine_id, score)] sorted desc by score. Score is cosine in [0, 1]
    (vectors are non-negative). Empty if the matrix isn't built or the recipe
    produced a zero vector.
    """
    st = _load()
    if st is None:
        return []

    rvec = recipe_vector(ingredients)          # 12-dim, L2-normalized
    if not np.any(rvec):
        return []

    mat = st["mat"]
    ids = st["ids"]
    rule_mat = st.get("rule_mat")
    if candidate_rows is not None:
        mat = mat[candidate_rows]
        ids = ids[candidate_rows]
        if rule_mat is not None:
            rule_mat = rule_mat[candidate_rows]

    # cosine == dot product: both sides already unit-normalized. In [0,1].
    cosine = mat.dot(rvec)                        # (n_wines,)

    if rule_mat is not None:
        # rule score: each wine's empirical quality (by style) against the
        # recipe's food-category mix. rvec is the (normalized) category weighting,
        # so this is a weighted average of the wine's per-category rule quality.
        rule = rule_mat.dot(rvec) / (rvec.sum() or 1.0)   # ~[0,1]
        scores = ALPHA_COSINE * cosine + BETA_RULES * rule
    else:
        scores = cosine

    k = min(top_n, scores.shape[0])
    if k <= 0:
        return []
    # argpartition for the top-k, then sort just those.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(ids[i]), float(scores[i])) for i in top_idx if scores[i] > 0.0]
