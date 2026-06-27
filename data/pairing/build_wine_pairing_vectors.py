"""
build_wine_pairing_vectors.py
=============================
MODULE 2 of the wine<->recipe pairing feature: wine -> category vector.

WHAT IT DOES
------------
Turns every wine into a 12-dim vector over the canonical food categories
(Module 1, pairing_vocabulary.CATEGORIES), using only the wine's harmonize_csv.

    Cabernet Sauvignon  harmonize="Beef,Lamb,Poultry"
        Beef    -> Red Meat
        Lamb    -> Red Meat   (same category, counts once)
        Poultry -> Poultry
        => presence set: {Red Meat, Poultry}
        => vector (before norm): [1,1,0,0,0,0,0,0,0,0,0,0]
        => L2-normalized:        [0.71,0.71,0, ...]

WHY PRESENCE (BINARY), THEN L2-NORMALIZE
----------------------------------------
- Presence, not counts: a category is either part of the wine's pairing profile
  or it isn't. Listing Beef AND Lamb doesn't make a wine "more Red Meat" than one
  that lists only Beef -- both pair with red meat. So each category contributes at
  most 1, regardless of how many tokens mapped to it.
- L2 normalize: makes every wine vector unit length, so later cosine similarity
  against a recipe vector is a clean dot product and wines with broad harmonize
  lists (many categories) don't dominate purely by spanning more categories. This
  matches the convention already used by the structured CB matrix
  (backend/ml/wine/training/train_cb.py).

SAVED ARTIFACTS (mirrors the existing wine_cb_* naming)
-------------------------------------------------------
    models/wine_pair_matrix.npz   sparse (n_wines x 12) L2-normalized matrix
    models/wine_pair_ids.npy      wine_id for each row
    models/wine_pair_meta.json    categories, counts, coverage stats

USED BY
-------
    Module 4 (pairing scorer) loads these to rank wines against a recipe vector.

Run:
    python -m data.pairing.build_wine_pairing_vectors
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import Wine
from data.pairing.pairing_vocabulary import (
    CATEGORIES,
    CATEGORY_INDEX,
    HARMONIZE_TO_CATEGORY,
)

MODELS_DIR  = Path("models")
PAIR_MATRIX = MODELS_DIR / "wine_pair_matrix.npz"
PAIR_IDS    = MODELS_DIR / "wine_pair_ids.npy"
PAIR_META   = MODELS_DIR / "wine_pair_meta.json"


def _tokens(harmonize_csv: str | None) -> list[str]:
    return [t.strip() for t in (harmonize_csv or "").split(",") if t.strip()]


def build() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading wines from DB ...")
    db = SessionLocal()
    try:
        wines = db.query(Wine.id, Wine.harmonize_csv).all()
    finally:
        db.close()
    if not wines:
        print("No wines. Run `python -m backend.db.wine.seed_wines` first.")
        sys.exit(1)
    print(f"  {len(wines):,} wines.")

    n_cat = len(CATEGORIES)
    rows_i, cols_i, vals = [], [], []
    ids = np.empty(len(wines), dtype=np.int64)

    n_empty = 0              # wines whose harmonize maps to NO category
    unknown_tokens: set[str] = set()

    for row, (wid, harmonize) in enumerate(wines):
        ids[row] = wid
        present = np.zeros(n_cat, dtype=np.float64)   # binary: category present or not
        for tok in _tokens(harmonize):
            if tok not in HARMONIZE_TO_CATEGORY:
                unknown_tokens.add(tok)             # not in Module 1's map
                continue
            cats = HARMONIZE_TO_CATEGORY[tok]
            if cats is None:                        # intentionally unmapped
                continue
            for c in cats:
                present[CATEGORY_INDEX[c]] = 1.0    # presence only; counts once

        counts = present
        norm = np.linalg.norm(counts)
        if norm == 0:
            n_empty += 1
            continue                                 # leave row all-zero
        counts /= norm
        for col in np.nonzero(counts)[0]:
            rows_i.append(row); cols_i.append(int(col)); vals.append(float(counts[col]))

    mat = sp.coo_matrix((vals, (rows_i, cols_i)),
                        shape=(len(wines), n_cat)).tocsr()

    sp.save_npz(PAIR_MATRIX, mat)
    np.save(PAIR_IDS, ids)
    meta = {
        "built_at": datetime.now().isoformat(),
        "n_wines": len(wines),
        "n_categories": n_cat,
        "categories": CATEGORIES,
        "n_empty_vectors": n_empty,
        "pct_empty": round(100 * n_empty / len(wines), 2),
        "unknown_tokens": sorted(unknown_tokens),
        "normalization": "per-wine L2 over binary category presence",
    }
    PAIR_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  Saved -> {PAIR_MATRIX}  ({mat.shape[0]:,}x{mat.shape[1]}, nnz={mat.nnz:,})")
    print(f"  Saved -> {PAIR_IDS}")
    print(f"  Saved -> {PAIR_META}")
    print(f"\n  empty (no category) wines: {n_empty:,} ({meta['pct_empty']}%)")
    if unknown_tokens:
        print(f"  !! UNKNOWN tokens (add to pairing_vocabulary): {sorted(unknown_tokens)}")
    else:
        print("  OK: no unknown tokens.")


if __name__ == "__main__":
    build()
