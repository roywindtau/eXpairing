"""
train_cb.py
-----------
Precompute the structured content-based feature matrix for every wine.

NOT TF-IDF, NOT neural embeddings. X-Wines has zero free text, so each wine is
encoded as a structured weighted vector (see memory wine-cb-design):

    grape    multi-hot   (block unit-normalized)
    region   one-hot on rolled-up parent (region_rollup.json), unit-length
    acidity  ordinal     Low/Med/High -> 0/.5/1
    body     ordinal     5 levels -> 0..1
    abv      numeric     min-max over a clipped [5,16] band

IMPORTANT: this saves the matrix UNWEIGHTED (each block normalized, no sommelier
weights applied). Weights are applied at SERVE time so they can be retuned — or
overridden per-user/per-request ("emphasize grape") — without recomputing 100k
vectors. style is NOT in the matrix; it's a serve-time hard filter.

Saved artifacts
---------------
    models/wine_cb_matrix.npz    sparse unweighted matrix (n_wines x dim)
    models/wine_cb_ids.npy       wine_id for each row
    models/wine_cb_blocks.json   block layout + vocab (col ranges per field, index maps)
    models/wine_cb_meta.json     stats

Run:
    python -m backend.ml.wine.training.train_cb
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.db.database import SessionLocal
from backend.db.models import Wine

MODELS_DIR = Path("models")
CB_MATRIX  = MODELS_DIR / "wine_cb_matrix.npz"
CB_IDS     = MODELS_DIR / "wine_cb_ids.npy"
CB_BLOCKS  = MODELS_DIR / "wine_cb_blocks.json"
CB_META    = MODELS_DIR / "wine_cb_meta.json"
ROLLUP     = MODELS_DIR / "region_rollup.json"

ACIDITY = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
BODY = {"Very light-bodied": 0.0, "Light-bodied": 0.25, "Medium-bodied": 0.5,
        "Full-bodied": 0.75, "Very full-bodied": 1.0}
ABV_LO, ABV_HI = 5.0, 16.0   # clip band: raw abv range is a dirty 0..50


def _grapes(w: Wine) -> list[str]:
    return [g.strip() for g in (w.grapes_csv or "").split(",") if g.strip()]


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    if not ROLLUP.exists():
        print(f"Missing {ROLLUP}. Run: python -m backend.ml.wine.region_rollup")
        sys.exit(1)
    rollup = json.loads(ROLLUP.read_text(encoding="utf-8"))

    print("Loading wines from DB ...")
    db = SessionLocal()
    try:
        wines = db.query(Wine).all()
    finally:
        db.close()
    if not wines:
        print("No wines. Run `python -m backend.db.wine.seed_wines` first.")
        sys.exit(1)
    print(f"  {len(wines):,} wines.")

    # ── build vocabularies ───────────────────────────────────────────────
    grapes = sorted({g for w in wines for g in _grapes(w)})
    regions = sorted({rollup.get(w.region, w.country or "?") for w in wines if w.region})
    g_idx = {g: i for i, g in enumerate(grapes)}
    r_idx = {r: i for i, r in enumerate(regions)}
    n_g, n_r = len(grapes), len(regions)
    # column layout: [grape block | region block | acidity | body | abv]
    off_grape, off_region = 0, n_g
    col_acid, col_body, col_abv = n_g + n_r, n_g + n_r + 1, n_g + n_r + 2
    dim = n_g + n_r + 3
    print(f"  vocab: {n_g} grapes, {n_r} parent-regions  ->  dim {dim}")

    # ── build sparse matrix (COO) ────────────────────────────────────────
    rows_i, cols_i, vals = [], [], []
    ids = np.empty(len(wines), dtype=np.int64)
    for row, w in enumerate(wines):
        ids[row] = w.id
        # grape multi-hot, block unit-normalized
        gs = _grapes(w)
        if gs:
            v = 1.0 / np.sqrt(len(gs))   # unit norm of an all-ones multi-hot
            for g in gs:
                rows_i.append(row); cols_i.append(off_grape + g_idx[g]); vals.append(v)
        # region one-hot (already unit length)
        if w.region:
            rows_i.append(row); cols_i.append(off_region + r_idx[rollup.get(w.region, w.country or "?")]); vals.append(1.0)
        # ordinal/numeric scalars
        rows_i.append(row); cols_i.append(col_acid); vals.append(ACIDITY.get(w.acidity, 0.5))
        rows_i.append(row); cols_i.append(col_body); vals.append(BODY.get(w.body, 0.5))
        abv = min(max(w.abv if w.abv is not None else 13.0, ABV_LO), ABV_HI)
        rows_i.append(row); cols_i.append(col_abv); vals.append((abv - ABV_LO) / (ABV_HI - ABV_LO))

    mat = sp.coo_matrix((vals, (rows_i, cols_i)), shape=(len(wines), dim)).tocsr()

    # ── save ─────────────────────────────────────────────────────────────
    sp.save_npz(CB_MATRIX, mat)
    np.save(CB_IDS, ids)
    blocks = {
        "dim": dim,
        "blocks": {
            "grape":   {"start": off_grape,  "end": n_g,        "type": "multihot"},
            "region":  {"start": off_region, "end": n_g + n_r,  "type": "onehot"},
            "acidity": {"col": col_acid, "type": "scalar"},
            "body":    {"col": col_body, "type": "scalar"},
            "abv":     {"col": col_abv,  "type": "scalar"},
        },
        "grape_vocab":  g_idx,
        "region_vocab": r_idx,
    }
    CB_BLOCKS.write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    meta = {
        "trained_at": datetime.now().isoformat(),
        "approach": "structured weighted vector (unweighted at rest; weights applied at serve)",
        "n_wines": len(wines), "dim": dim,
        "n_grapes": n_g, "n_parent_regions": n_r,
        "nnz": int(mat.nnz), "avg_nnz_per_wine": round(mat.nnz / len(wines), 2),
    }
    CB_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  Saved -> {CB_MATRIX}  ({mat.shape[0]:,}x{mat.shape[1]}, nnz={mat.nnz:,})")
    print(f"  Saved -> {CB_IDS}")
    print(f"  Saved -> {CB_BLOCKS}")
    print(f"  Saved -> {CB_META}")
    print(f"\nDone. {len(wines):,} wines -> {dim}-dim structured vectors (unweighted).")


if __name__ == "__main__":
    train()
