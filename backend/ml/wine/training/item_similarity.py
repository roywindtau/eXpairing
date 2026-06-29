"""
item_similarity.py
------------------------
Builds an item-item cosine similarity matrix for wine from real
(non-synthetic) WineEvent ratings. Mirrors backend/ml/item_similarity.py.

Synthetic ratings are EXCLUDED
------------------------------
Real expressed preferences only.

Saved artifacts
---------------
    models/wine_sim_wine.npz       sparse top-K wine sim matrix
    models/wine_sim_wine_ids.npy   wine wine_id per row/col
    models/wine_sim_meta.json      counts + thresholds + timestamp

Run:
    python -m backend.ml.wine.training.item_similarity
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import WineEvent

MODELS_DIR = Path("models")
SIM_WINE     = MODELS_DIR / "wine_sim_wine.npz"
SIM_WINE_IDS = MODELS_DIR / "wine_sim_wine_ids.npy"
SIM_META     = MODELS_DIR / "wine_sim_meta.json"

TOP_K          = 50
MIN_RATINGS_WINE = 2    # X-Wines Test is small — 5 would drop almost all wines
CHUNK_SIZE     = 200


def _load_wine_ratings() -> pd.DataFrame:
    """All real (synthetic=False) wine ratings."""
    db = SessionLocal()
    try:
        rows = (
            db.query(WineEvent.user_id, WineEvent.wine_id, WineEvent.rating)
            .filter(WineEvent.event_type == "rate")
            .filter(WineEvent.rating.isnot(None))
            .filter(WineEvent.synthetic == False)  # noqa: E712
            .all()
        )
    finally:
        db.close()
    return pd.DataFrame(rows, columns=["user_id", "wine_id", "rating"])


def _build_sim(df: pd.DataFrame, min_ratings: int) -> tuple[sp.csr_matrix, list[int]]:
    """Mean-center per user, L2-normalize, sparse chunked cosine top-K."""
    if df.empty:
        return sp.csr_matrix((0, 0), dtype=np.float32), []

    counts = df["wine_id"].value_counts()
    valid  = set(counts[counts >= min_ratings].index)
    df     = df[df["wine_id"].isin(valid)].copy()
    print(f"  Kept {len(valid):,} wines with >= {min_ratings} ratings  "
          f"({len(df):,} rating rows remaining)")

    if df.empty:
        return sp.csr_matrix((0, 0), dtype=np.float32), []

    wine_ids = sorted(df["wine_id"].unique())
    user_ids  = sorted(df["user_id"].unique())
    d_map = {d: i for i, d in enumerate(wine_ids)}
    u_map = {u: i for i, u in enumerate(user_ids)}
    n_wines = len(wine_ids)
    n_users  = len(user_ids)

    user_mean = df.groupby("user_id")["rating"].transform("mean")
    centered  = (df["rating"] - user_mean).astype(np.float32).values
    r_idx = df["wine_id"].map(d_map).values
    u_idx = df["user_id"].map(u_map).values

    R_T = sp.csr_matrix(
        (centered, (r_idx, u_idx)),
        shape=(n_wines, n_users),
        dtype=np.float32,
    )
    print(f"  wine x user matrix: {R_T.shape}  ({R_T.nnz:,} non-zero)")

    R_T_norm = normalize(R_T, norm="l2", axis=1)
    sim = _chunked_sparse_topk(R_T_norm, TOP_K, CHUNK_SIZE)
    print(f"  similarity matrix: {sim.shape}  ({sim.nnz:,} non-zero)")
    return sim, wine_ids


def _chunked_sparse_topk(
    matrix: sp.csr_matrix,
    top_k: int,
    chunk: int,
) -> sp.csr_matrix:
    """Compute cosine sim in chunks, keeping only top-K per row, zeroing diagonal."""
    n = matrix.shape[0]
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = (matrix[start:end] @ matrix.T).toarray().astype(np.float32)
        for local_i, sims in enumerate(block):
            global_i = start + local_i
            sims[global_i] = 0.0
            k = min(top_k, n - 1)
            if k <= 0:
                continue
            top_idx = np.argpartition(sims, -k)[-k:]
            for j in top_idx:
                v = float(sims[j])
                if v > 0:
                    rows.append(global_i)
                    cols.append(int(j))
                    vals.append(v)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n=== Wine item-similarity ===")
    wine_df = _load_wine_ratings()
    print(f"  Loaded {len(wine_df):,} wine ratings.")
    wine_sim, wine_ids = _build_sim(wine_df, MIN_RATINGS_WINE)
    sp.save_npz(SIM_WINE, wine_sim)
    np.save(SIM_WINE_IDS, np.array(wine_ids, dtype=np.int64))
    print(f"  Saved -> {SIM_WINE}  ({wine_sim.shape}, {wine_sim.nnz:,} nnz)")

    meta = {
        "trained_at":      datetime.now().isoformat(),
        "n_wines":         len(wine_ids),
        "wine_sim_nnz":    int(wine_sim.nnz),
        "top_k":           TOP_K,
        "min_ratings_wine": MIN_RATINGS_WINE,
        "synthetic_excluded": True,
        "algorithm":       "item-based CF (cosine, mean-centered, sparse top-K)",
    }
    with open(SIM_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Saved meta -> {SIM_META}")


if __name__ == "__main__":
    train()
