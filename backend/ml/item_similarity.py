"""
item_similarity.py
------------------
Builds the item-item cosine similarity matrix from user rating data.

COURSE ALGORITHM: ITEM-BASED COLLABORATIVE FILTERING
-----------------------------------------------------
This implements item-based CF exactly as taught in the course:

    sim(i, j) = cos(R_T[i], R_T[j]) = (R_T[i] · R_T[j]) / (‖R_T[i]‖ · ‖R_T[j]‖)

Where R_T is the recipe×user matrix (transpose of the user×recipe matrix),
mean-centered per user to remove rating-scale bias.

Two recipes are similar if the SAME USERS rated them highly — no content
or ingredient information enters the computation.

WHY ITEM-BASED OVER USER-BASED FOR THIS SYSTEM
-----------------------------------------------
  - More robust to data sparsity (~99% of matrix is unknown)
  - Item similarities are more stable than user similarities
    (user tastes change; recipe quality is fixed)
  - Enables cold-start: new users can be scored via item neighborhoods
    without needing a user vector

DATA SPARSITY
-------------
The Food.com user-item matrix has ~1M ratings across 230k recipes
and ~230k users → density ≈ 1M / (230k × 230k) ≈ 0.002%

This extreme sparsity is why matrix factorization (SVD) and item-based
CF are chosen over simple nearest-neighbor approaches.

SAVED ARTIFACTS
---------------
    models/item_sim_matrix.npz     sparse top-K similarity matrix
    models/item_sim_recipe_ids.npy recipe_id for each row/col
    models/item_sim_meta.json      training metadata

Run:
    python -m backend.ml.item_similarity
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import UserEvent, Recipe

MODELS_DIR      = Path("models")
SIM_MATRIX      = MODELS_DIR / "item_sim_matrix.npz"
SIM_RECIPE_IDS  = MODELS_DIR / "item_sim_recipe_ids.npy"
SIM_META        = MODELS_DIR / "item_sim_meta.json"

TOP_K_SIMILAR = 50   # keep only top-K per recipe (memory efficiency)
MIN_RATINGS   = 5    # ignore recipes with fewer ratings (too sparse to be useful)
CHUNK_SIZE    = 500  # rows processed per batch during cosine computation


def load_ratings() -> pd.DataFrame:
    print("Loading ratings ...")
    db = SessionLocal()
    try:
        rows = (
            db.query(UserEvent.user_id, UserEvent.recipe_id, UserEvent.rating)
            .filter(UserEvent.event_type == "rate")
            .filter(UserEvent.rating.isnot(None))
            .all()
        )
    finally:
        db.close()
    df = pd.DataFrame(rows, columns=["user_id", "recipe_id", "rating"])
    print(f"  {len(df):,} ratings, {df['user_id'].nunique():,} users, "
          f"{df['recipe_id'].nunique():,} recipes")
    return df


def build_item_similarity(
    df: pd.DataFrame,
    min_ratings: int = MIN_RATINGS,
) -> tuple[sp.csr_matrix, list[int]]:
    """
    Build item-item cosine similarity using fully sparse operations.

    The full Food.com matrix (231k recipes × 196k users) cannot be
    pivoted into a dense array — 44 billion cells. This implementation:
      1. Filters to recipes with ≥ min_ratings (removes near-empty rows)
      2. Builds a sparse COO/CSR recipe×user matrix directly
      3. Mean-centers per user in-place on the sparse values
      4. L2-normalises each recipe vector (sparse)
      5. Computes cosine in CHUNK_SIZE-row batches, keeping top-K per row
         — the n×n output is never fully materialised in memory

    Returns:
        (sparse top-K sim matrix, list of recipe_ids)
    """
    # Step 1: filter to recipes with enough ratings
    counts = df["recipe_id"].value_counts()
    valid  = set(counts[counts >= min_ratings].index)
    df     = df[df["recipe_id"].isin(valid)].copy()
    print(f"  Kept {len(valid):,} recipes with ≥{min_ratings} ratings "
          f"({len(df):,} ratings remaining)")

    # Step 2: encode IDs → contiguous integers
    recipe_ids  = sorted(df["recipe_id"].unique())
    user_ids    = sorted(df["user_id"].unique())
    recipe_map  = {r: i for i, r in enumerate(recipe_ids)}
    user_map    = {u: i for i, u in enumerate(user_ids)}
    n_recipes   = len(recipe_ids)
    n_users     = len(user_ids)

    r_idx = df["recipe_id"].map(recipe_map).values
    u_idx = df["user_id"].map(user_map).values

    # Step 3: mean-center per user (subtract each user's mean rating)
    user_mean  = df.groupby("user_id")["rating"].transform("mean")
    centered   = (df["rating"] - user_mean).astype(np.float32).values

    # Build sparse recipe×user matrix (R_T in course notation)
    R_T = sp.csr_matrix(
        (centered, (r_idx, u_idx)),
        shape=(n_recipes, n_users),
        dtype=np.float32,
    )
    print(f"  Matrix shape (recipes×users): {R_T.shape}")
    density = R_T.nnz / (n_recipes * n_users) * 100
    print(f"  Density: {density:.4f}%  ({R_T.nnz:,} non-zero entries)")

    # Step 4: L2-normalise recipe vectors (in-place on sparse)
    R_T_norm = normalize(R_T, norm="l2", axis=1)

    # Step 5: chunked cosine → sparse top-K output
    print(f"  Computing cosine similarities in chunks of {CHUNK_SIZE} ...")
    sparse_sim = _chunked_sparse_topk(R_T_norm, top_k=TOP_K_SIMILAR)

    print(f"  Similarity matrix: {sparse_sim.shape}, "
          f"{sparse_sim.nnz:,} non-zero entries")
    return sparse_sim, recipe_ids


def sparsify_top_k(
    matrix: sp.csr_matrix,
    k: int = TOP_K_SIMILAR,
) -> sp.csr_matrix:
    """
    Given an existing similarity matrix, keep only the top-K positive
    values per row and zero out the diagonal (self-similarity).

    Useful in tests and for re-sparsifying a loaded matrix to a lower K.
    """
    mat = sp.csr_matrix(matrix, copy=True)
    # Zero diagonal in-place
    mat.setdiag(0.0)
    mat.eliminate_zeros()

    rows, cols, vals = [], [], []
    for i in range(mat.shape[0]):
        row = mat.getrow(i)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        if len(data) > k:
            top_idx = np.argpartition(data, -k)[-k:]
        else:
            top_idx = np.arange(len(data))
        for idx in top_idx:
            v = float(data[idx])
            if v > 0:
                rows.append(i)
                cols.append(int(indices[idx]))
                vals.append(v)

    return sp.csr_matrix(
        (vals, (rows, cols)), shape=mat.shape, dtype=np.float32
    )


def _chunked_sparse_topk(
    matrix: sp.csr_matrix,
    top_k: int = TOP_K_SIMILAR,
    chunk: int = CHUNK_SIZE,
) -> sp.csr_matrix:
    """
    Compute cosine similarity chunk by chunk, keeping only top-K per row.
    Never allocates the full n×n output matrix — builds COO lists instead.
    """
    n = matrix.shape[0]
    all_rows, all_cols, all_vals = [], [], []

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # (chunk × n_users) @ (n_users × n_recipes) → (chunk × n_recipes) dense
        block = (matrix[start:end] @ matrix.T).toarray().astype(np.float32)

        for local_i, sims in enumerate(block):
            global_i = start + local_i
            sims[global_i] = 0.0          # zero self-similarity
            k = min(top_k, n - 1)
            top_idx = np.argpartition(sims, -k)[-k:]
            for j in top_idx:
                v = float(sims[j])
                if v > 0:
                    all_rows.append(global_i)
                    all_cols.append(j)
                    all_vals.append(v)

        if (start // chunk) % 20 == 0:
            print(f"    {end:,}/{n:,} recipes ...", end="\r")

    print()
    return sp.csr_matrix(
        (all_vals, (all_rows, all_cols)), shape=(n, n), dtype=np.float32
    )


def train() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_ratings()
    if len(df) == 0:
        print("No ratings found. Run seed_ratings.py first.")
        sys.exit(1)

    sparse_sim, recipe_ids = build_item_similarity(df)

    sp.save_npz(SIM_MATRIX, sparse_sim)
    np.save(SIM_RECIPE_IDS, np.array(recipe_ids, dtype=np.int32))

    meta = {
        "trained_at":    datetime.now().isoformat(),
        "n_recipes":     len(recipe_ids),
        "n_ratings":     int(len(df)),
        "n_users":       int(df["user_id"].nunique()),
        "top_k_similar": TOP_K_SIMILAR,
        "min_ratings":   MIN_RATINGS,
        "matrix_nnz":    sparse_sim.nnz,
        "algorithm":     "item-based CF (cosine similarity, mean-centered, sparse)",
    }
    with open(SIM_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {sparse_sim.shape}, {sparse_sim.nnz:,} non-zero entries.")
    print(f"  → {SIM_MATRIX}")


if __name__ == "__main__":
    train()
