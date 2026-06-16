"""
wine_data.py
------------
Shared data layer for the wine ALS pipeline: loading, sparsity filtering, and
confidence-matrix construction. Lives in its own module so both
build_wine_split.py and train_wine_als.py can import these helpers without a
circular dependency.

The confidence weighting (C = 1 + alpha * rating) and sparsity thresholds are
defined HERE so the frozen split and any trainer agree on them by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR   = Path("data/drinks")
MODELS_DIR = Path("models")

WINE_RATINGS_PATH = DATA_DIR / "wine" / "clean_ratings.csv"

MIN_RATINGS_PER_USER = 5   # wine median is 11; drop the long cold-user tail
MIN_RATINGS_PER_ITEM = 5   # drop barely-rated wines that can't get a stable factor
# C = 1 + alpha * rating. Tuned via sweep_wine_alpha.py: alpha=5 gave the best
# NDCG@10 (0.0291 vs 0.0263 at 40). Higher alpha over-saturates confidence so
# ALS can't separate 3-star from 5-star; 5 spreads the signal best.
CONFIDENCE_ALPHA     = 5.0
# The frozen split on disk (models/wine_split/) was built with this alpha when
# the matrix was created. Re-weighting to a different alpha is cheap (recover
# rating = (C-1)/SPLIT_BUILT_ALPHA, re-apply) — see sweep scripts.
SPLIT_BUILT_ALPHA    = 40.0


def load_ratings(path: Path = WINE_RATINGS_PATH) -> pd.DataFrame:
    print(f"Loading wine ratings from {path} ...")
    df = pd.read_csv(path, dtype={"drink_id": "int32", "rating": "float32"})
    print(f"  {len(df):,} ratings, {df['user_id'].nunique():,} users, "
          f"{df['drink_id'].nunique():,} wines")
    return df


def filter_sparse(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively drop cold users/items until both thresholds hold.

    A single pass isn't enough: dropping cold items can push a user below the
    user threshold and vice-versa, so we loop until stable.
    """
    print(f"Filtering: >= {min_user} ratings/user, >= {min_item} ratings/item ...")
    prev = -1
    rounds = 0
    while len(df) != prev:
        prev = len(df)
        rounds += 1
        uc = df.groupby("user_id").size()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        ic = df.groupby("drink_id").size()
        df = df[df["drink_id"].isin(ic[ic >= min_item].index)]
        print(f"  round {rounds}: {len(df):,} ratings, "
              f"{df['user_id'].nunique():,} users, {df['drink_id'].nunique():,} wines")
    return df


def build_matrices(df: pd.DataFrame, alpha: float = CONFIDENCE_ALPHA):
    """Build the user-item confidence CSR matrix and the id<->index maps.

    Returns (confidence_csr, user_ids, item_ids) where confidence_csr is
    shape (n_users, n_items) and the *_ids arrays map matrix index -> raw id.
    """
    print("Building sparse confidence matrix ...")
    user_cat = df["user_id"].astype("category")
    item_cat = df["drink_id"].astype("category")

    user_idx = user_cat.cat.codes.to_numpy()
    item_idx = item_cat.cat.codes.to_numpy()

    # Confidence weighting: C = 1 + alpha * rating. Higher rating -> stronger positive.
    confidence = (1.0 + alpha * df["rating"].to_numpy(dtype=np.float32))

    n_users = user_cat.cat.categories.size
    n_items = item_cat.cat.categories.size
    mat = sp.csr_matrix(
        (confidence, (user_idx, item_idx)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    print(f"  matrix shape {mat.shape}, nnz={mat.nnz:,}, "
          f"density={mat.nnz / (mat.shape[0] * mat.shape[1]) * 100:.4f}%")
    return (
        mat,
        np.asarray(user_cat.cat.categories),
        np.asarray(item_cat.cat.categories),
    )
