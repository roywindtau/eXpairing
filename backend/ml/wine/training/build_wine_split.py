"""
build_wine_split.py
-------------------
Builds and FREEZES a wine train/test split for trustworthy, repeatable
evaluation. Run this ONCE; every model experiment (ALS, popularity, future
tunings) then loads this exact split so all metrics are directly comparable.

WHY A FROZEN leave-k-out SPLIT
------------------------------
Earlier runs used implicit's random train_test_split, which has two problems:
  1. It scatters each user's ratings randomly, so the model can "see the
     future" — a late rating in train predicting an early one in test. Not how
     recommendation works in production (you always predict forward).
  2. It's a single random draw, regenerated per run. A tuning gain smaller than
     the split noise is indistinguishable from luck.

leave_k_out_split holds out exactly K items per eligible user (users with
> K+1 ratings), the standard recsys protocol. Saving it to disk makes the
benchmark fixed: load the same train/test for every experiment.

Saved artifacts (models/wine_split/)
------------------------------------
    train.npz      training CSR (confidence-weighted)
    test.npz       held-out CSR (K items per eligible user)
    user_ids.npy   matrix-row  -> raw user_id
    item_ids.npy   matrix-col  -> raw wine_id
    split_meta.json  parameters used to build the split

Run:
    python -m backend.ml.wine.training.build_wine_split
    python -m backend.ml.wine.training.build_wine_split --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import scipy.sparse as sp
from implicit.evaluation import leave_k_out_split

from backend.ml.wine.training.wine_data import (
    WINE_RATINGS_PATH,
    MIN_RATINGS_PER_USER,
    MIN_RATINGS_PER_ITEM,
    CONFIDENCE_ALPHA,
    load_ratings,
    filter_sparse,
    build_matrices,
)

SPLIT_DIR = Path("models/wine_split")

TRAIN_PATH    = SPLIT_DIR / "train.npz"
TEST_PATH     = SPLIT_DIR / "test.npz"
USER_IDS_PATH = SPLIT_DIR / "user_ids.npy"
ITEM_IDS_PATH = SPLIT_DIR / "item_ids.npy"
META_PATH     = SPLIT_DIR / "split_meta.json"

SPLIT_RANDOM_STATE = 42


def load_split():
    """Load the frozen split. Returns (train, test, user_ids, item_ids).

    Raises FileNotFoundError with a clear hint if the split hasn't been built.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"No frozen split at {SPLIT_DIR}. Build it first:\n"
            f"    python -m backend.ml.wine.training.build_wine_split"
        )
    train = sp.load_npz(TRAIN_PATH)
    test  = sp.load_npz(TEST_PATH)
    user_ids = np.load(USER_IDS_PATH, allow_pickle=True)
    item_ids = np.load(ITEM_IDS_PATH, allow_pickle=True)
    return train, test, user_ids, item_ids


def build(k: int = 5) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ratings(WINE_RATINGS_PATH)
    df = filter_sparse(df, MIN_RATINGS_PER_USER, MIN_RATINGS_PER_ITEM)
    mat, user_ids, item_ids = build_matrices(df, CONFIDENCE_ALPHA)

    print(f"\nleave-{k}-out split (users with > {k + 1} ratings get {k} held out) ...")
    train, test = leave_k_out_split(mat, K=k, random_state=SPLIT_RANDOM_STATE)
    n_test_users = int(np.diff(test.indptr).astype(bool).sum())
    print(f"  train nnz={train.nnz:,} | test nnz={test.nnz:,} | "
          f"{n_test_users:,} users have held-out items")

    print("\nSaving frozen split ...")
    sp.save_npz(TRAIN_PATH, train)
    sp.save_npz(TEST_PATH, test)
    np.save(USER_IDS_PATH, user_ids)
    np.save(ITEM_IDS_PATH, item_ids)

    meta = {
        "built_at":             datetime.now().isoformat(),
        "protocol":             f"leave-{k}-out",
        "k":                    k,
        "random_state":         SPLIT_RANDOM_STATE,
        "confidence_alpha":     CONFIDENCE_ALPHA,
        "min_ratings_per_user": MIN_RATINGS_PER_USER,
        "min_ratings_per_item": MIN_RATINGS_PER_ITEM,
        "n_users":              int(len(user_ids)),
        "n_items":              int(len(item_ids)),
        "train_nnz":            int(train.nnz),
        "test_nnz":             int(test.nnz),
        "n_test_users":         n_test_users,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {SPLIT_DIR}/  (train, test, user_ids, item_ids, split_meta.json)")
    print(f"\nDone. Frozen leave-{k}-out split ready. "
          f"All experiments should load it via build_wine_split.load_split().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Items held out per eligible user.")
    args = parser.parse_args()
    build(k=args.k)
