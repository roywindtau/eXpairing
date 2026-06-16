"""
eval_wine_popularity.py
-----------------------
Popularity baseline for wine ranking — the floor that ALS must beat.

Recommends the most-rated wines to EVERY user (no personalization). If ALS
can't beat this, its latent factors collapsed into popularity and the whole
point of matrix factorization (per-user preference) was lost.

To make the comparison apples-to-apples, this reuses the SAME pipeline as
train_wine_als.py: same load, same sparse matrix, same train_test_split
(same random_state), same ranking_metrics_at_k harness. The only difference
is the "model": instead of ALS factors, we hand the evaluator a rank-1
factorization where every user vector is identical and the single item
factor is item popularity. That makes every user's ranking = global
popularity order, scored by implicit's own metric code.

Run:
    python -m backend.ml.wine.training.eval_wine_popularity
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase

from backend.ml.wine.training.build_wine_split import load_split
from backend.ml.wine.training.eval_wine_model import evaluate, print_metrics


class PopularityModel(MatrixFactorizationBase):
    """A rank-1 MF whose score for (user, item) == item popularity.

    Every user shares the same factor [1.0] and the single item factor is
    item popularity, so dot(user, item) == popularity for all users. The
    per-user ranking is therefore identical and equals the global popularity
    order. This lets implicit's ranking_metrics_at_k evaluate popularity
    through the exact same code path used for ALS.
    """

    def __init__(self, item_popularity: np.ndarray, n_users: int):
        super().__init__()
        n_items = item_popularity.shape[0]
        self.item_factors = item_popularity.reshape(n_items, 1).astype(np.float32)
        self.user_factors = np.ones((n_users, 1), dtype=np.float32)

    def fit(self, *args, **kwargs):  # abstract in base; unused (no training)
        pass

    def save(self, *args, **kwargs):  # abstract in base; unused
        pass


def main(eval_k: int = 10) -> None:
    train_mat, _test_mat, _user_ids, _item_ids = load_split()

    # Popularity = number of users who rated each item, computed on TRAIN only
    # (binarize so confidence weighting doesn't inflate counts).
    print("Computing item popularity from train set ...")
    train_bin = train_mat.copy()
    train_bin.data[:] = 1.0
    item_pop = np.asarray(train_bin.sum(axis=0)).ravel().astype(np.float32)
    print(f"  most-rated item has {int(item_pop.max()):,} raters; "
          f"{int((item_pop == 0).sum()):,} items unseen in train")

    model = PopularityModel(item_pop, n_users=train_mat.shape[0])

    metrics = evaluate(model, k=eval_k)
    print_metrics("POPULARITY BASELINE", metrics)


if __name__ == "__main__":
    main()
