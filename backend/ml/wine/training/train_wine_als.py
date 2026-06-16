"""
train_wine_als.py
-----------------
Trains a confidence-weighted ALS matrix factorization model for WINE ratings,
using the `implicit` library.

WHY ALS, NOT SVD, FOR WINE
--------------------------
The wine dataset is ~21M ratings, ~1.06M users, ~100K items at 0.02% density.
Surprise's pure-Python SGD SVD does not scale to this comfortably (hours per
fit, high RAM, 3x for CV). `implicit`'s ALS is compiled, multi-threaded, and
built for exactly this sparse regime. (See docs/wine-cf-experiments.md for the
ALS-vs-SVD ranking comparison.)

EXPLICIT -> IMPLICIT CONFIDENCE
-------------------------------
ALS in `implicit` models IMPLICIT feedback: it factorizes a preference matrix
P (1 = interacted) weighted by a confidence matrix C. Our wine ratings are
EXPLICIT 1-5, so we map rating -> confidence:  C = 1 + alpha * rating.
A 5-star rating becomes a high-confidence positive; a 1-star a low-confidence
one. We do NOT try to predict the 1-5 value back (so RMSE is meaningless here);
we evaluate RANKING quality instead — Precision@K, MAP@K, NDCG@K — which is what
a recommender is actually judged on.

This is an OFFLINE EXPERIMENT: it saves the model + ranking metrics. It does NOT
touch the serving layer (serve_cf.py still serves wine via item-sim).

Saved artifacts
---------------
    models/drink_wine_als_model.npz   user_factors + item_factors + id maps
    models/drink_wine_als_meta.json   ranking metrics + hyperparams + timestamp

Run:
    python -m backend.ml.wine.training.train_wine_als
    python -m backend.ml.wine.training.train_wine_als --factors 64 --iterations 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
from implicit.als import AlternatingLeastSquares

from backend.ml.wine.training.wine_data import (
    MODELS_DIR, CONFIDENCE_ALPHA, SPLIT_BUILT_ALPHA,
)
from backend.ml.wine.training.build_wine_split import load_split
from backend.ml.wine.training.eval_wine_model import evaluate, print_metrics


def train(factors: int = 64, iterations: int = 20, regularization: float = 0.05,
          eval_k: int = 10) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "drink_wine_als_model.npz"
    meta_path  = MODELS_DIR / "drink_wine_als_meta.json"

    # Load the FROZEN leave-k-out split so every experiment is comparable.
    # The split already baked in confidence weighting + sparsity filtering,
    # so we no longer load/filter/split here (build_wine_split.py owns that).
    print("Loading frozen wine split (models/wine_split/) ...")
    train_mat, _test_mat, user_ids, item_ids = load_split()
    print(f"  train nnz={train_mat.nnz:,} | {len(user_ids):,} users x "
          f"{len(item_ids):,} wines")

    # The split baked in SPLIT_BUILT_ALPHA; re-weight to the tuned CONFIDENCE_ALPHA
    # (recover rating = (C-1)/SPLIT_BUILT_ALPHA, re-apply). See sweep_wine_alpha.py.
    ratings = (train_mat.data - 1.0) / SPLIT_BUILT_ALPHA
    train_mat.data = (1.0 + CONFIDENCE_ALPHA * ratings).astype("float32")
    print(f"  re-weighted to alpha={CONFIDENCE_ALPHA} (tuned)")

    print(f"\nTraining ALS (factors={factors}, iterations={iterations}, "
          f"reg={regularization}, alpha={CONFIDENCE_ALPHA}) ...")
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
        calculate_training_loss=True,   # prints loss per iteration
    )
    # implicit shows its own per-iteration tqdm bar when show_progress=True.
    model.fit(train_mat, show_progress=True)

    print(f"\nEvaluating ranking @ {eval_k} on frozen leave-k-out test set ...")
    metrics = evaluate(model, k=eval_k)
    print_metrics("WINE ALS", metrics)

    print("\nSaving model artifact ...")
    np.savez_compressed(
        model_path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
    )

    meta = {
        "trained_at":           datetime.now().isoformat(),
        "algorithm":            "Confidence-weighted ALS (implicit)",
        "kind":                 "wine",
        "source":               "frozen leave-k-out split (models/wine_split/)",
        "n_train_ratings":      int(train_mat.nnz),
        "n_users":              int(len(user_ids)),
        "n_wines":              int(len(item_ids)),
        "factors":              factors,
        "iterations":           iterations,
        "regularization":       regularization,
        "eval_k":               eval_k,
        f"precision_at_{eval_k}": round(metrics["precision"], 4),
        f"map_at_{eval_k}":       round(metrics["map"], 4),
        f"ndcg_at_{eval_k}":      round(metrics["ndcg"], 4),
        "auc":                  round(metrics["auc"], 4),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {model_path}")
    print(f"  Saved -> {meta_path}")
    print(f"\nDone. ALS trained on {train_mat.nnz:,} train ratings from "
          f"{len(user_ids):,} users x {len(item_ids):,} wines.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # alpha is baked into the frozen split (wine_data.CONFIDENCE_ALPHA), so it's
    # not a per-run arg here — train() consumes the split as-is.
    parser.add_argument("--factors",        type=int,   default=64)
    parser.add_argument("--iterations",     type=int,   default=15)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--eval-k",         type=int,   default=10)
    args = parser.parse_args()
    train(
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
        eval_k=args.eval_k,
    )
