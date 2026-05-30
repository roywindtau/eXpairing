"""
train_cf.py
-----------
Trains a biased MF (Funk SVD) model for beer or wine ratings.

Loads ratings directly from CSV — no DB dependency.
Produces one model per kind, trained independently.

Saved artifacts
---------------
    models/drink_beer_cf_model.pkl   trained Surprise SVD for beer
    models/drink_beer_cf_meta.json   RMSE/MAE + hyperparams + timestamp
    models/drink_wine_cf_model.pkl   trained Surprise SVD for wine
    models/drink_wine_cf_meta.json   RMSE/MAE + hyperparams + timestamp

Run:
    python -m backend.ml.drinks.training.train_cf --kind beer
    python -m backend.ml.drinks.training.train_cf --kind wine
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

DATA_DIR    = Path("data/drinks")
MODELS_DIR  = Path("models")

BEER_RATINGS_PATH = DATA_DIR / "beer" / "clean_beer_ratings.csv"
WINE_RATINGS_PATH = DATA_DIR / "wine" / "clean_ratings.csv"

MIN_RATINGS_PER_USER = 3


def load_ratings(path: Path, kind: str) -> pd.DataFrame:
    """Load clean ratings CSV (user_id, drink_id, rating).

    Both clean_beer_ratings.csv and clean_ratings.csv share this schema,
    produced by data/drinks/beer/clean_beer.py and wine/clean_wines.py respectively.
    Cleaning already happened upstream — this just loads.
    """
    print(f"Loading {kind} ratings from {path} ...")
    df = pd.read_csv(path, dtype={"drink_id": "int32", "rating": "float32"})
    print(f"  {len(df):,} ratings, {df['user_id'].nunique():,} users, "
          f"{df['drink_id'].nunique():,} {kind}s")
    density = len(df) / (df["user_id"].nunique() * df["drink_id"].nunique()) * 100
    print(f"  Matrix density: {density:.4f}%")
    return df


def filter_active_users(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    counts = df.groupby("user_id").size()
    active = counts[counts >= min_ratings].index
    out    = df[df["user_id"].isin(active)]
    dropped = df["user_id"].nunique() - out["user_id"].nunique()
    print(f"  Dropped {dropped:,} users with < {min_ratings} ratings. "
          f"{out['user_id'].nunique():,} users remain.")
    return out


def train(kind: str = "beer", n_factors: int = 30, n_epochs: int = 15) -> None:
    assert kind in ("beer", "wine"), f"kind must be 'beer' or 'wine', got {kind!r}"
    MODELS_DIR.mkdir(exist_ok=True)

    cf_model = MODELS_DIR / f"drink_{kind}_cf_model.pkl"
    cf_meta  = MODELS_DIR / f"drink_{kind}_cf_meta.json"

    ratings_path = BEER_RATINGS_PATH if kind == "beer" else WINE_RATINGS_PATH
    df = load_ratings(ratings_path, kind)
    if len(df) == 0:
        print(f"No {kind} ratings found. Check data/drinks/ for the ratings CSV.")
        sys.exit(1)

    df = filter_active_users(df, MIN_RATINGS_PER_USER)
    if len(df) == 0:
        print("After filtering, no ratings left to train on. Lower MIN_RATINGS_PER_USER.")
        sys.exit(1)

    # Beer is 0-5, wine is 1-5
    rating_scale = (0, 5) if kind == "beer" else (1, 5)
    reader  = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(df[["user_id", "drink_id", "rating"]], reader)

    print(f"\nTraining biased MF (n_factors={n_factors}, n_epochs={n_epochs}) on {kind} ratings ...")
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=True,   # Surprise prints per-epoch SGD progress
    )

    cv_rmse = cv_mae = None
    if len(df) >= 100:
        print("Running 3-fold cross-validation ...")
        cv = cross_validate(model, dataset, measures=["RMSE", "MAE"], cv=3, verbose=True)
        cv_rmse = float(cv["test_rmse"].mean())
        cv_mae  = float(cv["test_mae"].mean())
        print(f"  CV RMSE: {cv_rmse:.4f}  |  MAE: {cv_mae:.4f}")

    print("Training final model on full dataset ...")
    trainset = dataset.build_full_trainset()
    model.fit(trainset)

    with open(cf_model, "wb") as f:
        pickle.dump(model, f)

    item_key = "n_beers" if kind == "beer" else "n_wines"
    meta = {
        "trained_at":         datetime.now().isoformat(),
        "algorithm":          "Biased MF / Funk SVD (Surprise)",
        "kind":               kind,
        "source":             "CSV",
        "n_ratings":          int(len(df)),
        "n_users":            int(df["user_id"].nunique()),
        item_key:             int(df["drink_id"].nunique()),
        "n_factors":          n_factors,
        "n_epochs":           n_epochs,
        "cv_rmse":            round(cv_rmse, 4) if cv_rmse is not None else None,
        "cv_mae":             round(cv_mae,  4) if cv_mae  is not None else None,
        "rating_scale":       list(rating_scale),
        "min_ratings_per_user": MIN_RATINGS_PER_USER,
    }
    with open(cf_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {cf_model}")
    print(f"  Saved -> {cf_meta}")
    print(f"\nDone. SVD trained on {len(df):,} {kind} ratings from {df['user_id'].nunique():,} users.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind",      choices=["beer", "wine"], default="beer",
                        help="Which drink kind to train CF for.")
    parser.add_argument("--n-factors", type=int, default=30)
    parser.add_argument("--n-epochs",  type=int, default=15)
    args = parser.parse_args()
    train(kind=args.kind, n_factors=args.n_factors, n_epochs=args.n_epochs)
