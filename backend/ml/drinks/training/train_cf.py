"""
train_drink_cf.py
-----------------
Trains the biased MF (Funk SVD) model for BEER ratings only.

Why beer-only
-------------
BeerAdvocate gives us ~1.58M explicit ratings across 33k users — plenty
for biased MF to learn meaningful latent factors. X-Wines Test, in
contrast, has only ~1k ratings across 636 users; an SVD trained on that
collapses to global-mean and is worse than a smoothed popularity baseline.
So wine never gets a Surprise model — it runs entirely on item-sim and
Bayesian popularity (see drink_cold_start.py).

Synthetic ratings are EXCLUDED here
-----------------------------------
DrinkEvent.synthetic=True rows come from drink_synthesizer.py inferring
preferences from recipe ratings. They are useful as item-sim SEEDS at
serve time but are deliberately kept out of SVD training so the matrix-
factorization signal stays grounded in real, expressed user preferences.

Saved artifacts
---------------
    models/drink_cf_model.pkl    trained Surprise SVD
    models/drink_cf_meta.json    RMSE/MAE + hyperparams + timestamp

Run:
    python -m backend.ml.drinks.training.train_cf [--n-factors 30] [--n-epochs 15]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

from backend.db.database import SessionLocal
from backend.db.models import Drink, DrinkEvent

MODELS_DIR  = Path("models")
CF_MODEL    = MODELS_DIR / "drink_cf_model.pkl"
CF_META     = MODELS_DIR / "drink_cf_meta.json"

MIN_RATINGS_PER_USER = 3


def load_beer_ratings() -> pd.DataFrame:
    """All real (synthetic=False) beer rating events as a (user, item, rating) df."""
    print("Loading beer ratings from DB ...")
    db = SessionLocal()
    try:
        rows = (
            db.query(DrinkEvent.user_id, DrinkEvent.drink_id, DrinkEvent.rating)
            .join(Drink, Drink.id == DrinkEvent.drink_id)
            .filter(Drink.kind == "beer")
            .filter(DrinkEvent.event_type == "rate")
            .filter(DrinkEvent.rating.isnot(None))
            .filter(DrinkEvent.synthetic == False)  # noqa: E712 — SQLAlchemy needs ==
            .all()
        )
    finally:
        db.close()

    df = pd.DataFrame(rows, columns=["user_id", "drink_id", "rating"])
    if len(df) == 0:
        return df

    print(
        f"  {len(df):,} ratings, {df['user_id'].nunique():,} users, "
        f"{df['drink_id'].nunique():,} beers"
    )
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


def train(n_factors: int = 30, n_epochs: int = 15) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_beer_ratings()
    if len(df) == 0:
        print("No beer ratings found. Run `python -m backend.db.seed_drink_ratings` first.")
        sys.exit(1)

    df = filter_active_users(df, MIN_RATINGS_PER_USER)
    if len(df) == 0:
        print("After filtering, no ratings left to train on. Lower MIN_RATINGS_PER_USER.")
        sys.exit(1)

    # Beer reviews are on a 0-5 scale (0 occurs occasionally); use 0-5 reader.
    reader  = Reader(rating_scale=(0, 5))
    dataset = Dataset.load_from_df(df[["user_id", "drink_id", "rating"]], reader)

    print(f"\nTraining biased MF (n_factors={n_factors}, n_epochs={n_epochs}) on beer ratings ...")
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=False,
    )

    # Skip CV when corpus is tiny (test fixtures): cross_validate needs
    # at least cv * a-few-ratings-per-fold to be meaningful.
    cv_rmse = cv_mae = None
    if len(df) >= 100:
        print("Running 3-fold cross-validation ...")
        cv = cross_validate(model, dataset, measures=["RMSE", "MAE"], cv=3, verbose=False)
        cv_rmse = float(cv["test_rmse"].mean())
        cv_mae  = float(cv["test_mae"].mean())
        print(f"  CV RMSE: {cv_rmse:.4f}  |  MAE: {cv_mae:.4f}")

    print("Training final model on full dataset ...")
    trainset = dataset.build_full_trainset()
    model.fit(trainset)

    with open(CF_MODEL, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "trained_at":     datetime.now().isoformat(),
        "algorithm":      "Biased MF / Funk SVD (Surprise)",
        "kind":           "beer",
        "synthetic_excluded": True,
        "n_ratings":      int(len(df)),
        "n_users":        int(df["user_id"].nunique()),
        "n_beers":        int(df["drink_id"].nunique()),
        "n_factors":      n_factors,
        "n_epochs":       n_epochs,
        "cv_rmse":        round(cv_rmse, 4) if cv_rmse is not None else None,
        "cv_mae":         round(cv_mae,  4) if cv_mae  is not None else None,
        "rating_scale":   [0, 5],
        "min_ratings_for_cf": 5,
    }
    with open(CF_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {CF_MODEL}")
    print(f"  Saved -> {CF_META}")
    print(f"\nDone. SVD trained on {len(df):,} beer ratings from {df['user_id'].nunique():,} users.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-factors", type=int, default=30)
    parser.add_argument("--n-epochs",  type=int, default=15)
    args = parser.parse_args()
    train(n_factors=args.n_factors, n_epochs=args.n_epochs)
