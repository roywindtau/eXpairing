"""
train_cf.py
-----------
Trains the biased matrix factorization (Funk SVD) model on Food.com ratings.

ALGORITHM: BIASED MATRIX FACTORIZATION (Funk SVD)
--------------------------------------------------
We use Surprise's SVD class, which implements Funk SVD / biased MF —
NOT true truncated SVD of the rating matrix.

The model learns:
    predicted(u, r) = μ + b_u + b_r + p_u · q_r^T

Where:
    μ    ∈ R                           — global mean rating
    b_u  ∈ R^n_users                   — user biases
    b_r  ∈ R^n_recipes                 — recipe biases
    P    ∈ R^(n_users × n_factors)     — user latent vectors
    Q    ∈ R^(n_recipes × n_factors)   — recipe latent vectors

Parameters {b_u, b_r, P, Q} are learned by stochastic gradient descent
on the observed (user, recipe, rating) triples only — the ~99% of missing
entries are never materialized. This handles DATA SPARSITY efficiently.

Contrast with true SVD: truncated SVD factorizes a dense imputed matrix
R ≈ UΣV^T, which is both slower and less appropriate at ~99% sparsity.
Biased MF is strictly better here — the course calls this "Model-based CF
(Matrix Factorization)."

IMPLICIT vs EXPLICIT SIGNALS
-----------------------------
SVD trains on BOTH signal types:

  Explicit: star ratings 1-5            → primary training signal
  Implicit: cook events (n_missing)     → synthetic ratings added where no
                                           explicit rating exists

Implicit synthetic rating formula:
    implicit_rating = max(3.0, 4.0 - min(n_missing, 3) × 0.3)
    n_missing=0 → 4.0  (user had everything: strong positive signal)
    n_missing=1 → 3.7
    n_missing=2 → 3.4
    n_missing=3 → 3.1  (had to buy 3+ items: weak but still positive)

Explicit ratings always take precedence: if a user both cooked and
explicitly rated a recipe, only the explicit rating is used.

The net effect: more training signal per user, helping users who
cook frequently but rarely rate reach a meaningful SVD latent vector
sooner (and often: at all).

Skip events are NOT used as training data — they serve instead as
feed exclusions (see recipes.py: 7-day skip exclusion window).

COLD START NOTE
---------------
SVD cannot predict for users with no ratings — they have no P vector.
For those users, serve_cf.py routes to the item-based cold start path.
See cold_start.py for the full cold-start CF implementation.

EVALUATION
----------
3-fold cross-validated RMSE is computed during training.
Full offline evaluation (ablation, Precision@K, Recall@K, NDCG) is in
evaluate.py — run after training for complete metrics.

Baseline comparison:
  Global mean predictor:  RMSE ≈ 1.12  (predict mean for everyone)
  Per-user mean:          RMSE ≈ 1.05
  SVD (n_factors=50):     RMSE ≈ 0.82  (typical on Food.com subset)

SAVED ARTIFACTS
---------------
    models/cf_model.pkl    trained SVD model
    models/cf_meta.json    RMSE, n_users, n_recipes, hyperparams, date

Run:
    python -m backend.ml.train_cf [--n-factors 50] [--n-epochs 20]
    python -m backend.ml.train_cf --no-implicit   # explicit only (for comparison)

Typical training time on 1M ratings: ~5 minutes on a laptop CPU.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd

from backend.db.database import SessionLocal
from backend.db.models import UserEvent

MODELS_DIR = Path("models")
CF_MODEL   = MODELS_DIR / "cf_model.pkl"
CF_META    = MODELS_DIR / "cf_meta.json"

MIN_RATINGS_PER_USER  = 3    # exclude very sparse users from training

# Implicit signal parameters
IMPLICIT_RATING_BASE  = 4.0  # cook with 0 missing ingredients → this rating
IMPLICIT_RATING_DECAY = 0.3  # rating reduction per missing ingredient
IMPLICIT_RATING_FLOOR = 3.0  # minimum implicit rating (cooking is still positive)


def load_ratings() -> pd.DataFrame:
    print("Loading explicit ratings from DB ...")
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
    print(f"  {len(df):,} explicit ratings, {df['user_id'].nunique():,} users, "
          f"{df['recipe_id'].nunique():,} recipes")

    if len(df) > 0:
        density = len(df) / (df["user_id"].nunique() * df["recipe_id"].nunique()) * 100
        print(f"  Matrix density: {density:.4f}%  "
              f"(~{100-density:.1f}% of ratings unknown — typical sparsity problem)")
    return df


def load_cook_events() -> pd.DataFrame:
    """
    Load cook events and convert to synthetic implicit ratings.

    A cook event is a positive implicit signal: the user chose to cook
    this recipe given their pantry state. The confidence of the signal
    depends on how many ingredients they had to buy (n_missing):

        implicit_rating = max(3.0, 4.0 - min(n_missing, 3) × 0.3)

    This places cook events in the range [3.0, 4.0] — positive but below
    the "I loved it" threshold of an explicit 5-star rating.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(UserEvent.user_id, UserEvent.recipe_id, UserEvent.n_missing)
            .filter(UserEvent.event_type == "cook")
            .all()
        )
    finally:
        db.close()

    if not rows:
        return pd.DataFrame(columns=["user_id", "recipe_id", "rating"])

    df = pd.DataFrame(rows, columns=["user_id", "recipe_id", "n_missing"])
    df["rating"] = df["n_missing"].apply(
        lambda n: max(
            IMPLICIT_RATING_FLOOR,
            IMPLICIT_RATING_BASE - min(float(n or 0), 3) * IMPLICIT_RATING_DECAY
        )
    )
    print(f"  {len(df):,} cook events → synthetic ratings "
          f"(range [{IMPLICIT_RATING_FLOOR:.1f}, {IMPLICIT_RATING_BASE:.1f}])")
    return df[["user_id", "recipe_id", "rating"]]


def augment_with_cook_events(explicit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cook events as synthetic implicit ratings where the user has not
    provided an explicit rating for that recipe.

    Explicit ratings always take precedence over implicit cook signals.
    When a user both cooked and rated a recipe, only the explicit rating
    is included in the training set.

    For (user, recipe) pairs where only cook events exist, we take the
    highest synthetic rating (in case the user cooked the same recipe
    multiple times with different n_missing values).
    """
    cook_df = load_cook_events()
    if cook_df.empty:
        print("  No cook events found — training on explicit ratings only.")
        return explicit_df

    # Key set of explicitly-rated (user, recipe) pairs
    explicit_keys = set(zip(explicit_df["user_id"], explicit_df["recipe_id"]))

    # Filter to only cook events not already explicitly rated
    mask = ~cook_df.apply(
        lambda r: (r["user_id"], r["recipe_id"]) in explicit_keys, axis=1
    )
    new_rows = cook_df[mask]

    # Deduplicate: same (user, recipe) cooked multiple times → keep highest rating
    new_rows = (new_rows
                .groupby(["user_id", "recipe_id"])["rating"]
                .max()
                .reset_index())

    combined = pd.concat([explicit_df, new_rows]).reset_index(drop=True)
    n_added = len(combined) - len(explicit_df)
    print(f"  Augmented: +{n_added:,} implicit cook rows  "
          f"({len(combined):,} total training rows)")
    return combined


def filter_active_users(df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    counts   = df.groupby("user_id").size()
    active   = counts[counts >= min_ratings].index
    filtered = df[df["user_id"].isin(active)]
    dropped  = df["user_id"].nunique() - filtered["user_id"].nunique()
    print(f"  Dropped {dropped:,} users with < {min_ratings} ratings. "
          f"{filtered['user_id'].nunique():,} users remain.")
    return filtered


def train(n_factors: int = 50, n_epochs: int = 20, use_implicit: bool = True) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_ratings()
    if use_implicit:
        df = augment_with_cook_events(df)
    if len(df) == 0:
        print("No ratings or cook events found. Run seed_ratings.py first.")
        sys.exit(1)

    df = filter_active_users(df, MIN_RATINGS_PER_USER)

    reader  = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[["user_id", "recipe_id", "rating"]], reader)

    print(f"\nTraining biased MF / Funk SVD (n_factors={n_factors}, n_epochs={n_epochs}) ...")
    model = SVD(  # Surprise's SVD = Funk SVD / biased matrix factorization
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=False,
    )

    print("Running 3-fold cross-validation ...")
    cv_results = cross_validate(model, dataset,
                                measures=["RMSE", "MAE"], cv=3, verbose=False)
    mean_rmse  = float(cv_results["test_rmse"].mean())
    mean_mae   = float(cv_results["test_mae"].mean())
    print(f"  CV RMSE: {mean_rmse:.4f}  |  MAE: {mean_mae:.4f}")
    print(f"  (Global mean baseline RMSE ≈ 1.12 on Food.com)")

    print("Training final model on full dataset ...")
    trainset = dataset.build_full_trainset()
    model.fit(trainset)

    with open(CF_MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {CF_MODEL}")

    meta = {
        "trained_at":   datetime.now().isoformat(),
        "algorithm":    "Biased MF / Funk SVD (μ + b_u + b_r + p_u·q_r^T, SGD)",
        "signal_type":  "explicit ratings 1-5 + implicit cook events" if use_implicit else "explicit ratings 1-5",
        "use_implicit": use_implicit,
        "n_ratings":    len(df),
        "n_users":      int(df["user_id"].nunique()),
        "n_recipes":    int(df["recipe_id"].nunique()),
        "n_factors":    n_factors,
        "n_epochs":     n_epochs,
        "cv_rmse":      round(mean_rmse, 4),
        "cv_mae":       round(mean_mae, 4),
        "min_ratings_for_warmup": 5,
    }
    with open(CF_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {CF_META}")
    print(f"\nDone. RMSE {mean_rmse:.4f} on held-out folds.")
    print(f"Run 'python -m backend.ml.evaluate' for full offline metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-factors",   type=int,  default=50)
    parser.add_argument("--n-epochs",    type=int,  default=20)
    parser.add_argument("--no-implicit", action="store_true",
                        help="Train on explicit ratings only (no cook event augmentation)")
    args = parser.parse_args()
    train(n_factors=args.n_factors, n_epochs=args.n_epochs,
          use_implicit=not args.no_implicit)
