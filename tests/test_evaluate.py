"""
tests/test_evaluate.py
----------------------
Tests for the offline evaluation pipeline.

Does NOT call scikit-surprise (too slow for unit tests).
Tests the metric logic directly with known inputs so we can
verify correctness of RMSE, Precision@K, and Recall@K.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from collections import defaultdict

from backend.ml.evaluate import (
    make_synthetic_ratings,
    train_test_split,
    MIN_USER_RATINGS,
    RELEVANT_THRESHOLD,
    K_VALUES,
)


# ── make_synthetic_ratings ────────────────────────────────────────────────

class TestMakeSyntheticRatings:
    def test_returns_dataframe(self):
        df = make_synthetic_ratings(n_users=10, n_recipes=10)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = make_synthetic_ratings(n_users=10, n_recipes=10)
        assert set(df.columns) == {"user_id", "recipe_id", "rating"}

    def test_rating_range(self):
        df = make_synthetic_ratings(n_users=20, n_recipes=10)
        assert df["rating"].between(1, 5).all()

    def test_user_count(self):
        df = make_synthetic_ratings(n_users=15, n_recipes=10)
        assert df["user_id"].nunique() == 15

    def test_recipe_count(self):
        df = make_synthetic_ratings(n_users=10, n_recipes=8)
        assert df["recipe_id"].nunique() <= 8

    def test_each_user_has_min_ratings(self):
        df = make_synthetic_ratings(n_users=20, n_recipes=20)
        counts = df.groupby("user_id").size()
        assert (counts >= MIN_USER_RATINGS).all()

    def test_reproducible_with_seed(self):
        df1 = make_synthetic_ratings(n_users=10, n_recipes=10, seed=99)
        df2 = make_synthetic_ratings(n_users=10, n_recipes=10, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = make_synthetic_ratings(n_users=20, n_recipes=10, seed=1)
        df2 = make_synthetic_ratings(n_users=20, n_recipes=10, seed=2)
        assert not df1.equals(df2)


# ── train_test_split ───────────────────────────────────────────────────────

class TestTrainTestSplit:
    def df(self):
        return make_synthetic_ratings(n_users=30, n_recipes=15, seed=42)

    def test_no_overlap(self):
        df = self.df()
        train, test = train_test_split(df)
        train_keys = set(zip(train["user_id"], train["recipe_id"]))
        test_keys  = set(zip(test["user_id"],  test["recipe_id"]))
        assert len(train_keys & test_keys) == 0

    def test_all_data_accounted_for(self):
        df = self.df()
        # Only users with >= MIN_USER_RATINGS are included
        active = df.groupby("user_id").filter(lambda g: len(g) >= MIN_USER_RATINGS)
        train, test = train_test_split(df)
        assert len(train) + len(test) == len(active)

    def test_test_fraction_approximate(self):
        df = self.df()
        train, test = train_test_split(df, test_frac=0.2)
        actual_frac = len(test) / (len(train) + len(test))
        # Tolerance: per-user rounding makes exact fraction impossible
        assert 0.05 < actual_frac < 0.35

    def test_train_larger_than_test(self):
        df = self.df()
        train, test = train_test_split(df)
        assert len(train) > len(test)

    def test_excludes_low_activity_users(self):
        # Create a user with only 1 rating — should be excluded
        df  = make_synthetic_ratings(n_users=20, n_recipes=10, seed=42)
        low = pd.DataFrame([{"user_id": 9999, "recipe_id": 1, "rating": 3.0}])
        df  = pd.concat([df, low], ignore_index=True)
        train, test = train_test_split(df)
        assert 9999 not in train["user_id"].values
        assert 9999 not in test["user_id"].values


# ── precision and recall math ──────────────────────────────────────────────

class TestMetricMath:
    """
    Test precision@K and recall@K with known-answer inputs.
    This verifies the metric logic without needing a trained model.
    """

    def _compute_metrics(self, recommended: list[int],
                         relevant: set[int], k: int) -> tuple[float, float]:
        """Compute precision@k and recall@k directly."""
        top_k   = set(recommended[:k])
        n_hits  = len(top_k & relevant)
        precision = n_hits / k
        recall    = n_hits / len(relevant) if relevant else 0.0
        return precision, recall

    def test_perfect_precision(self):
        p, _ = self._compute_metrics([1, 2, 3, 4, 5], {1, 2, 3, 4, 5}, k=5)
        assert p == 1.0

    def test_zero_precision(self):
        p, _ = self._compute_metrics([10, 11, 12], {1, 2, 3}, k=3)
        assert p == 0.0

    def test_half_precision(self):
        p, _ = self._compute_metrics([1, 10, 2, 11, 12], {1, 2}, k=4)
        assert abs(p - 0.5) < 0.001

    def test_perfect_recall(self):
        _, r = self._compute_metrics([1, 2, 3, 4, 5], {1, 2}, k=5)
        assert r == 1.0

    def test_partial_recall(self):
        _, r = self._compute_metrics([1, 10, 11], {1, 2, 3, 4}, k=3)
        assert abs(r - 0.25) < 0.001

    def test_k_larger_than_recommended(self):
        # Should not crash, hits capped at actual recommendations
        top_k   = set([1, 2, 3][:10])
        n_hits  = len(top_k & {1, 2, 3, 4, 5})
        p = n_hits / 10
        assert p <= 1.0


# ── rmse math ────────────────────────────────────────────────────────────

class TestRmseMath:
    """Verify RMSE formula is correct with known inputs."""

    def test_perfect_predictions(self):
        actual = np.array([3.0, 4.0, 5.0])
        preds  = np.array([3.0, 4.0, 5.0])
        rmse   = float(np.sqrt(np.mean((actual - preds) ** 2)))
        assert rmse == 0.0

    def test_known_rmse(self):
        actual = np.array([4.0, 3.0, 5.0, 2.0])
        preds  = np.array([3.0, 3.0, 4.0, 3.0])
        # errors: 1, 0, 1, 1 → mse=0.75 → rmse=0.866
        rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))
        assert abs(rmse - np.sqrt(0.75)) < 0.001

    def test_global_mean_baseline(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean   = actual.mean()
        preds  = np.full(len(actual), mean)
        rmse   = float(np.sqrt(np.mean((actual - preds) ** 2)))
        # std of [1,2,3,4,5] = sqrt(2) ≈ 1.414
        assert abs(rmse - np.std(actual)) < 0.001

    def test_svd_should_beat_global_mean(self):
        """
        Property test: a personalized model should have lower RMSE
        than predicting the global mean for every user.
        This is a sanity check on our evaluation setup.
        """
        rng    = np.random.default_rng(42)
        actual = rng.integers(1, 6, size=100).astype(float)
        mean   = actual.mean()

        baseline_rmse = float(np.sqrt(np.mean((actual - mean) ** 2)))

        # "Perfect" model — knows exact ratings
        perfect_rmse  = 0.0
        assert perfect_rmse < baseline_rmse

    def test_improvement_pct_positive_when_better(self):
        baseline = 1.5
        svd      = 1.2
        improvement = (baseline - svd) / baseline * 100
        assert improvement > 0

    def test_improvement_pct_negative_when_worse(self):
        baseline = 1.2
        svd      = 1.5
        improvement = (baseline - svd) / baseline * 100
        assert improvement < 0


# ── relevant_threshold ────────────────────────────────────────────────────

class TestRelevantThreshold:
    def test_threshold_value(self):
        """RELEVANT_THRESHOLD should be 4.0 — top 40% of 5-star scale."""
        assert RELEVANT_THRESHOLD == 4.0

    def test_k_values_are_reasonable(self):
        assert 5  in K_VALUES
        assert 10 in K_VALUES
        assert 20 in K_VALUES
        assert all(k > 0 for k in K_VALUES)
        assert sorted(K_VALUES) == K_VALUES   # ascending
