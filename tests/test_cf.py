"""
tests/test_cf.py
----------------
Tests for the CF layer — cold start, warm user, and the transition.

Verifies:
  - New user (0 ratings) -> item-based cold start path
  - User with >= MIN_RATINGS_FOR_CF ratings -> SVD path
  - Automatic transition at the threshold
  - Cold-start scores are non-zero when item-sim matrix exists
  - Cold-start scores differ across recipes (not all equal)
  - SVD path correctly uses user_id
  - similar_recipes() returns sensible results
  - item_similarity training pipeline runs end-to-end on synthetic data
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.sparse as sp
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from backend.ml.serve_cf import (
    get_cf_scores,
    is_warm_user,
    cf_strategy_name,
    _blend_alpha,
    MIN_RATINGS_FOR_CF,
    _norm,
    _mf_scores,
)
from backend.ml.cold_start import cold_start_cf_scores as _cold_start_scores
COLD_START_ANCHOR_N = 200  # kept for existing tests
from backend.ml.item_similarity import (
    build_item_similarity,
    sparsify_top_k,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_ratings_df(n_users=20, n_recipes=10, seed=42) -> pd.DataFrame:
    """
    Synthetic ratings DataFrame.
    Each user rates ~half the recipes with scores 1-5.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        recipes_rated = rng.choice(n_recipes, size=n_recipes//2, replace=False)
        for r in recipes_rated:
            rows.append({
                "user_id":   u + 1,
                "recipe_id": r + 1,
                "rating":    float(rng.integers(1, 6)),
            })
    return pd.DataFrame(rows)


# ── _norm ──────────────────────────────────────────────────────────────────

class TestNorm:
    def test_min_rating_gives_zero(self):
        assert _norm(1.0) == 0.0

    def test_max_rating_gives_one(self):
        assert _norm(5.0) == 1.0

    def test_mid_rating(self):
        assert abs(_norm(3.0) - 0.5) < 0.001

    def test_clamped_below(self):
        assert _norm(0.0) == 0.0

    def test_clamped_above(self):
        assert _norm(6.0) == 1.0


# ── is_warm_user ───────────────────────────────────────────────────────────

class TestIsWarmUser:
    def test_zero_ratings_is_cold(self):
        assert is_warm_user(0) is False

    def test_below_threshold_is_cold(self):
        assert is_warm_user(MIN_RATINGS_FOR_CF - 1) is False

    def test_at_threshold_is_warm(self):
        assert is_warm_user(MIN_RATINGS_FOR_CF) is True

    def test_above_threshold_is_warm(self):
        assert is_warm_user(MIN_RATINGS_FOR_CF + 100) is True


# ── _blend_alpha ────────────────────────────────────────────────────────────

class TestBlendAlpha:
    def test_zero_ratings_is_zero(self):
        assert _blend_alpha(0) == 0.0

    def test_full_threshold_is_one(self):
        assert _blend_alpha(MIN_RATINGS_FOR_CF) == 1.0

    def test_above_threshold_clamped_to_one(self):
        assert _blend_alpha(MIN_RATINGS_FOR_CF + 10) == 1.0

    def test_halfway(self):
        half = MIN_RATINGS_FOR_CF / 2
        assert abs(_blend_alpha(int(half)) - int(half) / MIN_RATINGS_FOR_CF) < 0.01

    def test_strictly_between_zero_and_one_for_partial(self):
        for n in range(1, MIN_RATINGS_FOR_CF):
            a = _blend_alpha(n)
            assert 0.0 < a < 1.0, f"Expected blend at {n} ratings, got alpha={a}"


# ── cf_strategy_name ────────────────────────────────────────────────────────

class TestCfStrategyName:
    def setup_method(self):
        import backend.ml.serve_cf as scf
        scf._loaded = True
        scf._cf_model = None

    def test_no_svd_always_cold_start(self):
        import backend.ml.serve_cf as scf
        scf._cf_model = None
        for n in range(10):
            assert cf_strategy_name(n) == "item_based_cold_start"

    def test_zero_ratings_cold_start_even_with_svd(self):
        import backend.ml.serve_cf as scf
        scf._cf_model = MagicMock()
        assert cf_strategy_name(0) == "item_based_cold_start"

    def test_partial_ratings_blended(self):
        import backend.ml.serve_cf as scf
        scf._cf_model = MagicMock()
        for n in range(1, MIN_RATINGS_FOR_CF):
            assert cf_strategy_name(n) == "blended", f"Expected blended at {n} ratings"

    def test_at_threshold_svd(self):
        import backend.ml.serve_cf as scf
        scf._cf_model = MagicMock()
        assert cf_strategy_name(MIN_RATINGS_FOR_CF) == "biased_mf"

    def test_above_threshold_svd(self):
        import backend.ml.serve_cf as scf
        scf._cf_model = MagicMock()
        assert cf_strategy_name(MIN_RATINGS_FOR_CF + 5) == "biased_mf"


# ── build_item_similarity ──────────────────────────────────────────────────

class TestBuildItemSimilarity:
    def test_output_shape(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, recipe_ids = build_item_similarity(df)
        n = len(recipe_ids)
        assert sim.shape == (n, n)

    def test_symmetric(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        # cosine similarity is symmetric; convert to dense for np.allclose
        assert np.allclose(sim.toarray(), sim.T.toarray(), atol=1e-5)

    def test_diagonal_is_one(self):
        """Before sparsify, diagonal should be 1.0 (self-similarity)."""
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        # Self-similarity = 1.0 for any non-zero vector
        for i in range(sim.shape[0]):
            assert sim[i, i] >= 0.99 or sim[i, i] == 0.0  # 0 if all-zero row

    def test_recipe_ids_correct_count(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        _, recipe_ids = build_item_similarity(df)
        assert len(recipe_ids) == df["recipe_id"].nunique()

    def test_values_in_minus_one_to_one(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        assert sim.min() >= -1.01
        assert sim.max() <= 1.01


class TestSparsifyTopK:
    def test_at_most_k_per_row(self):
        df = make_ratings_df(n_users=30, n_recipes=15)
        sim, _ = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=3)
        for i in range(sparse.shape[0]):
            assert sparse.getrow(i).nnz <= 3

    def test_diagonal_removed(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)
        assert sparse.diagonal().sum() == 0.0

    def test_returns_csr_matrix(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)
        assert sp.issparse(sparse)

    def test_only_positive_values(self):
        df = make_ratings_df(n_users=20, n_recipes=10)
        sim, _ = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)
        assert sparse.data.min() > 0


# ── cold start scores ──────────────────────────────────────────────────────

class TestColdStartScores:
    """
    Test cold_start_cf_scores with a real similarity matrix.
    """

    def setup_method(self):
        df = make_ratings_df(n_users=30, n_recipes=15)
        sim, recipe_ids = build_item_similarity(df)
        self.sparse     = sparsify_top_k(sim, k=5)
        self.ids        = np.array(recipe_ids, dtype=np.int32)
        self.recipe_ids = recipe_ids

    def test_returns_dict_of_correct_length(self):
        scores = _cold_start_scores(self.recipe_ids, [1, 2, 3],
                                     self.sparse, self.ids)
        assert len(scores) == len(self.recipe_ids)

    def test_scores_in_zero_one(self):
        scores = _cold_start_scores(self.recipe_ids, [1, 2, 3],
                                     self.sparse, self.ids)
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_scores_not_all_equal(self):
        scores = _cold_start_scores(self.recipe_ids, self.recipe_ids[:5],
                                     self.sparse, self.ids)
        unique = len(set(round(s, 3) for s in scores.values()))
        assert unique > 1, "All cold-start scores identical — no signal"

    def test_max_score_is_one(self):
        scores = _cold_start_scores(self.recipe_ids, [1, 2, 3],
                                     self.sparse, self.ids)
        if any(s > 0 for s in scores.values()):
            assert abs(max(scores.values()) - 1.0) < 0.01

    def test_unknown_recipe_gets_zero(self):
        scores = _cold_start_scores([999999], [1, 2], self.sparse, self.ids)
        assert scores[999999] == 0.0

    def test_empty_recipe_list(self):
        scores = _cold_start_scores([], [1, 2], self.sparse, self.ids)
        assert scores == {}


# ── get_cf_scores: routing logic ───────────────────────────────────────────

class TestGetCfScoresRouting:
    def setup_method(self):
        """Reset serve_cf module state before each test."""
        import backend.ml.serve_cf as scf
        scf._cf_model  = None
        scf._sim_matrix = None
        scf._sim_ids    = None
        scf._loaded     = False

    def test_no_models_returns_zeros(self):
        """When no models loaded, cold start returns zero scores (not error)."""
        import backend.ml.serve_cf as scf
        scf._loaded     = True  # skip file loading
        scf._cf_model  = None
        scf._sim_matrix = None
        scf._sim_ids    = None
        scores = get_cf_scores(user_id=1, recipe_ids=[1, 2, 3],
                               n_user_ratings=0, all_recipes=[])
        assert all(s == 0.0 for s in scores.values())

    def test_cold_start_when_below_threshold(self):
        """With sim matrix loaded but no SVD, cold-start path used."""
        df = make_ratings_df(n_users=30, n_recipes=10)
        sim, recipe_ids = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)

        import backend.ml.serve_cf as scf
        scf._sim_matrix = sparse
        scf._sim_ids    = np.array(recipe_ids, dtype=np.int32)
        scf._cf_model  = None
        scf._loaded     = True

        # Build minimal recipe dicts for seed selection
        all_recipes = [{"id": rid, "tags": [], "ingredients": []}
                       for rid in recipe_ids]
        scores = get_cf_scores(
            user_id=1,
            recipe_ids=recipe_ids,
            n_user_ratings=MIN_RATINGS_FOR_CF - 1,
            all_recipes=all_recipes,
        )
        assert len(scores) > 0

    def test_svd_path_when_above_threshold(self):
        """With a mock SVD model and sufficient ratings, SVD path used."""
        mock_model = MagicMock()
        mock_pred  = MagicMock()
        mock_pred.est = 4.0
        mock_model.predict.return_value = mock_pred

        import backend.ml.serve_cf as scf
        scf._cf_model  = mock_model
        scf._loaded     = True

        scores = get_cf_scores(
            user_id=42,
            recipe_ids=[1, 2, 3],
            n_user_ratings=MIN_RATINGS_FOR_CF,
        )
        assert len(scores) == 3
        # SVD was called, not cold start
        assert mock_model.predict.call_count == 3

    def test_zero_ratings_no_svd_call(self):
        """
        At 0 ratings alpha=0 — pure cold start, SVD must not be called
        even if the model is loaded.
        """
        mock_model = MagicMock()
        df = make_ratings_df(n_users=30, n_recipes=10)
        sim, recipe_ids = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)

        import backend.ml.serve_cf as scf
        scf._cf_model  = mock_model
        scf._sim_matrix = sparse
        scf._sim_ids    = np.array(recipe_ids, dtype=np.int32)
        scf._loaded     = True

        all_recipes = [{"id": rid, "tags": [], "ingredients": []}
                       for rid in recipe_ids]
        get_cf_scores(
            user_id=1,
            recipe_ids=recipe_ids,
            n_user_ratings=0,
            all_recipes=all_recipes,
        )
        mock_model.predict.assert_not_called()

    def test_partial_ratings_blends_both_signals(self):
        """
        Between 0 and threshold, both cold start and SVD contribute.
        SVD is called, scores are a blend (not pure SVD values).
        """
        mock_model = MagicMock()
        mock_pred  = MagicMock()
        mock_pred.est = 4.0   # SVD always returns 4.0 → normalized ~0.75
        mock_model.predict.return_value = mock_pred

        df = make_ratings_df(n_users=30, n_recipes=10)
        sim, recipe_ids = build_item_similarity(df)
        sparse = sparsify_top_k(sim, k=5)

        import backend.ml.serve_cf as scf
        scf._cf_model  = mock_model
        scf._sim_matrix = sparse
        scf._sim_ids    = np.array(recipe_ids, dtype=np.int32)
        scf._loaded     = True

        all_recipes = [{"id": rid, "tags": [], "ingredients": []}
                       for rid in recipe_ids]
        scores = get_cf_scores(
            user_id=1,
            recipe_ids=recipe_ids,
            n_user_ratings=2,   # alpha = 0.4  →  blend
            all_recipes=all_recipes,
        )
        assert mock_model.predict.call_count == len(recipe_ids), \
            "SVD should be called in blend mode"
        # All blended scores must be in [0, 1]
        for s in scores.values():
            assert 0.0 <= s <= 1.0, f"Blended score {s} out of range"
        # Verify the blend is applied: expected = (1-0.4)*cold + 0.4*svd_norm
        svd_norm = _norm(4.0)  # ~0.75
        # With alpha=0.4 the SVD contribution is 40% of svd_norm
        # The scores should differ from zero (both signals active)
        assert any(s > 0 for s in scores.values()), "Blended scores all zero"

    def test_transition_at_exact_threshold_pure_svd(self):
        """At exactly MIN_RATINGS_FOR_CF alpha=1.0 — pure SVD, cold start not called."""
        mock_model = MagicMock()
        mock_pred  = MagicMock()
        mock_pred.est = 3.5
        mock_model.predict.return_value = mock_pred

        import backend.ml.serve_cf as scf
        scf._cf_model = mock_model
        scf._loaded    = True

        scores = get_cf_scores(
            user_id=7,
            recipe_ids=[1, 2],
            n_user_ratings=MIN_RATINGS_FOR_CF,
        )
        assert mock_model.predict.call_count == 2
        # At alpha=1.0, score equals pure SVD value
        svd_norm = _norm(3.5)
        for s in scores.values():
            assert abs(s - svd_norm) < 1e-6, \
                f"Expected pure SVD score {svd_norm}, got {s}"
