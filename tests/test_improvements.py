"""
tests/test_improvements.py
--------------------------
Tests for the RecSys improvements:

  1. Cook event augmentation (train_cf.py)
  2. Score calibration (scoring.py)
  3. MMR diversity reranking (scoring.py)
  4. CB taste profile (serve_cb.py)
  5. Revealed beta in stats endpoint (users.py)
  6. NDCG metric (evaluate.py)
  7. Skip exclusion (recipes router — via direct scoring logic test)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock


# ── helpers ────────────────────────────────────────────────────────────────

def future(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def make_pantry(items: list[tuple[str, int]]) -> list[dict]:
    return [{"ingredient": name, "expiry_date": future(days)}
            for name, days in items]


# ── 1. Cook event augmentation ─────────────────────────────────────────────

class TestCookEventAugmentation:
    """Tests for train_cf.augment_with_cook_events / load_cook_events."""

    def test_implicit_rating_formula_zero_missing(self):
        from backend.ml.train_cf import (
            IMPLICIT_RATING_BASE, IMPLICIT_RATING_DECAY, IMPLICIT_RATING_FLOOR
        )
        n = 0
        r = max(IMPLICIT_RATING_FLOOR,
                IMPLICIT_RATING_BASE - min(float(n), 3) * IMPLICIT_RATING_DECAY)
        assert r == 4.0

    def test_implicit_rating_formula_one_missing(self):
        from backend.ml.train_cf import (
            IMPLICIT_RATING_BASE, IMPLICIT_RATING_DECAY, IMPLICIT_RATING_FLOOR
        )
        n = 1
        r = max(IMPLICIT_RATING_FLOOR,
                IMPLICIT_RATING_BASE - min(float(n), 3) * IMPLICIT_RATING_DECAY)
        assert abs(r - 3.7) < 0.001

    def test_implicit_rating_capped_at_three_missing(self):
        """n_missing is capped at 3 in the formula, so max decay = 3 * 0.3 = 0.9."""
        from backend.ml.train_cf import (
            IMPLICIT_RATING_BASE, IMPLICIT_RATING_DECAY, IMPLICIT_RATING_FLOOR
        )
        # n=100 → min(100, 3) = 3 → 4.0 - 0.9 = 3.1 (floor=3.0 not triggered)
        n = 100
        r = max(IMPLICIT_RATING_FLOOR,
                IMPLICIT_RATING_BASE - min(float(n), 3) * IMPLICIT_RATING_DECAY)
        assert abs(r - 3.1) < 0.001

    def test_implicit_rating_floor_is_minimum(self):
        """The floor constant itself is the minimum possible implicit rating."""
        from backend.ml.train_cf import (
            IMPLICIT_RATING_BASE, IMPLICIT_RATING_DECAY, IMPLICIT_RATING_FLOOR
        )
        # Confirm: 4.0 - 3 * 0.3 = 3.1 >= 3.0 (floor not actually hit with default params)
        min_possible = max(
            IMPLICIT_RATING_FLOOR,
            IMPLICIT_RATING_BASE - 3 * IMPLICIT_RATING_DECAY
        )
        assert min_possible >= IMPLICIT_RATING_FLOOR

    def test_augment_adds_cook_events(self):
        from backend.ml.train_cf import augment_with_cook_events

        explicit = pd.DataFrame([
            {"user_id": 1, "recipe_id": 10, "rating": 5.0},
        ])
        cook_rows = pd.DataFrame([
            {"user_id": 1, "recipe_id": 20, "rating": 4.0},  # new pair
        ])
        with patch("backend.ml.train_cf.load_cook_events", return_value=cook_rows):
            result = augment_with_cook_events(explicit)

        assert len(result) == 2
        assert 20 in result["recipe_id"].values

    def test_augment_does_not_override_explicit(self):
        from backend.ml.train_cf import augment_with_cook_events

        explicit = pd.DataFrame([
            {"user_id": 1, "recipe_id": 10, "rating": 5.0},
        ])
        cook_rows = pd.DataFrame([
            {"user_id": 1, "recipe_id": 10, "rating": 3.5},  # same pair — should be excluded
        ])
        with patch("backend.ml.train_cf.load_cook_events", return_value=cook_rows):
            result = augment_with_cook_events(explicit)

        # Only the explicit row should remain
        assert len(result) == 1
        assert result.iloc[0]["rating"] == 5.0

    def test_augment_deduplicates_cook_events(self):
        """Same (user, recipe) cooked twice → keep highest synthetic rating."""
        from backend.ml.train_cf import augment_with_cook_events

        explicit = pd.DataFrame([], columns=["user_id", "recipe_id", "rating"])
        cook_rows = pd.DataFrame([
            {"user_id": 1, "recipe_id": 10, "rating": 3.1},
            {"user_id": 1, "recipe_id": 10, "rating": 3.7},  # same pair, higher rating
        ])
        with patch("backend.ml.train_cf.load_cook_events", return_value=cook_rows):
            result = augment_with_cook_events(explicit)

        matches = result[result["recipe_id"] == 10]
        assert len(matches) == 1
        assert matches.iloc[0]["rating"] == 3.7

    def test_augment_empty_cook_events(self):
        from backend.ml.train_cf import augment_with_cook_events

        explicit = pd.DataFrame([{"user_id": 1, "recipe_id": 10, "rating": 5.0}])
        empty = pd.DataFrame([], columns=["user_id", "recipe_id", "rating"])
        with patch("backend.ml.train_cf.load_cook_events", return_value=empty):
            result = augment_with_cook_events(explicit)

        assert len(result) == len(explicit)

    def test_use_implicit_false_skips_augmentation(self):
        """
        When use_implicit=False (--no-implicit flag), cook events must not be
        added to the training data. This is the correct mode when cook events
        are demo/test interactions rather than real user behaviour.
        """
        from backend.ml.train_cf import train
        import pandas as pd

        explicit = pd.DataFrame([
            {"user_id": 1, "recipe_id": 10, "rating": 4.0},
            {"user_id": 1, "recipe_id": 11, "rating": 3.0},
            {"user_id": 1, "recipe_id": 12, "rating": 5.0},
            {"user_id": 2, "recipe_id": 10, "rating": 3.0},
            {"user_id": 2, "recipe_id": 11, "rating": 4.0},
            {"user_id": 2, "recipe_id": 12, "rating": 2.0},
        ])
        cook_rows = pd.DataFrame([
            {"user_id": 3, "recipe_id": 99, "rating": 4.0},  # should NOT appear
        ])

        captured = {}

        def fake_load_ratings():
            return explicit.copy()

        def fake_augment(df):
            captured["augment_called"] = True
            return pd.concat([df, cook_rows])

        with patch("backend.ml.train_cf.load_ratings", side_effect=fake_load_ratings), \
             patch("backend.ml.train_cf.augment_with_cook_events", side_effect=fake_augment), \
             patch("backend.ml.train_cf.MODELS_DIR") as mock_dir, \
             patch("backend.ml.train_cf.CF_MODEL", "/dev/null"), \
             patch("backend.ml.train_cf.CF_META", "/dev/null"), \
             patch("builtins.open", create=True), \
             patch("pickle.dump"), \
             patch("json.dump"):
            mock_dir.mkdir = lambda **kw: None
            try:
                train(n_factors=2, n_epochs=1, use_implicit=False)
            except Exception:
                pass  # model save will fail with /dev/null — that's fine

        assert "augment_called" not in captured, \
            "augment_with_cook_events must NOT be called when use_implicit=False"

    def test_use_implicit_true_calls_augmentation(self):
        """When use_implicit=True (default), cook events ARE merged into training."""
        from backend.ml.train_cf import train
        import pandas as pd

        explicit = pd.DataFrame([
            {"user_id": 1, "recipe_id": i, "rating": 4.0} for i in range(3)
        ] + [
            {"user_id": 2, "recipe_id": i, "rating": 3.0} for i in range(3)
        ])

        captured = {}

        def fake_augment(df):
            captured["augment_called"] = True
            return df

        with patch("backend.ml.train_cf.load_ratings", return_value=explicit), \
             patch("backend.ml.train_cf.augment_with_cook_events", side_effect=fake_augment), \
             patch("backend.ml.train_cf.MODELS_DIR") as mock_dir, \
             patch("backend.ml.train_cf.CF_MODEL", "/dev/null"), \
             patch("backend.ml.train_cf.CF_META", "/dev/null"), \
             patch("builtins.open", create=True), \
             patch("pickle.dump"), \
             patch("json.dump"):
            mock_dir.mkdir = lambda **kw: None
            try:
                train(n_factors=2, n_epochs=1, use_implicit=True)
            except Exception:
                pass

        assert captured.get("augment_called") is True, \
            "augment_with_cook_events MUST be called when use_implicit=True"


# ── 2. Score calibration ───────────────────────────────────────────────────

class TestCalibrate:
    def test_min_becomes_zero(self):
        from backend.services.scoring import _calibrate
        assert _calibrate([0.0, 0.5, 1.0])[0] == 0.0

    def test_max_becomes_one(self):
        from backend.services.scoring import _calibrate
        assert _calibrate([0.0, 0.5, 1.0])[-1] == 1.0

    def test_uniform_values_become_half(self):
        from backend.services.scoring import _calibrate
        result = _calibrate([0.3, 0.3, 0.3])
        assert all(v == 0.5 for v in result)

    def test_preserves_order(self):
        from backend.services.scoring import _calibrate
        vals = [0.1, 0.4, 0.2, 0.9, 0.3]
        cal  = _calibrate(vals)
        for i in range(len(vals) - 1):
            if vals[i] <= vals[i + 1]:
                assert cal[i] <= cal[i + 1]

    def test_all_in_zero_one(self):
        from backend.services.scoring import _calibrate
        for v in _calibrate([0.05, 0.2, 0.7, 0.99]):
            assert 0.0 <= v <= 1.0

    def test_calibration_applied_in_rank_recipes(self):
        """After calibration, final_scores must still be in [0,1]."""
        from backend.services.scoring import rank_recipes

        pantry  = make_pantry([("eggs", 1), ("milk", 5)])
        recipes = [
            {"id": 1, "name": "A", "ingredients": ["eggs", "milk"]},
            {"id": 2, "name": "B", "ingredients": ["pasta", "cheese"]},
            {"id": 3, "name": "C", "ingredients": ["lobster", "cream"]},
        ]
        ranked = rank_recipes(pantry, recipes)
        for r in ranked:
            assert 0.0 <= r.final_score <= 1.0


# ── 3. MMR diversity ───────────────────────────────────────────────────────

class TestMMR:
    def _make_score(self, recipe_id, final_score, matched, missing):
        from backend.services.scoring import RecipeScore
        return RecipeScore(
            recipe_id=recipe_id,
            recipe_name=f"Recipe {recipe_id}",
            final_score=final_score,
            expiry_urgency=0.0,
            match_ratio=0.5,
            cf_score=0.5,
            cb_score=0.0,
            matched_ingredients=matched,
            missing_ingredients=missing,
            total_ingredients=len(matched) + len(missing),
        )

    def test_returns_top_n_items(self):
        from backend.services.scoring import _mmr_rerank
        candidates = [self._make_score(i, 1.0 - i * 0.1, ["eggs"], ["milk"])
                      for i in range(10)]
        result = _mmr_rerank(candidates, top_n=3)
        assert len(result) == 3

    def test_no_rerank_when_fewer_than_top_n(self):
        from backend.services.scoring import _mmr_rerank
        candidates = [self._make_score(1, 0.9, ["eggs"], []),
                      self._make_score(2, 0.8, ["milk"], [])]
        result = _mmr_rerank(candidates, top_n=5)
        assert len(result) == 2

    def test_first_selected_is_highest_scored(self):
        from backend.services.scoring import _mmr_rerank
        candidates = [
            self._make_score(1, 0.9, ["pasta", "cheese"], []),
            self._make_score(2, 0.95, ["eggs", "milk"], []),
            self._make_score(3, 0.7, ["lobster"], []),
        ]
        result = _mmr_rerank(candidates, top_n=2)
        assert result[0].recipe_id == 2

    def test_diversity_pushes_down_similar_recipe(self):
        """
        Two nearly-identical recipes: MMR should prefer the diverse third
        over the redundant clone of the top recipe.
        """
        from backend.services.scoring import _mmr_rerank
        same_ingredients = ["pasta", "tomato", "garlic", "olive_oil"]
        r1 = self._make_score(1, 0.90, same_ingredients, [])
        r2 = self._make_score(2, 0.85, same_ingredients, [])   # clone
        r3 = self._make_score(3, 0.80, ["chicken", "lemon", "herbs"], [])  # diverse

        result = _mmr_rerank([r1, r2, r3], top_n=2, lambda_=0.7)
        selected_ids = [r.recipe_id for r in result]
        # r3 should beat r2 because of diversity bonus
        assert 1 in selected_ids
        assert 3 in selected_ids
        assert 2 not in selected_ids

    def test_lambda_one_is_pure_relevance(self):
        """lambda=1.0 disables diversity → picks top-2 by score."""
        from backend.services.scoring import _mmr_rerank
        candidates = [
            self._make_score(1, 0.9, ["x", "y"], []),
            self._make_score(2, 0.8, ["x", "y"], []),
            self._make_score(3, 0.7, ["a", "b"], []),
        ]
        result = _mmr_rerank(candidates, top_n=2, lambda_=1.0)
        assert [r.recipe_id for r in result] == [1, 2]


class TestIngredientJaccard:
    def _score(self, matched, missing):
        from backend.services.scoring import RecipeScore
        return RecipeScore(
            recipe_id=1, recipe_name="X", final_score=0.5,
            expiry_urgency=0.0, match_ratio=0.5, cf_score=0.5, cb_score=0.0,
            matched_ingredients=matched, missing_ingredients=missing,
            total_ingredients=len(matched) + len(missing),
        )

    def test_identical_sets_is_one(self):
        from backend.services.scoring import _ingredient_jaccard
        a = self._score(["eggs", "milk"], [])
        b = self._score(["eggs", "milk"], [])
        assert _ingredient_jaccard(a, b) == 1.0

    def test_disjoint_sets_is_zero(self):
        from backend.services.scoring import _ingredient_jaccard
        a = self._score(["eggs"], [])
        b = self._score(["lobster"], [])
        assert _ingredient_jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        from backend.services.scoring import _ingredient_jaccard
        a = self._score(["eggs", "milk"], [])
        b = self._score(["eggs", "butter"], [])
        j = _ingredient_jaccard(a, b)
        assert abs(j - 1/3) < 0.001

    def test_empty_sets_is_zero(self):
        from backend.services.scoring import _ingredient_jaccard
        a = self._score([], [])
        b = self._score([], [])
        assert _ingredient_jaccard(a, b) == 0.0


# ── 4. CB taste profile ────────────────────────────────────────────────────

class TestCBTasteProfile:
    def setup_method(self):
        import backend.ml.serve_cb as scb
        scb._loaded = False
        scb._matrix = None
        scb._recipe_ids = None
        scb._vectorizer = None

    def test_returns_empty_when_no_model(self):
        import backend.ml.serve_cb as scb
        scb._loaded = True
        scb._matrix = None
        from backend.ml.serve_cb import cb_taste_profile_batch
        result = cb_taste_profile_batch([1, 2], [4.0, 3.0], [1, 2, 3])
        assert result == {}

    def test_returns_empty_for_no_rated_recipes(self):
        import backend.ml.serve_cb as scb
        scb._loaded = True
        scb._matrix = MagicMock()
        from backend.ml.serve_cb import cb_taste_profile_batch
        result = cb_taste_profile_batch([], [], [1, 2, 3])
        assert result == {}

    def test_scores_in_zero_one_with_mock(self):
        """With a real TF-IDF setup, all taste-profile scores are in [0,1]."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse as sp
        from backend.ml.serve_cb import cb_taste_profile_batch
        import backend.ml.serve_cb as scb

        docs = [
            "eggs milk butter",
            "pasta tomato garlic",
            "chicken lemon herbs",
            "eggs flour butter",
        ]
        recipe_ids = [1, 2, 3, 4]
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(docs)

        scb._matrix     = matrix
        scb._recipe_ids = np.array(recipe_ids)
        scb._vectorizer = vec
        scb._loaded     = True

        # User rated recipe 1 (5★) and recipe 2 (2★)
        result = cb_taste_profile_batch(
            rated_recipe_ids=[1, 2],
            ratings=[5.0, 2.0],
            candidate_recipe_ids=[1, 2, 3, 4],
        )
        assert len(result) == 4
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_negative_sims_clipped_to_zero(self):
        """Taste profile with a disliked recipe should not produce negative scores."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from backend.ml.serve_cb import cb_taste_profile_batch
        import backend.ml.serve_cb as scb

        docs = ["eggs milk", "pasta tomato"]
        vec  = TfidfVectorizer()
        mat  = vec.fit_transform(docs)

        scb._matrix     = mat
        scb._recipe_ids = np.array([1, 2])
        scb._vectorizer = vec
        scb._loaded     = True

        result = cb_taste_profile_batch(
            rated_recipe_ids=[1],
            ratings=[1.0],   # strongly disliked → weight = -2.0
            candidate_recipe_ids=[1, 2],
        )
        for v in result.values():
            assert v >= 0.0


# ── 5. Revealed beta in stats ──────────────────────────────────────────────

class TestRevealedBeta:
    def test_compute_revealed_beta_zero_missing(self):
        from backend.services.beta_updater import _compute_revealed_beta
        df = pd.DataFrame({"n_missing": [0, 0, 0]})
        rev, avg = _compute_revealed_beta(df)
        assert rev == 1.0
        assert avg == 0.0

    def test_compute_revealed_beta_high_missing(self):
        from backend.services.beta_updater import _compute_revealed_beta
        df = pd.DataFrame({"n_missing": [4.0, 4.0, 4.0]})
        rev, avg = _compute_revealed_beta(df)
        # avg_missing=4.0, half_point=2.0 → 1/(1+4/2)=1/3 ≈ 0.333
        assert abs(rev - 1/3) < 0.01

    def test_compute_revealed_beta_none_when_no_valid(self):
        from backend.services.beta_updater import _compute_revealed_beta
        df = pd.DataFrame({"n_missing": [None, None]})
        rev, avg = _compute_revealed_beta(df)
        assert rev is None
        assert avg is None

    def test_compute_revealed_beta_ignores_null(self):
        from backend.services.beta_updater import _compute_revealed_beta
        df = pd.DataFrame({"n_missing": [0.0, None, 0.0]})
        rev, avg = _compute_revealed_beta(df)
        assert rev is not None
        assert avg == 0.0

    def test_revealed_beta_in_zero_one(self):
        from backend.services.beta_updater import _compute_revealed_beta
        for missing_val in [0, 1, 2, 3, 5, 10]:
            df = pd.DataFrame({"n_missing": [float(missing_val)] * 5})
            rev, _ = _compute_revealed_beta(df)
            assert 0.0 <= rev <= 1.0


# ── 6. NDCG metric ────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_ranking_is_one(self):
        from backend.ml.evaluate import ndcg_at_k
        relevance = {1: 1.0, 2: 0.8, 3: 0.5}
        predicted = [1, 2, 3]
        score = ndcg_at_k(predicted, relevance, k=3)
        assert abs(score - 1.0) < 0.001

    def test_reversed_ranking_less_than_perfect(self):
        from backend.ml.evaluate import ndcg_at_k
        relevance = {1: 1.0, 2: 0.8, 3: 0.5}
        worst = [3, 2, 1]   # least relevant first
        perfect = [1, 2, 3]
        assert ndcg_at_k(worst, relevance, k=3) < ndcg_at_k(perfect, relevance, k=3)

    def test_all_irrelevant_is_zero(self):
        from backend.ml.evaluate import ndcg_at_k
        relevance = {}  # no relevant items
        score = ndcg_at_k([1, 2, 3], relevance, k=3)
        assert score == 0.0

    def test_top1_hit_beats_top2_hit(self):
        from backend.ml.evaluate import ndcg_at_k
        relevance = {99: 1.0}
        score_top1 = ndcg_at_k([99, 1, 2], relevance, k=3)
        score_top2 = ndcg_at_k([1, 99, 2], relevance, k=3)
        assert score_top1 > score_top2

    def test_k_truncation(self):
        """Relevant item ranked at k+1 should not contribute to NDCG@k."""
        from backend.ml.evaluate import ndcg_at_k
        relevance = {5: 1.0}
        # relevant item at position 3 (0-indexed 2), k=2 → should not be seen
        predicted = [1, 2, 5]
        score = ndcg_at_k(predicted, relevance, k=2)
        assert score == 0.0

    def test_output_in_zero_one(self):
        from backend.ml.evaluate import ndcg_at_k
        relevance = {1: 0.9, 2: 0.4, 3: 0.1}
        for perm in [[1, 2, 3], [2, 1, 3], [3, 2, 1]]:
            s = ndcg_at_k(perm, relevance, k=3)
            assert 0.0 <= s <= 1.0


# ── 7. Skip exclusion (scoring-level validation) ───────────────────────────

class TestSkipExclusion:
    """
    The 7-day skip exclusion lives in the recipes router (DB query).
    Here we test that the scoring layer correctly handles a reduced
    candidate set (i.e., skipped recipes simply absent from the input).
    """

    def test_absent_recipe_not_in_output(self):
        from backend.services.scoring import rank_recipes

        pantry = make_pantry([("eggs", 2)])
        recipes = [
            {"id": 1, "name": "Omelette", "ingredients": ["eggs"]},
            {"id": 2, "name": "Pasta",    "ingredients": ["pasta"]},
        ]
        # Simulate skip exclusion: recipe 2 removed before scoring
        filtered = [r for r in recipes if r["id"] != 2]
        ranked = rank_recipes(pantry, filtered)
        ids = [r.recipe_id for r in ranked]
        assert 2 not in ids
        assert 1 in ids

    def test_full_candidate_set_vs_filtered(self):
        """Removing a recipe from candidates changes the ranked list."""
        from backend.services.scoring import rank_recipes

        pantry  = make_pantry([("eggs", 1)])
        recipes = [{"id": i, "name": f"R{i}", "ingredients": ["eggs"]} for i in range(5)]

        full     = rank_recipes(pantry, recipes)
        filtered = rank_recipes(pantry, [r for r in recipes if r["id"] != 0])

        full_ids     = [r.recipe_id for r in full]
        filtered_ids = [r.recipe_id for r in filtered]

        assert 0 not in filtered_ids
        assert len(filtered_ids) == len(full_ids) - 1
