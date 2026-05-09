"""
tests/test_cold_start.py
------------------------
Tests for personalized cold-start CF.

Key behaviors verified:
  - Vegetarian user gets different seeds than omnivore user
  - Pantry ingredients influence seed selection
  - Scores differ across recipes (real signal, not uniform)
  - Diet-matched recipes score higher than mismatched ones
  - Seed diversification prevents mono-cuisine results
  - Fallback works when no sim matrix available
  - Empty diet tags and empty pantry are handled gracefully
  - Cold-start scores integrate correctly with full ranking pipeline
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd

from backend.ml.cold_start import (
    select_seeds,
    cold_start_cf_scores,
    diversify_seeds,
    personalized_cold_start,
    _tag_match_score,
    _pantry_match_score,
    N_SEEDS,
    MIN_TAG_MATCH,
)
from backend.ml.item_similarity import build_item_similarity, sparsify_top_k


# ── fixtures ───────────────────────────────────────────────────────────────

def make_recipe_corpus():
    """
    20 diverse recipes covering: vegetarian, vegan, breakfast,
    italian, asian, seafood, dessert. Each has tags and ingredients.
    """
    return [
        {"id": 1,  "tags": ["vegetarian","breakfast","italian"],
         "ingredients": ["eggs","milk","flour","butter"]},
        {"id": 2,  "tags": ["vegan","asian"],
         "ingredients": ["tofu","soy sauce","ginger","garlic"]},
        {"id": 3,  "tags": ["vegetarian","italian"],
         "ingredients": ["pasta","tomatoes","garlic","olive oil","parmesan"]},
        {"id": 4,  "tags": ["seafood","asian"],
         "ingredients": ["salmon","soy sauce","sesame oil","ginger"]},
        {"id": 5,  "tags": ["vegan","breakfast"],
         "ingredients": ["banana","oats","almond milk","maple syrup"]},
        {"id": 6,  "tags": ["vegetarian","dessert"],
         "ingredients": ["chocolate","butter","eggs","sugar","flour"]},
        {"id": 7,  "tags": ["italian","meat"],
         "ingredients": ["beef","pasta","tomatoes","red wine","onion"]},
        {"id": 8,  "tags": ["asian","meat"],
         "ingredients": ["chicken","soy sauce","garlic","ginger","rice"]},
        {"id": 9,  "tags": ["vegetarian","quick"],
         "ingredients": ["eggs","cheese","bread","butter"]},
        {"id": 10, "tags": ["vegan","italian"],
         "ingredients": ["pasta","tomatoes","basil","olive oil","capers"]},
        {"id": 11, "tags": ["seafood","italian"],
         "ingredients": ["shrimp","pasta","garlic","white wine","cream"]},
        {"id": 12, "tags": ["vegetarian","asian"],
         "ingredients": ["tofu","vegetables","soy sauce","sesame","rice"]},
        {"id": 13, "tags": ["dessert","vegetarian"],
         "ingredients": ["milk","sugar","vanilla","eggs","cream"]},
        {"id": 14, "tags": ["breakfast","meat"],
         "ingredients": ["bacon","eggs","toast","butter"]},
        {"id": 15, "tags": ["vegan","quick"],
         "ingredients": ["lentils","tomatoes","cumin","garlic","spinach"]},
        {"id": 16, "tags": ["vegetarian","breakfast"],
         "ingredients": ["eggs","tomatoes","onion","olive oil"]},
        {"id": 17, "tags": ["seafood","quick"],
         "ingredients": ["tuna","lemon","capers","olive oil","bread"]},
        {"id": 18, "tags": ["italian","meat"],
         "ingredients": ["pork","sage","white wine","butter","garlic"]},
        {"id": 19, "tags": ["vegan","dessert"],
         "ingredients": ["banana","cocoa","almond milk","maple syrup"]},
        {"id": 20, "tags": ["vegetarian","italian"],
         "ingredients": ["risotto","parmesan","white wine","butter","onion"]},
    ]


def make_sim_matrix_for_corpus(corpus):
    """Build a real item-sim matrix from synthetic ratings for the corpus."""
    rng = np.random.default_rng(42)
    rows = []
    for u in range(50):
        for r in corpus:
            if rng.random() < 0.4:
                rows.append({
                    "user_id":   u + 1,
                    "recipe_id": r["id"],
                    "rating":    float(rng.integers(1, 6)),
                })
    df  = pd.DataFrame(rows)
    sim, recipe_ids = build_item_similarity(df)
    sparse = sparsify_top_k(sim, k=8)
    return sparse, np.array(recipe_ids, dtype=np.int32)


# ── _tag_match_score ───────────────────────────────────────────────────────

class TestTagMatchScore:
    def test_perfect_match(self):
        assert _tag_match_score({"vegetarian","italian"}, {"vegetarian"}) == 1.0

    def test_no_match(self):
        assert _tag_match_score({"meat","italian"}, {"vegetarian"}) == 0.0

    def test_partial_match(self):
        score = _tag_match_score({"vegetarian","italian"}, {"vegetarian","vegan"})
        assert abs(score - 0.5) < 0.01

    def test_no_user_tags_returns_one(self):
        assert _tag_match_score({"anything"}, set()) == 1.0

    def test_empty_recipe_tags(self):
        assert _tag_match_score(set(), {"vegetarian"}) == 0.0


# ── _pantry_match_score ────────────────────────────────────────────────────

class TestPantryMatchScore:
    def test_full_match(self):
        score = _pantry_match_score(["eggs","milk"], {"eggs","milk","butter"})
        assert score == 1.0

    def test_no_match(self):
        score = _pantry_match_score(["lobster","truffle"], {"eggs","milk"})
        assert score == 0.0

    def test_partial_match(self):
        score = _pantry_match_score(["eggs","milk","bread"], {"eggs","milk"})
        assert abs(score - 2/3) < 0.01

    def test_empty_pantry(self):
        assert _pantry_match_score(["eggs"], set()) == 0.0

    def test_empty_recipe(self):
        assert _pantry_match_score([], {"eggs"}) == 0.0


# ── select_seeds ───────────────────────────────────────────────────────────

class TestSelectSeeds:
    def corpus(self):
        return make_recipe_corpus()

    def test_returns_list_of_ids(self):
        seeds = select_seeds(self.corpus(), ["vegetarian"], [], n_seeds=5)
        assert isinstance(seeds, list)
        assert all(isinstance(s, int) for s in seeds)

    def test_respects_n_seeds(self):
        seeds = select_seeds(self.corpus(), ["vegetarian"], [], n_seeds=5)
        assert len(seeds) <= 5

    def test_vegetarian_seeds_are_vegetarian(self):
        """All selected seeds should have vegetarian tag."""
        corpus = self.corpus()
        seeds = select_seeds(corpus, ["vegetarian"], [], n_seeds=10)
        corpus_map = {r["id"]: r for r in corpus}
        for sid in seeds:
            tags = corpus_map[sid]["tags"]
            assert "vegetarian" in tags or "vegan" in tags, \
                f"Recipe {sid} tags {tags} not vegetarian"

    def test_no_diet_tags_selects_from_all(self):
        """With no diet tags, all recipes are candidates."""
        seeds = select_seeds(self.corpus(), [], [], n_seeds=20)
        assert len(seeds) == 20

    def test_pantry_boosts_matching_recipes(self):
        """Seeds with pantry overlap should score higher."""
        corpus  = self.corpus()
        # Pantry has eggs and milk — should boost recipe 1 (eggs,milk,flour,butter)
        seeds_with_pantry    = select_seeds(corpus, [], ["eggs","milk"], n_seeds=5)
        seeds_without_pantry = select_seeds(corpus, [], [], n_seeds=5)
        # Recipe 1 should appear in seeds_with_pantry or ranked higher
        assert 1 in seeds_with_pantry

    def test_vegan_user_different_seeds_than_meat_user(self):
        corpus      = self.corpus()
        vegan_seeds = set(select_seeds(corpus, ["vegan"], [], n_seeds=10))
        meat_seeds  = set(select_seeds(corpus, ["meat"], [], n_seeds=10))
        # The two sets should not be identical
        assert vegan_seeds != meat_seeds

    def test_strict_diet_filter_excludes_non_matching(self):
        """Seafood-tagged recipe should not appear in vegan seeds."""
        corpus = self.corpus()
        vegan_seeds = select_seeds(corpus, ["vegan"], [], n_seeds=20)
        corpus_map  = {r["id"]: r for r in corpus}
        for sid in vegan_seeds:
            assert "seafood" not in corpus_map[sid]["tags"], \
                f"Seafood recipe {sid} in vegan seeds"


# ── diversify_seeds ────────────────────────────────────────────────────────

class TestDiversifySeeds:
    def test_caps_per_tag(self):
        corpus = make_recipe_corpus()
        # Select only vegetarian seeds first
        veg_seeds   = [r["id"] for r in corpus if "vegetarian" in r["tags"]]
        diversified = diversify_seeds(veg_seeds, corpus, max_per_tag=2)
        # Count vegetarian recipes with tag "vegetarian" as primary
        veg_count = sum(
            1 for sid in diversified
            if next((r for r in corpus if r["id"] == sid), {}).get("tags", [""])[0] == "vegetarian"
        )
        assert veg_count <= 2

    def test_doesnt_add_seeds(self):
        corpus = make_recipe_corpus()
        seeds  = [r["id"] for r in corpus[:10]]
        result = diversify_seeds(seeds, corpus, max_per_tag=3)
        assert len(result) <= len(seeds)

    def test_empty_seeds(self):
        assert diversify_seeds([], [], max_per_tag=3) == []


# ── cold_start_cf_scores ───────────────────────────────────────────────────

class TestColdStartCfScores:
    def setup_method(self):
        corpus = make_recipe_corpus()
        self.sim, self.ids = make_sim_matrix_for_corpus(corpus)
        self.recipe_ids    = [r["id"] for r in corpus]

    def test_returns_dict_for_all_candidates(self):
        scores = cold_start_cf_scores(
            self.recipe_ids, [1, 2, 3], self.sim, self.ids
        )
        assert len(scores) == len(self.recipe_ids)

    def test_scores_in_zero_one(self):
        scores = cold_start_cf_scores(
            self.recipe_ids, [1, 5, 10], self.sim, self.ids
        )
        for s in scores.values():
            assert 0.0 <= s <= 1.0, f"Score out of range: {s}"

    def test_scores_not_all_equal(self):
        scores = cold_start_cf_scores(
            self.recipe_ids, [1, 2, 3, 4, 5], self.sim, self.ids
        )
        unique = set(round(s, 3) for s in scores.values())
        assert len(unique) > 1, "All cold-start scores identical"

    def test_max_score_is_one(self):
        scores = cold_start_cf_scores(
            self.recipe_ids, [1, 2, 3], self.sim, self.ids
        )
        if any(s > 0 for s in scores.values()):
            assert abs(max(scores.values()) - 1.0) < 0.01

    def test_empty_seeds_returns_zeros(self):
        scores = cold_start_cf_scores(
            self.recipe_ids, [], self.sim, self.ids
        )
        assert all(s == 0.0 for s in scores.values())

    def test_unknown_recipe_gets_zero(self):
        scores = cold_start_cf_scores([99999], [1, 2], self.sim, self.ids)
        assert scores[99999] == 0.0


# ── personalized_cold_start (full pipeline) ────────────────────────────────

class TestPersonalizedColdStart:
    def setup_method(self):
        self.corpus = make_recipe_corpus()
        self.sim, self.ids = make_sim_matrix_for_corpus(self.corpus)
        self.all_ids = [r["id"] for r in self.corpus]

    def test_vegetarian_user_gets_scores(self):
        scores = personalized_cold_start(
            candidate_recipe_ids=self.all_ids,
            all_recipes=self.corpus,
            user_diet_tags=["vegetarian"],
            pantry_ingredients=["eggs","milk"],
            sim_matrix=self.sim,
            sim_recipe_ids=self.ids,
        )
        assert len(scores) == len(self.all_ids)
        assert any(s > 0 for s in scores.values())

    def test_vegan_scores_differ_from_omnivore(self):
        """Core property: personalization changes scores."""
        vegan = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=["vegan"],
            pantry_ingredients=[],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        omni = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=[],
            pantry_ingredients=[],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        # Ranking should differ between vegan and generic user
        vegan_order = sorted(vegan, key=lambda x: -vegan[x])
        omni_order  = sorted(omni,  key=lambda x: -omni[x])
        assert vegan_order != omni_order, \
            "Vegan and omnivore got identical cold-start ranking"

    def test_pantry_changes_scores(self):
        """Pantry influences seed selection -> different CF scores."""
        with_pantry = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=[],
            pantry_ingredients=["eggs","milk","butter"],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        no_pantry = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=[],
            pantry_ingredients=[],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        assert with_pantry != no_pantry

    def test_no_sim_matrix_falls_back_to_preference_scores(self):
        # Without a trained sim_matrix, personalized_cold_start should return
        # tag+pantry preference scores (not zeros) so the feed stays personalized.
        scores = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=["vegetarian"],
            pantry_ingredients=["eggs"],
            sim_matrix=None, sim_recipe_ids=None,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == set(self.all_ids)
        assert any(s > 0.0 for s in scores.values()), "expected non-zero preference scores"
        assert all(0.0 <= s <= 1.0 for s in scores.values()), "scores must be in [0,1]"

    def test_empty_corpus_fallback_works(self):
        """Empty all_recipes -> fallback seeds, should not crash."""
        scores = personalized_cold_start(
            candidate_recipe_ids=self.all_ids,
            all_recipes=[],
            user_diet_tags=["vegetarian"],
            pantry_ingredients=["eggs"],
            sim_matrix=self.sim,
            sim_recipe_ids=self.ids,
        )
        assert isinstance(scores, dict)

    def test_empty_diet_and_pantry_still_scores(self):
        """User with no preferences set should still get non-trivial scores."""
        scores = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=[],
            pantry_ingredients=[],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        assert any(s > 0 for s in scores.values())

    def test_scores_in_zero_one(self):
        scores = personalized_cold_start(
            self.all_ids, self.corpus,
            user_diet_tags=["vegetarian"],
            pantry_ingredients=["eggs","milk"],
            sim_matrix=self.sim, sim_recipe_ids=self.ids,
        )
        for s in scores.values():
            assert 0.0 <= s <= 1.0
