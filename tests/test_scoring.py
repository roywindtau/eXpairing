"""
tests/test_scoring.py
---------------------
Unit tests for expiry.py, ingredient_match.py, and scoring.py.
Run with:  cd fridge2fork && python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date, timedelta

from backend.services.expiry import (
    days_until_expiry, urgency_score, pantry_urgency_map,
)
from backend.services.ingredient_match import (
    match_ingredients, expiry_weighted_match,
)
from backend.services.scoring import rank_recipes, RecipeScore, DEFAULT_BETA


# ── helpers ──────────────────────────────────────────────────────────────────

def future(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()

def past(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


# ── expiry.py ─────────────────────────────────────────────────────────────────

class TestDaysUntilExpiry:
    def test_today_is_zero(self):
        assert days_until_expiry(future(0)) == 0

    def test_tomorrow_is_one(self):
        assert days_until_expiry(future(1)) == 1

    def test_yesterday_is_negative(self):
        assert days_until_expiry(past(1)) == -1

    def test_accepts_date_object(self):
        assert days_until_expiry(date.today() + timedelta(days=5)) == 5


class TestUrgencyScore:
    def test_today_is_one(self):
        assert urgency_score(future(0)) == 1.0

    def test_expired_is_one(self):
        assert urgency_score(past(3)) == 1.0

    def test_half_life_correct(self):
        assert abs(urgency_score(future(3), half_life_days=3.0) - 0.5) < 0.001

    def test_two_half_lives(self):
        assert abs(urgency_score(future(6), half_life_days=3.0) - 0.25) < 0.001

    def test_far_future_near_zero(self):
        assert urgency_score(future(60)) < 0.01

    def test_always_in_range(self):
        for d in [0, 1, 3, 7, 14, 30]:
            s = urgency_score(future(d))
            assert 0.0 <= s <= 1.0


class TestPantryUrgencyMap:
    def test_basic(self):
        items = [
            {"ingredient": "milk",  "expiry_date": future(1)},
            {"ingredient": "eggs",  "expiry_date": future(10)},
        ]
        m = pantry_urgency_map(items)
        assert "milk" in m and "eggs" in m
        assert m["milk"] > m["eggs"]

    def test_duplicate_keeps_highest(self):
        items = [
            {"ingredient": "milk", "expiry_date": future(10)},
            {"ingredient": "milk", "expiry_date": future(1)},
        ]
        m = pantry_urgency_map(items)
        assert abs(m["milk"] - urgency_score(future(1))) < 0.001

    def test_empty_returns_empty(self):
        assert pantry_urgency_map([]) == {}


# ── ingredient_match.py ───────────────────────────────────────────────────────

class TestMatchIngredients:
    def test_perfect_match(self):
        r = match_ingredients(["eggs", "milk"], ["eggs", "milk", "butter"])
        assert r["match_ratio"] == 1.0
        assert r["missing"] == []

    def test_zero_match(self):
        r = match_ingredients(["lobster", "truffle"], ["eggs", "milk"])
        assert r["match_ratio"] == 0.0
        assert len(r["missing"]) == 2

    def test_partial_match(self):
        r = match_ingredients(["eggs", "milk", "bread"], ["eggs", "milk"])
        assert abs(r["match_ratio"] - 2/3) < 0.01
        assert "bread" in r["missing"]

    def test_fuzzy_whole_milk(self):
        r = match_ingredients(["whole milk"], ["milk"])
        assert r["match_ratio"] == 1.0

    def test_fuzzy_cherry_tomatoes(self):
        r = match_ingredients(["cherry tomatoes"], ["tomatoes"])
        assert r["match_ratio"] == 1.0

    def test_empty_recipe(self):
        r = match_ingredients([], ["eggs"])
        assert r["match_ratio"] == 1.0 and r["total"] == 0

    def test_empty_pantry(self):
        r = match_ingredients(["eggs", "milk"], [])
        assert r["match_ratio"] == 0.0
        assert len(r["missing"]) == 2

    def test_missing_list_correct(self):
        r = match_ingredients(["eggs", "milk", "cheese"], ["eggs"])
        assert "milk" in r["missing"]
        assert "cheese" in r["missing"]
        assert "eggs" not in r["missing"]


class TestExpiryWeightedMatch:
    def test_urgent_ingredient_scores_higher(self):
        urgency = {"milk": 0.9, "butter": 0.1}
        assert expiry_weighted_match(["milk"], urgency) > \
               expiry_weighted_match(["butter"], urgency)

    def test_empty_recipe_returns_zero(self):
        assert expiry_weighted_match([], {"milk": 0.9}) == 0.0

    def test_no_match_returns_zero(self):
        assert expiry_weighted_match(["lobster"], {"milk": 0.9}) == 0.0

    def test_complex_recipe_using_both_expiring_beats_simple_using_one(self):
        # garlic expires today (urgency 1.0), milk tomorrow (urgency ~0.84)
        urgency = {"garlic": 1.0, "milk": 0.84}
        # fettuccine-style: 13 ingredients, uses garlic + milk
        fettuccine = ["fettuccine", "olive oil", "garlic", "milk", "parmesan",
                      "dill", "chives", "nutmeg", "salt", "pepper",
                      "asparagus", "smoked salmon", "lemon juice"]
        # hot choc: 2 ingredients, uses only milk
        hot_choc = ["nutella", "milk"]
        assert expiry_weighted_match(fettuccine, urgency) > \
               expiry_weighted_match(hot_choc, urgency)

    def test_empty_urgency_map_returns_zero(self):
        assert expiry_weighted_match(["milk"], {}) == 0.0

    def test_result_bounded_zero_one(self):
        urgency = {"garlic": 1.0, "milk": 1.0}
        assert 0.0 <= expiry_weighted_match(["garlic", "milk"], urgency) <= 1.0


# ── scoring.py ────────────────────────────────────────────────────────────────

class TestRankRecipes:
    def pantry(self):
        return [
            {"ingredient": "eggs",   "expiry_date": future(1)},
            {"ingredient": "milk",   "expiry_date": future(2)},
            {"ingredient": "butter", "expiry_date": future(14)},
        ]

    def recipes(self):
        return [
            {"id": 1, "name": "French toast",
             "ingredients": ["eggs", "milk", "bread", "butter"]},
            {"id": 2, "name": "Pasta carbonara",
             "ingredients": ["pasta", "eggs", "pancetta", "parmesan"]},
            {"id": 3, "name": "Lobster bisque",
             "ingredients": ["lobster", "cream", "brandy", "shallots"]},
        ]

    def test_returns_list_of_recipe_scores(self):
        r = rank_recipes(self.pantry(), self.recipes())
        assert all(isinstance(x, RecipeScore) for x in r)

    def test_french_toast_beats_carbonara(self):
        ranked = rank_recipes(self.pantry(), self.recipes())
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    def test_lobster_bisque_is_last(self):
        ranked = rank_recipes(self.pantry(), self.recipes())
        assert ranked[-1].recipe_id == 3

    def test_scores_in_zero_one(self):
        for r in rank_recipes(self.pantry(), self.recipes()):
            assert 0.0 <= r.final_score <= 1.0

    def test_top_item_is_highest_scored(self):
        # MMR reranks for diversity; the first item is always the highest-scored.
        ranked = rank_recipes(self.pantry(), self.recipes())
        scores = [r.final_score for r in ranked]
        assert scores[0] == max(scores)

    def test_top_n_respected(self):
        ranked = rank_recipes(self.pantry(), self.recipes(), top_n=2)
        assert len(ranked) == 2

    def test_empty_pantry_still_ranks(self):
        ranked = rank_recipes([], self.recipes())
        assert len(ranked) == 3
        for r in ranked:
            assert r.match_ratio == 0.0

    def test_empty_recipes_returns_empty(self):
        assert rank_recipes(self.pantry(), []) == []

    def test_explainability_fields_present(self):
        top = rank_recipes(self.pantry(), self.recipes())[0]
        assert isinstance(top.matched_ingredients, list)
        assert isinstance(top.missing_ingredients, list)
        assert top.total_ingredients > 0

    def test_cf_scores_influence_ranking(self):
        """
        CF-first architecture: cf_scores always influence ranking.
        has_cf=True means SVD is active; has_cf=False means cold-start
        item-based CF is used instead. Either way CF scores matter.
        Passing cf_scores changes the ranking relative to no cf_scores.
        """
        cf = {1: 0.01, 2: 0.01, 3: 0.99}
        no_cf = rank_recipes(self.pantry(), self.recipes(),
                             user_profile={"has_cf": True, "beta": 0.05},
                             cf_scores=None)
        with_cf = rank_recipes(self.pantry(), self.recipes(),
                               user_profile={"has_cf": True, "beta": 0.05},
                               cf_scores=cf)
        # With a huge CF boost on recipe 3, it should rise vs no-CF ranking
        no_cf_ids   = [r.recipe_id for r in no_cf]
        with_cf_ids = [r.recipe_id for r in with_cf]
        assert no_cf_ids != with_cf_ids, "CF scores should change the ranking"

    def test_cf_influences_rank_when_has_cf_true(self):
        cf = {1: 0.01, 2: 0.01, 3: 0.99}
        ranked = rank_recipes(
            self.pantry(), self.recipes(),
            user_profile={"has_cf": True, "has_cb": False, "beta": 0.05},
            cf_scores=cf,
        )
        ids = [r.recipe_id for r in ranked]
        # lobster bisque should climb with a huge CF boost
        assert ids.index(3) < len(ids) - 1

    def test_high_beta_still_sorts_correctly(self):
        ranked = rank_recipes(
            self.pantry(), self.recipes(),
            user_profile={"beta": 0.99},
        )
        # french toast (3/4 match) still beats lobster bisque (0/4)
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(3)

class TestIngredientMatcherFixes:
    """
    Regression tests for the three-pass ingredient matcher.
    Verifies fixes for partial_ratio false positives (reviewer bug #1).
    """

    def test_corn_does_not_match_peppercorns(self):
        result = match_ingredients(["peppercorns"], ["corn"])
        assert result["match_ratio"] == 0.0
        assert "peppercorns" in result["missing"]

    def test_corn_does_not_match_popcorn(self):
        result = match_ingredients(["popcorn"], ["corn"])
        assert result["match_ratio"] == 0.0

    def test_egg_does_not_match_eggplant(self):
        result = match_ingredients(["eggplant"], ["egg"])
        assert result["match_ratio"] == 0.0

    def test_butter_does_not_match_peanut_butter(self):
        result = match_ingredients(["peanut butter"], ["butter"])
        assert result["match_ratio"] == 0.0

    def test_oil_does_not_match_essential_oils(self):
        result = match_ingredients(["essential oils"], ["oil"])
        assert result["match_ratio"] == 0.0

    def test_milk_matches_whole_milk(self):
        result = match_ingredients(["whole milk"], ["milk"])
        assert result["match_ratio"] == 1.0

    def test_tomato_matches_cherry_tomatoes(self):
        result = match_ingredients(["cherry tomatoes"], ["tomato"])
        assert result["match_ratio"] == 1.0

    def test_butter_matches_unsalted_butter(self):
        result = match_ingredients(["unsalted butter"], ["butter"])
        assert result["match_ratio"] == 1.0

    def test_garlic_matches_garlic_powder(self):
        result = match_ingredients(["garlic powder"], ["garlic"])
        assert result["match_ratio"] == 1.0

    def test_eggs_matches_scrambled_eggs(self):
        result = match_ingredients(["scrambled eggs"], ["eggs"])
        assert result["match_ratio"] == 1.0

    def test_cheese_matches_cream_cheese(self):
        result = match_ingredients(["cream cheese"], ["cheese"])
        assert result["match_ratio"] == 1.0

    def test_corn_matches_corn_starch(self):
        """corn starch is still corn — valid pantry match."""
        result = match_ingredients(["corn starch"], ["corn"])
        assert result["match_ratio"] == 1.0

    def test_chicken_matches_chicken_breast(self):
        result = match_ingredients(["chicken breast"], ["chicken"])
        assert result["match_ratio"] == 1.0

    def test_plural_singular_eggs(self):
        result = match_ingredients(["eggs"], ["egg"])
        assert result["match_ratio"] == 1.0

    def test_multiple_ingredients_mixed(self):
        """Real-world scenario: some match, some are false-positive-prone."""
        recipe = ["peppercorns", "whole milk", "chicken breast", "eggplant"]
        pantry = ["milk", "chicken", "corn", "egg"]
        result = match_ingredients(recipe, pantry)
        # milk and chicken should match; peppercorns and eggplant should not
        assert "peppercorns" in result["missing"]
        assert "eggplant"    in result["missing"]
        assert "whole milk"  in result["matched"]
        assert "chicken breast" in result["matched"]
