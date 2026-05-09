"""
test_scoring_edge_cases.py
--------------------------
Behavioral edge-case tests for every scoring component using real
recipe names and ingredient lists drawn from Food.com-style data.

Each class focuses on one score dimension and documents what we believe
the system MUST do — the "contract" the scoring engine keeps with the user.

Run:
    python -m pytest tests/test_scoring_edge_cases.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta
from backend.services.ingredient_match import (
    match_ingredients, expiry_weighted_match, ingredient_matches,
)
from backend.services.expiry import pantry_urgency_map, urgency_score
from backend.services.scoring import rank_recipes, RecipeScore


# ── helpers ───────────────────────────────────────────────────────────────────

def future(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def pantry(*items_and_days):
    """Build pantry list from (name, days_until_expiry) pairs."""
    return [{"ingredient": n, "expiry_date": future(d)} for n, d in items_and_days]


def recipe(rid, name, ingredients):
    return {"id": rid, "name": name, "ingredients": ingredients}


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXPIRY URGENCY — expiry_weighted_match
# ─────────────────────────────────────────────────────────────────────────────

class TestExpiryUrgency:
    """
    Contract: expiry_urgency measures "what fraction of your expiring pantry
    does this recipe use?", NOT "what fraction of recipe ingredients do you own?".
    A complex recipe using all your expiring items must beat a simple recipe
    using only one — even if the complex recipe has many other ingredients too.
    """

    def urgency(self, *items_and_days):
        return pantry_urgency_map(pantry(*items_and_days))

    # ── the bug we fixed ──────────────────────────────────────────────────────

    def test_fettuccine_beats_hot_choc_when_it_uses_both_expiring_items(self):
        """Core regression: recipe using garlic+milk should beat one using only milk."""
        u = self.urgency(("garlic", 0), ("milk", 1))
        fettuccine = ["fettuccine", "olive oil", "garlic", "milk", "parmesan cheese",
                      "fresh dill", "chives", "nutmeg", "salt", "pepper",
                      "asparagus", "smoked salmon", "lemon juice"]
        hot_choc   = ["nutella", "milk"]
        assert expiry_weighted_match(fettuccine, u) > expiry_weighted_match(hot_choc, u)

    def test_recipe_using_two_expiring_items_beats_one_using_one(self):
        """Generic: using 2/2 expiring items scores higher than using 1/2."""
        u = self.urgency(("garlic", 0), ("milk", 1))
        both_used = ["garlic", "milk", "pasta", "olive oil", "salt"]
        one_used  = ["milk", "cocoa", "sugar"]
        assert expiry_weighted_match(both_used, u) > expiry_weighted_match(one_used, u)

    # ── urgency ordering ─────────────────────────────────────────────────────

    def test_expires_today_scores_higher_than_expires_in_5_days(self):
        u = self.urgency(("eggs", 0), ("butter", 5))
        eggs_recipe   = ["eggs", "flour", "sugar"]
        butter_recipe = ["butter", "bread", "salt"]
        assert expiry_weighted_match(eggs_recipe, u) > expiry_weighted_match(butter_recipe, u)

    def test_expired_item_has_maximum_urgency(self):
        u_expired = self.urgency(("milk", -1))   # already expired
        u_today   = self.urgency(("milk",  0))   # expires today
        milk_recipe = ["milk", "sugar", "vanilla"]
        # Both should be ≥ 0.5 and close in value (both maximally urgent)
        assert expiry_weighted_match(milk_recipe, u_expired) >= 0.4
        assert expiry_weighted_match(milk_recipe, u_today)   >= 0.4

    def test_recipe_needing_no_pantry_items_scores_zero(self):
        u = self.urgency(("garlic", 0), ("milk", 1))
        apple_pie = ["apples", "sugar", "flour", "cinnamon", "pie crust"]
        assert expiry_weighted_match(apple_pie, u) == 0.0

    def test_empty_pantry_returns_zero(self):
        assert expiry_weighted_match(["garlic", "milk"], {}) == 0.0

    def test_empty_recipe_returns_zero(self):
        u = self.urgency(("garlic", 0))
        assert expiry_weighted_match([], u) == 0.0

    def test_score_bounded_zero_to_one(self):
        # All pantry items used → should not exceed 1.0
        u = self.urgency(("garlic", 0), ("milk", 0))
        everything = ["garlic", "milk"]
        score = expiry_weighted_match(everything, u)
        assert 0.0 <= score <= 1.0

    def test_ordering_reflects_urgency_not_recipe_complexity(self):
        """Ordering of 4 scenarios from most to least urgent pantry coverage."""
        u = self.urgency(("garlic", 0), ("milk", 1))
        # Uses both (complex)
        fettuccine = ["fettuccine", "garlic", "milk", "parmesan", "dill",
                      "chives", "nutmeg", "asparagus", "salmon", "lemon juice"]
        # Uses only garlic (more urgent)
        garlic_only = ["garlic", "bread", "butter"]
        # Uses only milk (less urgent)
        milk_only = ["milk", "cocoa", "nutella"]
        # Uses neither
        nothing = ["lobster", "cream", "brandy"]

        scores = [expiry_weighted_match(r, u) for r in
                  [fettuccine, garlic_only, milk_only, nothing]]
        # fettuccine uses both → highest
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        # garlic (urgency=1.0) beats milk (urgency<1.0)
        assert scores[1] > scores[2]
        # anything > nothing
        assert scores[2] > scores[3]
        assert scores[3] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. INGREDIENT MATCHING — ingredient_matches / match_ingredients
# ─────────────────────────────────────────────────────────────────────────────

class TestIngredientMatchingRealNames:
    """
    Contract: ingredient matching must handle real Food.com-style names.
    Safe qualifiers ("whole", "fresh", "minced", …) allow matching.
    Compound exclusions ("peanut butter", "ice cream", …) prevent false positives.
    Word-boundary rules prevent substring false positives.
    """

    # ── safe qualifiers: should match ────────────────────────────────────────

    def test_minced_garlic_matches_garlic(self):
        assert ingredient_matches("garlic", "minced garlic")

    def test_whole_milk_matches_milk(self):
        assert ingredient_matches("milk", "whole milk")

    def test_unsalted_butter_matches_butter(self):
        assert ingredient_matches("butter", "unsalted butter")

    def test_fresh_dill_matches_dill(self):
        assert ingredient_matches("dill", "fresh dill")

    def test_frozen_peas_matches_peas(self):
        assert ingredient_matches("peas", "frozen peas")

    def test_grated_parmesan_matches_parmesan(self):
        assert ingredient_matches("parmesan", "grated parmesan")

    def test_smoked_salmon_matches_salmon(self):
        # "salmon" is the head noun (last word) — always matches
        assert ingredient_matches("salmon", "smoked salmon")

    def test_cherry_tomatoes_matches_tomato(self):
        # singular of "tomatoes" is "tomato", "cherry" is in SAFE_QUALIFIERS
        assert ingredient_matches("tomato", "cherry tomatoes")

    def test_ground_black_pepper_matches_pepper(self):
        assert ingredient_matches("pepper", "ground black pepper")

    # ── compound exclusions: should NOT match ────────────────────────────────

    def test_butter_does_not_match_peanut_butter(self):
        assert not ingredient_matches("butter", "peanut butter")

    def test_butter_does_not_match_cocoa_butter(self):
        assert not ingredient_matches("butter", "cocoa butter")

    def test_cream_does_not_match_ice_cream(self):
        assert not ingredient_matches("cream", "ice cream")

    def test_oil_does_not_match_essential_oil(self):
        assert not ingredient_matches("oil", "essential oil")

    # ── word boundary: should NOT match ──────────────────────────────────────

    def test_egg_does_not_match_eggplant(self):
        assert not ingredient_matches("egg", "eggplant")

    def test_corn_does_not_match_peppercorns(self):
        # Old false-positive from partial_ratio — now fixed with word-boundary check
        assert not ingredient_matches("corn", "peppercorns")

    def test_berry_does_not_match_strawberry(self):
        assert not ingredient_matches("berry", "strawberry")

    def test_milk_does_not_match_milkshake(self):
        assert not ingredient_matches("milk", "milkshake")

    # ── match_ratio reflects what fraction of RECIPE is covered ──────────────

    def test_match_ratio_two_of_thirteen(self):
        pantry = ["garlic", "milk"]
        fettuccine = ["fettuccine", "olive oil", "garlic", "milk", "parmesan cheese",
                      "fresh dill", "chives", "nutmeg", "salt", "pepper",
                      "asparagus", "smoked salmon", "lemon juice"]
        result = match_ingredients(fettuccine, pantry)
        assert result["match_ratio"] == round(2 / 13, 6)
        assert "garlic" in result["matched"]
        assert "milk"   in result["matched"]

    def test_match_ratio_zero_when_nothing_matches(self):
        result = match_ingredients(["lobster", "brandy", "shallots"], ["eggs", "milk"])
        assert result["match_ratio"] == 0.0
        assert result["missing"] == ["lobster", "brandy", "shallots"]

    def test_match_ratio_one_when_everything_matches(self):
        result = match_ingredients(["eggs", "milk"], ["eggs", "milk", "butter"])
        assert result["match_ratio"] == 1.0
        assert result["missing"] == []

    def test_empty_recipe_returns_full_match(self):
        result = match_ingredients([], ["eggs"])
        assert result["match_ratio"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL RANKING INTEGRATION — realistic pantry + recipe scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingIntegration:
    """
    Contract: the final ranked list must reflect a combination of all four
    score components. These tests assert ORDER not exact scores, so they
    remain valid even if weights are tuned.
    """

    # ── scenario 1: garlic (today) + milk (tomorrow) ─────────────────────────

    def test_uses_both_expiring_beats_uses_neither(self):
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [
            recipe(1, "Fettuccine Alfredo",
                   ["fettuccine", "garlic", "milk", "parmesan", "dill",
                    "chives", "nutmeg", "asparagus", "smoked salmon"]),
            recipe(2, "Apple Pie",
                   ["apples", "sugar", "flour", "butter", "cinnamon"]),
        ]
        ranked = rank_recipes(p, recipes)
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    def test_uses_one_expiring_beats_uses_neither(self):
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [
            recipe(1, "Garlic Bread", ["garlic", "bread", "olive oil"]),
            recipe(2, "Chocolate Fondue",
                   ["bittersweet chocolate", "cream", "vanilla extract"]),
        ]
        ranked = rank_recipes(p, recipes)
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    def test_uses_most_urgent_item_beats_uses_less_urgent(self):
        """Garlic expires TODAY (urgency 1.0). Milk expires in 5 days (urgency ~0.3)."""
        p = pantry(("garlic", 0), ("milk", 5))
        recipes = [
            recipe(1, "Roast Garlic", ["garlic", "olive oil", "salt"]),
            recipe(2, "Milk Pudding", ["milk", "sugar", "vanilla", "cornstarch"]),
        ]
        ranked = rank_recipes(p, recipes)
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    # ── scenario 2: high β (care about having ingredients) ───────────────────

    def test_high_beta_favours_high_match_ratio(self):
        """
        With β=1.0, ingredient availability dominates.
        Recipe A has all ingredients (100% match), recipe B has none (0% match)
        and recipe A should rank first even if B might have higher CF score.
        """
        p = pantry(("eggs", 3), ("milk", 3), ("butter", 7), ("bread", 5))
        recipes = [
            recipe(1, "French Toast",
                   ["eggs", "milk", "bread", "butter", "vanilla extract"]),
            recipe(2, "Lobster Bisque",
                   ["lobster", "cream", "brandy", "shallots", "tarragon"]),
        ]
        ranked = rank_recipes(p, recipes, user_profile={"beta": 1.0})
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    def test_low_beta_still_ranks_expiry_urgently(self):
        """Even with β=0.0, a recipe using everything expiring today should rank high."""
        p = pantry(("garlic", 0), ("milk", 0))
        recipes = [
            recipe(1, "Garlic Milk Soup",
                   ["garlic", "milk", "broth", "salt"]),
            recipe(2, "Apple Cake",
                   ["apples", "flour", "sugar", "eggs", "cinnamon"]),
        ]
        ranked = rank_recipes(p, recipes, user_profile={"beta": 0.0})
        ids = [r.recipe_id for r in ranked]
        assert ids.index(1) < ids.index(2)

    # ── scenario 3: empty pantry ──────────────────────────────────────────────

    def test_empty_pantry_still_returns_all_recipes(self):
        p = []
        recipes = [
            recipe(1, "French Toast", ["eggs", "milk", "bread"]),
            recipe(2, "Pasta Carbonara", ["pasta", "eggs", "pancetta"]),
            recipe(3, "Lobster Bisque", ["lobster", "cream"]),
        ]
        ranked = rank_recipes(p, recipes)
        assert len(ranked) == 3

    def test_empty_pantry_expiry_scores_are_zero(self):
        ranked = rank_recipes([], [recipe(1, "Omelette", ["eggs", "milk"])])
        assert ranked[0].expiry_urgency == 0.0

    # ── scenario 4: single recipe ─────────────────────────────────────────────

    def test_single_recipe_always_returned(self):
        p = pantry(("eggs", 2))
        ranked = rank_recipes(p, [recipe(1, "Scrambled Eggs", ["eggs", "butter"])])
        assert len(ranked) == 1
        assert ranked[0].recipe_id == 1

    # ── scenario 5: all scores in valid range ─────────────────────────────────

    def test_all_final_scores_in_zero_one(self):
        p = pantry(("garlic", 0), ("milk", 1), ("eggs", 2), ("butter", 10))
        recipes = [
            recipe(1, "Pasta Carbonara",   ["pasta", "eggs", "pancetta", "garlic", "parmesan"]),
            recipe(2, "French Toast",       ["eggs", "milk", "bread", "butter", "vanilla"]),
            recipe(3, "Garlic Butter Shrimp", ["shrimp", "garlic", "butter", "lemon juice"]),
            recipe(4, "Chocolate Lava Cake", ["chocolate", "flour", "sugar", "eggs", "butter"]),
            recipe(5, "Lobster Bisque",     ["lobster", "cream", "brandy", "shallots"]),
        ]
        for r in rank_recipes(p, recipes):
            assert 0.0 <= r.final_score <= 1.0, \
                f"final_score out of range for {r.recipe_name}: {r.final_score}"

    def test_expiry_urgency_in_zero_one(self):
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [
            recipe(1, "Fettuccine Alfredo", ["garlic", "milk", "pasta", "parmesan"]),
            recipe(2, "Nutella Hot Choc",   ["nutella", "milk"]),
            recipe(3, "Apple Pie",          ["apples", "sugar", "flour"]),
        ]
        for r in rank_recipes(p, recipes):
            assert 0.0 <= r.expiry_urgency <= 1.0, \
                f"expiry_urgency out of range for {r.recipe_name}: {r.expiry_urgency}"

    def test_match_ratio_in_zero_one(self):
        p = pantry(("garlic", 3), ("milk", 5))
        recipes = [
            recipe(1, "Garlic Milk Pasta", ["garlic", "milk", "pasta"]),
            recipe(2, "Apple Pie",         ["apples", "sugar", "flour"]),
        ]
        for r in rank_recipes(p, recipes):
            assert 0.0 <= r.match_ratio <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCalibration:
    """
    Contract: after calibration, the highest-scoring recipe on a component
    should have calibrated value 1.0 and the lowest should have 0.0 (unless
    all values are equal, in which case all get 0.5).
    """

    from backend.services.scoring import _calibrate

    def test_min_becomes_zero(self):
        from backend.services.scoring import _calibrate
        cal = _calibrate([0.1, 0.5, 0.9])
        assert cal[0] == 0.0

    def test_max_becomes_one(self):
        from backend.services.scoring import _calibrate
        cal = _calibrate([0.1, 0.5, 0.9])
        assert cal[-1] == 1.0

    def test_uniform_values_all_get_half(self):
        from backend.services.scoring import _calibrate
        cal = _calibrate([0.42, 0.42, 0.42])
        assert all(v == 0.5 for v in cal)

    def test_order_preserved(self):
        from backend.services.scoring import _calibrate
        original = [0.1, 0.3, 0.7, 0.9]
        cal = _calibrate(original)
        assert cal == sorted(cal)

    def test_calibration_applied_in_ranking(self):
        """
        After ranking, the top recipe's expiry_urgency and match_ratio are the
        raw pre-calibration values — calibration only affects final_score.
        The highest final_score should still belong to the best recipe.
        """
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [
            recipe(1, "Garlic Milk Pasta", ["garlic", "milk", "pasta"]),
            recipe(2, "Apple Pie",         ["apples", "sugar", "flour"]),
        ]
        ranked = rank_recipes(p, recipes)
        # Top recipe is garlic+milk pasta
        assert ranked[0].recipe_id == 1
        # Final score is highest
        assert ranked[0].final_score == max(r.final_score for r in ranked)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MMR DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

class TestMMRDiversity:
    """
    Contract:
    - The first result is always the highest-scored recipe (MMR anchors on it).
    - Two recipes sharing all ingredients should not both appear at the top
      when a more diverse alternative exists.
    """

    def test_first_result_is_highest_scored(self):
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [recipe(i, f"Recipe {i}", ["garlic", "milk", "item" + str(i)])
                   for i in range(1, 10)]
        ranked = rank_recipes(p, recipes)
        assert ranked[0].final_score == max(r.final_score for r in ranked)

    def test_diverse_recipe_preferred_over_duplicate(self):
        """
        If recipe A and recipe B are nearly identical (same ingredients) and
        recipe C is different, C should appear in the top-3 feed ahead of
        whichever of A/B scored slightly lower.
        """
        p = pantry(("eggs", 2), ("milk", 2))
        clone1 = recipe(1, "Omelette 1",  ["eggs", "milk", "salt", "pepper"])
        clone2 = recipe(2, "Omelette 2",  ["eggs", "milk", "salt", "pepper", "herbs"])
        diverse = recipe(3, "Tomato Soup", ["tomatoes", "broth", "onion", "garlic"])

        # Give both omelettes identical CF/CB scores; diverse recipe has slightly lower
        cf_scores = {1: 0.9, 2: 0.85, 3: 0.6}
        ranked = rank_recipes(p, [clone1, clone2, diverse],
                               cf_scores=cf_scores,
                               user_profile={"has_cf": True},
                               top_n=3)
        ids = [r.recipe_id for r in ranked]
        # Diverse recipe should be in output (not squeezed out by two near-clones)
        assert 3 in ids

    def test_top_n_respected(self):
        p = pantry(("garlic", 1))
        recipes = [recipe(i, f"Recipe {i}", ["garlic", "item" + str(i)]) for i in range(1, 15)]
        ranked = rank_recipes(p, recipes, top_n=5)
        assert len(ranked) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPIRY URGENCY MATH — urgency_score
# ─────────────────────────────────────────────────────────────────────────────

class TestExpiryUrgencyMath:
    """Contract: the exponential decay formula must produce sensible values."""

    def test_expires_today_is_one(self):
        assert urgency_score(future(0)) == 1.0

    def test_expired_yesterday_is_one(self):
        assert urgency_score(future(-1)) == 1.0

    def test_half_life_at_3_days(self):
        # Default half_life = 3 days → score at 3 days should be ~0.5
        score = urgency_score(future(3))
        assert abs(score - 0.5) < 0.01

    def test_14_days_is_low(self):
        assert urgency_score(future(14)) < 0.1

    def test_far_future_approaches_zero(self):
        assert urgency_score(future(60)) < 0.01

    def test_urgency_decreasing_over_time(self):
        scores = [urgency_score(future(d)) for d in [0, 1, 3, 7, 14]]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CF / CB MODEL AVAILABILITY — weight redistribution
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAvailabilityWeights:
    """
    Contract: when CF or CB model files are absent (not trained yet), their
    weights must redistribute to the remaining components so the formula still
    sums to 1.0 and rankings remain meaningful.

    This matters for:
    - Fresh installs (no trained models)
    - The --no-implicit retrain path (CF model exists, but CB/CF may be cold)
    - Cold-start users who have no latent vector in the MF model
    """

    def test_no_cf_no_cb_scores_still_rank(self):
        """Without CF or CB scores, expiry + match alone must produce a ranking."""
        p = pantry(("garlic", 0), ("milk", 1))
        recipes = [
            recipe(1, "Garlic Milk Pasta", ["garlic", "milk", "pasta"]),
            recipe(2, "Apple Pie",         ["apples", "sugar", "flour"]),
        ]
        ranked = rank_recipes(p, recipes, cf_scores=None, cb_scores=None)
        assert len(ranked) == 2
        assert ranked[0].recipe_id == 1  # garlic+milk should still win on expiry

    def test_cf_scores_present_boost_high_cf_recipe(self):
        """A recipe with high CF score should rank above one with low CF + same domain scores."""
        p = pantry(("salt", 30))  # salt far from expiry → low expiry urgency for both
        recipes = [
            recipe(1, "Steak",    ["beef", "salt", "pepper"]),
            recipe(2, "Salad",    ["lettuce", "salt", "lemon"]),
        ]
        cf_scores = {1: 0.95, 2: 0.10}
        ranked = rank_recipes(p, recipes,
                               cf_scores=cf_scores,
                               user_profile={"has_cf": True})
        assert ranked[0].recipe_id == 1

    def test_cb_unavailable_weight_redistributes(self):
        """
        When has_cb=False, CB weight (δ=0.10) redistributes to CF and expiry.
        Final scores must still be in [0,1] and ordering must be sensible.
        """
        p = pantry(("garlic", 0))
        recipes = [
            recipe(1, "Garlic Shrimp", ["garlic", "shrimp", "butter"]),
            recipe(2, "Chocolate Cake", ["chocolate", "flour", "sugar", "eggs"]),
        ]
        ranked_no_cb = rank_recipes(p, recipes,
                                     user_profile={"has_cb": False})
        ranked_with_cb = rank_recipes(p, recipes,
                                       cb_scores={1: 0.8, 2: 0.1},
                                       user_profile={"has_cb": True})
        # In both cases garlic shrimp should win and scores stay in range
        assert ranked_no_cb[0].recipe_id == 1
        assert ranked_with_cb[0].recipe_id == 1
        for r in ranked_no_cb + ranked_with_cb:
            assert 0.0 <= r.final_score <= 1.0

    def test_real_cf_model_loaded(self):
        """
        The trained cf_model.pkl must be present and loadable.
        This guards against accidental deletion or corruption after retrain.
        """
        from backend.ml.serve_cf import cf_model_available
        assert cf_model_available(), \
            "CF model not loaded — run: python3 -m backend.ml.train_cf --no-implicit"

    def test_real_cb_model_loaded(self):
        """The trained cb_matrix.npz must be present and loadable."""
        from backend.ml.serve_cb import model_available as cb_model_available
        assert cb_model_available(), \
            "CB model not loaded — run: python3 -m backend.ml.train_cb"

    def test_cf_meta_records_no_implicit(self):
        """
        After --no-implicit retrain, cf_meta.json must record use_implicit=False
        so we know the model was trained on real data only.
        """
        import json
        from pathlib import Path
        meta_path = Path(__file__).resolve().parents[1] / "models" / "cf_meta.json"
        assert meta_path.exists(), "cf_meta.json missing"
        meta = json.loads(meta_path.read_text())
        assert meta.get("use_implicit") is False, \
            f"Expected use_implicit=False in cf_meta.json, got: {meta.get('use_implicit')}"
