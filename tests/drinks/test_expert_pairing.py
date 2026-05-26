"""
test_expert_pairing.py
----------------------
Behavior tests for backend/services/expert_pairing.py.

We use SimpleNamespace fakes (no DB) — expert_pairing only reads
attributes, so duck typing is enough.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.drinks.expert_pairing import (
    BEER_STYLE_RULES,
    MAX_BOOST,
    WINE_BOOST_PER_MATCH,
    expert_boost,
    expert_boost_batch,
)


# ── recipe fakes ─────────────────────────────────────────────────────────

def _recipe(ingredients_csv: str, tags_csv: str = "") -> SimpleNamespace:
    return SimpleNamespace(ingredients_csv=ingredients_csv, tags_csv=tags_csv)


# ── drink fakes ──────────────────────────────────────────────────────────

def _wine(harmonize: str, id_: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id=id_, kind="wine", style=None,
        harmonize_csv=harmonize, name="Test Wine",
    )


def _beer(style: str, id_: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        id=id_, kind="beer", style=style,
        harmonize_csv=None, name="Test Beer",
    )


# ── wine Harmonize matching ─────────────────────────────────────────────

def test_steak_recipe_with_beef_wine_gets_boost():
    recipe = _recipe("beef,steak,garlic,butter", tags_csv="bbq,american")
    wine = _wine("Beef,Lamb,Grilled")
    boost = expert_boost(recipe, wine)
    # recipe has beef → bridges to {beef, red, full, bold, ...}
    # wine harmonize = {beef, lamb, grilled} → overlap {beef} → 1 match
    assert boost >= WINE_BOOST_PER_MATCH


def test_seafood_recipe_with_seafood_wine_gets_boost():
    recipe = _recipe("shrimp,pasta,garlic,lemon")
    wine = _wine("Fish,Seafood,Salads")
    boost = expert_boost(recipe, wine)
    assert boost > 0
    # shrimp bridges to {seafood, shellfish, white, light, crisp}
    # harmonize {fish, seafood, salads} → overlap {seafood} → 1 match
    assert boost >= WINE_BOOST_PER_MATCH


def test_steak_with_seafood_wine_no_boost():
    recipe = _recipe("beef,steak,garlic")
    wine = _wine("Fish,Seafood")
    assert expert_boost(recipe, wine) == 0.0


def test_wine_boost_caps_at_max():
    """Many overlapping harmonize categories should not exceed MAX_BOOST."""
    recipe = _recipe(
        "beef,lamb,fish,pasta,chicken,shrimp,chocolate",
        tags_csv="italian,seafood,bbq,vegetarian",
    )
    wine = _wine("Beef,Lamb,Fish,Pasta,Poultry,Seafood,Cheese,Dessert,Italian,Vegetarian")
    assert expert_boost(recipe, wine) == MAX_BOOST


def test_multi_word_harmonize_category_tokenized():
    """'Rich Fish' should tokenize so that 'fish' from a recipe matches."""
    recipe = _recipe("fish,lemon")
    wine = _wine("Rich Fish,Seafood")
    boost = expert_boost(recipe, wine)
    assert boost >= WINE_BOOST_PER_MATCH


# ── beer style heuristics ────────────────────────────────────────────────

def test_spicy_recipe_with_ipa_gets_boost():
    recipe = _recipe("chicken,curry,chili,onion", tags_csv="indian,spicy")
    beer = _beer("American IPA")
    boost = expert_boost(recipe, beer)
    assert boost > 0


def test_chocolate_dessert_with_stout_gets_boost():
    recipe = _recipe("chocolate,butter,sugar,eggs", tags_csv="dessert")
    beer = _beer("Russian Imperial Stout")
    assert expert_boost(recipe, beer) > 0


def test_porter_with_chocolate_gets_boost():
    recipe = _recipe("chocolate,cream,sugar")
    assert expert_boost(recipe, _beer("Robust Porter")) > 0


def test_pilsner_with_fish_gets_boost():
    recipe = _recipe("fish,lemon,salad")
    beer = _beer("Czech Pilsner")
    assert expert_boost(recipe, beer) > 0


def test_steak_with_pilsner_no_boost():
    """Pilsner + beef has no rule -> 0."""
    recipe = _recipe("beef,steak")
    assert expert_boost(recipe, _beer("Pilsner")) == 0.0


def test_dessert_with_pilsner_no_boost():
    recipe = _recipe("chocolate,sugar")
    assert expert_boost(recipe, _beer("Pilsner")) == 0.0


def test_unknown_style_returns_zero():
    recipe = _recipe("chicken,curry")
    assert expert_boost(recipe, _beer("Some Made-Up Style")) == 0.0


def test_multiple_beer_rules_can_stack():
    """Stout + chocolate + beef should trigger both stout rules."""
    recipe = _recipe("beef,chocolate,butter")  # both 'beef' and 'chocolate'
    beer = _beer("Imperial Stout")
    boost = expert_boost(recipe, beer)
    # one rule for chocolate (+0.10), one for beef (+0.05) = 0.15
    assert boost >= 0.10  # at least the chocolate rule fires


# ── edge cases ───────────────────────────────────────────────────────────

def test_none_inputs_return_zero():
    assert expert_boost(None, _wine("Beef")) == 0.0
    assert expert_boost(_recipe("beef"), None) == 0.0
    assert expert_boost(None, None) == 0.0


def test_empty_recipe_returns_zero():
    assert expert_boost(_recipe(""), _wine("Beef,Lamb")) == 0.0


def test_wine_with_no_harmonize_returns_zero():
    assert expert_boost(_recipe("beef"), _wine("")) == 0.0


def test_beer_with_no_style_returns_zero():
    beer = SimpleNamespace(id=1, kind="beer", style=None,
                           harmonize_csv=None, name="Mystery Beer")
    assert expert_boost(_recipe("chicken,curry,spicy"), beer) == 0.0


def test_rule_count_sanity():
    """Smoke check: rule table is non-trivial."""
    assert len(BEER_STYLE_RULES) >= 5


# ── batch ────────────────────────────────────────────────────────────────

def test_expert_boost_batch_returns_only_positive():
    recipe = _recipe("beef,steak,butter")
    drinks = [
        _wine("Beef,Lamb",          id_=1),
        _wine("Fish,Seafood",       id_=2),
        _beer("Russian Imperial Stout", id_=100),  # stout + beef rule fires (+0.05)
        _beer("Pilsner",            id_=101),  # no match
    ]
    out = expert_boost_batch(recipe, drinks)
    assert set(out.keys()) == {1, 100}
    assert out[1] > 0
    assert out[100] > 0
    assert 2 not in out
    assert 101 not in out


def test_expert_boost_batch_empty_inputs():
    assert expert_boost_batch(None, [_wine("Beef")]) == {}
    assert expert_boost_batch(_recipe("beef"), []) == {}


def test_expert_boost_batch_skips_none_drinks():
    recipe = _recipe("beef")
    out = expert_boost_batch(recipe, [None, _wine("Beef", id_=7), None])
    assert 7 in out
