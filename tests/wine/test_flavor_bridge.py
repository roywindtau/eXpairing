"""
test_flavor_bridge.py
---------------------
Tests for backend/ml/flavor_bridge.py.

Coverage:
  - Known single-token ingredients map correctly
  - Multi-word ingredients ("chicken breast") match via substring
  - Repetition is preserved (TF weighting later relies on it)
  - Empty / unknown / malformed input returns empty list (no exceptions)
  - bridge_recipe_doc combines ingredients + tags + bridged tokens
  - bridge_tags only emits tags in the known cuisine set
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ml.wine.serving.flavor_bridge import (
    INGREDIENT_FLAVORS,
    bridge_ingredients,
    bridge_recipe_doc,
    bridge_tags,
    bridge_text,
)


# ── bridge_ingredients ───────────────────────────────────────────────────

def test_known_ingredient_maps_to_flavors():
    tokens = bridge_ingredients("beef")
    assert "beef" in tokens
    assert "red" in tokens
    assert "full" in tokens


def test_multi_word_ingredient_substring_match():
    """'boneless chicken breast' should still trigger 'chicken'."""
    tokens = bridge_ingredients("boneless chicken breast")
    assert "poultry" in tokens
    assert "white" in tokens


def test_multiple_ingredients_csv():
    tokens = bridge_ingredients("shrimp,garlic,lemon")
    assert "seafood" in tokens
    assert "savory" in tokens
    assert "acidic" in tokens


def test_unknown_ingredient_returns_empty():
    assert bridge_ingredients("quinoa") == []


def test_empty_input_safe():
    assert bridge_ingredients("") == []
    assert bridge_ingredients(",,,") == []


def test_repetition_preserved_for_tf_weighting():
    """If two ingredients both bridge to 'red', it appears twice."""
    tokens = bridge_ingredients("beef,lamb")
    assert tokens.count("red") == 2


def test_ingredient_with_punctuation_and_case():
    """Mixed case + extra whitespace shouldn't break the lookup."""
    tokens = bridge_ingredients(" CHICKEN  Stock, GARLIC ")
    assert "poultry" in tokens
    assert "savory" in tokens


# ── bridge_tags ──────────────────────────────────────────────────────────

def test_bridge_tags_keeps_known_cuisines():
    assert "italian" in bridge_tags("italian,30-minutes-or-less,easy")


def test_bridge_tags_filters_unknown():
    """Food.com tags like '30-minutes-or-less' should not leak through."""
    assert "30-minutes-or-less" not in bridge_tags("30-minutes-or-less,italian")


def test_bridge_tags_empty_safe():
    assert bridge_tags("") == []
    assert bridge_tags(None) == []


# ── bridge_recipe_doc ────────────────────────────────────────────────────

def test_bridge_recipe_doc_combines_all_sources():
    recipe = SimpleNamespace(
        ingredients_csv="shrimp,garlic,white wine",
        tags_csv="italian,seafood,30-minutes-or-less",
    )
    doc = bridge_recipe_doc(recipe)

    assert "shrimp" in doc            # original ingredient word
    assert "garlic" in doc            # original ingredient word
    assert "italian" in doc           # known cuisine tag passed through
    assert "seafood" in doc           # bridged from shrimp + also a kept tag
    assert "savory" in doc            # bridged from garlic
    assert "30-minutes-or-less" not in doc  # unknown tag filtered


def test_bridge_recipe_doc_missing_fields_safe():
    """Should not crash if recipe has no ingredients/tags."""
    recipe = SimpleNamespace(ingredients_csv=None, tags_csv=None)
    assert bridge_recipe_doc(recipe) == ""


def test_bridge_recipe_doc_lowercase():
    recipe = SimpleNamespace(
        ingredients_csv="BEEF,SALT",
        tags_csv="ITALIAN",
    )
    doc = bridge_recipe_doc(recipe)
    assert doc == doc.lower()


# ── bridge_text (Path B helper) ──────────────────────────────────────────

def test_bridge_text_with_extra_tags():
    """Aggregated ingredient blob + extra cuisine tags from history."""
    doc = bridge_text("chicken tomato basil", also_tags=["italian", "weeknight"])
    assert "chicken" in doc
    assert "tomato" in doc
    assert "italian" in doc
    assert "weeknight" not in doc      # not a known cuisine
    assert "poultry" in doc            # bridged
    assert "red" in doc                # bridged from tomato


def test_bridge_text_handles_none():
    assert bridge_text(None) == ""
    assert bridge_text("") == ""


# ── lexicon sanity ───────────────────────────────────────────────────────

def test_lexicon_values_are_lowercase_and_non_empty():
    for k, vs in INGREDIENT_FLAVORS.items():
        assert k == k.lower(), f"lexicon key '{k}' must be lowercase"
        assert vs, f"lexicon value for '{k}' must be non-empty"
        for v in vs:
            assert isinstance(v, str) and v == v.lower(), \
                f"flavor token '{v}' for '{k}' must be lowercase string"


def test_lexicon_covers_core_categories():
    """Smoke check: lexicon must cover at least these protein categories."""
    must_have = ["beef", "chicken", "fish", "shrimp", "chocolate", "tomato"]
    for ing in must_have:
        assert ing in INGREDIENT_FLAVORS, f"Missing core ingredient '{ing}'"
