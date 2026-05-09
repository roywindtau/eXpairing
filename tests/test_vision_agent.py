"""
tests/test_vision_agent.py
--------------------------
Tests for the vision agent canonicalization pipeline and
ingredient mapper. These tests do NOT call the OpenAI API —
they test all the deterministic parts: cleaning, fuzzy matching,
and the mock scan.

Tests cover:
  - clean_product_name: brand stripping, quantity removal, noise removal
  - IngredientMapper: fuzzy matching, fallback to cleaned name
  - IngredientMapper.map_batch: batch processing
  - mock_scan: correct structure and non-null required fields
  - openfoodfacts: _extract_primary_ingredient parsing
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date

from backend.services.vision_agent import (
    _clean_raw_name,
    IngredientCanonicalizer,
    mock_scan,
)
from backend.canonicalizer.ingredient_map import (
    clean_product_name,
    IngredientMapper,
    BRAND_WORDS,
    NOISE_WORDS,
)
from backend.canonicalizer.openfoodfacts import _extract_primary_ingredient


# ── clean_product_name ─────────────────────────────────────────────────────

class TestCleanProductName:
    def test_strips_brand_name(self):
        assert "milk" in clean_product_name("Tnuva 3% Milk")

    def test_strips_quantity_ml(self):
        result = clean_product_name("Milk 500ml")
        assert "500ml" not in result
        assert "500" not in result

    def test_strips_quantity_g(self):
        result = clean_product_name("Butter 250g")
        assert "250g" not in result

    def test_strips_noise_organic(self):
        result = clean_product_name("Organic Eggs")
        assert "organic" not in result

    def test_strips_noise_free_range(self):
        result = clean_product_name("Free Range Eggs")
        assert "free" not in result
        assert "range" not in result

    def test_keeps_core_ingredient(self):
        assert "eggs" in clean_product_name("Free Range Large Eggs 6 pack")

    def test_multiword_ingredient_preserved(self):
        result = clean_product_name("Heinz Tomato Ketchup 300ml")
        assert "tomato" in result
        assert "ketchup" in result

    def test_lowercase_output(self):
        result = clean_product_name("WHOLE MILK")
        assert result == result.lower()

    def test_empty_string_fallback(self):
        # Should not crash, returns something
        result = clean_product_name("")
        assert isinstance(result, str)

    def test_strips_percentage(self):
        result = clean_product_name("Milk 3%")
        assert "3%" not in result

    def test_strips_x_quantity(self):
        result = clean_product_name("Yogurt 6 x 125g")
        assert "6" not in result


# ── vision_agent _clean_raw_name ───────────────────────────────────────────

class TestVisionCleanRawName:
    """vision_agent._clean_raw_name uses the same logic."""
    def test_milk(self):
        result = _clean_raw_name("Tnuva 3% Fresh Milk 500ml")
        assert "milk" in result

    def test_eggs(self):
        result = _clean_raw_name("Free Range Large Eggs")
        assert "eggs" in result

    def test_butter(self):
        result = _clean_raw_name("Anchor Unsalted Butter 200g")
        assert "butter" in result


# ── IngredientMapper ───────────────────────────────────────────────────────

SAMPLE_VOCAB = [
    "milk", "eggs", "butter", "flour", "sugar", "salt", "pepper",
    "tomato", "tomato ketchup", "tomatoes", "chicken", "chicken breast",
    "pasta", "olive oil", "garlic", "onion", "cheese", "cheddar cheese",
    "cream cheese", "yogurt", "rice", "bread",
]


class TestIngredientMapper:
    def mapper(self):
        return IngredientMapper.from_vocab_list(SAMPLE_VOCAB)

    def test_exact_match(self):
        m = self.mapper()
        assert m.map("milk") == "milk"

    def test_brand_stripped_then_matched(self):
        m = self.mapper()
        result = m.map("Tnuva 3% Milk 500ml")
        assert result == "milk"

    def test_fuzzy_whole_milk_to_milk(self):
        m = self.mapper()
        result = m.map("whole milk")
        assert result == "milk"

    def test_fuzzy_cherry_tomatoes(self):
        m = self.mapper()
        result = m.map("cherry tomatoes")
        # Should match "tomatoes" or "tomato"
        assert "tomato" in result

    def test_multiword_match(self):
        m = self.mapper()
        result = m.map("Heinz Tomato Ketchup")
        assert "tomato" in result

    def test_unknown_returns_cleaned_name(self):
        m = self.mapper()
        result = m.map("dragon fruit 200g")
        # Not in vocab — returns cleaned version
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_vocab_returns_cleaned(self):
        m = IngredientMapper(vocab=[])
        result = m.map("Tnuva 3% Milk 500ml")
        # Falls back to text cleaning
        assert "milk" in result

    def test_map_batch_preserves_order(self):
        m = self.mapper()
        inputs  = ["Tnuva Milk", "Free Range Eggs", "Anchor Butter"]
        results = m.map_batch(inputs)
        assert len(results) == 3
        assert "milk"   in results[0]
        assert "eggs"   in results[1]
        assert "butter" in results[2]

    def test_vocab_size(self):
        m = self.mapper()
        assert m.vocab_size() == len(SAMPLE_VOCAB)

    def test_cheddar_cheese(self):
        m = self.mapper()
        result = m.map("Mature Cheddar Cheese 200g")
        assert "cheese" in result


# ── IngredientCanonicalizer (vision_agent) ─────────────────────────────────

class TestIngredientCanonicalizer:
    def canon(self):
        return IngredientCanonicalizer(vocab=SAMPLE_VOCAB)

    def test_canonicalize_milk(self):
        result = self.canon().canonicalize("Tnuva 3% Milk 500ml")
        assert result == "milk"

    def test_canonicalize_eggs(self):
        result = self.canon().canonicalize("Free Range Eggs 6 pack")
        assert result == "eggs"

    def test_no_vocab_fallback(self):
        c = IngredientCanonicalizer(vocab=[])
        result = c.canonicalize("Tnuva 3% Milk")
        assert "milk" in result

    def test_returns_string(self):
        result = self.canon().canonicalize("anything at all")
        assert isinstance(result, str)


# ── mock_scan ──────────────────────────────────────────────────────────────

class TestMockScan:
    def test_returns_list(self):
        items = mock_scan()
        assert isinstance(items, list)

    def test_nonempty(self):
        items = mock_scan()
        assert len(items) > 0

    def test_required_fields_present(self):
        for item in mock_scan():
            assert "ingredient"  in item
            assert "expiry_date" in item
            assert "raw_name"    in item
            assert "quantity"    in item

    def test_all_have_ingredient(self):
        for item in mock_scan():
            assert isinstance(item["ingredient"], str)
            assert len(item["ingredient"]) > 0

    def test_all_have_expiry_date(self):
        """Mock scan should always return expiry dates (not null)."""
        for item in mock_scan():
            assert item["expiry_date"] is not None

    def test_expiry_dates_are_future(self):
        today = date.today().isoformat()
        for item in mock_scan():
            assert item["expiry_date"] >= today, \
                f"Mock expiry {item['expiry_date']} is in the past"

    def test_expiry_date_format(self):
        import re
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for item in mock_scan():
            assert pattern.match(item["expiry_date"]), \
                f"Bad date format: {item['expiry_date']}"


# ── openfoodfacts _extract_primary_ingredient ──────────────────────────────

class TestExtractPrimaryIngredient:
    def test_from_ingredients_tags(self):
        product = {"ingredients_tags": ["en:wheat-flour", "en:sugar", "en:eggs"]}
        result  = _extract_primary_ingredient(product)
        assert result is not None
        assert "wheat" in result or "flour" in result

    def test_from_ingredients_list(self):
        product = {
            "ingredients": [{"text": "Skimmed Milk"}, {"text": "Sugar"}]
        }
        result = _extract_primary_ingredient(product)
        assert result is not None
        assert "milk" in result

    def test_from_product_name_fallback(self):
        product = {"product_name_en": "Whole Milk 500ml"}
        result  = _extract_primary_ingredient(product)
        assert result is not None
        assert "milk" in result

    def test_empty_product_returns_none(self):
        result = _extract_primary_ingredient({})
        assert result is None

    def test_strips_language_prefix(self):
        product = {"ingredients_tags": ["en:tomato-paste"]}
        result  = _extract_primary_ingredient(product)
        assert "en:" not in (result or "")
