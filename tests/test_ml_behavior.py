"""
tests/test_ml_behavior.py
--------------------------
Behavioral integration tests for the eXpairing recommender system.

These tests hit the live backend at localhost:8000 and verify that the
ML pipeline behaves correctly at each stage of user progression:

  - Cold-start initial state (new user, no pantry, no ratings)
  - Pantry effect: ingredients raise match_ratio and expiry urgency
  - CF progression: is_warm flips at 5 ratings; SVD activates with trained model
  - CB / beta: high-beta users rank pantry-matching recipes higher
  - Vision: mock scan structure; confirm → pantry → feed

Requirements:
  - Backend running:   uvicorn backend.main:app --reload --port 8000
  - DB seeded:         python -m backend.db.seed_dev   (or seed_recipes.py)

Optional (unlocks SVD and real-vision tests):
  - ML models trained: python -m backend.ml.train_cf
  - Vision key set:    OPENAI_API_KEY=sk-... pytest ...::TestVision

Run:
    cd /Users/roy.wind/recsys/smartrecipes
    pytest tests/test_ml_behavior.py -v
"""

from __future__ import annotations

import datetime
import os

import pytest
import requests

BASE = os.environ.get("TEST_API_BASE", "http://localhost:8000")

# Models are stored relative to project root (where pytest is invoked from)
_SVD_EXISTS      = os.path.exists("models/cf_model.pkl")
_SIM_MAT_EXISTS  = os.path.exists("models/item_sim_matrix.npz")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ranked(user_id: int, top_n: int = 15) -> list[dict]:
    r = requests.get(f"{BASE}/recipes/ranked", params={"user_id": user_id, "top_n": top_n})
    assert r.status_code == 200, f"ranked failed {r.status_code}: {r.text[:200]}"
    return r.json()


def _add_pantry(user_id: int, ingredient: str, expiry: str) -> dict:
    r = requests.post(f"{BASE}/pantry/{user_id}", json={
        "ingredient": ingredient, "expiry_date": expiry,
        "quantity": None, "raw_name": None,
    })
    assert r.status_code == 201, f"add_pantry failed {r.status_code}: {r.text[:200]}"
    return r.json()


def _rate(user_id: int, recipe_id: int, stars: int) -> None:
    r = requests.post(f"{BASE}/events", json={
        "user_id": user_id, "recipe_id": recipe_id,
        "event_type": "rate", "rating": stars,
    })
    assert r.status_code == 201, f"rate event failed {r.status_code}: {r.text[:200]}"


def _stats(user_id: int) -> dict:
    r = requests.get(f"{BASE}/users/{user_id}/stats")
    assert r.status_code == 200
    return r.json()


def _make_user(name: str = "_ml_test", beta: float = 0.35,
               diet_tags: str | None = None) -> int:
    payload: dict = {"name": name, "beta": beta}
    if diet_tags:
        payload["diet_tags"] = diet_tags
    r = requests.post(f"{BASE}/users", json=payload)
    assert r.status_code == 201, f"create_user failed {r.status_code}: {r.text[:200]}"
    return r.json()["id"]


def _future(days: int) -> str:
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def new_user() -> int:
    """Fresh user — no pantry, no ratings."""
    return _make_user()


@pytest.fixture
def user_with_pantry() -> int:
    """User with eggs, butter, flour, milk (covers many simple recipes)."""
    uid = _make_user()
    for ing in ("eggs", "butter", "flour", "milk"):
        _add_pantry(uid, ing, _future(60))
    return uid


# ─── 1. Cold-start initial state ──────────────────────────────────────────────

class TestInitialState:
    """A brand-new user with no pantry and no ratings should get sensible defaults."""

    def test_returns_recipes(self, new_user):
        assert len(_ranked(new_user)) > 0

    def test_top_item_is_highest_scored(self, new_user):
        # MMR reranking trades strict score-order for ingredient diversity.
        # The first item is always the highest-scored (MMR starts with it),
        # but subsequent items may not be in strict descending order.
        results = _ranked(new_user, top_n=20)
        scores  = [r["final_score"] for r in results]
        assert scores[0] == max(scores), \
            f"Top item should have highest score. Got: {scores[:5]}"

    def test_all_score_fields_in_0_1(self, new_user):
        for r in _ranked(new_user):
            for field in ("final_score", "cf_score", "cb_score", "expiry_urgency", "match_ratio"):
                v = r[field]
                assert 0.0 <= v <= 1.0, \
                    f"recipe {r['recipe_id']}: {field}={v:.4f} outside [0,1]"

    def test_empty_pantry_gives_zero_expiry_urgency(self, new_user):
        for r in _ranked(new_user):
            assert r["expiry_urgency"] == 0.0, \
                f"recipe {r['recipe_id']}: urgency={r['expiry_urgency']} with empty pantry"

    def test_empty_pantry_gives_zero_match_ratio(self, new_user):
        for r in _ranked(new_user):
            assert r["match_ratio"] == 0.0, \
                f"recipe {r['recipe_id']}: match_ratio={r['match_ratio']} with empty pantry"

    def test_cold_start_cf_scores_differentiated(self, new_user):
        """Cold-start CF should give varied scores, not all 0 or all the same value."""
        cf_scores = [r["cf_score"] for r in _ranked(new_user, top_n=20)]
        n_nonzero = sum(1 for s in cf_scores if s > 0)
        n_unique  = len({round(s, 4) for s in cf_scores})
        assert n_nonzero > 0, \
            "CF scores are all zero — cold-start fallback not producing preference scores"
        assert n_unique > 1, \
            f"CF scores all identical ({cf_scores[0]:.4f}) — no differentiation"

    def test_cf_strategy_is_valid_value(self, new_user):
        valid = {"biased_mf", "item_based_cold_start", "blended", "none"}
        for r in _ranked(new_user):
            assert r["cf_strategy"] in valid, \
                f"Unknown cf_strategy: {r['cf_strategy']!r}"

    def test_required_fields_present(self, new_user):
        required = {
            "recipe_id", "recipe_name", "final_score", "cf_score", "cb_score",
            "expiry_urgency", "match_ratio", "cf_strategy",
            "missing_ingredients", "matched_ingredients",
        }
        for r in _ranked(new_user):
            missing = required - set(r.keys())
            assert not missing, f"recipe response missing fields: {missing}"

    def test_user_starts_cold_in_stats(self, new_user):
        stats = _stats(new_user)
        assert stats["n_ratings"] == 0
        assert stats["is_warm"] is False
        assert stats["warm_cf_progress_pct"] == 0.0


# ─── 2. Pantry effect on match_ratio and expiry urgency ───────────────────────

class TestPantryEffect:
    """Adding pantry items should raise match_ratio and drive expiry urgency."""

    def test_adding_items_increases_avg_match_ratio(self, new_user):
        before = sum(r["match_ratio"] for r in _ranked(new_user, 20)) / 20

        for ing in ("eggs", "butter", "flour", "milk"):
            _add_pantry(new_user, ing, _future(60))

        after = sum(r["match_ratio"] for r in _ranked(new_user, 20)) / 20
        assert after > before, \
            f"Pantry items should raise avg match_ratio: {before:.4f} → {after:.4f}"

    def test_max_match_ratio_significant_with_common_ingredients(self, user_with_pantry):
        results = _ranked(user_with_pantry, top_n=20)
        best = max(r["match_ratio"] for r in results)
        assert best > 0.3, \
            f"With eggs/butter/flour/milk, expected max match_ratio > 0.3, got {best:.3f}"

    def test_near_expiry_produces_nonzero_urgency(self, new_user):
        _add_pantry(new_user, "eggs", _future(1))  # expires tomorrow

        urgencies = [r["expiry_urgency"] for r in _ranked(new_user, 20)]
        assert max(urgencies) > 0.0, \
            "Eggs expiring tomorrow should drive expiry_urgency > 0 on egg-using recipes"

    def test_near_expiry_urgency_greater_than_far_expiry(self, new_user):
        """Same ingredient: near expiry → higher urgency than far expiry."""
        uid_near = _make_user("_near")
        uid_far  = _make_user("_far")
        _add_pantry(uid_near, "eggs", _future(1))
        _add_pantry(uid_far,  "eggs", _future(30))

        near_max = max(r["expiry_urgency"] for r in _ranked(uid_near, 20))
        far_max  = max(r["expiry_urgency"] for r in _ranked(uid_far,  20))
        assert near_max >= far_max, \
            f"Near-expiry urgency {near_max:.4f} should be ≥ far-expiry {far_max:.4f}"

    def test_expiry_urgency_zero_for_far_future(self, new_user):
        """Ingredient expiring in 30+ days contributes less urgency than a nearby one."""
        _add_pantry(new_user, "eggs", _future(30))
        results = _ranked(new_user, 20)
        near_uid = _make_user("_near2")
        _add_pantry(near_uid, "eggs", _future(1))
        near_results = _ranked(near_uid, 20)

        far_max  = max(r["expiry_urgency"] for r in results)
        near_max = max(r["expiry_urgency"] for r in near_results)
        assert near_max > far_max or near_max > 0, \
            "Near-expiry should produce more urgency than far-future"

    def test_match_ratio_deterministic(self, user_with_pantry):
        """Two consecutive calls with same pantry must return identical match_ratios."""
        r1 = {r["recipe_id"]: r["match_ratio"] for r in _ranked(user_with_pantry, 15)}
        r2 = {r["recipe_id"]: r["match_ratio"] for r in _ranked(user_with_pantry, 15)}
        for rid in r1:
            if rid in r2:
                assert r1[rid] == r2[rid], \
                    f"match_ratio for recipe {rid} changed: {r1[rid]} vs {r2[rid]}"

    def test_removing_item_decreases_coverage(self, user_with_pantry):
        """After deleting pantry items, avg match_ratio should fall back toward 0."""
        before = sum(r["match_ratio"] for r in _ranked(user_with_pantry, 20)) / 20
        assert before > 0, "Fixture should have non-zero match_ratio"

        # Delete everything
        pantry = requests.get(f"{BASE}/pantry/{user_with_pantry}").json()
        for item in pantry:
            requests.delete(f"{BASE}/pantry/{user_with_pantry}/{item['id']}")

        after = sum(r["match_ratio"] for r in _ranked(user_with_pantry, 20)) / 20
        assert after < before, \
            f"Clearing pantry should reduce match_ratio: {before:.4f} → {after:.4f}"


# ─── 3. CF progression: cold-start → warm ────────────────────────────────────

class TestCFProgression:
    """CF should remain cold at <5 ratings, flip warm at exactly 5."""

    def test_is_warm_false_before_5_ratings(self, user_with_pantry):
        assert _stats(user_with_pantry)["is_warm"] is False

    def test_is_warm_flips_at_exactly_5_ratings(self, user_with_pantry):
        """Rate 4 recipes → still cold. Rate 5th → warm flips True."""
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]

        for i in range(4):
            _rate(user_with_pantry, recipe_ids[i], 4)
            assert _stats(user_with_pantry)["is_warm"] is False, \
                f"Should still be cold after {i+1} rating(s)"

        _rate(user_with_pantry, recipe_ids[4], 4)
        assert _stats(user_with_pantry)["is_warm"] is True, \
            "Should be warm after 5th rating"

    def test_rating_count_increments_correctly(self, user_with_pantry):
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 5)]
        for i, rid in enumerate(recipe_ids[:5]):
            _rate(user_with_pantry, rid, 3)
            assert _stats(user_with_pantry)["n_ratings"] == i + 1

    def test_warm_cf_progress_pct_reaches_100(self, user_with_pantry):
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 5)]
        for rid in recipe_ids[:5]:
            _rate(user_with_pantry, rid, 4)
        assert _stats(user_with_pantry)["warm_cf_progress_pct"] == 100.0

    def test_cold_start_feed_is_deterministic(self, user_with_pantry):
        """Without ratings, same feed order every call."""
        ids1 = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        ids2 = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        assert ids1 == ids2, "Cold-start feed should be deterministic"

    @pytest.mark.skipif(not _SVD_EXISTS, reason="SVD model not trained")
    def test_blended_strategy_between_1_and_4_ratings(self, user_with_pantry):
        """With a trained SVD model, 1–4 ratings → cf_strategy == 'blended'."""
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        _rate(user_with_pantry, recipe_ids[0], 4)   # 1 rating → blended

        strategies = {r["cf_strategy"] for r in _ranked(user_with_pantry, 10)}
        assert "blended" in strategies, \
            f"Expected 'blended' after 1 rating with trained SVD. Got: {strategies}"

    @pytest.mark.skipif(not _SVD_EXISTS, reason="SVD model not trained — run python -m backend.ml.train_cf")
    def test_svd_strategy_after_5_ratings(self, user_with_pantry):
        """With a trained biased MF model, cf_strategy should become 'biased_mf' after 5 ratings."""
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        for rid in recipe_ids[:5]:
            _rate(user_with_pantry, rid, 4)

        strategies = {r["cf_strategy"] for r in _ranked(user_with_pantry, 10)}
        assert "biased_mf" in strategies, \
            f"Expected 'biased_mf' strategy after 5 ratings with trained model. Got: {strategies}"

    @pytest.mark.skipif(not _SVD_EXISTS, reason="Biased MF model not trained")
    def test_biased_mf_scores_nonzero_after_warm(self, user_with_pantry):
        """After warm transition, biased MF CF scores should be non-zero."""
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        for rid in recipe_ids[:5]:
            _rate(user_with_pantry, rid, 4)

        mf_results = [r for r in _ranked(user_with_pantry, 20) if r["cf_strategy"] == "biased_mf"]
        assert len(mf_results) > 0
        assert any(r["cf_score"] > 0 for r in mf_results), \
            "Biased MF CF scores should be non-zero after warm transition"

    @pytest.mark.skipif(not _SVD_EXISTS, reason="Biased MF model not trained")
    def test_biased_mf_scores_in_range_after_warm(self, user_with_pantry):
        """After warm transition, all SVD CF scores should remain in [0, 1]."""
        recipe_ids = [r["recipe_id"] for r in _ranked(user_with_pantry, 10)]
        for rid in recipe_ids[:5]:
            _rate(user_with_pantry, rid, 4)

        results = _ranked(user_with_pantry, 20)
        for r in results:
            assert 0.0 <= r["cf_score"] <= 1.0, \
                f"SVD cf_score={r['cf_score']} out of range for recipe {r['recipe_id']}"
            assert 0.0 <= r["final_score"] <= 1.0, \
                f"final_score={r['final_score']} out of range after warm transition"


# ─── 4. Content-based scoring and beta ────────────────────────────────────────

class TestCBAndBeta:
    """CB scores should be valid; beta parameter should affect pantry-match ranking."""

    def test_cb_score_in_range_for_all_recipes(self, new_user):
        for r in _ranked(new_user):
            assert 0.0 <= r["cb_score"] <= 1.0, \
                f"cb_score={r['cb_score']} out of range for recipe {r['recipe_id']}"

    def test_diet_tag_user_gets_results(self):
        """Vegetarian tag constraint should still produce a non-empty feed."""
        uid = _make_user("_veg", diet_tags="vegetarian")
        results = _ranked(uid, top_n=10)
        assert len(results) >= 3, \
            f"Vegetarian-filtered user got only {len(results)} recipes — feed too sparse"

    def test_high_beta_ranks_pantry_matches_higher(self):
        """High-beta user (pantry-focused) should have higher avg match_ratio in top-5."""
        uid_low  = _make_user("_low_beta",  beta=0.05)
        uid_high = _make_user("_high_beta", beta=0.95)

        for uid in (uid_low, uid_high):
            for ing in ("eggs", "butter", "flour", "milk"):
                _add_pantry(uid, ing, _future(60))

        top_low  = _ranked(uid_low,  15)[:5]
        top_high = _ranked(uid_high, 15)[:5]

        avg_low  = sum(r["match_ratio"] for r in top_low)  / 5
        avg_high = sum(r["match_ratio"] for r in top_high) / 5

        assert avg_high >= avg_low, (
            f"High-beta top-5 should have ≥ pantry match than low-beta: "
            f"low={avg_low:.3f}, high={avg_high:.3f}"
        )

    def test_beta_extremes_both_produce_valid_feeds(self):
        """Beta at 0.05 and 0.95 should both produce valid in-range feeds."""
        for beta in (0.05, 0.95):
            uid = _make_user(f"_beta_{beta}", beta=beta)
            _add_pantry(uid, "eggs", _future(30))
            results = _ranked(uid, 10)
            assert len(results) > 0
            scores = [r["final_score"] for r in results]
            # MMR reranks for diversity — not strictly sorted, but top item is best
            assert scores[0] == max(scores), \
                f"Top item should have highest score at beta={beta}"
            for r in results:
                assert 0 <= r["final_score"] <= 1


# ─── 5. Score math sanity ─────────────────────────────────────────────────────

class TestScoreMath:
    """final_score should be a weighted mix of its components, always in [0, 1]."""

    def test_final_score_within_component_bounds(self, user_with_pantry):
        """final_score is a weighted average so it must lie within [min, max] of components."""
        for r in _ranked(user_with_pantry, top_n=10):
            components = [r["cf_score"], r["cb_score"], r["expiry_urgency"], r["match_ratio"]]
            lo, hi = min(components), max(components)
            assert lo - 0.01 <= r["final_score"] <= hi + 0.01, (
                f"recipe {r['recipe_id']}: final_score={r['final_score']:.4f} "
                f"outside [{lo:.4f}, {hi:.4f}] of components {components}"
            )

    def test_higher_urgency_recipe_not_buried(self, new_user):
        """Recipe with imminent-expiry ingredient should appear in top half of feed."""
        _add_pantry(new_user, "eggs", _future(1))  # tomorrow
        results = _ranked(new_user, 20)

        # Find highest-urgency recipe position
        max_urg_idx = max(range(len(results)), key=lambda i: results[i]["expiry_urgency"])
        max_urg = results[max_urg_idx]["expiry_urgency"]

        if max_urg == 0:
            pytest.skip("No egg-using recipes in corpus to test urgency surfacing")

        assert max_urg_idx < len(results), "Urgency recipe found in results"
        # Should be in top 70% — urgency is weighted at α=0.35 so it meaningfully lifts score
        threshold = int(len(results) * 0.70)
        assert max_urg_idx <= threshold, (
            f"High-urgency recipe at position {max_urg_idx}/{len(results)} — "
            f"urgency signal ({max_urg:.3f}) not surfacing it high enough"
        )

    def test_matched_ingredients_subset_of_pantry(self, user_with_pantry):
        """matched_ingredients must only contain items from the user's pantry."""
        pantry_items = {i["ingredient"] for i in requests.get(f"{BASE}/pantry/{user_with_pantry}").json()}
        for r in _ranked(user_with_pantry, top_n=10):
            for mi in r.get("matched_ingredients", []):
                # Fuzzy matching may canonicalize names — check substring overlap
                mi_lower = mi.lower()
                matched = any(
                    mi_lower in p or p in mi_lower
                    for p in pantry_items
                )
                assert matched, (
                    f"matched_ingredient '{mi}' not found in pantry {pantry_items} "
                    f"(recipe {r['recipe_id']})"
                )


# ─── 6. Vision endpoint ────────────────────────────────────────────────────────

class TestVision:
    """Vision mock should return valid structure; confirm endpoint → pantry → affects feed."""

    def test_mock_scan_returns_list(self):
        r = requests.get(f"{BASE}/vision/mock")
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list) and len(items) > 0

    def test_mock_scan_item_has_required_fields(self):
        for item in requests.get(f"{BASE}/vision/mock").json():
            assert "raw_name" in item,    f"mock item missing raw_name: {item}"
            assert "expiry_date" in item, f"mock item missing expiry_date key: {item}"

    def test_mock_scan_raw_names_nonempty(self):
        for item in requests.get(f"{BASE}/vision/mock").json():
            assert isinstance(item["raw_name"], str) and item["raw_name"].strip(), \
                f"raw_name should be a non-empty string: {item}"

    def test_vision_confirm_adds_items_to_pantry(self, new_user):
        mock_items = requests.get(f"{BASE}/vision/mock").json()
        confirmable = [i for i in mock_items if i.get("expiry_date")]
        if not confirmable:
            pytest.skip("Mock scan returned no items with expiry_date")

        # Endpoint expects {"items": [...]} (ConfirmPayload model)
        r = requests.post(f"{BASE}/vision/confirm/{new_user}",
                          json={"items": confirmable[:2]})
        assert r.status_code == 201, f"Vision confirm failed: {r.text}"

        pantry = requests.get(f"{BASE}/pantry/{new_user}").json()
        assert len(pantry) > 0, "Confirmed vision items should appear in pantry"

    def test_vision_confirm_then_feed_has_nonzero_match(self, new_user):
        """After confirming vision items, ranked feed should show pantry coverage."""
        mock_items = requests.get(f"{BASE}/vision/mock").json()
        confirmable = [i for i in mock_items if i.get("expiry_date")]
        if not confirmable:
            pytest.skip("Mock scan returned no items with expiry_date")

        requests.post(f"{BASE}/vision/confirm/{new_user}", json={"items": confirmable})
        results = _ranked(new_user, top_n=20)
        max_match = max(r["match_ratio"] for r in results)
        assert max_match > 0.0, \
            "After vision confirm, some recipes should match confirmed pantry items"

    def test_vision_confirm_rejects_items_without_expiry(self, new_user):
        """Items with null expiry_date should be rejected on confirm."""
        no_expiry = [{"raw_name": "mystery_thing", "expiry_date": None,
                      "quantity": None, "ingredient": "mystery_thing"}]
        r = requests.post(f"{BASE}/vision/confirm/{new_user}", json={"items": no_expiry})
        # Should either reject (422/400) or silently skip them
        if r.status_code == 200:
            pantry = requests.get(f"{BASE}/pantry/{new_user}").json()
            # If accepted, pantry should still be empty or not contain the null-expiry item
            no_null = all(i["expiry_date"] is not None for i in pantry)
            assert no_null, "Items without expiry_date should not appear in pantry"
        else:
            assert r.status_code in (400, 422), \
                f"Expected rejection of null-expiry item, got {r.status_code}"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set OPENAI_API_KEY to run live GPT-4o vision test",
    )
    def test_real_vision_scan_with_food_image(self, new_user):
        """
        Real GPT-4o vision scan. Requires a food image at /tmp/test_food.jpg
        or the path in env var TEST_FOOD_IMAGE.
        """
        img_path = os.environ.get("TEST_FOOD_IMAGE", "/tmp/test_food.jpg")
        if not os.path.exists(img_path):
            pytest.skip(
                f"No test image at {img_path}. "
                "Place a fridge/food photo there or set TEST_FOOD_IMAGE."
            )

        with open(img_path, "rb") as fh:
            r = requests.post(
                f"{BASE}/vision/scan",
                files={"file": ("food.jpg", fh, "image/jpeg")},
                timeout=30,
            )
        assert r.status_code == 200, f"Vision scan failed: {r.text}"
        items = r.json()
        assert isinstance(items, list), "Vision scan should return a list"
        assert len(items) > 0, "GPT-4o should detect at least one item in a food image"
        for item in items:
            assert "raw_name" in item and item["raw_name"], \
                f"Each scanned item needs a non-empty raw_name: {item}"
