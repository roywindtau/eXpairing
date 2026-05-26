"""
test_drink_scoring.py
---------------------
Tests for backend/services/drink_scoring.py.

We use SimpleNamespace drinks (no DB) — drink_scoring only reads .id,
.name, .kind, .avg_rating, .n_ratings.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from backend.services.drinks.scoring import (
    DrinkScore,
    WEIGHTS_PATH_A,
    WEIGHTS_PATH_B,
    _calibrate,
    _popularity_prior,
    rank_drinks_for_recipe,
    rank_drinks_for_user,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _drink(id_, name, kind="beer", avg=4.0, n=100):
    return SimpleNamespace(id=id_, name=name, kind=kind, avg_rating=avg, n_ratings=n)


# ── weights sanity ───────────────────────────────────────────────────────

def test_path_a_weights_sum_to_one():
    assert abs(sum(WEIGHTS_PATH_A.values()) - 1.0) < 1e-9


def test_path_b_weights_sum_to_one():
    assert abs(sum(WEIGHTS_PATH_B.values()) - 1.0) < 1e-9


def test_path_b_has_no_expert_weight():
    assert "expert" not in WEIGHTS_PATH_B


# ── _calibrate ───────────────────────────────────────────────────────────

def test_calibrate_min_max_normalizes():
    assert _calibrate([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]


def test_calibrate_all_equal_returns_half():
    assert _calibrate([0.7, 0.7, 0.7]) == [0.5, 0.5, 0.5]


def test_calibrate_empty():
    assert _calibrate([]) == []


def test_calibrate_single_value_returns_half():
    assert _calibrate([0.5]) == [0.5]


# ── _popularity_prior ───────────────────────────────────────────────────

def test_popularity_prior_uses_log_n():
    prior = _popularity_prior(_drink(1, "X", avg=4.0, n=99))
    assert prior == pytest.approx(4.0 * math.log1p(99))


def test_popularity_prior_zero_when_no_ratings():
    assert _popularity_prior(_drink(1, "X", avg=4.0, n=0)) == 0.0
    assert _popularity_prior(_drink(1, "X", avg=0.0, n=100)) == 0.0


def test_popularity_prior_handles_missing_attrs():
    d = SimpleNamespace(id=1, name="X", kind="beer", avg_rating=None, n_ratings=None)
    assert _popularity_prior(d) == 0.0


def test_popularity_prior_higher_n_higher_score_same_avg():
    a = _popularity_prior(_drink(1, "A", avg=4.0, n=100))
    b = _popularity_prior(_drink(2, "B", avg=4.0, n=1000))
    assert b > a


# ── Path A end-to-end ────────────────────────────────────────────────────

def test_rank_drinks_for_recipe_empty_candidates():
    assert rank_drinks_for_recipe(
        recipe=None, candidates=[],
        cb_scores={}, cf_scores={}, expert_boosts={},
    ) == []


def test_rank_drinks_for_recipe_basic_ordering():
    """Drink with highest CB should win when only CB differs."""
    drinks = [_drink(1, "A"), _drink(2, "B"), _drink(3, "C")]
    cb = {1: 0.1, 2: 0.9, 3: 0.5}
    cf = {1: 0.5, 2: 0.5, 3: 0.5}
    expert = {1: 0.0, 2: 0.0, 3: 0.0}
    ranked = rank_drinks_for_recipe(None, drinks, cb, cf, expert)
    assert [s.drink_id for s in ranked] == [2, 3, 1]


def test_rank_drinks_for_recipe_expert_boost_breaks_tie():
    """When CB+CF are equal, expert boost should determine order."""
    drinks = [_drink(1, "A"), _drink(2, "B")]
    cb = {1: 0.5, 2: 0.5}
    cf = {1: 0.5, 2: 0.5}
    expert = {1: 0.0, 2: 0.20}
    ranked = rank_drinks_for_recipe(None, drinks, cb, cf, expert)
    assert ranked[0].drink_id == 2
    assert ranked[0].expert_boost == 0.20


def test_rank_drinks_for_recipe_calibration_per_pool():
    """All-equal scores → calibration returns 0.5 → all final_scores equal."""
    drinks = [_drink(1, "A"), _drink(2, "B"), _drink(3, "C")]
    cb = {1: 0.5, 2: 0.5, 3: 0.5}
    cf = {1: 0.7, 2: 0.7, 3: 0.7}
    expert = {1: 0.0, 2: 0.0, 3: 0.0}
    ranked = rank_drinks_for_recipe(None, drinks, cb, cf, expert)
    assert len({round(s.final_score, 6) for s in ranked}) == 1


def test_rank_drinks_for_recipe_top_n_truncation():
    drinks = [_drink(i, f"D{i}") for i in range(10)]
    cb = {i: i / 10 for i in range(10)}
    ranked = rank_drinks_for_recipe(None, drinks, cb, {}, {}, top_n=3)
    assert len(ranked) == 3
    assert ranked[0].drink_id == 9  # highest CB


def test_rank_drinks_for_recipe_top_n_zero_returns_all():
    drinks = [_drink(i, f"D{i}") for i in range(5)]
    ranked = rank_drinks_for_recipe(None, drinks, {}, {}, {}, top_n=0)
    assert len(ranked) == 5


def test_rank_drinks_for_recipe_passes_cf_strategy_through():
    drinks = [_drink(1, "A")]
    ranked = rank_drinks_for_recipe(
        None, drinks, {}, {}, {},
        cf_strategies={1: "biased_mf"},
    )
    assert ranked[0].cf_strategy == "biased_mf"


def test_rank_drinks_for_recipe_handles_missing_signals():
    """Drinks not in any signal dict get 0 for that component, no crash."""
    drinks = [_drink(1, "A"), _drink(2, "B")]
    ranked = rank_drinks_for_recipe(None, drinks, cb_scores={1: 0.7},
                                    cf_scores={}, expert_boosts={})
    assert {s.drink_id for s in ranked} == {1, 2}


def test_rank_drinks_for_recipe_golden_case():
    """
    Two-drink hand-computed case to verify the formula end-to-end.
    Raw signals:
        D1: cb=0.0, cf=0.0, expert=0.0, prior=very small
        D2: cb=1.0, cf=1.0, expert=0.2, prior=very big
    After calibration (per-component min-max): D1 all 0.0, D2 all 1.0.
    final_A(D1) = 0.45*0 + 0.25*0 + 0.20*0 + 0.10*0 = 0.0
    final_A(D2) = 0.45*1 + 0.25*1 + 0.20*1 + 0.10*1 = 1.0
    """
    drinks = [_drink(1, "Bad", n=1, avg=1.0), _drink(2, "Good", n=10000, avg=5.0)]
    cb = {1: 0.0, 2: 1.0}
    cf = {1: 0.0, 2: 1.0}
    expert = {1: 0.0, 2: 0.20}
    ranked = rank_drinks_for_recipe(None, drinks, cb, cf, expert)
    assert ranked[0].drink_id == 2
    assert ranked[0].final_score == 1.0
    assert ranked[1].final_score == 0.0


# ── Path B end-to-end ────────────────────────────────────────────────────

def test_rank_drinks_for_user_no_expert_in_formula():
    """Verify expert_boost is absent from the output (always 0)."""
    drinks = [_drink(1, "A")]
    ranked = rank_drinks_for_user(drinks, cb_scores={1: 1.0}, cf_scores={1: 1.0})
    assert ranked[0].expert_boost == 0.0


def test_rank_drinks_for_user_cb_dominates():
    """CB weight 0.55 > CF weight 0.30 in Path B."""
    drinks = [_drink(1, "CB-wins", n=10), _drink(2, "CF-wins", n=10)]
    cb = {1: 1.0, 2: 0.0}
    cf = {1: 0.0, 2: 1.0}
    ranked = rank_drinks_for_user(drinks, cb, cf)
    assert ranked[0].drink_id == 1


def test_rank_drinks_for_user_golden_case():
    """
    final_B = 0.55*cb + 0.30*cf + 0.15*prior
    After calibration (min=0, max=1):
        D1 (worst): 0.0
        D2 (best):  0.55 + 0.30 + 0.15 = 1.0
    """
    drinks = [_drink(1, "Bad", n=1, avg=1.0), _drink(2, "Good", n=10000, avg=5.0)]
    cb = {1: 0.0, 2: 1.0}
    cf = {1: 0.0, 2: 1.0}
    ranked = rank_drinks_for_user(drinks, cb, cf)
    assert ranked[0].final_score == 1.0
    assert ranked[1].final_score == 0.0


def test_rank_drinks_for_user_empty_candidates():
    assert rank_drinks_for_user([], {}, {}) == []


def test_rank_drinks_for_user_top_n_truncation():
    drinks = [_drink(i, f"D{i}") for i in range(10)]
    ranked = rank_drinks_for_user(drinks, {i: i for i in range(10)}, {}, top_n=4)
    assert len(ranked) == 4


def test_rank_drinks_for_user_passes_cf_strategy():
    drinks = [_drink(1, "A")]
    ranked = rank_drinks_for_user(
        drinks, {}, {}, cf_strategies={1: "popularity_cold_start"},
    )
    assert ranked[0].cf_strategy == "popularity_cold_start"


# ── DrinkScore dataclass sanity ─────────────────────────────────────────

def test_drink_score_is_constructible_with_all_fields():
    s = DrinkScore(
        drink_id=1, drink_name="X", kind="beer",
        final_score=0.5, cb_score=0.3, cf_score=0.4,
        expert_boost=0.1, prior_score=2.0, cf_strategy="biased_mf",
    )
    assert s.drink_id == 1
    assert s.matched_harmonize == []


def test_drink_scores_are_sortable_by_final():
    a = DrinkScore(1, "A", "beer", 0.3, 0, 0, 0, 0)
    b = DrinkScore(2, "B", "wine", 0.7, 0, 0, 0, 0)
    out = sorted([a, b], key=lambda s: -s.final_score)
    assert out[0].drink_id == 2


# ── mixed beer + wine in same pool ──────────────────────────────────────

def test_rank_drinks_mixed_kinds():
    """Beer and wine candidates can be ranked together with one calibration pass."""
    drinks = [
        _drink(1, "IPA",  kind="beer", avg=4.5, n=200),
        _drink(2, "Malbec", kind="wine", avg=4.2, n=100),
        _drink(3, "Stout", kind="beer", avg=4.0, n=50),
    ]
    cb = {1: 0.5, 2: 0.8, 3: 0.4}
    cf = {1: 0.6, 2: 0.5, 3: 0.7}
    expert = {1: 0.0, 2: 0.1, 3: 0.0}
    ranked = rank_drinks_for_recipe(None, drinks, cb, cf, expert)
    assert len(ranked) == 3
    assert {s.kind for s in ranked} == {"beer", "wine"}
    # Malbec has highest cb + expert match → should be top
    assert ranked[0].drink_id == 2
