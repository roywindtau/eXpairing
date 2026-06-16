"""
test_scoring.py
---------------------
Tests for backend/services/scoring.py.

We use SimpleNamespace wines (no DB) — scoring only reads .id,
.name, .kind, .avg_rating, .n_ratings.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.services.wine.scoring import (
    WineScore,
    WEIGHTS_PATH_A,
    WEIGHTS_PATH_B,
    _calibrate,
    _popularity_prior,
    rank_wines_for_recipe,
    rank_wines_for_user,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _wine(id_, name, avg=4.0, n=100):
    return SimpleNamespace(id=id_, name=name, avg_rating=avg, n_ratings=n)


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
    prior = _popularity_prior(_wine(1, "X", avg=4.0, n=99))
    assert prior == pytest.approx(4.0 * math.log1p(99))


def test_popularity_prior_zero_when_no_ratings():
    assert _popularity_prior(_wine(1, "X", avg=4.0, n=0)) == 0.0
    assert _popularity_prior(_wine(1, "X", avg=0.0, n=100)) == 0.0


def test_popularity_prior_handles_missing_attrs():
    d = SimpleNamespace(id=1, name="X", avg_rating=None, n_ratings=None)
    assert _popularity_prior(d) == 0.0


def test_popularity_prior_higher_n_higher_score_same_avg():
    a = _popularity_prior(_wine(1, "A", avg=4.0, n=100))
    b = _popularity_prior(_wine(2, "B", avg=4.0, n=1000))
    assert b > a


# ── Path A end-to-end ────────────────────────────────────────────────────

def test_rank_wines_for_recipe_empty_candidates():
    assert rank_wines_for_recipe(
        recipe=None, candidates=[],
        cb_scores={}, cf_scores={}, expert_boosts={},
    ) == []


def test_rank_wines_for_recipe_basic_ordering():
    """Wine with highest CB should win when only CB differs."""
    wines = [_wine(1, "A"), _wine(2, "B"), _wine(3, "C")]
    cb = {1: 0.1, 2: 0.9, 3: 0.5}
    cf = {1: 0.5, 2: 0.5, 3: 0.5}
    expert = {1: 0.0, 2: 0.0, 3: 0.0}
    ranked = rank_wines_for_recipe(None, wines, cb, cf, expert)
    assert [s.wine_id for s in ranked] == [2, 3, 1]


def test_rank_wines_for_recipe_expert_boost_breaks_tie():
    """When CB+CF are equal, expert boost should determine order."""
    wines = [_wine(1, "A"), _wine(2, "B")]
    cb = {1: 0.5, 2: 0.5}
    cf = {1: 0.5, 2: 0.5}
    expert = {1: 0.0, 2: 0.20}
    ranked = rank_wines_for_recipe(None, wines, cb, cf, expert)
    assert ranked[0].wine_id == 2
    assert ranked[0].expert_boost == 0.20


def test_rank_wines_for_recipe_calibration_per_pool():
    """All-equal scores → calibration returns 0.5 → all final_scores equal."""
    wines = [_wine(1, "A"), _wine(2, "B"), _wine(3, "C")]
    cb = {1: 0.5, 2: 0.5, 3: 0.5}
    cf = {1: 0.7, 2: 0.7, 3: 0.7}
    expert = {1: 0.0, 2: 0.0, 3: 0.0}
    ranked = rank_wines_for_recipe(None, wines, cb, cf, expert)
    assert len({round(s.final_score, 6) for s in ranked}) == 1


def test_rank_wines_for_recipe_top_n_truncation():
    wines = [_wine(i, f"D{i}") for i in range(10)]
    cb = {i: i / 10 for i in range(10)}
    ranked = rank_wines_for_recipe(None, wines, cb, {}, {}, top_n=3)
    assert len(ranked) == 3
    assert ranked[0].wine_id == 9  # highest CB


def test_rank_wines_for_recipe_top_n_zero_returns_all():
    wines = [_wine(i, f"D{i}") for i in range(5)]
    ranked = rank_wines_for_recipe(None, wines, {}, {}, {}, top_n=0)
    assert len(ranked) == 5


def test_rank_wines_for_recipe_passes_cf_strategy_through():
    wines = [_wine(1, "A")]
    ranked = rank_wines_for_recipe(
        None, wines, {}, {}, {},
        cf_strategies={1: "biased_mf"},
    )
    assert ranked[0].cf_strategy == "biased_mf"


def test_rank_wines_for_recipe_handles_missing_signals():
    """Wines not in any signal dict get 0 for that component, no crash."""
    wines = [_wine(1, "A"), _wine(2, "B")]
    ranked = rank_wines_for_recipe(None, wines, cb_scores={1: 0.7},
                                    cf_scores={}, expert_boosts={})
    assert {s.wine_id for s in ranked} == {1, 2}


def test_rank_wines_for_recipe_golden_case():
    """
    Two-wine hand-computed case to verify the formula end-to-end.
    Raw signals:
        D1: cb=0.0, cf=0.0, expert=0.0, prior=very small
        D2: cb=1.0, cf=1.0, expert=0.2, prior=very big
    After calibration (per-component min-max): D1 all 0.0, D2 all 1.0.
    final_A(D1) = 0.45*0 + 0.25*0 + 0.20*0 + 0.10*0 = 0.0
    final_A(D2) = 0.45*1 + 0.25*1 + 0.20*1 + 0.10*1 = 1.0
    """
    wines = [_wine(1, "Bad", n=1, avg=1.0), _wine(2, "Good", n=10000, avg=5.0)]
    cb = {1: 0.0, 2: 1.0}
    cf = {1: 0.0, 2: 1.0}
    expert = {1: 0.0, 2: 0.20}
    ranked = rank_wines_for_recipe(None, wines, cb, cf, expert)
    assert ranked[0].wine_id == 2
    assert ranked[0].final_score == 1.0
    assert ranked[1].final_score == 0.0


# ── Path B end-to-end ────────────────────────────────────────────────────

def test_rank_wines_for_user_no_expert_in_formula():
    """Verify expert_boost is absent from the output (always 0)."""
    wines = [_wine(1, "A")]
    ranked = rank_wines_for_user(wines, cb_scores={1: 1.0}, cf_scores={1: 1.0})
    assert ranked[0].expert_boost == 0.0


def test_rank_wines_for_user_cb_dominates():
    """CB weight 0.55 > CF weight 0.30 in Path B."""
    wines = [_wine(1, "CB-wins", n=10), _wine(2, "CF-wins", n=10)]
    cb = {1: 1.0, 2: 0.0}
    cf = {1: 0.0, 2: 1.0}
    ranked = rank_wines_for_user(wines, cb, cf)
    assert ranked[0].wine_id == 1


def test_rank_wines_for_user_golden_case():
    """
    final_B = 0.55*cb + 0.30*cf + 0.15*prior
    After calibration (min=0, max=1):
        D1 (worst): 0.0
        D2 (best):  0.55 + 0.30 + 0.15 = 1.0
    """
    wines = [_wine(1, "Bad", n=1, avg=1.0), _wine(2, "Good", n=10000, avg=5.0)]
    cb = {1: 0.0, 2: 1.0}
    cf = {1: 0.0, 2: 1.0}
    ranked = rank_wines_for_user(wines, cb, cf)
    assert ranked[0].final_score == 1.0
    assert ranked[1].final_score == 0.0


def test_rank_wines_for_user_empty_candidates():
    assert rank_wines_for_user([], {}, {}) == []


def test_rank_wines_for_user_top_n_truncation():
    wines = [_wine(i, f"D{i}") for i in range(10)]
    ranked = rank_wines_for_user(wines, {i: i for i in range(10)}, {}, top_n=4)
    assert len(ranked) == 4


def test_rank_wines_for_user_passes_cf_strategy():
    wines = [_wine(1, "A")]
    ranked = rank_wines_for_user(
        wines, {}, {}, cf_strategies={1: "popularity_cold_start"},
    )
    assert ranked[0].cf_strategy == "popularity_cold_start"


# ── WineScore dataclass sanity ─────────────────────────────────────────

def test_wine_score_is_constructible_with_all_fields():
    s = WineScore(
        wine_id=1, wine_name="X",
        final_score=0.5, cb_score=0.3, cf_score=0.4,
        expert_boost=0.1, prior_score=2.0, cf_strategy="wine_item_sim",
    )
    assert s.wine_id == 1
    assert s.matched_harmonize == []


def test_wine_scores_are_sortable_by_final():
    a = WineScore(1, "A", 0.3, 0, 0, 0, 0)
    b = WineScore(2, "B", 0.7, 0, 0, 0, 0)
    out = sorted([a, b], key=lambda s: -s.final_score)
    assert out[0].wine_id == 2


# ── ranking a wine pool ─────────────────────────────────────────────────

def test_rank_wines_pool():
    """Wine candidates are ranked together with one calibration pass."""
    wines = [
        _wine(1, "Cabernet", avg=4.5, n=200),
        _wine(2, "Malbec",   avg=4.2, n=100),
        _wine(3, "Pinot",    avg=4.0, n=50),
    ]
    cb = {1: 0.5, 2: 0.8, 3: 0.4}
    cf = {1: 0.6, 2: 0.5, 3: 0.7}
    expert = {1: 0.0, 2: 0.1, 3: 0.0}
    ranked = rank_wines_for_recipe(None, wines, cb, cf, expert)
    assert len(ranked) == 3
    # Malbec has highest cb + expert match → should be top
    assert ranked[0].wine_id == 2
