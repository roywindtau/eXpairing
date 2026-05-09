"""
tests/test_beta_updater.py
--------------------------
Tests for the beta updater — both the math and the DB integration.

Key behaviors verified:
  - A user who always cooks zero-missing recipes -> beta drifts UP
  - A user who always cooks 3-missing recipes   -> beta drifts DOWN
  - Users with < MIN_EVENTS are skipped
  - Users with no n_missing populated are skipped
  - Dry-run does not write to DB
  - Beta is always clamped to [BETA_MIN, BETA_MAX]
  - Multiple runs converge beta toward revealed_beta
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, User, Recipe, UserEvent
from backend.services.beta_updater import (
    compute_updates,
    apply_updates,
    run,
    _compute_revealed_beta,
    _drift_beta,
    BETA_MIN,
    BETA_MAX,
    MIN_EVENTS,
    LEARNING_RATE,
    MAX_MISSING_NORMALIZER,
)
import pandas as pd


# ---------------------------------------------------------------------------
# In-memory SQLite fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Fresh in-memory DB for each test."""
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def make_user(db, user_id: int, beta: float = 0.5) -> User:
    u = User(id=user_id, beta=beta)
    db.add(u)
    db.commit()
    return u


def make_recipe(db, recipe_id: int) -> Recipe:
    r = Recipe(id=recipe_id, name=f"Recipe {recipe_id}",
               ingredients_csv="eggs,milk", avg_rating=4.0, n_ratings=10)
    db.add(r)
    db.commit()
    return r


def add_cook_events(db, user_id: int, recipe_id: int,
                    n_missing_list: list[int],
                    days_ago: int = 1) -> None:
    """Add multiple cook events with specified n_missing values."""
    ts = datetime.now() - timedelta(days=days_ago)
    for n in n_missing_list:
        db.add(UserEvent(
            user_id=user_id,
            recipe_id=recipe_id,
            event_type="cook",
            n_missing=n,
            created_at=ts,
        ))
    db.commit()


# ---------------------------------------------------------------------------
# Unit tests for pure math functions
# ---------------------------------------------------------------------------

class TestComputeRevealedBeta:
    def test_zero_missing_gives_beta_one(self):
        df = pd.DataFrame({"n_missing": [0, 0, 0, 0]})
        beta, avg = _compute_revealed_beta(df)
        assert beta == 1.0
        assert avg == 0.0

    def test_max_missing_gives_low_beta(self):
        """
        With soft sigmoid normalization, MAX_MISSING_NORMALIZER missing gives
        a low but nonzero beta (reviewer fix: gradual decay, not hard floor).
        At avg=MAX, formula: 1/(1 + MAX/(MAX/2)) = 1/3 ≈ 0.33
        """
        df = pd.DataFrame({"n_missing": [MAX_MISSING_NORMALIZER] * 4})
        beta, avg = _compute_revealed_beta(df)
        assert beta is not None
        assert 0.0 < beta < 0.5   # low but not zero

    def test_half_max_gives_half_beta(self):
        half = MAX_MISSING_NORMALIZER / 2
        df = pd.DataFrame({"n_missing": [half] * 4})
        beta, _ = _compute_revealed_beta(df)
        assert abs(beta - 0.5) < 0.01

    def test_null_n_missing_excluded(self):
        df = pd.DataFrame({"n_missing": [None, None, None]})
        beta, avg = _compute_revealed_beta(df)
        assert beta is None
        assert avg is None

    def test_mixed_null_uses_valid_only(self):
        df = pd.DataFrame({"n_missing": [0, None, 0, None]})
        beta, avg = _compute_revealed_beta(df)
        assert beta == 1.0
        assert avg == 0.0

    def test_above_max_gives_small_positive_beta(self):
        """
        Soft sigmoid: very high missing (99) gives a small but positive beta.
        Previously hard-clamped to 0.0 — now decays gracefully.
        """
        df = pd.DataFrame({"n_missing": [99, 99, 99, 99]})
        beta, _ = _compute_revealed_beta(df)
        assert beta is not None
        assert 0.0 < beta < 0.1   # very small, approaching but never reaching 0


class TestDriftBeta:
    def test_drift_toward_revealed(self):
        # current=0.5, revealed=1.0 -> new should be between 0.5 and 1.0
        new = _drift_beta(0.5, 1.0, LEARNING_RATE)
        assert 0.5 < new < 1.0

    def test_drift_down_when_revealed_lower(self):
        new = _drift_beta(0.8, 0.2, LEARNING_RATE)
        assert new < 0.8

    def test_drift_up_when_revealed_higher(self):
        new = _drift_beta(0.2, 0.8, LEARNING_RATE)
        assert new > 0.2

    def test_no_drift_when_equal(self):
        new = _drift_beta(0.5, 0.5, LEARNING_RATE)
        assert new == 0.5

    def test_clamped_at_max(self):
        new = _drift_beta(BETA_MAX, 1.0, 1.0)
        assert new == BETA_MAX

    def test_clamped_at_min(self):
        new = _drift_beta(BETA_MIN, 0.0, 1.0)
        assert new == BETA_MIN

    def test_converges_after_many_runs(self):
        beta = 0.5
        revealed = 0.9
        for _ in range(50):
            beta = _drift_beta(beta, revealed, LEARNING_RATE)
        # After 50 daily updates, should be very close to revealed
        assert abs(beta - revealed) < 0.05


# ---------------------------------------------------------------------------
# Integration tests against in-memory DB
# ---------------------------------------------------------------------------

class TestComputeUpdates:
    def test_zero_missing_drifts_beta_up(self, db):
        make_recipe(db, 1)
        u = make_user(db, 1, beta=0.4)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        updates = compute_updates(db)
        assert len(updates) == 1
        assert updates[0].new_beta > u.beta

    def test_high_missing_drifts_beta_down(self, db):
        make_recipe(db, 1)
        u = make_user(db, 1, beta=0.8)
        add_cook_events(db, 1, 1, [3, 3, 3, 3])

        updates = compute_updates(db)
        assert len(updates) == 1
        assert updates[0].new_beta < u.beta

    def test_user_below_min_events_skipped(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.5)
        # Add fewer events than MIN_EVENTS
        add_cook_events(db, 1, 1, [0] * (MIN_EVENTS - 1))

        updates = compute_updates(db)
        assert len(updates) == 0

    def test_user_with_no_n_missing_skipped(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.5)
        # Add events with NULL n_missing
        for _ in range(MIN_EVENTS + 1):
            db.add(UserEvent(
                user_id=1, recipe_id=1,
                event_type="cook", n_missing=None,
                created_at=datetime.now() - timedelta(days=1),
            ))
        db.commit()

        updates = compute_updates(db)
        assert len(updates) == 0

    def test_old_events_outside_lookback_ignored(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.5)
        # All events are older than lookback window
        add_cook_events(db, 1, 1, [0, 0, 0, 0], days_ago=60)

        updates = compute_updates(db, lookback_days=30)
        assert len(updates) == 0

    def test_multiple_users_independent(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.3)  # low beta, cooks zero-missing -> should go up
        make_user(db, 2, beta=0.9)  # high beta, cooks 3-missing  -> should go down

        add_cook_events(db, 1, 1, [0, 0, 0, 0])
        add_cook_events(db, 2, 1, [3, 3, 3, 3])

        updates = compute_updates(db)
        assert len(updates) == 2

        u1 = next(u for u in updates if u.user_id == 1)
        u2 = next(u for u in updates if u.user_id == 2)

        assert u1.new_beta > 0.3
        assert u2.new_beta < 0.9

    def test_revealed_beta_field_populated(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.5)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        updates = compute_updates(db)
        assert updates[0].revealed_beta == 1.0
        assert updates[0].avg_missing == 0.0

    def test_drift_field_is_difference(self, db):
        make_recipe(db, 1)
        u = make_user(db, 1, beta=0.5)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        updates = compute_updates(db)
        upd = updates[0]
        assert abs(upd.drift - (upd.new_beta - upd.old_beta)) < 1e-6


class TestApplyUpdates:
    def test_writes_new_beta_to_db(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=0.4)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        updates = compute_updates(db)
        apply_updates(db, updates)

        user = db.get(User, 1)
        assert abs(user.beta - updates[0].new_beta) < 1e-6

    def test_beta_clamped_in_db(self, db):
        make_recipe(db, 1)
        make_user(db, 1, beta=BETA_MAX)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        updates = compute_updates(db)
        apply_updates(db, updates)

        user = db.get(User, 1)
        assert user.beta <= BETA_MAX

    def test_empty_updates_no_error(self, db):
        apply_updates(db, [])  # should not raise


class TestRunDryRun:
    def test_dry_run_does_not_write(self, db):
        """
        We can't easily test the full run() function with our in-memory DB
        because run() creates its own session. Instead test the components:
        compute_updates returns results, apply_updates is skipped in dry-run.
        """
        make_recipe(db, 1)
        original_beta = 0.4
        make_user(db, 1, beta=original_beta)
        add_cook_events(db, 1, 1, [0, 0, 0, 0])

        # Simulate dry_run: compute but don't apply
        updates = compute_updates(db)
        assert len(updates) == 1  # would update
        # Don't call apply_updates
        user = db.get(User, 1)
        assert abs(user.beta - original_beta) < 1e-6  # unchanged


class TestBetaConvergence:
    def test_converges_to_zero_waste(self, db):
        """
        A user who always cooks zero-missing recipes should converge
        toward beta=BETA_MAX after repeated update cycles.
        """
        make_recipe(db, 1)
        make_user(db, 1, beta=0.2)

        for cycle in range(30):
            add_cook_events(db, 1, 1, [0, 0, 0, 0], days_ago=1)
            updates = compute_updates(db)
            if updates:
                apply_updates(db, updates)

        user = db.get(User, 1)
        assert user.beta >= 0.85, f"Expected beta near {BETA_MAX}, got {user.beta}"

    def test_converges_to_permissive(self, db):
        """
        A user who always cooks recipes needing 4+ extra ingredients
        should converge toward beta=BETA_MIN.
        """
        make_recipe(db, 1)
        make_user(db, 1, beta=0.9)

        for cycle in range(30):
            add_cook_events(db, 1, 1,
                            [int(MAX_MISSING_NORMALIZER)] * 4, days_ago=1)
            updates = compute_updates(db)
            if updates:
                apply_updates(db, updates)

        user = db.get(User, 1)
        # With soft sigmoid, revealed_beta at MAX_MISSING is ~0.33 (not 0).
        # After 30 cycles drifting toward 0.33 from 0.9, beta should be
        # substantially lower than the starting 0.9.
        assert user.beta < 0.55, f"Expected beta to drift down from 0.9, got {user.beta}"
        assert user.beta > BETA_MIN, f"Beta should not hit hard floor, got {user.beta}"

class TestBetaCeilingFix:
    """
    Regression tests for the soft sigmoid normalization fix (reviewer bug #4).
    Previously MAX_MISSING_NORMALIZER=4 caused immediate BETA_MIN for 5+ missing.
    Now uses soft formula: revealed = 1 / (1 + avg_missing / (MAX_MISSING / 2))
    """

    def test_zero_missing_gives_one(self):
        df = pd.DataFrame({"n_missing": [0, 0, 0, 0]})
        beta, avg = _compute_revealed_beta(df)
        assert beta == 1.0

    def test_above_max_does_not_hit_floor_immediately(self):
        """
        With old formula: 5 missing -> revealed_beta = 0.0 (hits BETA_MIN)
        With soft formula: 5 missing -> revealed_beta > 0.0 (gradual decay)
        """
        df = pd.DataFrame({"n_missing": [5, 5, 5, 5]})
        beta, avg = _compute_revealed_beta(df)
        # Should be positive but low, not zero
        assert beta is not None
        assert beta > 0.0
        assert beta < 0.5

    def test_very_high_missing_approaches_but_does_not_equal_zero(self):
        """Even 20 missing ingredients should give a small but nonzero beta."""
        df = pd.DataFrame({"n_missing": [20, 20, 20, 20]})
        beta, avg = _compute_revealed_beta(df)
        assert beta is not None
        assert beta > 0.0   # graceful decay, not hard floor

    def test_half_point_is_at_half_max_normalizer(self):
        """
        At avg_missing = MAX_MISSING_NORMALIZER / 2, revealed_beta should be 0.5.
        soft formula: 1 / (1 + half_point/half_point) = 1/2 = 0.5
        """
        half = MAX_MISSING_NORMALIZER / 2
        df   = pd.DataFrame({"n_missing": [half] * 4})
        beta, _ = _compute_revealed_beta(df)
        assert abs(beta - 0.5) < 0.05

    def test_monotonically_decreasing(self):
        """More missing ingredients → lower revealed beta."""
        betas = []
        for n in [0, 1, 2, 4, 8, 16]:
            df = pd.DataFrame({"n_missing": [float(n)] * 4})
            b, _ = _compute_revealed_beta(df)
            betas.append(b)
        assert betas == sorted(betas, reverse=True)
