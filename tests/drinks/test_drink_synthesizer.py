"""
test_drink_synthesizer.py
-------------------------
Tests for backend/services/drink_synthesizer.py.

Strategy: in-memory SQLite DB with a small set of drinks + a beef recipe.
We monkey-patch serve_drink_cb.cb_for_recipe to return canned scores so
we don't need a trained CB model — that lets us assert exactly which
drinks get synthesized for each fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, Wine, DrinkEvent, Recipe, User
from backend.services.drinks import synthesizer as drink_synthesizer


# ── fixture data ─────────────────────────────────────────────────────────

def _seed(db):
    db.add_all([
        # Wines that should match a beef recipe via Harmonize
        Wine(id=1, name="Beef Wine",   style="Red",
             grapes_csv="Malbec", harmonize_csv="Beef,Lamb,Grilled",
             avg_rating=4.2, n_ratings=20),
        Wine(id=2, name="Fish Wine",   style="White",
             grapes_csv="Sauv Blanc", harmonize_csv="Fish,Seafood",
             avg_rating=3.9, n_ratings=15),
        Wine(id=3, name="Random Wine", style="Sparkling",
             grapes_csv="Chardonnay", harmonize_csv="Appetizer",
             avg_rating=4.0, n_ratings=10),
        # Recipes
        Recipe(id=1001, name="Beef Stew", ingredients_csv="beef,onion,garlic,potato",
               tags_csv="american,beef"),
        Recipe(id=1002, name="Spicy Curry", ingredients_csv="chicken,curry,chili,ginger",
               tags_csv="indian,spicy"),
        Recipe(id=1003, name="Chocolate Cake", ingredients_csv="chocolate,butter,eggs,sugar",
               tags_csv="dessert"),
        User(id=42, beta=0.5),
        User(id=43, beta=0.5),
    ])
    db.commit()


@pytest.fixture
def db_session(monkeypatch):
    """Fresh in-memory DB per test + canned CB scores so we don't need artifacts."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    _seed(db)

    # Stub out the CB module so we don't need trained artifacts.
    # Each test can re-patch with its own canned scores if needed.
    from backend.ml.drinks.serving import serve_cb as serve_drink_cb

    def fake_model_available():
        return True

    def fake_cb_for_recipe(recipe, kind_filter=None):
        # Default: zero CB signal across the board. Tests can override.
        all_ids = {1: 0.0, 2: 0.0, 3: 0.0}
        if kind_filter == "wine":
            return {d: s for d, s in all_ids.items() if d < 100}
        return all_ids

    monkeypatch.setattr(serve_drink_cb, "model_available", fake_model_available)
    monkeypatch.setattr(serve_drink_cb, "cb_for_recipe",   fake_cb_for_recipe)

    # Re-enable kill switch every test (sometimes flipped off in previous test).
    monkeypatch.setattr(drink_synthesizer, "ENABLE_SYNTHETIC_DRINK_RATINGS", True)

    yield db
    db.close()


# ── threshold / kill switch ──────────────────────────────────────────────

def test_low_rating_writes_nothing(db_session):
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=3.5, db=db_session
    )
    assert n == 0
    assert db_session.query(DrinkEvent).count() == 0


def test_high_rating_writes_events(db_session):
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=4.5, db=db_session
    )
    assert n > 0
    events = db_session.query(DrinkEvent).all()
    assert len(events) == n
    assert all(e.synthetic is True for e in events)
    assert all(e.rating == drink_synthesizer.SYNTHETIC_RATING for e in events)
    assert all(e.event_type == "rate" for e in events)


def test_exactly_threshold_fires(db_session):
    """rating == 4.0 should trigger (boundary inclusive)."""
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=4.0, db=db_session
    )
    assert n > 0


def test_kill_switch_disables_synthesis(db_session, monkeypatch):
    monkeypatch.setattr(drink_synthesizer, "ENABLE_SYNTHETIC_DRINK_RATINGS", False)
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    assert n == 0
    assert db_session.query(DrinkEvent).count() == 0


def test_unknown_recipe_no_crash(db_session):
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=999999, rating=5.0, db=db_session
    )
    assert n == 0


# ── correctness of picks ─────────────────────────────────────────────────

def test_beef_recipe_picks_beef_wine(db_session):
    """Beef recipe should produce a synthetic event for Wine id=1 (Beef harmonize)."""
    drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    wine_event_ids = {
        e.drink_id for e in db_session.query(DrinkEvent).all() if e.drink_id < 100
    }
    assert 1 in wine_event_ids   # Beef Wine got picked
    assert 2 not in wine_event_ids  # Fish Wine did NOT get picked (only 3 wines, but Fish should rank lowest for beef)


def test_caps_at_n_per_kind(db_session):
    """No more than N_SYNTHETIC_PER_KIND events per kind."""
    drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    n_wine = db_session.query(DrinkEvent).filter(DrinkEvent.drink_id < 100).count()
    assert n_wine <= drink_synthesizer.N_SYNTHETIC_PER_KIND


# ── dedup / explicit-wins ────────────────────────────────────────────────

def test_does_not_overwrite_explicit_rating(db_session):
    """User already explicitly rated Wine 1 → synthesizer must skip it."""
    db_session.add(DrinkEvent(user_id=42, drink_id=1, event_type="rate",
                               rating=2.0, synthetic=False))
    db_session.commit()

    drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )

    # The original explicit row is still 2.0; no new row for (42, 1) created.
    wine_1_events = (
        db_session.query(DrinkEvent)
        .filter(DrinkEvent.user_id == 42, DrinkEvent.drink_id == 1)
        .all()
    )
    assert len(wine_1_events) == 1
    assert wine_1_events[0].synthetic is False
    assert wine_1_events[0].rating == 2.0


def test_does_not_double_synthesize_same_pair(db_session):
    """Running synthesizer twice on the same recipe should not duplicate rows."""
    drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    n_first = db_session.query(DrinkEvent).count()

    drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    n_second = db_session.query(DrinkEvent).count()

    assert n_first == n_second  # nothing new added


def test_different_users_get_independent_history(db_session):
    drink_synthesizer.maybe_synthesize_on_recipe_rating(42, 1001, 5.0, db_session)
    drink_synthesizer.maybe_synthesize_on_recipe_rating(43, 1001, 5.0, db_session)

    n_42 = db_session.query(DrinkEvent).filter(DrinkEvent.user_id == 42).count()
    n_43 = db_session.query(DrinkEvent).filter(DrinkEvent.user_id == 43).count()
    assert n_42 > 0 and n_43 > 0
    assert n_42 == n_43


# ── fail-soft ───────────────────────────────────────────────────────────

def test_synthesizer_swallows_exceptions(db_session, monkeypatch):
    """A bug in CB or expert_pairing must not break the caller."""
    from backend.ml.drinks.serving import serve_cb as serve_drink_cb

    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(serve_drink_cb, "cb_for_recipe", crash)

    # Should not raise — must return 0 even though CB crashed
    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    assert n == 0


# ── CB fallback to popularity ───────────────────────────────────────────

def test_falls_back_to_popularity_when_cb_unavailable(db_session, monkeypatch):
    """No CB model → still synthesizes top-popularity candidates."""
    from backend.ml.drinks.serving import serve_cb as serve_drink_cb

    monkeypatch.setattr(serve_drink_cb, "model_available", lambda: False)

    n = drink_synthesizer.maybe_synthesize_on_recipe_rating(
        user_id=42, recipe_id=1001, rating=5.0, db=db_session
    )
    # Beef recipe + popularity-only candidates: Beef Wine should still
    # win the wine pick via expert_boost rules.
    assert n > 0


# ── recipe-router hook ──────────────────────────────────────────────────

def test_log_event_triggers_synthesizer(db_session):
    """Calling the router's log_event with a high rating should fire the synthesizer."""
    from backend.routers.recipes import log_event, EventIn

    payload = EventIn(
        user_id=42, recipe_id=1001, event_type="rate",
        rating=5.0, n_missing=None,
    )
    result = log_event(payload, db_session)
    assert result["status"] == "ok"
    # Synthesizer should have fired and written some drink events
    assert db_session.query(DrinkEvent).count() > 0


def test_log_event_no_synthesis_for_cook_event(db_session):
    """Only 'rate' events trigger synthesis, not 'cook' events."""
    from backend.routers.recipes import log_event, EventIn

    payload = EventIn(
        user_id=42, recipe_id=1001, event_type="cook",
        rating=None, n_missing=0,
    )
    log_event(payload, db_session)
    assert db_session.query(DrinkEvent).count() == 0


def test_log_event_no_synthesis_for_low_rating(db_session):
    from backend.routers.recipes import log_event, EventIn

    payload = EventIn(
        user_id=42, recipe_id=1001, event_type="rate",
        rating=2.0, n_missing=None,
    )
    log_event(payload, db_session)
    assert db_session.query(DrinkEvent).count() == 0
