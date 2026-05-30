"""
test_seed_drinks.py
-------------------
Tests for backend/db/drinks/seed_drinks.py and backend/db/drinks/seed_ratings.py.

Strategy: write tiny mock CSV fixtures into a tmp dir, point the seed scripts
at them via monkeypatched module-level Path constants, run the seed against
an in-memory SQLite DB, and verify row counts + key fields.

Wine seeding is intentionally absent — the wine-data branch is choosing a
new source. These tests only cover the beer path.
"""

import sys
from pathlib import Path

# allow `import backend.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.drinks import seed_drinks
from backend.db.drinks import seed_ratings as seed_drink_ratings
from backend.db.models import Base, Drink, DrinkEvent, User


# ── fixtures ─────────────────────────────────────────────────────────────

MOCK_BEER_CSV = """\
brewery_id,brewery_name,review_time,review_overall,review_aroma,review_appearance,review_profilename,beer_style,review_palate,review_taste,beer_name,beer_abv,beer_beerid
10,Acme Brewing,100,4.5,4.0,4.0,alice,IPA,4.5,5.0,Hop Bomb,6.5,1001
10,Acme Brewing,200,4.0,4.0,3.5,bob,IPA,4.0,4.0,Hop Bomb,6.5,1001
20,Beta Brewing,300,3.5,3.5,4.0,alice,Stout,4.0,3.5,Dark Velvet,8.0,1002
20,Beta Brewing,400,5.0,5.0,5.0,charlie,Stout,5.0,5.0,Dark Velvet,8.0,1002
30,Gamma Brewing,500,2.5,3.0,3.0,bob,Pilsner,2.5,2.5,Crisp Light,4.5,1003
"""


@pytest.fixture
def mock_data(tmp_path, monkeypatch):
    """Write mock CSVs to tmp_path and rebind the seed modules' file paths."""
    beer = tmp_path / "beer_reviews.csv"
    beer.write_text(MOCK_BEER_CSV)

    monkeypatch.setattr(seed_drinks, "BEER_CSV", beer)
    monkeypatch.setattr(seed_drink_ratings, "BEER_CSV", beer)
    return tmp_path


@pytest.fixture
def in_memory_db(monkeypatch):
    """Point the seed scripts at an in-memory SQLite DB."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def fake_init():
        Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(seed_drinks, "init_db", fake_init)
    monkeypatch.setattr(seed_drinks, "SessionLocal", Session)
    monkeypatch.setattr(seed_drink_ratings, "init_db", fake_init)
    monkeypatch.setattr(seed_drink_ratings, "SessionLocal", Session)

    return Session


# ── tests: seed_drinks ───────────────────────────────────────────────────

def test_seed_drinks_inserts_correct_counts(mock_data, in_memory_db):
    seed_drinks.seed()
    db = in_memory_db()
    assert db.query(Drink).filter_by(kind="beer").count() == 3   # 3 unique beer_beerid


def test_seed_drinks_beer_aggregates(mock_data, in_memory_db):
    seed_drinks.seed()
    db = in_memory_db()
    hop_bomb = db.query(Drink).filter_by(id=1001).one()
    assert hop_bomb.kind == "beer"
    assert hop_bomb.name == "Hop Bomb"
    assert hop_bomb.style == "IPA"
    assert hop_bomb.n_ratings == 2
    assert hop_bomb.avg_rating == pytest.approx((4.5 + 4.0) / 2, abs=0.01)
    assert hop_bomb.avg_taste == pytest.approx((5.0 + 4.0) / 2, abs=0.01)
    assert hop_bomb.abv == 6.5


def test_seed_drinks_is_idempotent(mock_data, in_memory_db):
    seed_drinks.seed()
    first = in_memory_db().query(Drink).count()
    seed_drinks.seed()    # second run should be a no-op
    second = in_memory_db().query(Drink).count()
    assert first == second == 3


# ── tests: seed_ratings ──────────────────────────────────────────────────

def test_seed_drink_ratings_inserts_events_and_users(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    db = in_memory_db()

    # 5 beer events from the mock
    assert db.query(DrinkEvent).filter_by(event_type="rate").count() == 5

    # 3 unique beer profilenames (alice, bob, charlie)
    assert db.query(User).filter(User.id >= 100_000).count() == 3

    # spot-check a beer event
    e = db.query(DrinkEvent).filter_by(drink_id=1001).first()
    assert e.rating in (4.5, 4.0)
    assert e.user_id >= 100_000
    assert e.synthetic is False


def test_seed_drink_ratings_user_id_offsets(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    db = in_memory_db()
    # beer users land in [100_000, 200_000)
    beer_user_ids = [u.id for u in db.query(User).filter(
        User.id >= 100_000, User.id < 200_000
    ).all()]
    assert len(beer_user_ids) == 3


def test_seed_drink_ratings_skips_unknown_drinks(mock_data, in_memory_db):
    """If a beer_beerid in the ratings CSV isn't in the Drink table, skip it."""
    seed_drinks.seed()
    # delete one beer so its events get skipped
    db = in_memory_db()
    db.query(Drink).filter_by(id=1003).delete()
    db.commit()

    seed_drink_ratings.seed()
    # 5 mock beer rows minus 1 for the deleted beer_id (1003 appears once)
    n_beer_events = db.query(DrinkEvent).count()
    assert n_beer_events == 4


def test_seed_drink_ratings_is_idempotent(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    first = in_memory_db().query(DrinkEvent).count()
    seed_drink_ratings.seed()    # no-op
    second = in_memory_db().query(DrinkEvent).count()
    assert first == second == 5
