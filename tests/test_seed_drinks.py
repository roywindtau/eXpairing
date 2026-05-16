"""
test_seed_drinks.py
-------------------
Tests for backend/db/seed_drinks.py and backend/db/seed_drink_ratings.py.

Strategy: write tiny mock CSV fixtures into a tmp dir, point the seed scripts
at them via monkeypatched module-level Path constants, run the seed against
an in-memory SQLite DB, and verify row counts + key fields.
"""

import os
import sys
from pathlib import Path

# allow `import backend.*`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import seed_drinks, seed_drink_ratings
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

MOCK_WINE_CSV = """\
WineID,WineName,Type,Elaborate,Grapes,Harmonize,ABV,Body,Acidity,Code,Country,RegionID,RegionName,WineryID,WineryName,Website,Vintages
2001,Test Malbec,Red,Varietal/100%,['Malbec'],"['Beef', 'Lamb', 'Grilled']",13.5,Full-bodied,Medium,AR,Argentina,500,Mendoza,9000,Test Winery,,"[2020, 2019]"
2002,Test Chardonnay,White,Varietal/100%,['Chardonnay'],"['Fish', 'Seafood', 'Poultry']",13.0,Medium-bodied,High,FR,France,501,Burgundy,9001,Other Winery,,"[2020, 2019]"
"""

MOCK_WINE_RATINGS_CSV = """\
RatingID,UserID,WineID,Vintage,Rating,Date
1,7001,2001,2020,4.5,2023-01-01
2,7001,2002,2019,3.5,2023-01-02
3,7002,2001,2020,5.0,2023-01-03
"""


@pytest.fixture
def mock_data(tmp_path, monkeypatch):
    """Write mock CSVs to tmp_path and rebind the seed modules' file paths."""
    beer = tmp_path / "beer_reviews.csv"
    wines = tmp_path / "xwines_wines.csv"
    wine_ratings = tmp_path / "xwines_ratings.csv"
    beer.write_text(MOCK_BEER_CSV)
    wines.write_text(MOCK_WINE_CSV)
    wine_ratings.write_text(MOCK_WINE_RATINGS_CSV)

    monkeypatch.setattr(seed_drinks, "BEER_CSV", beer)
    monkeypatch.setattr(seed_drinks, "WINE_CSV", wines)
    monkeypatch.setattr(seed_drink_ratings, "BEER_CSV", beer)
    monkeypatch.setattr(seed_drink_ratings, "WINE_RATINGS_CSV", wine_ratings)
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
    assert db.query(Drink).filter_by(kind="wine").count() == 2


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


def test_seed_drinks_wine_parses_harmonize_and_grapes(mock_data, in_memory_db):
    seed_drinks.seed()
    db = in_memory_db()
    malbec = db.query(Drink).filter_by(id=2001).one()
    assert malbec.kind == "wine"
    assert malbec.wine_type == "Red"
    assert malbec.variety == "Malbec"
    assert malbec.grapes_csv == "Malbec"
    # values from Python-list literal in CSV should be parsed and joined
    assert malbec.harmonize_csv == "Beef,Lamb,Grilled"
    assert malbec.body == "Full-bodied"
    assert malbec.acidity == "Medium"


def test_seed_drinks_is_idempotent(mock_data, in_memory_db):
    seed_drinks.seed()
    first = in_memory_db().query(Drink).count()
    seed_drinks.seed()    # second run should be a no-op
    second = in_memory_db().query(Drink).count()
    assert first == second == 5


# ── tests: seed_drink_ratings ────────────────────────────────────────────

def test_seed_drink_ratings_inserts_events_and_users(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    db = in_memory_db()

    # 5 beer events from the mock + 3 wine events
    assert db.query(DrinkEvent).filter_by(event_type="rate").count() == 8

    # 3 unique beer profilenames (alice, bob, charlie) + 2 unique wine UserIDs (7001, 7002)
    assert db.query(User).filter(User.id >= 100_000).count() == 5

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
    # wine users land at 200_000+ (7001 + 200_000, 7002 + 200_000)
    wine_user_ids = sorted(u.id for u in db.query(User).filter(User.id >= 200_000).all())
    assert wine_user_ids == [207001, 207002]


def test_seed_drink_ratings_skips_unknown_drinks(mock_data, in_memory_db):
    """If a beer_beerid in the ratings CSV isn't in the Drink table, skip it."""
    seed_drinks.seed()
    # delete one beer so its events get skipped
    db = in_memory_db()
    db.query(Drink).filter_by(id=1003).delete()
    db.commit()

    seed_drink_ratings.seed()
    # 5 mock beer rows minus 1 for the deleted beer_id (1003 appears once)
    n_beer_events = db.query(DrinkEvent).filter(DrinkEvent.drink_id < 2000).count()
    assert n_beer_events == 4


def test_seed_drink_ratings_is_idempotent(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    first = in_memory_db().query(DrinkEvent).count()
    seed_drink_ratings.seed()    # no-op
    second = in_memory_db().query(DrinkEvent).count()
    assert first == second == 8


def test_seed_drink_ratings_recomputes_avg(mock_data, in_memory_db):
    seed_drinks.seed()
    seed_drink_ratings.seed()
    db = in_memory_db()
    # wine 2001 has two ratings: 4.5 and 5.0 → avg 4.75
    malbec = db.query(Drink).filter_by(id=2001).one()
    assert malbec.avg_rating == pytest.approx(4.75, abs=0.01)
    assert malbec.n_ratings == 2
