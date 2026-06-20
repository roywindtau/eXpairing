"""
test_wine_scoring.py
--------------------
Unit tests for backend/services/wine/scoring.py — the personalized
"recommend me a wine" ranking.

These run WITHOUT model artifacts (no wine_cb_matrix.npz / wine_als_model.npz).
The scorer degrades gracefully when CB/CF are unavailable: cold start, the style
hard-filter, no-re-recommend, and the popularity fallback are all testable on
their own. The CB/CF math is monkeypatched where we want to exercise the blend.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, User, Wine, WineEvent
from backend.services.wine import scoring


# ── fixture ──────────────────────────────────────────────────────────────

def _db():
    engine = create_engine("sqlite:///:memory:",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    # 4 reds (varying popularity) + 2 whites + a user
    db.add_all([
        Wine(id=1, name="Top Red",   style="Red",   grapes_csv="Malbec",
             body="Full-bodied",  acidity="High", abv=14.0, avg_rating=4.9, n_ratings=900),
        Wine(id=2, name="Mid Red",   style="Red",   grapes_csv="Merlot",
             body="Full-bodied",  acidity="High", abv=13.5, avg_rating=4.4, n_ratings=300),
        Wine(id=3, name="Low Red",   style="Red",   grapes_csv="Syrah",
             body="Medium-bodied", acidity="Medium", abv=13.0, avg_rating=3.8, n_ratings=50),
        Wine(id=4, name="Liked Red", style="Red",   grapes_csv="Cabernet Sauvignon",
             body="Full-bodied",  acidity="High", abv=14.0, avg_rating=4.0, n_ratings=120),
        Wine(id=5, name="Top White", style="White", grapes_csv="Chardonnay",
             body="Light-bodied", acidity="High", abv=12.0, avg_rating=4.7, n_ratings=800),
        Wine(id=6, name="Mid White", style="White", grapes_csv="Riesling",
             body="Light-bodied", acidity="High", abv=11.0, avg_rating=4.1, n_ratings=200),
        User(id=1, beta=0.5),
    ])
    db.commit()
    return db


@pytest.fixture
def db():
    d = _db()
    yield d
    d.close()


def _rate(db, user_id, wine_id, rating):
    db.add(WineEvent(user_id=user_id, wine_id=wine_id, event_type="rate",
                     rating=rating, synthetic=False))
    db.commit()


# ── cold start ───────────────────────────────────────────────────────────

def test_cold_user_gets_popularity(db):
    """0 ratings -> top popularity across all styles, most popular first."""
    out = scoring.rank_wines(db, user_id=1, top_n=3)
    assert [w.id for w in out][0] == 1            # Top Red is most popular
    assert len(out) == 3


def test_cold_user_not_style_filtered(db):
    """Cold start spans styles (whites can appear)."""
    out = scoring.rank_wines(db, user_id=1, top_n=6)
    styles = {w.style for w in out}
    assert "White" in styles and "Red" in styles


# ── style hard-filter ────────────────────────────────────────────────────

def test_warm_user_style_filtered_to_drunk_styles(db):
    """User who only rated reds never gets a white recommended."""
    _rate(db, 1, 4, 5.0)                            # likes a Red
    out = scoring.rank_wines(db, user_id=1, top_n=5)
    assert out, "expected recommendations"
    assert all(w.style == "Red" for w in out)


# ── no re-recommend ──────────────────────────────────────────────────────

def test_does_not_recommend_already_rated_wine(db):
    _rate(db, 1, 4, 5.0)
    out = scoring.rank_wines(db, user_id=1, top_n=5)
    assert 4 not in [w.id for w in out]


# ── warming path uses popularity ordering when CB absent ─────────────────

def test_warming_falls_back_to_popularity_without_cb(db, monkeypatch):
    """1-4 ratings, no CB artifact -> within-style popularity order."""
    monkeypatch.setattr(scoring.serve_cb, "cb_available", lambda: False)
    monkeypatch.setattr(scoring.serve_cf, "cf_available", lambda: False)
    _rate(db, 1, 4, 5.0)                            # 1 red rating -> warming
    out = scoring.rank_wines(db, user_id=1, top_n=3)
    ids = [w.id for w in out]
    assert ids[0] == 1                              # Top Red, most popular red
    assert all(w.style == "Red" for w in out)


# ── warm blend routes through CF+CB ──────────────────────────────────────

def test_warm_user_blends_cf_and_cb(db, monkeypatch):
    """>=5 ratings with CF+CB available -> blend picks the CF/CB favorite."""
    monkeypatch.setattr(scoring.serve_cb, "cb_available", lambda: True)
    monkeypatch.setattr(scoring.serve_cf, "cf_available", lambda: True)
    # CB/CF both strongly prefer wine 3 (the otherwise-least-popular red)
    monkeypatch.setattr(scoring.serve_cb, "cb_scores",
                        lambda liked, cands, **k: {c: (1.0 if c == 3 else 0.0) for c in cands})
    monkeypatch.setattr(scoring.serve_cf, "cf_scores",
                        lambda liked, cands: {c: (1.0 if c == 3 else 0.0) for c in cands})
    for wid in (1, 2, 4):                           # rate 3 reds...
        _rate(db, 1, wid, 5.0)
    # ...and pad to >=5 ratings on reds so the warm CF+CB path fires.
    db.add_all([Wine(id=10, name="Pad A", style="Red", grapes_csv="Tempranillo",
                     body="Full-bodied", acidity="High", avg_rating=4.0, n_ratings=10),
                Wine(id=11, name="Pad B", style="Red", grapes_csv="Grenache",
                     body="Full-bodied", acidity="High", avg_rating=4.0, n_ratings=10)])
    db.commit()
    _rate(db, 1, 10, 5.0)
    _rate(db, 1, 11, 5.0)
    out = scoring.rank_wines(db, user_id=1, top_n=3)
    # CB+CF both spike wine 3 (unrated) -> it should be the top pick.
    assert out[0].id == 3


def test_empty_db_user_cold_path_safe(db):
    """A user with no ratings on an otherwise-populated DB never errors."""
    out = scoring.rank_wines(db, user_id=999, top_n=5)
    assert isinstance(out, list)
