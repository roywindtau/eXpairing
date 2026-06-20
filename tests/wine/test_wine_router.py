"""
test_wine_router.py
-------------------
HTTP-level tests for backend/routers/wine.py.

Strategy:
  - Spin up the FastAPI app with an in-memory SQLite DB
  - Override the get_db dependency so every request uses our test DB
  - Stub serve_cb.cb_for_recipe / cb_for_user with canned scores
    (no trained CB artifacts needed)
  - Use TestClient to make real HTTP calls and assert response shape +
    semantics
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.database import get_db
from backend.db.models import Base, Wine, WineEvent, Recipe, User
from backend.main import app


# ── fixture ──────────────────────────────────────────────────────────────

def _seed(db):
    db.add_all([
        # 3 wines
        Wine(id=1, name="Estate Malbec", style="Red",
             grapes_csv="Malbec", harmonize_csv="Beef,Lamb,Grilled",
             avg_rating=4.2, n_ratings=20),
        Wine(id=2, name="Coastal Sauvignon", style="White",
             grapes_csv="Sauvignon Blanc", harmonize_csv="Fish,Seafood",
             avg_rating=3.9, n_ratings=15),
        Wine(id=3, name="Sparkling Bubbly", style="Sparkling",
             grapes_csv="Chardonnay", harmonize_csv="Appetizer",
             avg_rating=4.0, n_ratings=10),
        # Recipe + user
        Recipe(id=1001, name="Grilled Ribeye",
               ingredients_csv="beef,steak,garlic,butter",
               tags_csv="bbq,american"),
        Recipe(id=1002, name="Shrimp Linguine",
               ingredients_csv="shrimp,pasta,garlic,lemon",
               tags_csv="italian,seafood"),
        User(id=42, beta=0.5),
        User(id=99, beta=0.5),
    ])
    db.commit()


@pytest.fixture
def client(monkeypatch):
    """TestClient with an in-memory DB + stubbed CB serving."""
    # StaticPool keeps one shared connection so every Session sees the same
    # in-memory schema — without it, each Session opens a fresh empty DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    _seed(db)
    db.close()

    def _override_get_db():
        d = Session()
        try:
            yield d
        finally:
            d.close()

    app.dependency_overrides[get_db] = _override_get_db

    # Stub CB module so we don't need trained artifacts.
    # NOTE: the router does `from backend.ml.wine.serving.serve_cb import cb_for_recipe`
    # which binds the function at import time, so we patch the ROUTER's local
    # reference (`backend.routers.wine.cb_for_recipe`) — patching the source
    # module wouldn't affect the already-imported binding.
    from backend.routers import wine as wine_router

    monkeypatch.setattr(wine_router, "cb_available", lambda: True)

    def fake_cb_for_recipe(recipe):
        # Make recipe 1001 (beef) score the red wine high; recipe 1002 (shrimp)
        # score the white wine high. Realistic semantic ordering.
        rid = getattr(recipe, "id", None)
        if rid == 1001:
            scores = {1: 0.9, 2: 0.1, 3: 0.3}
        elif rid == 1002:
            scores = {1: 0.1, 2: 0.9, 3: 0.5}
        else:
            scores = {1: 0.5, 2: 0.5, 3: 0.5}
        return scores

    def fake_cb_for_user(user_id, db, min_rating=1.0):
        # Stub: user 42 likes beef wines; user 99 has no preference
        if user_id == 42:
            return {1: 0.9, 2: 0.1, 3: 0.3}
        return {}

    monkeypatch.setattr(wine_router, "cb_for_recipe", fake_cb_for_recipe)
    monkeypatch.setattr(wine_router, "cb_for_user",   fake_cb_for_user)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── GET /wine/ranked  (Path B) ──────────────────────────────────────────

def test_ranked_returns_valid_shape(client):
    r = client.get("/wine/ranked", params={"user_id": 42, "top_n": 5})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0 and len(data) <= 5
    item = data[0]
    for key in ("wine_id", "wine_name", "final_score",
                "cb_score", "cf_score", "expert_boost", "prior_score",
                "cf_strategy", "avg_rating", "n_ratings"):
        assert key in item, f"missing field '{key}'"


def test_ranked_sorted_by_final_score(client):
    r = client.get("/wine/ranked", params={"user_id": 42, "top_n": 6})
    data = r.json()
    finals = [x["final_score"] for x in data]
    assert finals == sorted(finals, reverse=True)


def test_ranked_no_expert_in_path_b(client):
    r = client.get("/wine/ranked", params={"user_id": 42, "top_n": 6})
    for item in r.json():
        assert item["expert_boost"] == 0.0


def test_ranked_unknown_user_returns_404(client):
    r = client.get("/wine/ranked", params={"user_id": 999999})
    assert r.status_code == 404


# ── GET /wine/pairings/{recipe_id}  (Path A) ────────────────────────────

def test_pairings_beef_recipe_ranks_red_wine_first(client):
    r = client.get(
        "/wine/pairings/1001",
        params={"user_id": 99, "top_n": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data[0]["wine_id"] == 1   # Estate Malbec wins on CB + Beef harmonize


def test_pairings_shrimp_recipe_ranks_white_wine_first(client):
    r = client.get(
        "/wine/pairings/1002",
        params={"user_id": 99, "top_n": 3},
    )
    data = r.json()
    assert data[0]["wine_id"] == 2   # Coastal Sauvignon


def test_pairings_includes_expert_boost(client):
    """At least one wine should have a non-zero expert_boost for a beef recipe."""
    r = client.get("/wine/pairings/1001", params={"user_id": 99})
    data = r.json()
    assert any(x["expert_boost"] > 0 for x in data)


def test_pairings_unknown_recipe_returns_404(client):
    r = client.get("/wine/pairings/999999", params={"user_id": 99})
    assert r.status_code == 404


def test_pairings_unknown_user_returns_404(client):
    r = client.get("/wine/pairings/1001", params={"user_id": 99999})
    assert r.status_code == 404


def test_pairings_top_n_truncation(client):
    r = client.get("/wine/pairings/1001", params={"user_id": 99, "top_n": 2})
    assert len(r.json()) == 2


# ── GET /wine/search ────────────────────────────────────────────────────

def test_search_by_name_substring(client):
    r = client.get("/wine/search", params={"q": "malbec"})
    data = r.json()
    assert any("Malbec" in d["name"] for d in data)


def test_search_by_style(client):
    r = client.get("/wine/search", params={"q": "sparkling"})
    data = r.json()
    assert any("Sparkling" in (d["style"] or "") for d in data)


def test_search_empty_query_returns_all(client):
    r = client.get("/wine/search", params={"q": "", "limit": 100})
    assert len(r.json()) == 3


# ── GET /wine/{wine_id} ─────────────────────────────────────────────────

def test_wine_detail_returns_full_object(client):
    r = client.get("/wine/1")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == 1
    assert data["style"] == "Red"
    assert data["harmonize_csv"] == "Beef,Lamb,Grilled"


def test_wine_detail_unknown_returns_404(client):
    r = client.get("/wine/999999")
    assert r.status_code == 404


# ── POST /wine-events ───────────────────────────────────────────────────

def test_post_wine_event_creates_row(client):
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    1,
        "event_type": "rate",
        "rating":     4.5,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "ok"
    assert "event_id" in body


def test_post_wine_event_unknown_wine_returns_404(client):
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    999999,
        "event_type": "rate",
        "rating":     4.0,
    })
    assert r.status_code == 404


def test_post_wine_event_rejects_invalid_event_type(client):
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    1,
        "event_type": "cook",   # not allowed for wine events
        "rating":     4.0,
    })
    assert r.status_code == 422


def test_post_wine_event_rejects_missing_rating(client):
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    1,
        "event_type": "rate",
    })
    assert r.status_code == 422


def test_post_wine_event_rejects_out_of_range_rating(client):
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    1,
        "event_type": "rate",
        "rating":     6.5,
    })
    assert r.status_code == 422


def test_post_wine_event_writes_explicit_not_synthetic(client):
    """The rating must land as synthetic=False (only synthesizer writes synthetic)."""
    r = client.post("/wine-events", json={
        "user_id":    42,
        "wine_id":    1,
        "event_type": "rate",
        "rating":     4.0,
    })
    assert r.status_code == 201

    # Verify directly in DB via the override session.
    db_gen = app.dependency_overrides[get_db]()
    db = next(db_gen)
    try:
        ev = (
            db.query(WineEvent)
            .filter(WineEvent.user_id == 42, WineEvent.wine_id == 1)
            .first()
        )
        assert ev is not None
        assert ev.synthetic is False
        assert ev.rating == 4.0
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
