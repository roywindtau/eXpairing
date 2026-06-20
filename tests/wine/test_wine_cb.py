"""
test_wine_cb.py
----------------
End-to-end tests for backend/ml/train_cb.py and serve_cb.py.

Strategy:
  - Build an in-memory SQLite DB with a small but representative fixture
    (3 wines, 4 recipes, 4 user-rating events)
  - Train the wine CB into a tmp `models/` dir (monkeypatched paths)
  - Verify artifacts exist, then exercise cb_for_recipe / cb_for_user
  - Validate semantics: a beef-heavy recipe must rank Red wine above White,
    a seafood-heavy recipe must rank White wine above Red.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base, Wine, Recipe, User, UserEvent
from backend.ml.wine.serving import serve_cb as serve_cb
from backend.ml.wine.training import train_cb as train_cb


# ── fixture data ─────────────────────────────────────────────────────────

def _seed_fixture(db):
    db.add_all([
        # Wines
        Wine(id=1, name="Estate Malbec", style="Red",
             grapes_csv="Malbec", harmonize_csv="Beef,Lamb,Grilled",
             body="Full-bodied", acidity="Medium",
             review_tokens_csv="beef,lamb,grilled,malbec"),
        Wine(id=2, name="Coastal Sauvignon", style="White",
             grapes_csv="Sauvignon Blanc", harmonize_csv="Fish,Seafood,Salads",
             body="Light-bodied", acidity="High",
             review_tokens_csv="fish,seafood,salads,sauvignon"),
        Wine(id=3, name="Sparkling Bubbly", style="Sparkling",
             grapes_csv="Chardonnay", harmonize_csv="Appetizer,Cheese",
             body="Light-bodied", acidity="High",
             review_tokens_csv="appetizer,cheese,sparkling"),
        # Recipes (used by cb_for_user)
        Recipe(id=1001, name="Grilled Ribeye", ingredients_csv="beef,steak,garlic,butter",
               tags_csv="american,bbq"),
        Recipe(id=1002, name="Shrimp Linguine", ingredients_csv="shrimp,pasta,garlic,lemon",
               tags_csv="italian,seafood"),
        Recipe(id=1003, name="Spicy Chicken Curry", ingredients_csv="chicken,curry,chili,onion",
               tags_csv="indian,spicy"),
        Recipe(id=1004, name="Chocolate Lava Cake", ingredients_csv="chocolate,butter,sugar,eggs",
               tags_csv="dessert"),
        User(id=42, beta=0.5),
    ])
    db.commit()


@pytest.fixture
def trained_cb(tmp_path, monkeypatch):
    """
    Build a fixture DB, point both train + serve modules at tmp paths,
    train the CB, and yield (Session, models_dir).
    Resets the serve module's lazy-load singleton between tests.
    """
    # In-memory DB
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    _seed_fixture(db)
    db.close()
    monkeypatch.setattr(train_cb, "SessionLocal", Session)

    # Redirect artifact paths into tmp_path
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(train_cb, "MODELS_DIR",    models_dir)
    monkeypatch.setattr(train_cb, "CB_MATRIX",     models_dir / "wine_cb_matrix.npz")
    monkeypatch.setattr(train_cb, "CB_IDS",        models_dir / "wine_cb_ids.npy")
    monkeypatch.setattr(train_cb, "CB_KINDS",      models_dir / "wine_cb_kinds.npy")
    monkeypatch.setattr(train_cb, "CB_VECTORIZER", models_dir / "wine_cb_vectorizer.pkl")
    monkeypatch.setattr(train_cb, "CB_META",       models_dir / "wine_cb_meta.json")

    monkeypatch.setattr(serve_cb, "CB_MATRIX_PATH",     models_dir / "wine_cb_matrix.npz")
    monkeypatch.setattr(serve_cb, "CB_IDS_PATH",        models_dir / "wine_cb_ids.npy")
    monkeypatch.setattr(serve_cb, "CB_KINDS_PATH",      models_dir / "wine_cb_kinds.npy")
    monkeypatch.setattr(serve_cb, "CB_VECTORIZER_PATH", models_dir / "wine_cb_vectorizer.pkl")

    train_cb.train()
    serve_cb._reset_for_tests()

    yield Session, models_dir

    serve_cb._reset_for_tests()


# ── training artifacts ───────────────────────────────────────────────────

def test_training_writes_all_artifacts(trained_cb):
    _, models_dir = trained_cb
    for name in (
        "wine_cb_matrix.npz",
        "wine_cb_ids.npy",
        "wine_cb_kinds.npy",
        "wine_cb_vectorizer.pkl",
        "wine_cb_meta.json",
    ):
        assert (models_dir / name).exists(), f"missing {name}"


def test_artifact_shapes_match(trained_cb):
    serve_cb._load()
    assert serve_cb._matrix.shape[0] == 3   # 3 wines
    assert len(serve_cb._wine_ids) == 3
    assert len(serve_cb._kinds) == 3
    assert set(serve_cb._kinds.tolist()) == {"wine"}


def test_model_available_true_after_train(trained_cb):
    assert serve_cb.model_available() is True


def test_model_available_false_without_artifacts(tmp_path, monkeypatch):
    """No artifacts -> model_available is False, scores are empty."""
    monkeypatch.setattr(serve_cb, "CB_MATRIX_PATH",     tmp_path / "x.npz")
    monkeypatch.setattr(serve_cb, "CB_IDS_PATH",        tmp_path / "x.npy")
    monkeypatch.setattr(serve_cb, "CB_KINDS_PATH",      tmp_path / "x.npy")
    monkeypatch.setattr(serve_cb, "CB_VECTORIZER_PATH", tmp_path / "x.pkl")
    serve_cb._reset_for_tests()
    assert serve_cb.model_available() is False


# ── cb_for_recipe semantics ─────────────────────────────────────────────

def test_cb_for_recipe_beef_prefers_red_wine(trained_cb):
    Session, _ = trained_cb
    db = Session()
    try:
        steak = db.query(Recipe).get(1001)
        scores = serve_cb.cb_for_recipe(steak)
    finally:
        db.close()
    assert scores, "expected non-empty scores"
    # Malbec (id=1, Red) must outrank Sauvignon (id=2, White)
    assert scores[1] > scores[2], f"Red {scores[1]} should beat White {scores[2]} for steak"


def test_cb_for_recipe_seafood_prefers_white_wine(trained_cb):
    Session, _ = trained_cb
    db = Session()
    try:
        shrimp = db.query(Recipe).get(1002)
        scores = serve_cb.cb_for_recipe(shrimp)
    finally:
        db.close()
    assert scores
    # Sauvignon (white) must outrank Malbec (red) for shrimp pasta
    assert scores[2] > scores[1], f"White {scores[2]} should beat Red {scores[1]} for shrimp"


def test_cb_for_recipe_scores_all_wines(trained_cb):
    Session, _ = trained_cb
    db = Session()
    try:
        recipe = db.query(Recipe).get(1001)
        scores = serve_cb.cb_for_recipe(recipe)
    finally:
        db.close()
    # All three seeded wines should be scored.
    assert set(scores.keys()) == {1, 2, 3}


def test_cb_for_recipe_empty_recipe_returns_empty(trained_cb):
    from types import SimpleNamespace
    empty = SimpleNamespace(ingredients_csv="", tags_csv="")
    assert serve_cb.cb_for_recipe(empty) == {}


# ── cb_for_user semantics ───────────────────────────────────────────────

def test_cb_for_user_no_history_returns_empty(trained_cb):
    Session, _ = trained_cb
    db = Session()
    try:
        scores = serve_cb.cb_for_user(user_id=42, db=db)
    finally:
        db.close()
    assert scores == {}


def test_cb_for_user_beef_lover_ranks_red_above_white(trained_cb):
    """A user who rated the steak recipe 5 should get Red wine > White."""
    Session, _ = trained_cb
    db = Session()
    try:
        db.add(UserEvent(user_id=42, recipe_id=1001, event_type="rate", rating=5.0))
        db.commit()
        scores = serve_cb.cb_for_user(user_id=42, db=db)
    finally:
        db.close()
    assert scores
    assert scores[1] > scores[2], \
        f"Beef lover: Red {scores[1]} should beat White {scores[2]}"


def test_cb_for_user_seafood_lover_ranks_white_above_red(trained_cb):
    Session, _ = trained_cb
    db = Session()
    try:
        db.add(UserEvent(user_id=42, recipe_id=1002, event_type="rate", rating=5.0))
        db.commit()
        scores = serve_cb.cb_for_user(user_id=42, db=db)
    finally:
        db.close()
    assert scores
    assert scores[2] > scores[1], \
        f"Seafood lover: White {scores[2]} should beat Red {scores[1]}"


def test_cb_for_user_negative_weight_pushes_wine_down(trained_cb):
    """
    User loves steak (5) but hates shrimp (1). The Red wine should still win
    among wines because the steak signal dominates and shrimp's negative
    weight pushes White further down.
    """
    Session, _ = trained_cb
    db = Session()
    try:
        db.add_all([
            UserEvent(user_id=42, recipe_id=1001, event_type="rate", rating=5.0),
            UserEvent(user_id=42, recipe_id=1002, event_type="rate", rating=1.0),
        ])
        db.commit()
        scores = serve_cb.cb_for_user(user_id=42, db=db)
    finally:
        db.close()
    assert scores
    assert scores[1] > scores[2]
