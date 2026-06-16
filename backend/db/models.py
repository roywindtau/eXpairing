"""
models.py
---------
SQLAlchemy ORM table definitions.

Tables:
    User        -- one row per user, stores beta + preference flags
    PantryItem  -- one row per ingredient a user currently has
    Recipe      -- seeded from Food.com CSV; never written by the app
    UserEvent   -- every cook/skip/rate action (feeds beta_updater)
    Drink       -- parent drink catalog (shared columns)
    Wine        -- wine-specific attributes (joined to Drink)
    DrinkEvent  -- every drink rating (explicit or synthesizer-derived)
"""

from datetime import date
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, Date, DateTime, ForeignKey, Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, nullable=True)
    beta        = Column(Float, default=0.35, nullable=False)
    has_cf      = Column(Boolean, default=False, nullable=False)
    has_cb      = Column(Boolean, default=False, nullable=False)
    # dietary preferences stored as comma-separated tags e.g. "vegetarian,gluten-free"
    diet_tags   = Column(String, nullable=True)
    created_at  = Column(DateTime, server_default=func.now())

    pantry_items   = relationship("PantryItem", back_populates="user",
                                  cascade="all, delete-orphan")
    events         = relationship("UserEvent", back_populates="user",
                                  cascade="all, delete-orphan")
    shopping_items = relationship("ShoppingListItem", back_populates="user",
                                  cascade="all, delete-orphan")


class PantryItem(Base):
    __tablename__ = "pantry_items"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ingredient  = Column(String, nullable=False)   # canonical name e.g. "milk"
    raw_name    = Column(String, nullable=True)    # original scan name e.g. "Tnuva 3%"
    expiry_date = Column(Date, nullable=False)
    quantity    = Column(String, nullable=True)    # free text e.g. "500ml"
    added_at    = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="pantry_items")


class Recipe(Base):
    __tablename__ = "recipes"

    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String, nullable=False, index=True)
    # comma-separated canonical ingredient names for fast loading
    ingredients_csv = Column(Text, nullable=False)
    tags_csv        = Column(String, nullable=True)   # e.g. "vegetarian,quick"
    minutes         = Column(Integer, nullable=True)
    n_steps         = Column(Integer, nullable=True)
    avg_rating      = Column(Float, nullable=True)
    n_ratings       = Column(Integer, default=0)
    description     = Column(Text, nullable=True)
    steps_json      = Column(Text, nullable=True)    # JSON array of step strings

    @property
    def ingredients(self) -> list[str]:
        return [i.strip() for i in self.ingredients_csv.split(",") if i.strip()]

    @property
    def steps(self) -> list[str]:
        if not self.steps_json:
            return []
        import json
        try:
            return json.loads(self.steps_json)
        except Exception:
            return []


class ShoppingListItem(Base):
    __tablename__ = "shopping_list_items"

    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ingredient         = Column(String, nullable=False)
    source_recipe_id   = Column(Integer, nullable=True)
    source_recipe_name = Column(String, nullable=True)
    is_checked         = Column(Boolean, default=False, nullable=False)
    added_at           = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="shopping_items")


class UserEvent(Base):
    __tablename__ = "user_events"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipe_id   = Column(Integer, ForeignKey("recipes.id"), nullable=False)
    event_type  = Column(String, nullable=False)   # "cook" | "skip" | "rate"
    rating      = Column(Float, nullable=True)     # only for event_type="rate"
    # snapshot of how many missing ingredients there were at the time
    n_missing   = Column(Integer, nullable=True)
    created_at  = Column(DateTime, server_default=func.now(), index=True)

    user = relationship("User", back_populates="events")


class Drink(Base):
    """
    Parent table for all drinks — shared columns.
    Uses joined table inheritance: Wine extends this with
    kind-specific columns in its own table.

    Query all drinks  → query Drink
    Query wines only  → query Wine (auto-joins drinks + wines)
    """
    __tablename__ = "drinks"
    __mapper_args__ = {
        "polymorphic_on": "kind",
        "polymorphic_identity": "drink",
    }

    id                = Column(Integer, primary_key=True, index=True)
    kind              = Column(String, nullable=False, index=True)   # "wine"
    name              = Column(String, nullable=False, index=True)
    producer          = Column(String, nullable=True)                # winery / brewery
    country           = Column(String, nullable=True)
    style             = Column(String, nullable=True)                # "Red"/"IPA"/"Stout" etc.
    abv               = Column(Float,  nullable=True)
    avg_rating        = Column(Float,  nullable=True)
    n_ratings         = Column(Integer, default=0)
    harmonize_csv     = Column(Text,   nullable=True)                # comma-sep food pairings
    review_tokens_csv = Column(Text,   nullable=True)                # CB signal tokens


class Wine(Drink):
    """Wine-specific attributes. Joined to drinks on id."""
    __tablename__ = "wines"
    __mapper_args__ = {"polymorphic_identity": "wine"}

    id          = Column(Integer, ForeignKey("drinks.id"), primary_key=True)
    grapes_csv  = Column(Text,   nullable=True)                      # comma-sep grape varieties
    body        = Column(String, nullable=True)                      # "Full-bodied" etc.
    acidity     = Column(String, nullable=True)                      # "High" | "Medium" | "Low"
    region      = Column(String, nullable=True)


class DrinkEvent(Base):
    """
    User x drink rating events.
    `synthetic=True` rows are written by drink_synthesizer.py from high recipe
    ratings and are EXCLUDED from CF training (see train_drink_cf.py).
    """
    __tablename__ = "drink_events"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    drink_id    = Column(Integer, ForeignKey("drinks.id"), nullable=False, index=True)
    event_type  = Column(String, nullable=False)        # v1: "rate" only
    rating      = Column(Float, nullable=True)
    synthetic   = Column(Boolean, default=False, nullable=False)
    created_at  = Column(DateTime, server_default=func.now(), index=True)
