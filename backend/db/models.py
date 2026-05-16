"""
models.py
---------
SQLAlchemy ORM table definitions.

Tables:
    User        -- one row per user, stores beta + preference flags
    PantryItem  -- one row per ingredient a user currently has
    Recipe      -- seeded from Food.com CSV; never written by the app
    UserEvent   -- every cook/skip/rate action (feeds beta_updater)
    Drink       -- beer + wine catalog, seeded from drinks CSVs
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
    Unified beer + wine catalog, discriminated by `kind`.
    Beer-specific and wine-specific columns are nullable.
    Seeded from data/beer_reviews.csv and data/xwines_wines.csv.
    """
    __tablename__ = "drinks"

    id              = Column(Integer, primary_key=True, index=True)
    kind            = Column(String, nullable=False, index=True)   # "beer" | "wine"
    name            = Column(String, nullable=False, index=True)
    producer        = Column(String, nullable=True)                # brewer / winery
    country         = Column(String, nullable=True)
    abv             = Column(Float,  nullable=True)
    avg_rating      = Column(Float,  nullable=True)
    n_ratings       = Column(Integer, default=0)
    # space-separated top-N most-frequent non-stopword tokens from reviews
    # used by train_drink_cb.py as part of each drink's CB document
    review_tokens_csv = Column(Text, nullable=True)

    # ── beer-specific (nullable) ─────────────────────────────────────────
    style           = Column(String, nullable=True)                # "IPA", "Stout", ...
    ibu             = Column(Float,  nullable=True)
    avg_aroma       = Column(Float,  nullable=True)
    avg_taste       = Column(Float,  nullable=True)
    avg_palate      = Column(Float,  nullable=True)
    avg_appearance  = Column(Float,  nullable=True)

    # ── wine-specific (nullable; from X-Wines) ───────────────────────────
    wine_type       = Column(String, nullable=True)                # "Red" | "White" | "Rose" | "Sparkling" | "Dessert" | "Dessert/Port"
    variety         = Column(String, nullable=True)                # primary grape
    grapes_csv      = Column(Text,   nullable=True)                # full grape list, comma-sep
    region          = Column(String, nullable=True)
    body            = Column(String, nullable=True)                # "Full-bodied" | "Medium-bodied" | "Light-bodied" | "Very full-bodied"
    acidity         = Column(String, nullable=True)                # "High" | "Medium" | "Low"
    sweetness       = Column(Integer, nullable=True)               # only populated by Slim/Full X-Wines, not Test
    tannin          = Column(Integer, nullable=True)               # only populated by Slim/Full X-Wines, not Test
    harmonize_csv   = Column(Text,   nullable=True)                # comma-sep food categories, e.g. "Beef,Lamb,Pasta"


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
