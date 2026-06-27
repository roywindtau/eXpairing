"""
recipe_categories.py
====================
MODULE 3 of the wine<->recipe pairing feature: recipe -> category vector.

WHAT IT DOES
------------
Turns a recipe (a list of ingredient strings) into a 12-dim vector over the same
canonical food categories used for wines (Module 1). This is the piece that puts
recipes into the SAME vector space as wines, so they can be compared.

    Lobster bisque  ["lobster","cream","shallots","brandy","tomato paste",
                     "fish stock","butter","tarragon"]
        lobster    -> Seafood
        cream      -> Creamy
        fish stock -> Seafood
        => category weights {Seafood: 2, Creamy: 1}
        => L2-normalized 12-dim vector

WHY A SIGNAL LEXICON WITH WORD MATCHING (not exact match, not a model)
---------------------------------------------------------------------
Real Food.com data has ~15k distinct ingredient strings with endless variants
("garlic", "garlic cloves", "garlic powder", "minced garlic"). Exact-match would
miss almost everything; a trained model would just relearn this lookup from the
synthetic pairing rules (see check_ingredient_signal.py). So we use a hand-written
lexicon of SIGNAL keywords and match them as whole words inside each ingredient:

    keyword "salmon" matches "salmon", "smoked salmon", "salmon fillet"
    keyword "ham"    matches "ham" but NOT "graham" (word-boundary guarded)

Most ingredients (salt, butter, water, flour, sugar) are NOISE -- they appear in
everything and signal no category, so they are simply absent from the lexicon and
contribute nothing.

DESIGN NOTES
------------
- A recipe gets a WEIGHTED SET of categories, not a single label. "Lobster bisque"
  is both Seafood and Creamy. Weights = how many signal ingredients hit each
  category, then L2-normalized (same convention as the wine vectors, Module 2).
- A keyword may map to several categories (e.g. "sausage" -> Pork + Smoky BBQ).
- Recipes that hit NO signal keyword fall back to Vegetarian (the catch-all for
  produce/grain/dairy dishes with no dominant protein or sweet signal).
- This is intentionally a hand-written rule. We can state the mapping plainly, so
  we write it rather than train a model to rediscover it.

PUBLIC API
----------
    recipe_vector(ingredients: list[str]) -> np.ndarray   # 12-dim, L2-normalized
    recipe_categories(ingredients: list[str]) -> dict[str, float]  # for display

Run (prints category coverage over the live recipe DB):
    python -m data.pairing.recipe_categories
"""

from __future__ import annotations

import re

import numpy as np

from data.pairing.pairing_vocabulary import CATEGORIES, CATEGORY_INDEX

# ── Signal keyword -> category(ies) ──────────────────────────────────────────
# Keywords are matched as whole words inside an ingredient string (case-insensitive).
# Order does not matter; all matches accumulate.
KEYWORD_TO_CATEGORY: dict[str, list[str]] = {
    # ── Red Meat ──
    "beef": ["Red Meat"], "steak": ["Red Meat"], "lamb": ["Red Meat"],
    "veal": ["Red Meat"], "venison": ["Red Meat"], "ground beef": ["Red Meat"],
    "sirloin": ["Red Meat"], "brisket": ["Red Meat"], "ribeye": ["Red Meat"],
    "roast beef": ["Red Meat"], "oxtail": ["Red Meat"],
    # ── Poultry ──
    "chicken": ["Poultry"], "turkey": ["Poultry"], "duck": ["Poultry"],
    "quail": ["Poultry"], "hen": ["Poultry"],
    # ── Pork ──
    "pork": ["Pork"], "bacon": ["Pork", "Smoky BBQ"], "ham": ["Pork"],
    "prosciutto": ["Pork", "Salty Snack"], "pancetta": ["Pork"],
    "sausage": ["Pork"], "chorizo": ["Pork", "Spicy"], "salami": ["Pork", "Salty Snack"],
    # ── Seafood ──
    "fish": ["Seafood"], "salmon": ["Seafood"], "tuna": ["Seafood"],
    "shrimp": ["Seafood"], "prawn": ["Seafood"], "crab": ["Seafood"],
    "lobster": ["Seafood"], "cod": ["Seafood"], "scallop": ["Seafood"],
    "clam": ["Seafood"], "oyster": ["Seafood"], "mussel": ["Seafood"],
    "anchovy": ["Seafood"], "anchovies": ["Seafood"], "halibut": ["Seafood"],
    "tilapia": ["Seafood"], "trout": ["Seafood"], "sardine": ["Seafood"],
    "haddock": ["Seafood"], "calamari": ["Seafood"], "squid": ["Seafood"],
    "seafood": ["Seafood"],
    # ── Cheese ──
    "cheese": ["Cheese"], "parmesan": ["Cheese"], "cheddar": ["Cheese"],
    "mozzarella": ["Cheese"], "feta": ["Cheese"], "gouda": ["Cheese"],
    "brie": ["Cheese"], "ricotta": ["Cheese"], "gruyere": ["Cheese"],
    "blue cheese": ["Cheese"], "goat cheese": ["Cheese"],
    # ── Creamy ──
    "cream": ["Creamy"], "heavy cream": ["Creamy"], "sour cream": ["Creamy"],
    "cream cheese": ["Creamy"], "creme fraiche": ["Creamy"], "alfredo": ["Creamy"],
    "bechamel": ["Creamy"], "custard": ["Creamy"], "mascarpone": ["Creamy"],
    # ── Spicy ──
    "chili": ["Spicy"], "chilli": ["Spicy"], "jalapeno": ["Spicy"],
    "sriracha": ["Spicy"], "cayenne": ["Spicy"], "curry": ["Spicy"],
    "habanero": ["Spicy"], "chipotle": ["Spicy", "Smoky BBQ"], "harissa": ["Spicy"],
    "tabasco": ["Spicy"], "hot sauce": ["Spicy"], "red pepper flakes": ["Spicy"],
    "chili powder": ["Spicy"], "wasabi": ["Spicy"],
    # ── Acidic ──
    "lemon": ["Acidic"], "lime": ["Acidic"], "vinegar": ["Acidic"],
    "tomato": ["Acidic"], "tomatoes": ["Acidic"], "citrus": ["Acidic"],
    "lemon juice": ["Acidic"], "lime juice": ["Acidic"],
    # ── Salty Snack ──
    "chips": ["Salty Snack"], "pretzel": ["Salty Snack"], "crackers": ["Salty Snack"],
    "olives": ["Salty Snack"], "popcorn": ["Salty Snack"], "nuts": ["Salty Snack"],
    "peanuts": ["Salty Snack"],
    # ── Smoky BBQ ──
    "barbecue": ["Smoky BBQ"], "bbq": ["Smoky BBQ"], "smoked": ["Smoky BBQ"],
    "grilled": ["Smoky BBQ"], "mesquite": ["Smoky BBQ"],
    # ── Dessert ──
    "chocolate": ["Dessert"], "cocoa": ["Dessert"], "honey": ["Dessert"],
    "caramel": ["Dessert"], "maple syrup": ["Dessert"], "marshmallow": ["Dessert"],
    "frosting": ["Dessert"], "icing": ["Dessert"], "fudge": ["Dessert"],
    # NOTE: plain "sugar"/"vanilla" are NOT here -- too common (baking staples),
    # they would tag almost everything Dessert. Only stronger sweet signals.
}

# Precompile a word-boundary regex per keyword. \b guards against "ham" in "graham".
_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), cats)
    for kw, cats in KEYWORD_TO_CATEGORY.items()
]

FALLBACK = "Vegetarian"


def recipe_categories(ingredients: list[str]) -> dict[str, float]:
    """Raw category weights for a recipe (how many ingredients hit each category)."""
    weights: dict[str, float] = {}
    for ing in ingredients:
        for pat, cats in _PATTERNS:
            if pat.search(ing):
                for c in cats:
                    weights[c] = weights.get(c, 0.0) + 1.0
    if not weights:
        weights[FALLBACK] = 1.0
    return weights


def recipe_vector(ingredients: list[str]) -> np.ndarray:
    """12-dim L2-normalized category vector for a recipe (same space as wines)."""
    vec = np.zeros(len(CATEGORIES), dtype=np.float64)
    for cat, w in recipe_categories(ingredients).items():
        vec[CATEGORY_INDEX[cat]] = w
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _report() -> None:
    """Coverage report over the live recipe DB: how many recipes hit a signal."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db.database import SessionLocal
    from backend.db.models import Recipe

    db = SessionLocal()
    try:
        recipes = db.query(Recipe.ingredients_csv).all()
    finally:
        db.close()

    import collections
    cat_hits = collections.Counter()
    fallback_only = 0
    for (csv,) in recipes:
        ings = [i.strip() for i in (csv or "").split(",") if i.strip()]
        cats = recipe_categories(ings)
        if set(cats) == {FALLBACK}:
            fallback_only += 1
        for c in cats:
            cat_hits[c] += 1

    n = len(recipes)
    print(f"recipes: {n:,}")
    print(f"fell back to {FALLBACK} only (no signal): "
          f"{fallback_only:,} ({100*fallback_only/n:.1f}%)\n")
    print("recipes touching each category:")
    for cat in CATEGORIES:
        h = cat_hits[cat]
        print(f"  {cat:14s} {h:7,d}  ({100*h/n:4.1f}%)")


if __name__ == "__main__":
    _report()
