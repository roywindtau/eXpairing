"""
pairing_vocabulary.py
=====================
MODULE 1 of the wine<->recipe pairing feature: the shared vocabulary.

WHY THIS EXISTS
---------------
A wine and a recipe can only be compared if they live in the SAME vector space
(same dimensions, each dimension meaning the same thing). The shared space we
use is a fixed list of FOOD CATEGORIES. Three different data sources speak three
different vocabularies, and this module is the single place that reconciles them:

    source                vocabulary                  -> mapped to
    --------------------  --------------------------  ----------------------
    wine harmonize_csv    66 tokens (Beef, Shellfish) -> 12 categories
    pairing CSV           12 categories (the target)  -> used as-is
    recipe ingredients    free text (lobster, lamb)   -> 12 categories (Module 3)

The 12 categories come from the labeled pairing dataset
(data/pairing/wine_food_pairings.csv, column `food_category`). We adopt them as the
canonical axes because the pairing RULES (and the test labels) are expressed in
exactly these terms.

DESIGN NOTES
------------
- Not every harmonize token is a "food category" in the sensory sense. Tokens
  like "Aperitif", "Appetizer", "Salad", "Pasta", "Pizza" describe a COURSE or
  DISH TYPE, not a dominant ingredient/sensory class. Those map to None and are
  deliberately NOT forced into a category. Honest coverage > fake precision.
- A token may legitimately carry more than one category (e.g. "Lasagna" is both
  Red Meat and Cheese). The map value is therefore a LIST of categories.
- This is a hand-written lookup, on purpose. The mapping is a rule we can state
  plainly, so we write it rather than train a model to rediscover it.

USED BY
-------
    Module 2 (wine -> category vectors)  reads HARMONIZE_TO_CATEGORY
    Module 3 (recipe -> category vector) reads CATEGORIES (and its own ing map)

Run (prints a coverage report over the live wine DB):
    python -m data.pairing.pairing_vocabulary
"""

from __future__ import annotations

# ── The 12 canonical categories (the dimensions of the shared space) ──────────
# Order is fixed: a category's index here is its column in every vector.
CATEGORIES: list[str] = [
    "Red Meat",
    "Poultry",
    "Pork",
    "Seafood",
    "Cheese",
    "Creamy",
    "Spicy",
    "Acidic",
    "Salty Snack",
    "Smoky BBQ",
    "Dessert",
    "Vegetarian",
]

CATEGORY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

# ── Wine harmonize token -> category(ies) ────────────────────────────────────
# Keys are the 66 distinct tokens found in wines.harmonize_csv.
# Value None means "intentionally unmapped" (a course/dish-type, not a category).
HARMONIZE_TO_CATEGORY: dict[str, list[str] | None] = {
    # red meats
    "Beef":            ["Red Meat"],
    "Lamb":            ["Red Meat"],
    "Veal":            ["Red Meat"],
    "Game Meat":       ["Red Meat"],
    "Meat":            ["Red Meat"],
    "Roast":           ["Red Meat"],
    "Cured Meat":      ["Red Meat", "Salty Snack"],
    "Cold Cuts":       ["Red Meat", "Salty Snack"],
    "Ham":             ["Pork", "Salty Snack"],
    # poultry
    "Chicken":         ["Poultry"],
    "Duck":            ["Poultry"],
    "Poultry":         ["Poultry"],
    "Curry Chicken":   ["Poultry", "Spicy"],
    # pork
    "Pork":            ["Pork"],
    # seafood
    "Fish":            ["Seafood"],
    "Lean Fish":       ["Seafood"],
    "Rich Fish":       ["Seafood"],
    "Codfish":         ["Seafood"],
    "Shellfish":       ["Seafood"],
    "Seafood":         ["Seafood"],
    "Sashimi":         ["Seafood"],
    "Sushi":           ["Seafood"],
    "Paella":          ["Seafood"],
    # cheese
    "Cheese":          ["Cheese"],
    "Blue Cheese":     ["Cheese"],
    "Goat Cheese":     ["Cheese"],
    "Hard Cheese":     ["Cheese"],
    "Soft Cheese":     ["Cheese"],
    "Mild Cheese":     ["Cheese"],
    "Maturated Cheese": ["Cheese"],
    "Medium-cured Cheese": ["Cheese"],
    "Eggplant Parmigiana": ["Cheese", "Vegetarian"],
    # creamy / rich
    "Cream":           ["Creamy"],
    "Risotto":         ["Creamy"],
    "Soufflé":         ["Creamy"],
    "Light Stews":     ["Creamy"],
    # spicy
    "Spicy Food":      ["Spicy"],
    "Curry":           ["Spicy"],
    "Asian Food":      ["Spicy"],
    "Yakissoba":       ["Spicy"],
    # smoky / grilled
    "Barbecue":        ["Smoky BBQ"],
    "Grilled":         ["Smoky BBQ"],
    # salty snacks
    "French Fries":    ["Salty Snack"],
    "Baked Potato":    ["Salty Snack"],
    "Snack":           ["Salty Snack"],
    "Beans":           ["Salty Snack", "Vegetarian"],
    "Chestnut":        ["Salty Snack"],
    # dessert / sweet
    "Dessert":         ["Dessert"],
    "Sweet Dessert":   ["Dessert"],
    "Fruit Dessert":   ["Dessert"],
    "Citric Dessert":  ["Dessert", "Acidic"],
    "Cake":            ["Dessert"],
    "Cookies":         ["Dessert"],
    "Chocolate":       ["Dessert"],
    "Spiced Fruit Cake": ["Dessert"],
    "Dried Fruits":    ["Dessert"],
    "Fruit":           ["Dessert"],
    # acidic / tomato
    "Tomato Dishes":   ["Acidic"],
    # vegetarian
    "Vegetarian":      ["Vegetarian"],
    "Salad":           ["Vegetarian"],
    "Mushrooms":       ["Vegetarian"],
    # pasta / pizza family: cheese-forward, often a creamy/rich base
    "Pasta":           ["Cheese", "Creamy"],
    "Pizza":           ["Cheese", "Creamy"],
    "Lasagna":         ["Cheese", "Creamy"],
    "Tagliatelle":     ["Cheese", "Creamy"],
    # ── intentionally unmapped: course / serving-occasion, not a sensory category ──
    "Aperitif":        None,
    "Appetizer":       None,
}


def _distinct_harmonize_tokens() -> list[str]:
    """Pull the live set of harmonize tokens from the wine DB."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db.database import SessionLocal
    from backend.db.models import Wine

    db = SessionLocal()
    try:
        toks: set[str] = set()
        for (h,) in db.query(Wine.harmonize_csv).filter(Wine.harmonize_csv.isnot(None)):
            toks.update(t.strip() for t in (h or "").split(",") if t.strip())
        return sorted(toks)
    finally:
        db.close()


def report() -> None:
    """Coverage report: which live tokens are mapped / unmapped / unknown."""
    tokens = _distinct_harmonize_tokens()
    mapped, unmapped, unknown = [], [], []
    for t in tokens:
        if t not in HARMONIZE_TO_CATEGORY:
            unknown.append(t)                       # token in DB, missing from map
        elif HARMONIZE_TO_CATEGORY[t] is None:
            unmapped.append(t)                      # deliberately not a category
        else:
            mapped.append(t)

    print(f"{len(CATEGORIES)} categories: {CATEGORIES}\n")
    print(f"live harmonize tokens: {len(tokens)}")
    print(f"  mapped to >=1 category : {len(mapped)}")
    print(f"  intentionally unmapped : {len(unmapped)}  {unmapped}")
    print(f"  UNKNOWN (fix the map)  : {len(unknown)}  {unknown}")

    if unknown:
        print("\n!! Add the UNKNOWN tokens above to HARMONIZE_TO_CATEGORY.")
    else:
        print("\nOK: every live token is accounted for.")


if __name__ == "__main__":
    report()
