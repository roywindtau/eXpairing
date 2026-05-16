"""
flavor_bridge.py
----------------
Bridges recipe-side vocabulary (ingredients, tags) to drink-side vocabulary
so the TF-IDF cosine similarity in train_drink_cb.py / serve_drink_cb.py has
real token overlap to work with.

Without this module, "shrimp scampi" and a wine harmonized with "Seafood"
would have ~zero shared tokens and content-based pairing would collapse.

Output vocabulary is intentionally chosen to overlap with what's already in
the drink documents (Step 4 builds the docs from these sources):
  - X-Wines Harmonize values: Beef, Lamb, Fish, Seafood, Poultry, Pasta,
    Cheese, Spicy, Vegetarian, Dessert, ...
  - Wine types:               red, white, rose, sparkling, dessert
  - Wine body / acidity:      full, medium, light, crisp
  - Beer styles:              ipa, stout, lager, pilsner, ale, wheat, sour

The lexicon is hand-curated and deliberately small. Extend it when you see
specific pairings going sideways during eval.
"""

from __future__ import annotations

import re
from typing import Iterable

# ── lexicon ──────────────────────────────────────────────────────────────
# Map single-word ingredient tokens to lists of drink-side flavor tokens.
# Bridge tokens are repeated as needed so that an ingredient triggering
# both "seafood" and "white" doesn't lose either signal when TF-IDF later
# weights them.

INGREDIENT_FLAVORS: dict[str, list[str]] = {
    # ── proteins: red meat ─────────────────────────────────────────────
    "beef":     ["beef", "red", "full", "bold"],
    "steak":    ["beef", "red", "full", "bold"],
    "lamb":     ["lamb", "red", "full"],
    "venison":  ["red", "full", "bold"],
    "pork":     ["pork", "medium", "red"],
    "bacon":    ["pork", "savory", "smoky"],
    "ham":      ["pork", "savory"],
    "sausage":  ["pork", "savory", "spicy"],

    # ── proteins: poultry ──────────────────────────────────────────────
    "chicken":  ["poultry", "white", "medium", "light"],
    "turkey":   ["poultry", "white", "medium"],
    "duck":     ["poultry", "red", "medium"],

    # ── proteins: seafood ──────────────────────────────────────────────
    "fish":     ["fish", "seafood", "white", "light", "crisp"],
    "salmon":   ["fish", "seafood", "white", "medium"],
    "tuna":     ["fish", "seafood", "white", "medium"],
    "cod":      ["fish", "seafood", "white", "light"],
    "shrimp":   ["seafood", "shellfish", "white", "light", "crisp"],
    "prawn":    ["seafood", "shellfish", "white", "light"],
    "lobster":  ["seafood", "shellfish", "white", "rich"],
    "crab":     ["seafood", "shellfish", "white", "light"],
    "scallop":  ["seafood", "shellfish", "white", "light"],
    "oyster":   ["seafood", "shellfish", "white", "crisp"],
    "mussel":   ["seafood", "shellfish", "white"],

    # ── eggs / dairy ───────────────────────────────────────────────────
    "egg":      ["light", "white"],
    "eggs":     ["light", "white"],
    "cheese":   ["cheese", "white", "rich"],
    "parmesan": ["cheese", "italian", "savory"],
    "mozzarella": ["cheese", "italian"],
    "cream":    ["rich", "white", "medium"],
    "butter":   ["rich", "white"],

    # ── starches / pasta / bread ───────────────────────────────────────
    "pasta":    ["pasta", "italian", "red", "medium"],
    "noodle":   ["pasta", "asian", "light"],
    "rice":     ["rice", "light", "white"],
    "bread":    ["light"],
    "potato":   ["medium", "light"],

    # ── vegetables / aromatics ─────────────────────────────────────────
    "tomato":   ["red", "acidic", "italian"],
    "mushroom": ["earthy", "red", "medium"],
    "garlic":   ["savory", "bold"],
    "onion":    ["savory"],
    "spinach":  ["vegetarian", "light"],
    "salad":    ["vegetarian", "salads", "white", "crisp", "light"],

    # ── spices / heat ──────────────────────────────────────────────────
    "chili":    ["spicy", "ipa", "lager", "bold"],
    "chile":    ["spicy", "ipa", "lager", "bold"],
    "pepper":   ["spicy", "bold"],
    "jalapeno": ["spicy", "ipa", "lager"],
    "curry":    ["spicy", "indian", "ipa"],
    "cumin":    ["spicy", "savory"],
    "paprika":  ["spicy"],
    "ginger":   ["spicy", "asian", "crisp"],
    "cinnamon": ["sweet", "dessert"],

    # ── sweets / desserts ──────────────────────────────────────────────
    "chocolate": ["dessert", "sweet", "stout", "rich"],
    "vanilla":  ["dessert", "sweet"],
    "sugar":    ["sweet"],
    "honey":    ["sweet"],
    "caramel":  ["dessert", "sweet"],
    "fruit":    ["sweet", "white", "rose"],
    "berry":    ["sweet", "rose"],
    "lemon":    ["acidic", "crisp", "white"],
    "lime":     ["acidic", "crisp"],
    "vinegar":  ["acidic"],
}

# ── tags that already match drink vocabulary (no bridge needed) ─────────
# Food.com tags like "italian", "mexican" already exist in (or near) the
# drink vocabulary — we just pass them through.
KNOWN_CUISINE_TAGS = frozenset({
    "italian", "french", "mexican", "spanish", "indian", "asian",
    "chinese", "japanese", "thai", "mediterranean", "american",
    "spicy", "vegetarian", "vegan", "dessert", "appetizer",
    "appetizers", "salads", "seafood",
})

_TOKEN_RE = re.compile(r"[a-z][a-z']+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + alphabetic-only tokens; preserves repetition."""
    return _TOKEN_RE.findall((text or "").lower())


def bridge_ingredients(ingredients_csv: str) -> list[str]:
    """
    Apply the INGREDIENT_FLAVORS lookup to each word in each comma-separated
    ingredient. Returns the flat list of bridged flavor tokens (with repetition).

    Substring match: "boneless chicken breast" splits into
    ["boneless", "chicken", "breast"] and matches "chicken" in the lexicon.
    """
    if not ingredients_csv:
        return []

    out: list[str] = []
    for ing in ingredients_csv.split(","):
        for word in _tokenize(ing):
            mapping = INGREDIENT_FLAVORS.get(word)
            if mapping:
                out.extend(mapping)
    return out


def bridge_tags(tags_csv: str) -> list[str]:
    """Pass through any tag that already overlaps the drink vocabulary."""
    if not tags_csv:
        return []
    return [
        t.lower().strip()
        for t in tags_csv.split(",")
        if t.lower().strip() in KNOWN_CUISINE_TAGS
    ]


def bridge_recipe_doc(recipe) -> str:
    """
    Produce the lowercased, space-separated augmented document for a recipe,
    ready to be fed to a TfidfVectorizer alongside drink documents.

    Composition:
        original ingredient words + relevant tag words + bridged flavor tokens

    `recipe` is duck-typed: anything with `.ingredients_csv` and `.tags_csv`
    attributes works (Recipe ORM object, or a SimpleNamespace from tests).
    """
    ingredients_csv = getattr(recipe, "ingredients_csv", "") or ""
    tags_csv        = getattr(recipe, "tags_csv", "")        or ""

    parts: list[str] = []
    parts.extend(_tokenize(ingredients_csv.replace(",", " ")))
    parts.extend(bridge_tags(tags_csv))
    parts.extend(bridge_ingredients(ingredients_csv))
    return " ".join(parts)


def bridge_text(text: str, also_tags: Iterable[str] | None = None) -> str:
    """
    Convenience for callers that don't have a Recipe ORM object — e.g. the
    standalone Path-B "For You" flow that aggregates a user's history into
    one big ingredient string.
    """
    parts = _tokenize(text or "")
    parts.extend(bridge_ingredients(text or ""))
    if also_tags:
        for tag in also_tags:
            if tag and tag.lower() in KNOWN_CUISINE_TAGS:
                parts.append(tag.lower())
    return " ".join(parts)
