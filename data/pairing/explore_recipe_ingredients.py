"""
explore_recipe_ingredients.py
=============================
EXPLORATION script (not part of the serving pipeline) used to design Module 3's
ingredient->category lexicon (recipe_categories.py).

WHY THIS EXISTS
---------------
Before writing the signal-ingredient lexicon, we needed to know what the real
Food.com ingredient text actually looks like:
  1. How many DISTINCT ingredient strings are there? (Can we hand-map them? No.)
  2. What are the highest-frequency ingredients? (These are mostly NOISE -- salt,
     butter, water -- which is why exact-match fails and why noise words are
     deliberately absent from the lexicon.)
  3. For each candidate SIGNAL keyword (chicken, salmon, chili...), how many
     recipes would a whole-word match catch? (Confirms substring/keyword matching
     is the right strategy and that the keywords cover real volume.)

This is the evidence behind the design decisions documented in recipe_categories.py.
Re-run it any time the recipe data changes to re-check coverage.

Run:
    python -m data.pairing.explore_recipe_ingredients
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import Recipe

# Candidate signal keywords we probed while building the lexicon. Each is matched
# as a whole word (\b) so "ham" does not match "graham".
PROBE_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "bacon", "ham", "sausage", "turkey", "duck",
    "salmon", "tuna", "shrimp", "fish", "crab", "lobster", "cod", "scallop",
    "clam", "oyster", "anchovy",
    "cheese", "cream", "chili", "jalapeno", "sriracha", "curry", "cayenne",
    "chocolate", "honey", "lemon", "tomato",
]

TOP_N = 60


def _ingredient_counts() -> collections.Counter:
    db = SessionLocal()
    try:
        rows = db.query(Recipe.ingredients_csv).all()
    finally:
        db.close()
    c: collections.Counter = collections.Counter()
    for (csv,) in rows:
        for ing in (csv or "").split(","):
            ing = ing.strip().lower()
            if ing:
                c[ing] += 1
    return c


def main() -> None:
    counts = _ingredient_counts()
    total_recipes = sum(1 for _ in [0])  # placeholder; recomputed below
    print(f"distinct ingredient strings: {len(counts):,}\n")

    print(f"--- top {TOP_N} most common ingredients (mostly NOISE) ---")
    for ing, n in counts.most_common(TOP_N):
        print(f"  {n:7,d}  {ing}")

    print("\n--- signal-keyword coverage (whole-word match across all ingredients) ---")
    # Sum the counts of every distinct ingredient whose text contains the keyword
    # as a whole word. This is the volume each lexicon keyword would capture.
    for kw in PROBE_KEYWORDS:
        pat = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        hits = sum(n for ing, n in counts.items() if pat.search(ing))
        print(f"  {hits:7,d}  *{kw}*")


if __name__ == "__main__":
    main()
