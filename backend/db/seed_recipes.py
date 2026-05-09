"""
seed_recipes.py
---------------
Loads the Food.com Kaggle dataset into the Recipe table.

Expected input files (download via data/download_foodcom.py):
    data/RAW_recipes.csv     -- recipe metadata + ingredients
    data/RAW_interactions.csv -- user ratings (used by seed_ratings.py)

Column mapping from Food.com CSV:
    id          -> Recipe.id
    name        -> Recipe.name
    ingredients -> Recipe.ingredients_csv  (JSON array -> comma-sep string)
    tags        -> Recipe.tags_csv         (JSON array -> comma-sep string)
    minutes     -> Recipe.minutes
    (avg rating computed from interactions in seed_ratings.py)

Run:
    python -m backend.db.seed_recipes [--limit 50000]

For dev/POC without the full dataset, seed_dev.py is faster.
"""

import argparse
import ast
import csv
import json
import sys
import os
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import Recipe

RECIPES_CSV = Path("data/RAW_recipes.csv")
BATCH_SIZE  = 5_000


def parse_list_field(raw: str) -> list[str]:
    """
    Food.com stores ingredients and tags as Python-literal lists
    e.g.  "['eggs', 'whole milk', 'butter']"
    ast.literal_eval handles this safely.
    """
    try:
        result = ast.literal_eval(raw)
        return [str(x).strip() for x in result if str(x).strip()]
    except Exception:
        return []


def clean_ingredient(name: str) -> str:
    """
    Lowercase and strip quantity hints that sometimes appear in Food.com
    ingredient names e.g. '1 cup whole milk' -> 'whole milk'
    """
    # Remove leading digits / fractions
    parts = name.lower().strip().split()
    cleaned = []
    for p in parts:
        # skip pure numbers and common unit words
        if p in {"1","2","3","4","½","¼","¾","cup","cups","tbsp","tsp",
                 "oz","lb","g","kg","ml","l","tablespoon","teaspoon",
                 "tablespoons","teaspoons","pound","pounds","ounce","ounces"}:
            continue
        cleaned.append(p)
    return " ".join(cleaned) if cleaned else name.lower().strip()


def seed(limit: int = 0) -> None:
    if not RECIPES_CSV.exists():
        print(f"ERROR: {RECIPES_CSV} not found.")
        print("Run:  python data/download_foodcom.py  first.")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    try:
        existing = db.query(Recipe).count()
        if existing > 0:
            print(f"Recipes table already has {existing} rows. Skipping.")
            print("To re-seed, truncate the table first.")
            return

        print(f"Loading {RECIPES_CSV} ...")
        batch   = []
        total   = 0
        skipped = 0

        with open(RECIPES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and total >= limit:
                    break

                # parse ingredients
                ingredients = parse_list_field(row.get("ingredients", "[]"))
                if not ingredients:
                    skipped += 1
                    continue

                # clean and deduplicate ingredient names
                cleaned = list(dict.fromkeys(
                    clean_ingredient(i) for i in ingredients
                ))

                tags = parse_list_field(row.get("tags", "[]"))

                try:
                    minutes = int(float(row.get("minutes", 0) or 0))
                    minutes = min(minutes, 9999)  # cap outliers
                except (ValueError, TypeError):
                    minutes = None

                try:
                    recipe_id = int(row["id"])
                except (ValueError, KeyError):
                    skipped += 1
                    continue

                steps = parse_list_field(row.get("steps", "[]"))
                description = (row.get("description") or "").strip() or None

                batch.append(Recipe(
                    id=recipe_id,
                    name=row.get("name", f"Recipe {recipe_id}").strip(),
                    ingredients_csv=",".join(cleaned),
                    tags_csv=",".join(tags[:20]),  # cap tag count
                    minutes=minutes,
                    n_steps=len(steps) or None,
                    description=description,
                    steps_json=json.dumps(steps) if steps else None,
                    avg_rating=None,
                    n_ratings=0,
                ))
                total += 1

                if len(batch) >= BATCH_SIZE:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
                    print(f"  Inserted {total:,} recipes ...", end="\r")

        if batch:
            db.bulk_save_objects(batch)
            db.commit()

        print(f"\nDone. Inserted {total:,} recipes. Skipped {skipped:,}.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max recipes to load (0 = all)")
    args = parser.parse_args()
    seed(limit=args.limit)
