"""
seed_ratings.py
---------------
Loads Food.com user interactions into UserEvent and updates
Recipe.avg_rating / Recipe.n_ratings.

Expected input:
    data/RAW_interactions.csv
    columns: user_id, recipe_id, date, rating, review

Also creates a User row for each unique Food.com user_id so the CF
model has real user->recipe rating triples to train on.

Run AFTER seed_recipes.py:
    python -m backend.db.seed_ratings [--limit 200000]
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import User, Recipe, UserEvent

INTERACTIONS_CSV = Path("data/RAW_interactions.csv")
BATCH_SIZE = 10_000


def seed(limit: int = 0) -> None:
    if not INTERACTIONS_CSV.exists():
        print(f"ERROR: {INTERACTIONS_CSV} not found.")
        print("Run:  python data/download_foodcom.py  first.")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    try:
        # Only treat the Food.com seed as "already done" if Food.com rows exist.
        # Food.com users are offset by USER_ID_OFFSET (1000); app/dev users live
        # below that, so their rating events must not block this seed.
        existing = (
            db.query(UserEvent)
            .filter(UserEvent.event_type == "rate", UserEvent.user_id >= 1000)
            .count()
        )
        if existing > 0:
            print(f"UserEvent table already has {existing} Food.com ratings. Skipping.")
            return

        # Build set of valid recipe ids already in DB
        print("Loading valid recipe IDs ...")
        valid_recipe_ids = {r[0] for r in db.query(Recipe.id).all()}
        print(f"  {len(valid_recipe_ids):,} recipes in DB.")

        # Track which Food.com user IDs we've already created User rows for
        # Food.com user IDs start from 1 — offset by 1000 to avoid clashing
        # with our app's demo users (id=1, id=2)
        USER_ID_OFFSET = 1000
        seen_users: set[int] = set()

        # Accumulate ratings per recipe for avg computation
        recipe_ratings: dict[int, list[float]] = defaultdict(list)

        print(f"Loading {INTERACTIONS_CSV} ...")
        batch   = []
        total   = 0
        skipped = 0

        with open(INTERACTIONS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and total >= limit:
                    break

                try:
                    foodcom_user_id = int(row["user_id"])
                    recipe_id       = int(row["recipe_id"])
                    rating          = float(row["rating"])
                except (ValueError, KeyError):
                    skipped += 1
                    continue

                if recipe_id not in valid_recipe_ids:
                    skipped += 1
                    continue

                if rating < 1 or rating > 5:
                    skipped += 1
                    continue

                app_user_id = foodcom_user_id + USER_ID_OFFSET

                # Create User row on first encounter
                if foodcom_user_id not in seen_users:
                    db.add(User(id=app_user_id, beta=0.35))
                    seen_users.add(foodcom_user_id)

                batch.append(UserEvent(
                    user_id=app_user_id,
                    recipe_id=recipe_id,
                    event_type="rate",
                    rating=rating,
                ))
                recipe_ratings[recipe_id].append(rating)
                total += 1

                if len(batch) >= BATCH_SIZE:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
                    print(f"  Inserted {total:,} ratings ...", end="\r")

        if batch:
            db.bulk_save_objects(batch)
            db.commit()

        # Update avg_rating and n_ratings on each Recipe
        print(f"\nUpdating recipe average ratings ...")
        update_batch = []
        for recipe_id, ratings in recipe_ratings.items():
            avg = round(sum(ratings) / len(ratings), 2)
            update_batch.append({"id": recipe_id,
                                  "avg_rating": avg,
                                  "n_ratings": len(ratings)})
            if len(update_batch) >= BATCH_SIZE:
                db.bulk_update_mappings(Recipe, update_batch)
                db.commit()
                update_batch = []
        if update_batch:
            db.bulk_update_mappings(Recipe, update_batch)
            db.commit()

        print(f"Done. Inserted {total:,} ratings. "
              f"Created {len(seen_users):,} users. "
              f"Skipped {skipped:,}.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max interactions to load (0 = all)")
    args = parser.parse_args()
    seed(limit=args.limit)
