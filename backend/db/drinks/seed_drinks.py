"""
seed_drinks.py
--------------
Seeds the beers table from data/drinks/clean_beer.csv.

Run AFTER data/drinks/clean_beer.py:
    python -m backend.db.drinks.seed_drinks [--limit 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import Beer

CLEAN_BEER_PATH = Path("data/drinks/clean_beer.csv")
BATCH_SIZE = 5_000


def seed(limit: int = 0) -> None:
    init_db()
    db = SessionLocal()

    try:
        existing = db.query(Beer).count()
        if existing > 0:
            print(f"Beers table already has {existing} rows. Skipping.")
            print("To re-seed, run python -m backend.db.reset_drinks first.")
            return

        if not CLEAN_BEER_PATH.exists():
            print(f"ERROR: {CLEAN_BEER_PATH} not found.")
            print("Run: python -m data.drinks.clean_beer")
            return

        df = pd.read_csv(CLEAN_BEER_PATH)
        if limit:
            df = df.head(limit)

        print(f"Seeding {len(df):,} beers...")

        batch: list[Beer] = []
        total = 0
        for _, row in df.iterrows():
            batch.append(Beer(
                id=int(row["id"]),
                name=str(row["name"]),
                producer=row.get("producer") or None,
                country=row.get("country") or None,
                style=row.get("style") or None,
                abv=float(row["abv"]) if pd.notna(row.get("abv")) else None,
                avg_rating=float(row["avg_rating"]) if pd.notna(row.get("avg_rating")) else None,
                n_ratings=int(row["n_ratings"]) if pd.notna(row.get("n_ratings")) else 0,
                harmonize_csv=row.get("harmonize_csv") or None,
                avg_aroma=float(row["avg_aroma"]) if pd.notna(row.get("avg_aroma")) else None,
                avg_taste=float(row["avg_taste"]) if pd.notna(row.get("avg_taste")) else None,
                avg_palate=float(row["avg_palate"]) if pd.notna(row.get("avg_palate")) else None,
                avg_appearance=float(row["avg_appearance"]) if pd.notna(row.get("avg_appearance")) else None,
            ))
            total += 1
            if len(batch) >= BATCH_SIZE:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
                print(f"  Inserted {total:,} beers...", end="\r")

        if batch:
            db.bulk_save_objects(batch)
            db.commit()

        print(f"\nDone. {total:,} beers seeded.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max beers to seed (0 = all).")
    args = parser.parse_args()
    seed(limit=args.limit)
