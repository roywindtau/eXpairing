"""
seed_wines.py
-------------
Seeds the wines table from data/wine/clean_wines.csv.

Run AFTER data/wine/clean_wines.py:
    python -m backend.db.wine.seed_wines [--limit 5000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import Wine

CLEAN_WINES_PATH = Path("data/wine/clean_wines.csv")
BATCH_SIZE = 5_000


def seed(limit: int = 0) -> None:
    init_db()
    db = SessionLocal()

    try:
        existing = db.query(Wine).count()
        if existing > 0:
            print(f"Wines table already has {existing} rows. Skipping.")
            print("To re-seed, run python -m backend.db.reset_wines first.")
            return

        if not CLEAN_WINES_PATH.exists():
            print(f"ERROR: {CLEAN_WINES_PATH} not found.")
            print("Run: python -m data.wine.clean_wines")
            return

        df = pd.read_csv(CLEAN_WINES_PATH)
        if limit:
            df = df.head(limit)

        print(f"Seeding {len(df):,} wines...")

        batch: list[Wine] = []
        total = 0
        for _, row in df.iterrows():
            batch.append(Wine(
                id=int(row["id"]),
                name=str(row["name"]),
                producer=row.get("producer") or None,
                country=row.get("country") or None,
                style=row.get("style") or None,
                abv=float(row["abv"]) if pd.notna(row.get("abv")) else None,
                avg_rating=None,      # computed after seeding ratings
                n_ratings=0,          # updated by train pipeline
                harmonize_csv=row.get("harmonize_csv") or None,
                grapes_csv=row.get("grapes_csv") or None,
                body=row.get("body") or None,
                acidity=row.get("acidity") or None,
                region=row.get("region") or None,
            ))
            total += 1
            if len(batch) >= BATCH_SIZE:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
                print(f"  Inserted {total:,} wines...", end="\r")

        if batch:
            db.bulk_save_objects(batch)
            db.commit()

        print(f"\nDone. {total:,} wines seeded.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max wines to seed (0 = all).")
    args = parser.parse_args()
    seed(limit=args.limit)
