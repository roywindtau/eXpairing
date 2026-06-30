"""
seed_wines.py
-------------
Seeds the wines table from data/wine/clean_wines.csv.

If that full cleaned CSV isn't present (e.g. a fresh clone without the X-Wines
download), falls back to the committed 100-wine sample so the demo still works
with zero downloads.

Run AFTER data/wine/clean_wines.py (or rely on the sample):
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

CLEAN_WINES_PATH  = Path("data/wine/clean_wines.csv")
SAMPLE_WINES_PATH = Path("data/wine/clean_wines.sample.csv")
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

        path = CLEAN_WINES_PATH if CLEAN_WINES_PATH.exists() else SAMPLE_WINES_PATH
        if not path.exists():
            print(f"ERROR: no wine CSV found ({CLEAN_WINES_PATH} or {SAMPLE_WINES_PATH}).")
            print("Run: python -m data.wine.clean_wines  (or restore the committed sample)")
            return
        if path == SAMPLE_WINES_PATH:
            print(f"Using committed sample ({SAMPLE_WINES_PATH}) — demo catalog, no download needed.")

        df = pd.read_csv(path)
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
                # popularity stats: present in the sample so cold-start ranking is
                # meaningful out of the box; the full CSV omits them and they're
                # filled later by compute_wine_stats from clean_ratings.csv.
                avg_rating=float(row["avg_rating"]) if pd.notna(row.get("avg_rating")) else None,
                n_ratings=int(row["n_ratings"]) if pd.notna(row.get("n_ratings")) else 0,
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
