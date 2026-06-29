"""
compute_wine_stats.py
---------------------
Populates wines.avg_rating and wines.n_ratings by aggregating the raw
wine ratings in data/wine/clean_ratings.csv.

These two columns drive the popularity prior used by the cold-start
recommender path (serve_cf.bayesian_popularity) — i.e. what a brand-new
user with no wine history is ranked by. seed_wines.py inserts them as
NULL/0; this script fills them in after seeding.

Run AFTER backend.db.wine.seed_wines:
    python -m backend.db.wine.compute_wine_stats

The ratings file is large (~21M rows), so it is streamed in chunks and
aggregated incrementally rather than loaded into memory at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.db.database import SessionLocal, engine
from backend.db.models import Wine

CLEAN_RATINGS_PATH = Path("data/wine/clean_ratings.csv")
CHUNK_SIZE = 2_000_000


def compute() -> dict[int, tuple[float, int]]:
    """Stream the ratings CSV and return {wine_id: (avg_rating, n_ratings)}."""
    if not CLEAN_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"{CLEAN_RATINGS_PATH} not found. Run data/wine/clean_wines.py first."
        )

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    total = 0
    for chunk in pd.read_csv(
        CLEAN_RATINGS_PATH,
        usecols=["wine_id", "rating"],
        dtype={"wine_id": "int64", "rating": "float64"},
        chunksize=CHUNK_SIZE,
    ):
        grp = chunk.groupby("wine_id")["rating"].agg(["sum", "count"])
        for wid, row in grp.iterrows():
            sums[wid] = sums.get(wid, 0.0) + float(row["sum"])
            counts[wid] = counts.get(wid, 0) + int(row["count"])
        total += len(chunk)
        print(f"  processed {total:,} ratings...", end="\r")

    print(f"\nAggregated {total:,} ratings across {len(counts):,} wines.")
    return {
        wid: (round(sums[wid] / counts[wid], 4), counts[wid])
        for wid in counts
    }


def write_back(stats: dict[int, tuple[float, int]]) -> None:
    """Bulk-update the wines table with computed avg_rating + n_ratings."""
    db = SessionLocal()
    try:
        updates = [
            {"id": wid, "avg_rating": avg, "n_ratings": n}
            for wid, (avg, n) in stats.items()
        ]
        # bulk_update_mappings is the fast path for many single-row updates.
        BATCH = 10_000
        for i in range(0, len(updates), BATCH):
            db.bulk_update_mappings(Wine, updates[i:i + BATCH])
            db.commit()
            print(f"  updated {min(i + BATCH, len(updates)):,} wines...", end="\r")
        print(f"\nWrote stats for {len(updates):,} wines.")
    finally:
        db.close()


def main() -> None:
    stats = compute()
    write_back(stats)

    # Quick sanity readout
    with engine.connect() as conn:
        from sqlalchemy import text
        rated = conn.execute(
            text("SELECT count(*) FROM wines WHERE n_ratings > 0")
        ).scalar()
        top = conn.execute(text(
            "SELECT name, avg_rating, n_ratings FROM wines "
            "WHERE n_ratings >= 50 ORDER BY avg_rating DESC LIMIT 3"
        )).fetchall()
    print(f"Done. {rated:,} wines now have ratings.")
    print("Top-rated (n>=50):")
    for name, avg, n in top:
        print(f"  {avg}  ({n:,})  {name}")


if __name__ == "__main__":
    main()
