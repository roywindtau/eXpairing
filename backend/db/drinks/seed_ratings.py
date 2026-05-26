"""
seed_ratings.py
---------------
Loads beer rating events into DrinkEvent, and creates a User row per
unique external user (mirrors seed_ratings.py for Food.com).

Expected input (download via data/drinks/download_beer.py):
    data/beer_reviews.csv     -- ~1.58M rows; user = review_profilename (string)

User ID mapping (offsets chosen to avoid clashes with other domains):
    Food.com (recipe ratings):   foodcom_user_id + 1_000
    Beer (this file):            idx_in_profilename_table + 100_000

Wine ratings are intentionally not seeded here yet — the wine-data branch
is choosing the new source(s). The 200_000 offset slot is reserved for
whichever wine dataset lands.

Run AFTER seed_drinks.py:
    python -m backend.db.drinks.seed_ratings [--limit 200000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import User, Drink, DrinkEvent

BEER_CSV = Path("data/beer_reviews.csv")

BEER_USER_OFFSET = 100_000
BATCH_SIZE       = 10_000


def _safe_float(s: str) -> float | None:
    try:
        v = float(s)
        return v if v == v else None
    except (ValueError, TypeError):
        return None


# ── beer ratings ─────────────────────────────────────────────────────────

def _seed_beer_events(db, valid_beer_ids: set[int], limit: int) -> tuple[int, int]:
    """Single pass: collect profilenames on the fly + insert events in batches."""
    if not BEER_CSV.exists():
        print(f"  WARN: {BEER_CSV} not found, skipping beer events.")
        return 0, 0

    profile_to_idx: dict[str, int] = {}
    user_batch:  list[User]       = []
    event_batch: list[DrinkEvent] = []
    rating_sum:  dict[int, float] = {}
    rating_n:    dict[int, int]   = {}

    total = 0
    skipped = 0
    print(f"  Streaming {BEER_CSV} ...")
    with open(BEER_CSV, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and total >= limit:
                break

            profile = (row.get("review_profilename") or "").strip()
            if not profile:
                skipped += 1
                continue

            try:
                beer_id = int(row["beer_beerid"])
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue

            if beer_id not in valid_beer_ids:
                skipped += 1
                continue

            rating = _safe_float(row.get("review_overall", ""))
            if rating is None or rating <= 0 or rating > 5:
                skipped += 1
                continue

            if profile not in profile_to_idx:
                idx = len(profile_to_idx)
                profile_to_idx[profile] = idx
                user_batch.append(User(id=idx + BEER_USER_OFFSET, beta=0.35))
                if len(user_batch) >= BATCH_SIZE:
                    db.bulk_save_objects(user_batch)
                    db.commit()
                    user_batch = []

            app_user_id = profile_to_idx[profile] + BEER_USER_OFFSET

            event_batch.append(DrinkEvent(
                user_id=app_user_id,
                drink_id=beer_id,
                event_type="rate",
                rating=rating,
                synthetic=False,
            ))
            rating_sum[beer_id] = rating_sum.get(beer_id, 0.0) + rating
            rating_n[beer_id]   = rating_n.get(beer_id,   0)   + 1
            total += 1

            if len(event_batch) >= BATCH_SIZE:
                db.bulk_save_objects(event_batch)
                db.commit()
                event_batch = []
                print(f"    Inserted {total:,} beer events ({len(profile_to_idx):,} users) ...", end="\r")

    if user_batch:
        db.bulk_save_objects(user_batch)
        db.commit()
    if event_batch:
        db.bulk_save_objects(event_batch)
        db.commit()

    # Refresh Drink.avg_rating / n_ratings from the freshly inserted events.
    # (seed_drinks.py already populated these from the same source, but a
    # --limit run produces a partial set, so we re-sync.)
    print(f"\n  Updating beer avg_rating / n_ratings ...")
    updates = []
    for beer_id, n in rating_n.items():
        updates.append({
            "id": beer_id,
            "avg_rating": round(rating_sum[beer_id] / n, 3),
            "n_ratings":  n,
        })
        if len(updates) >= BATCH_SIZE:
            db.bulk_update_mappings(Drink, updates)
            db.commit()
            updates = []
    if updates:
        db.bulk_update_mappings(Drink, updates)
        db.commit()

    print(f"  Done. {total:,} beer events, {len(profile_to_idx):,} users created, {skipped:,} skipped.")
    return total, len(profile_to_idx)


# ── entrypoint ───────────────────────────────────────────────────────────

def seed(limit: int = 0) -> None:
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(DrinkEvent).filter(DrinkEvent.event_type == "rate").count()
        if existing > 0:
            print(f"DrinkEvent already has {existing:,} rate rows. Skipping.")
            return

        print("Loading valid beer IDs from DB ...")
        valid_beer_ids = {d[0] for d in db.query(Drink.id).filter(Drink.kind == "beer").all()}
        print(f"  {len(valid_beer_ids):,} beers.")

        if not valid_beer_ids:
            print("No beers in DB. Run `python -m backend.db.drinks.seed_drinks` first.")
            sys.exit(1)

        print("\nSeeding beer events ...")
        n_beer_events, n_beer_users = _seed_beer_events(db, valid_beer_ids, limit)

        print(f"\nTotal: {n_beer_events:,} events from {n_beer_users:,} external users.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max beer review rows to load (0 = all).")
    args = parser.parse_args()
    seed(limit=args.limit)
