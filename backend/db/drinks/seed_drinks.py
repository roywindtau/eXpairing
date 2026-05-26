"""
seed_drinks.py
--------------
Loads the beer dataset into the Drink table.

Expected input file (download via data/drinks/download_beer.py):
    data/beer_reviews.csv     -- ~1.58M BeerAdvocate reviews; we group by beer

Per-beer aggregates computed during seed:
    avg_rating, n_ratings, avg_aroma/taste/palate/appearance
    review_tokens_csv: top-N most-frequent non-stopword words from this beer's
                      review text (used by train_cb.py)

Wines are intentionally not seeded here yet — the wine-data branch is
choosing the new source(s). See data/drinks/download_wines.py.

Run AFTER data/drinks/download_beer.py:
    python -m backend.db.drinks.seed_drinks [--limit 5000]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import init_db, SessionLocal
from backend.db.models import Drink

BEER_CSV  = Path("data/beer_reviews.csv")
BATCH_SIZE = 5_000
TOP_REVIEW_TOKENS = 25

# Small in-house stopword list; we don't want a heavy NLP dep
# in the seed pipeline. Words specific to beer reviews ("beer", "head",
# "pour", etc.) are kept because they ARE flavor-relevant context.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is",
    "it", "its", "just", "me", "more", "my", "no", "not", "of", "on", "one",
    "or", "out", "over", "she", "so", "some", "still", "than", "that", "the",
    "their", "them", "then", "there", "they", "this", "to", "too", "up",
    "very", "was", "we", "were", "what", "when", "which", "while", "who",
    "will", "with", "would", "you", "your", "all", "also", "any", "been",
    "can", "do", "does", "down", "even", "get", "got", "how", "i'm", "i've",
    "just", "like", "make", "much", "now", "off", "only", "other", "really",
    "see", "since", "small", "such", "take", "these", "those", "two", "well",
    "where", "did", "go", "good", "way", "us", "im", "ive", "dont", "didnt",
})

# Words longer than this are usually IDs / typos / non-words
MAX_TOKEN_LEN = 20

_TOKEN_RE = re.compile(r"[a-z][a-z']+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphabetic-only, stopword-filtered, length-bounded."""
    if not text:
        return []
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS and len(t) <= MAX_TOKEN_LEN
    ]


def _safe_float(s: str) -> float | None:
    try:
        v = float(s)
        return v if v == v else None  # filter NaN
    except (ValueError, TypeError):
        return None


# ── beer seeding ─────────────────────────────────────────────────────────

def _seed_beers(db, limit: int) -> int:
    """Stream beer_reviews.csv once, aggregate per-beer, bulk insert."""
    if not BEER_CSV.exists():
        print(f"  WARN: {BEER_CSV} not found, skipping beer seed.")
        return 0

    # Per-beer accumulators
    name:        dict[int, str]       = {}
    brewery:     dict[int, str]       = {}
    style:       dict[int, str]       = {}
    abv:         dict[int, float]     = {}
    ratings:     dict[int, list[float]] = defaultdict(list)
    aroma:       dict[int, list[float]] = defaultdict(list)
    taste:       dict[int, list[float]] = defaultdict(list)
    palate:      dict[int, list[float]] = defaultdict(list)
    appearance:  dict[int, list[float]] = defaultdict(list)
    token_counts: dict[int, Counter] = defaultdict(Counter)

    print(f"  Streaming {BEER_CSV} ...")
    rows_seen = 0
    with open(BEER_CSV, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and rows_seen >= limit:
                break
            rows_seen += 1

            try:
                beer_id = int(row["beer_beerid"])
            except (ValueError, KeyError, TypeError):
                continue

            if beer_id not in name:
                name[beer_id]    = (row.get("beer_name") or "").strip() or f"Beer {beer_id}"
                brewery[beer_id] = (row.get("brewery_name") or "").strip() or None
                style[beer_id]   = (row.get("beer_style") or "").strip() or None
                v = _safe_float(row.get("beer_abv", ""))
                if v is not None:
                    abv[beer_id] = v

            overall = _safe_float(row.get("review_overall", ""))
            if overall is not None and 0 <= overall <= 5:
                ratings[beer_id].append(overall)
            for src_key, bucket in (
                ("review_aroma", aroma),
                ("review_taste", taste),
                ("review_palate", palate),
                ("review_appearance", appearance),
            ):
                v = _safe_float(row.get(src_key, ""))
                if v is not None and 0 <= v <= 5:
                    bucket[beer_id].append(v)

            # Note: this CSV has no review_text column (only the numeric aspects
            # + style). We synthesize review_tokens from style words so CB still
            # has signal beyond just kind+style.
            if style.get(beer_id):
                token_counts[beer_id].update(_tokenize(style[beer_id]))

            if rows_seen % 200_000 == 0:
                print(f"    ... read {rows_seen:,} rows, {len(name):,} beers so far")

    print(f"  Aggregated {len(name):,} unique beers from {rows_seen:,} reviews.")

    batch: list[Drink] = []
    total = 0
    for beer_id, beer_name in name.items():
        r = ratings.get(beer_id) or []
        top_tokens = [t for t, _ in token_counts[beer_id].most_common(TOP_REVIEW_TOKENS)]
        batch.append(Drink(
            id=beer_id,
            kind="beer",
            name=beer_name,
            producer=brewery.get(beer_id),
            country=None,
            abv=abv.get(beer_id),
            avg_rating=round(sum(r) / len(r), 3) if r else None,
            n_ratings=len(r),
            review_tokens_csv=",".join(top_tokens) if top_tokens else None,
            style=style.get(beer_id),
            avg_aroma=round(sum(aroma[beer_id]) / len(aroma[beer_id]), 3) if aroma.get(beer_id) else None,
            avg_taste=round(sum(taste[beer_id]) / len(taste[beer_id]), 3) if taste.get(beer_id) else None,
            avg_palate=round(sum(palate[beer_id]) / len(palate[beer_id]), 3) if palate.get(beer_id) else None,
            avg_appearance=round(sum(appearance[beer_id]) / len(appearance[beer_id]), 3) if appearance.get(beer_id) else None,
        ))
        total += 1
        if len(batch) >= BATCH_SIZE:
            db.bulk_save_objects(batch)
            db.commit()
            batch = []
            print(f"    Inserted {total:,} beers ...", end="\r")

    if batch:
        db.bulk_save_objects(batch)
        db.commit()
    print(f"\n  Inserted {total:,} beers.")
    return total


# ── entrypoint ───────────────────────────────────────────────────────────

def seed(limit: int = 0) -> None:
    init_db()
    db = SessionLocal()

    try:
        existing = db.query(Drink).count()
        if existing > 0:
            print(f"Drinks table already has {existing} rows. Skipping.")
            print("To re-seed, truncate the table first.")
            return

        print("Seeding beers ...")
        n_beers = _seed_beers(db, limit)

        print(f"\nDone. {n_beers:,} beers.")

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max beer review rows to read (0 = all).")
    args = parser.parse_args()
    seed(limit=args.limit)
