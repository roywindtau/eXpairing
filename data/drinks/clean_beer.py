"""
clean_beer.py
-------------
Cleaning pipeline for the BeerAdvocate dataset.

Reads raw beer_reviews.csv (one row per review, 1.58M rows), aggregates
per beer, and writes one clean row per beer to clean_beer.csv.

Column names are aligned with clean_wines.csv for shared concepts:
    producer  (← brewery_name)
    style     (← beer_style)
    abv       (← beer_abv)
    avg_rating (← review_overall averaged)

Run from project root:
    python -m data.drinks.clean_beer

Raw file is never modified (raw data is immutable).
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent

RAW_BEER_PATH   = DATA_DIR / "beer_reviews.csv"
CLEAN_BEER_PATH = DATA_DIR / "clean_beer.csv"

CHUNK_SIZE = 200_000  # ~150MB file, smaller chunks than ratings

# Minimum ratings a beer must have to be included.
# Beers with fewer ratings have unreliable avg_rating for CF/CB.
MIN_RATINGS_PER_BEER = 5


# ── stage 1: aggregate reviews into per-beer stats ────────────────────────────

def aggregate_beers() -> pd.DataFrame:
    """Stream raw reviews, aggregate per beer, return one-row-per-beer DataFrame."""

    # Accumulators — keyed by beer_id
    name       = {}
    producer   = {}
    style      = {}
    abv        = {}
    ratings    = defaultdict(list)
    aroma      = defaultdict(list)
    taste      = defaultdict(list)
    palate     = defaultdict(list)
    appearance = defaultdict(list)

    print(f"Streaming {RAW_BEER_PATH} in {CHUNK_SIZE:,}-row chunks...")

    total_rows = 0
    for chunk in tqdm(pd.read_csv(RAW_BEER_PATH, chunksize=CHUNK_SIZE,
                                  dtype={"beer_beerid": "int32",
                                         "beer_abv": "float32",
                                         "review_overall": "float32",
                                         "review_aroma": "float32",
                                         "review_taste": "float32",
                                         "review_palate": "float32",
                                         "review_appearance": "float32"}),
                      desc="beers", unit="chunk"):
        total_rows += len(chunk)

        for _, row in chunk.iterrows():
            bid = row["beer_beerid"]

            if bid not in name:
                name[bid]     = str(row.get("beer_name") or "").strip() or f"Beer {bid}"
                producer[bid] = str(row.get("brewery_name") or "").strip() or None
                style[bid]    = str(row.get("beer_style") or "").strip() or None
                v = row.get("beer_abv")
                if pd.notna(v):
                    abv[bid] = float(v)

            for val, bucket in (
                (row.get("review_overall"),    ratings),
                (row.get("review_aroma"),      aroma),
                (row.get("review_taste"),      taste),
                (row.get("review_palate"),     palate),
                (row.get("review_appearance"), appearance),
            ):
                if pd.notna(val) and 0 <= float(val) <= 5:
                    bucket[bid].append(float(val))

    print(f"  {total_rows:,} reviews → {len(name):,} unique beers")

    # Build one row per beer
    rows = []
    for bid, beer_name in name.items():
        r = ratings[bid]
        if len(r) < MIN_RATINGS_PER_BEER:
            continue
        rows.append({
            "id":             bid,       # canonical — matches wine's "id"
            "name":           beer_name, # canonical — matches wine's "name"
            "producer":       producer.get(bid),
            "style":          style.get(bid),
            "abv":            abv.get(bid),
            "avg_rating":     round(sum(r) / len(r), 3),
            "n_ratings":      len(r),
            "country":        None,      # not in BeerAdvocate data
            "harmonize_csv":  None,      # food pairings — to be added later
            "avg_aroma":      round(sum(aroma[bid]) / len(aroma[bid]), 3) if aroma[bid] else None,
            "avg_taste":      round(sum(taste[bid]) / len(taste[bid]), 3) if taste[bid] else None,
            "avg_palate":     round(sum(palate[bid]) / len(palate[bid]), 3) if palate[bid] else None,
            "avg_appearance": round(sum(appearance[bid]) / len(appearance[bid]), 3) if appearance[bid] else None,
        })

    return pd.DataFrame(rows)


# ── stage 2: validate ─────────────────────────────────────────────────────────

def validate(beers: pd.DataFrame) -> None:
    assert beers["id"].is_unique, "Duplicate BeerIDs"
    assert beers["id"].notna().all(), "Null BeerIDs"
    assert beers["avg_rating"].between(0, 5).all(), "Ratings outside [0,5]"
    assert (beers["n_ratings"] >= MIN_RATINGS_PER_BEER).all(), "Beer below min ratings slipped through"
    print("Validation passed.")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    if not RAW_BEER_PATH.exists():
        print(f"ERROR: {RAW_BEER_PATH} not found.")
        print("Run: python -m data.drinks.download_beer")
        return

    print("=== Stage 1: Aggregate beer reviews ===")
    beers = aggregate_beers()
    print(f"  After cleaning: {len(beers):,} beers")

    print("\n=== Stage 2: Validate ===")
    validate(beers)

    print("\n=== Writing clean file ===")
    beers.to_csv(CLEAN_BEER_PATH, index=False)
    print(f"  {CLEAN_BEER_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
