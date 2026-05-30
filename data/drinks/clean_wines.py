"""
clean_wines.py
--------------
Cleaning pipeline for the X-Wines Full dataset.

Reads raw CSVs, applies cleaning rules documented in wine_data_quality_report.md,
and writes two clean CSVs ready for seeding:
    data/drinks/clean_wines.csv
    data/drinks/clean_ratings.csv

Run from project root:
    python -m data.drinks.clean_wines

Raw files are never modified (thumb rule: raw data is immutable).
"""

import ast
from pathlib import Path

import pandas as pd
from tqdm import tqdm  # pip install tqdm

DATA_DIR = Path(__file__).resolve().parent

RAW_WINES_PATH   = DATA_DIR / "XWines_Full_100K_wines.csv"
RAW_RATINGS_PATH = DATA_DIR / "XWines_Full_21M_ratings.csv"

CLEAN_WINES_PATH   = DATA_DIR / "clean_wines.csv"
CLEAN_RATINGS_PATH = DATA_DIR / "clean_ratings.csv"

# Minimum ratings a user must have to be included in CF training.
# Profiling showed the dataset is pre-filtered to >=5, but we set this
# explicitly so the rule is visible and easy to raise later.
MIN_RATINGS_PER_USER = 5

# Chunk size for streaming the 1GB ratings file.
# 500k rows ≈ ~50MB in memory — safe on a laptop, fast enough.
CHUNK_SIZE = 500_000


# ── stage 1: clean the wines catalog ─────────────────────────────────────────

def _parse_list_column(value: str) -> list[str]:
    """Parse a stringified Python list into a real list.

    The X-Wines CSV stores list columns as literal Python syntax:
    "['Beef', 'Lamb']". ast.literal_eval is the safe way to parse this —
    unlike eval() it only handles literals, not arbitrary code.
    Falls back to empty list if the value is malformed.
    """
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def clean_wines(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to the wines catalog."""

    # Drop Website — not used in recommendations, 18k nulls (from report)
    df = df.drop(columns=["Website"])

    # Parse list columns from strings into real Python lists
    for col in ["Harmonize", "Grapes", "Vintages"]:
        df[col] = df[col].apply(_parse_list_column)

    # Convert parsed lists to comma-separated strings.
    # Storing as strings keeps the clean CSV simple and format-agnostic —
    # the seeder will decide how to map these to DB columns.
    df["Harmonize"] = df["Harmonize"].apply(lambda x: ",".join(x))
    df["Grapes"]    = df["Grapes"].apply(lambda x: ",".join(x))

    # Drop columns not useful downstream — decided in wine_data_quality_report.md
    df = df.drop(columns=[
        "Elaborate", "Code",    # not used by any model or endpoint
        "RegionID", "WineryID", # X-Wines internal IDs, RegionName/WineryName are enough
        "Vintages",             # not used in recommendations
    ])

    return df


# ── stage 2: clean the ratings ────────────────────────────────────────────────

def clean_ratings() -> pd.DataFrame:
    """Stream the 1GB ratings file, apply cleaning rules, return clean DataFrame.

    Streaming (chunksize) is required here — loading 1GB into pandas
    would expand to ~3-5GB in RAM. We accumulate only what we need.
    """
    print(f"Streaming {RAW_RATINGS_PATH} in {CHUNK_SIZE:,}-row chunks...")

    chunks = []
    rows_read = 0

    total_chunks = 21_013_536 // CHUNK_SIZE + 1  # approx, from profiling
    for chunk in tqdm(pd.read_csv(RAW_RATINGS_PATH, chunksize=CHUNK_SIZE,
                                  dtype={"UserID": "int32", "WineID": "int32",
                                         "Rating": "float32"}),
                      total=total_chunks, unit="chunk", desc="ratings"):
        # dtype specified explicitly — prevents pandas from guessing wrong
        # types, and int32/float32 vs int64/float64 halves memory usage.

        rows_read += len(chunk)

        # Drop rows with missing ratings — can't train on them
        chunk = chunk.dropna(subset=["Rating"])

        # Clamp ratings to valid range [1, 5].
        # X-Wines uses a 1-5 scale; anything outside is a data error.
        chunk = chunk[(chunk["Rating"] >= 1) & (chunk["Rating"] <= 5)]

        # Keep only columns needed for CF training and seeding
        chunk = chunk[["UserID", "WineID", "Rating", "Date"]]

        chunks.append(chunk)

        if rows_read % 5_000_000 == 0:
            print(f"  ... {rows_read:,} rows read")

    print(f"  Done. {rows_read:,} rows read.")

    ratings = pd.concat(chunks, ignore_index=True)

    # Filter out users below the minimum rating threshold.
    # Users with too few ratings don't provide enough signal for CF.
    # Profiling showed the dataset is already pre-filtered to >=5,
    # but we enforce this explicitly so it's visible and adjustable.
    user_counts = ratings["UserID"].value_counts()
    valid_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    before = len(ratings)
    ratings = ratings[ratings["UserID"].isin(valid_users)]
    print(f"  Filtered users <{MIN_RATINGS_PER_USER} ratings: {before - len(ratings):,} rows dropped.")

    return ratings


# ── stage 3: validate ────────────────────────────────────────────────────────

def validate(wines: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Assert invariants that must hold after cleaning.

    Validation is not optional — cleaning code has bugs like any code.
    These assertions catch silent errors before they corrupt the DB.
    """
    assert wines["WineID"].is_unique, "Duplicate WineIDs in catalog"
    assert wines["WineID"].notna().all(), "Null WineIDs in catalog"
    assert wines["Harmonize"].notna().all(), "Null Harmonize — food pairing data lost"
    assert (wines["Harmonize"] != "").all(), "Empty Harmonize — expected all wines to have pairings"
    assert ratings["Rating"].between(1, 5).all(), "Ratings outside [1,5] slipped through"
    assert ratings["UserID"].notna().all(), "Null UserIDs in ratings"
    assert ratings["WineID"].notna().all(), "Null WineIDs in ratings"

    # Every rated wine must exist in the catalog
    orphan_wines = set(ratings["WineID"].unique()) - set(wines["WineID"].unique())
    assert not orphan_wines, f"{len(orphan_wines)} rated wines not in catalog"

    print("Validation passed.")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Stage 1: Clean wines catalog ===")
    raw_wines = pd.read_csv(RAW_WINES_PATH)
    print(f"  Loaded {len(raw_wines):,} wines.")
    wines = clean_wines(raw_wines)
    print(f"  After cleaning: {len(wines):,} wines, {len(wines.columns)} columns.")

    print("\n=== Stage 2: Clean ratings ===")
    ratings = clean_ratings()
    print(f"  After cleaning: {len(ratings):,} ratings, {ratings['UserID'].nunique():,} users.")

    print("\n=== Stage 3: Validate ===")
    validate(wines, ratings)

    print("\n=== Writing clean files ===")
    wines.to_csv(CLEAN_WINES_PATH, index=False)
    ratings.to_csv(CLEAN_RATINGS_PATH, index=False)
    print(f"  {CLEAN_WINES_PATH}")
    print(f"  {CLEAN_RATINGS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
