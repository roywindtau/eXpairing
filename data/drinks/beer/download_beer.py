"""
download_beer.py
----------------
Downloads the beer dataset used by the drink recommender.

Beer Reviews (Kaggle, ~150MB)
    ~1.58M reviews, ~33k users, ~66k beers from BeerAdvocate (1998-2011).
    Aspect ratings: appearance, aroma, palate, taste, overall.
    Kaggle slug: rdoume/beerreviews

Requirements:
    pip install kaggle    (already in requirements.txt)
    Set up ~/.kaggle/kaggle.json with your API key
        (Kaggle > Account > Create New API Token)
    Accept the Beer Reviews terms once at:
        https://www.kaggle.com/datasets/rdoume/beerreviews

Run:
    python -m data.drinks.beer.download_beer

Output file:
    data/beer_reviews.csv     ~150MB, ~1.5M rows
"""

import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR     = Path("data")
BEER_DATASET = "rdoume/beerreviews"
BEER_CSV     = DATA_DIR / "beer_reviews.csv"


def download_beer_reviews() -> None:
    print("Beer Reviews (Kaggle)")
    if BEER_CSV.exists():
        mb = BEER_CSV.stat().st_size / 1_048_576
        print(f"  Already present: {BEER_CSV} ({mb:.0f} MB) — skipping.")
        return

    print(f"  Downloading {BEER_DATASET} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", BEER_DATASET, "-p", str(DATA_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  Kaggle download failed:")
        print(result.stderr)
        print("\n  Make sure you have:")
        print("    1. pip install kaggle")
        print("    2. ~/.kaggle/kaggle.json with your API credentials")
        print(f"    3. Accepted dataset terms at https://www.kaggle.com/datasets/{BEER_DATASET}")
        sys.exit(1)

    zips = sorted(DATA_DIR.glob("beerreviews*.zip"))
    if not zips:
        print("  ERROR: zip file not found after download.")
        sys.exit(1)
    zip_path = zips[0]

    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv")]
        if not csvs:
            print("  ERROR: no CSV inside zip.")
            sys.exit(1)
        z.extract(csvs[0], DATA_DIR)
        extracted = DATA_DIR / csvs[0]
        if extracted != BEER_CSV:
            extracted.rename(BEER_CSV)
        print(f"  Extracted -> {BEER_CSV.name}")

    zip_path.unlink()


def report() -> None:
    print("\nFile ready:")
    if BEER_CSV.exists():
        size = BEER_CSV.stat().st_size
        unit = f"{size / 1_048_576:.0f} MB" if size > 1_048_576 else f"{size / 1024:.1f} KB"
        print(f"  {BEER_CSV}  ({unit})")
    else:
        print(f"  MISSING: {BEER_CSV}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    download_beer_reviews()
    report()


if __name__ == "__main__":
    main()
