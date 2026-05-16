"""
download_drinks.py
------------------
Downloads the two drink datasets used by the drink recommender:

  1. Beer Reviews (Kaggle, ~150MB)
        ~1.58M reviews, ~33k users, ~66k beers from BeerAdvocate (1998-2011).
        Aspect ratings: appearance, aroma, palate, taste, overall.
        Kaggle slug: rdoume/beerreviews

  2. X-Wines Test (GitHub raw, ~100KB total)
        100 wines + 1k ratings + Harmonize field (food categories per wine).
        Repo: https://github.com/rogerioxavier/X-Wines
        License: CC0 1.0 Universal.
        Citation: de Azambuja, R.X.; Morais, A.J.; Filipe, V.
                  X-Wines: A Wine Dataset for Recommender Systems and ML.
                  Big Data Cogn. Comput. 2023, 7, 20.

        NOTE: We use the small Test version because the larger Slim (1k wines,
        150k ratings) and Full (100k wines, 21M ratings) versions are hosted on
        Google Drive, not GitHub. To upgrade to Slim later, fetch
            XWines_Slim_1K_wines.csv
            XWines_Slim_1K_ratings.csv
        from the Drive folder linked in the X-Wines repo README and place them
        at data/xwines_wines.csv and data/xwines_ratings.csv.

Requirements:
    pip install kaggle    (already in requirements.txt)
    Set up ~/.kaggle/kaggle.json with your API key
        (Kaggle > Account > Create New API Token)
    Accept the Beer Reviews terms once at:
        https://www.kaggle.com/datasets/rdoume/beerreviews

Run:
    python data/download_drinks.py

Output files:
    data/beer_reviews.csv     ~150MB, ~1.5M rows
    data/xwines_wines.csv     ~40KB,  100 rows  (with Harmonize field)
    data/xwines_ratings.csv   ~50KB,  1k rows
"""

import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR     = Path("data")
BEER_DATASET = "rdoume/beerreviews"
BEER_CSV     = DATA_DIR / "beer_reviews.csv"

XWINES_BASE  = "https://raw.githubusercontent.com/rogerioxavier/X-Wines/main/Dataset/last"
XWINES_FILES = {
    DATA_DIR / "xwines_wines.csv":   f"{XWINES_BASE}/XWines_Test_100_wines.csv",
    DATA_DIR / "xwines_ratings.csv": f"{XWINES_BASE}/XWines_Test_1K_ratings.csv",
}


def download_beer_reviews():
    print("\n[1/2] Beer Reviews (Kaggle)")
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


def download_xwines():
    print("\n[2/2] X-Wines Test (GitHub raw)")
    for dst, url in XWINES_FILES.items():
        if dst.exists():
            kb = dst.stat().st_size / 1024
            print(f"  Already present: {dst} ({kb:.1f} KB) — skipping.")
            continue
        print(f"  Downloading {dst.name} ...")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"  Download failed: {e}")
            sys.exit(1)
        kb = dst.stat().st_size / 1024
        print(f"  Wrote {dst} ({kb:.1f} KB)")


def report():
    print("\nFiles ready:")
    for path in [BEER_CSV, *XWINES_FILES.keys()]:
        if path.exists():
            size = path.stat().st_size
            unit = f"{size / 1_048_576:.0f} MB" if size > 1_048_576 else f"{size / 1024:.1f} KB"
            print(f"  {path}  ({unit})")
        else:
            print(f"  MISSING: {path}")

    print("\nNext steps (run from recsys26/):")
    print("  python -m backend.db.seed_drinks         # ~66k beer + 100 wine rows")
    print("  python -m backend.db.seed_drink_ratings  # ~1.5M beer + 1k wine ratings")
    print("  python -m backend.ml.train_drink_cb      # TF-IDF over drink text")
    print("  python -m backend.ml.train_drink_cf      # Surprise SVD on beer ratings")
    print("  python -m backend.ml.drink_item_similarity  # per-kind item-item sim")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    download_beer_reviews()
    download_xwines()
    report()


if __name__ == "__main__":
    main()
