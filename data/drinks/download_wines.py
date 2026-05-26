"""
download_wines.py
-----------------
Downloads the wine datasets used by the drink recommender exploration phase.

Two sources, each fetchable independently or together:

  1. Wine Enthusiast (Kaggle, ~43MB)
        ~81k expert reviews, 2017-2020. One expert score per wine (80-100),
        no user dimension. Rich free-text descriptions — strong CB signal.
        Kaggle slug: manyregression/updated-wine-enthusiast-review
        License: CC0 1.0 Universal.

  2. X-Wines Full (Google Drive, ~several hundred MB)
        ~100k wines, ~21M user ratings, ~1M users. Real CF signal plus
        structured Harmonize food-pairing field.
        Repo: https://github.com/rogerioxavier/X-Wines
        License: CC0 1.0 Universal.

        TODO: not implemented yet. The full dataset lives on Google Drive,
        not GitHub. To wire this up we need:
          - `gdown` added to requirements.txt
          - the Drive file IDs for XWines_Full_*_wines.csv and ratings
          - then call gdown.download(id=..., output=...) here
        Until then this source is a no-op with a warning.

Requirements:
    pip install kaggle    (already in requirements.txt)
    ~/.kaggle/kaggle.json set up
    Accept Wine Enthusiast terms at:
        https://www.kaggle.com/datasets/manyregression/updated-wine-enthusiast-review

Run:
    python -m data.drinks.download_wines                    # both sources
    python -m data.drinks.download_wines --wine-enthusiast  # WE only
    python -m data.drinks.download_wines --xwines-full      # X-Wines Full only

Output files (when fully wired):
    data/winemag-data-2017-2020.csv   ~43MB, ~81k rows
    data/xwines_full_wines.csv        TBD (TODO)
    data/xwines_full_ratings.csv      TBD (TODO)
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")

# Wine Enthusiast
WE_DATASET = "manyregression/updated-wine-enthusiast-review"
WE_CSV     = DATA_DIR / "winemag-data-2017-2020.csv"

# X-Wines Full (TODO)
XWINES_FULL_WINES   = DATA_DIR / "xwines_full_wines.csv"
XWINES_FULL_RATINGS = DATA_DIR / "xwines_full_ratings.csv"


def download_wine_enthusiast() -> None:
    print("Wine Enthusiast (Kaggle)")
    if WE_CSV.exists():
        mb = WE_CSV.stat().st_size / 1_048_576
        print(f"  Already present: {WE_CSV} ({mb:.0f} MB) — skipping.")
        return

    print(f"  Downloading {WE_DATASET} ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", WE_DATASET, "-p", str(DATA_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  Kaggle download failed:")
        print(result.stderr)
        print("\n  Make sure you have:")
        print("    1. pip install kaggle")
        print("    2. ~/.kaggle/kaggle.json with your API credentials")
        print(f"    3. Accepted dataset terms at https://www.kaggle.com/datasets/{WE_DATASET}")
        sys.exit(1)

    # Kaggle's filename mangles the slug; locate the produced zip.
    zips = sorted(DATA_DIR.glob("updated-wine-enthusiast-review*.zip"))
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
        # Use the largest CSV in the archive (in case there are multiple).
        target = max(csvs, key=lambda n: z.getinfo(n).file_size)
        z.extract(target, DATA_DIR)
        extracted = DATA_DIR / target
        if extracted != WE_CSV:
            extracted.rename(WE_CSV)
        print(f"  Extracted -> {WE_CSV.name}")

    zip_path.unlink()


def download_xwines_full() -> None:
    print("X-Wines Full (Google Drive)")
    # TODO: implement once gdown is in requirements.txt and the Drive file IDs
    # are known. The dataset is hosted on Google Drive (linked from the X-Wines
    # repo README) and gdown.download(id=..., output=...) is the canonical way
    # to fetch it. Targets: xwines_full_wines.csv, xwines_full_ratings.csv.
    print("  Not implemented yet — see TODO in module docstring.")


def report() -> None:
    print("\nFiles ready:")
    for path in (WE_CSV, XWINES_FULL_WINES, XWINES_FULL_RATINGS):
        if path.exists():
            size = path.stat().st_size
            unit = f"{size / 1_048_576:.0f} MB" if size > 1_048_576 else f"{size / 1024:.1f} KB"
            print(f"  {path}  ({unit})")
        else:
            print(f"  MISSING: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wine-enthusiast", action="store_true",
                        help="Download Wine Enthusiast only.")
    parser.add_argument("--xwines-full", action="store_true",
                        help="Download X-Wines Full only (TODO — not implemented).")
    args = parser.parse_args()

    # Default (no flags) = both
    fetch_we    = args.wine_enthusiast or not (args.wine_enthusiast or args.xwines_full)
    fetch_full  = args.xwines_full     or not (args.wine_enthusiast or args.xwines_full)

    DATA_DIR.mkdir(exist_ok=True)
    if fetch_we:
        download_wine_enthusiast()
    if fetch_full:
        download_xwines_full()
    report()


if __name__ == "__main__":
    main()
