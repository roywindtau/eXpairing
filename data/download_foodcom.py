"""
download_foodcom.py
-------------------
Downloads the Food.com Kaggle dataset into the data/ directory.

Requirements:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API key
    (Kaggle > Account > Create New API Token)

Run:
    python data/download_foodcom.py

Output files:
    data/RAW_recipes.csv        ~230MB, 231k recipes
    data/RAW_interactions.csv   ~55MB,  1.1M ratings
"""

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET   = "shuyangli94/food-com-recipes-and-user-interactions"
DATA_DIR  = Path("data")
ZIP_NAME  = "food-com-recipes-and-user-interactions.zip"

def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Downloading Food.com dataset from Kaggle ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Kaggle download failed:")
        print(result.stderr)
        print("\nMake sure you have:")
        print("  1. pip install kaggle")
        print("  2. ~/.kaggle/kaggle.json with your API credentials")
        print("  3. Accepted the dataset terms on kaggle.com")
        sys.exit(1)

    zip_path = DATA_DIR / ZIP_NAME
    if not zip_path.exists():
        # kaggle sometimes names it differently
        zips = list(DATA_DIR.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print("ERROR: zip file not found after download.")
            sys.exit(1)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Only extract the two files we need
        for name in z.namelist():
            if name in ("RAW_recipes.csv", "RAW_interactions.csv"):
                z.extract(name, DATA_DIR)
                print(f"  Extracted {name}")

    zip_path.unlink()
    print("\nDone. Files ready:")
    for f in ["RAW_recipes.csv", "RAW_interactions.csv"]:
        p = DATA_DIR / f
        if p.exists():
            mb = p.stat().st_size / 1_048_576
            print(f"  {f}  ({mb:.0f} MB)")

    print("\nNext steps:")
    print("  python -m backend.db.seed_recipes")
    print("  python -m backend.db.seed_ratings")
    print("  python -m backend.ml.train_cf")
    print("  python -m backend.ml.train_cb")

if __name__ == "__main__":
    main()
