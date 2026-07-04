"""
download_pairing.py
--------------------
Downloads the wine-food pairing Kaggle dataset into data/pairing/.

Requirements:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API key
    (Kaggle > Account > Create New API Token)

Run:
    python -m data.pairing.download_pairing

Output file:
    data/pairing/wine_food_pairings.csv   ~3.2MB, ~35k labeled (wine, food) pairings
"""

import subprocess
import sys
import zipfile
from pathlib import Path

DATASET   = "wafaaelhusseini/wine-and-food-pairing-dataset"
DATA_DIR  = Path("data/pairing")
ZIP_NAME  = "wine-and-food-pairing-dataset.zip"
CSV_NAME  = "wine_food_pairings.csv"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading wine-food pairing dataset from Kaggle ...")
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
        for name in z.namelist():
            if name == CSV_NAME:
                z.extract(name, DATA_DIR)
                print(f"  Extracted {name}")

    zip_path.unlink()

    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.exists():
        print(f"ERROR: {CSV_NAME} not found in the downloaded archive.")
        sys.exit(1)

    mb = csv_path.stat().st_size / 1_048_576
    print(f"\nDone. {CSV_NAME} ready ({mb:.1f} MB)")

    print("\nNext steps:")
    print("  python -m data.pairing.extract_pairing_rules")
    print("  python -m data.pairing.build_wine_pairing_vectors")

if __name__ == "__main__":
    main()
