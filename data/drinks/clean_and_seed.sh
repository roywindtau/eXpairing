#!/usr/bin/env bash
# clean_and_seed.sh
# -----------------
# Full pre-training data refresh for drinks:
#   1. Clean wine raw CSVs   -> clean_wines.csv + clean_ratings.csv
#   2. Clean beer raw CSV    -> clean_beer.csv + clean_beer_ratings.csv
#   3. Drop & recreate the drink tables (overwrites existing rows)
#   4. Seed beers into the beers table
#   5. Seed wines into the wines table
#
# Run from project root:
#   ./data/drinks/clean_and_seed.sh
#
# Requires the raw files in data/drinks/:
#   XWines_Full_100K_wines.csv, XWines_Full_21M_ratings.csv, beer_reviews.csv

set -e   # stop on first error

echo "========================================"
echo " Drinks: clean + seed (overwrite)"
echo "========================================"

echo ""
echo "[1/5] Cleaning wines..."
python -m data.drinks.clean_wines

echo ""
echo "[2/5] Cleaning beer..."
python -m data.drinks.clean_beer

echo ""
echo "[3/5] Resetting drink tables (overwrite)..."
python -m backend.db.reset_drinks

echo ""
echo "[4/5] Seeding beers..."
python -m backend.db.drinks.seed_drinks

echo ""
echo "[5/5] Seeding wines..."
python -m backend.db.drinks.seed_wines

echo ""
echo "========================================"
echo " Done. Beers + wines cleaned and seeded."
echo "========================================"
