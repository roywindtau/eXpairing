#!/usr/bin/env bash
# clean_and_seed.sh
# -----------------
# Full pre-training data refresh for drinks:
#   1. Clean wine raw CSVs   -> clean_wines.csv + clean_ratings.csv
#   2. Drop & recreate the drink tables (overwrites existing rows)
#   3. Seed wines into the wines table
#
# Run from project root:
#   ./data/drinks/clean_and_seed.sh
#
# Requires the raw files in data/drinks/:
#   XWines_Full_100K_wines.csv, XWines_Full_21M_ratings.csv

set -e   # stop on first error

echo "========================================"
echo " Drinks: clean + seed (overwrite)"
echo "========================================"

echo ""
echo "[1/3] Cleaning wines..."
python -m data.drinks.wine.clean_wines

echo ""
echo "[2/3] Resetting drink tables (overwrite)..."
python -m backend.db.reset_drinks

echo ""
echo "[3/3] Seeding wines..."
python -m backend.db.drinks.seed_wines

echo ""
echo "========================================"
echo " Done. Wines cleaned and seeded."
echo "========================================"
