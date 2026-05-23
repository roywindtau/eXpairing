#!/usr/bin/env bash
# dev.sh — spin up Fridge2Fork in one command
# Usage: ./dev.sh [--seed] [--rebuild]
#   --seed     seed the DB with 20 dev recipes before starting (safe to re-run)
#   --rebuild  force Docker image rebuild (use after requirements/package changes)
set -euo pipefail

SEED=false
REBUILD_FLAG=""

for arg in "$@"; do
  case $arg in
    --seed)    SEED=true ;;
    --rebuild) REBUILD_FLAG="--build" ;;
  esac
done

# Seed the DB if requested or if it doesn't exist yet
if $SEED || [ ! -f fridge2fork.db ]; then
  echo "==> Seeding database..."
  python3 -m backend.db.seed_dev
fi

echo "==> Starting containers..."
docker compose up $REBUILD_FLAG --remove-orphans

# docker compose up blocks; Ctrl-C shuts everything down cleanly
