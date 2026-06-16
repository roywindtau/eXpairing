"""
reset_wines.py
--------------
Drops and recreates only the wine-related tables:
    wines, wine_events

All other tables (users, recipes, pantry_items, user_events, etc.)
are left completely untouched.

Run from project root:
    python -m backend.db.reset_wines
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import engine
from backend.db.models import Wine, WineEvent

WINE_TABLES = [
    WineEvent.__table__,
    Wine.__table__,
]

if __name__ == "__main__":
    print("Dropping wine tables...")
    # Drop in order to respect foreign keys (events → wines).
    for table in WINE_TABLES:
        table.drop(engine, checkfirst=True)
        print(f"  dropped: {table.name}")

    print("Recreating wine tables...")
    for table in reversed(WINE_TABLES):
        table.create(engine)
        print(f"  created: {table.name}")

    print("Done.")
