"""
reset_drinks.py
---------------
Drops and recreates only the drink-related tables:
    drinks, wines, drink_events

All other tables (users, recipes, pantry_items, user_events, etc.)
are left completely untouched.

Run from project root:
    python -m backend.db.reset_drinks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import engine
from backend.db.models import Base, Drink, Wine, DrinkEvent

DRINK_TABLES = [
    DrinkEvent.__table__,
    Wine.__table__,
    Drink.__table__,
]

if __name__ == "__main__":
    print("Dropping drink tables...")
    # Drop in reverse order to respect foreign keys
    for table in DRINK_TABLES:
        table.drop(engine, checkfirst=True)
        print(f"  dropped: {table.name}")

    print("Recreating drink tables...")
    for table in reversed(DRINK_TABLES):
        table.create(engine)
        print(f"  created: {table.name}")

    print("Done.")
