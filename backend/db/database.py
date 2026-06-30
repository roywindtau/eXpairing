"""
database.py
-----------
DB engine, session factory, and init_db().

Uses SQLite for local/POC. To switch to Postgres, replace DATABASE_URL
with "postgresql://user:pass@host/dbname" and install psycopg2.

Usage in routers:
    from backend.db.database import get_db
    def my_endpoint(db: Session = Depends(get_db)):
        ...
"""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from backend.db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fridge2fork.db")

# connect_args only needed for SQLite (allows multi-thread access)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables. Safe to call multiple times (no-op if exist)."""
    Base.metadata.create_all(bind=engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """Additive migration for DBs created before a new nullable column existed.

    create_all() never ALTERs existing tables, so a pre-existing dev DB would be
    missing recently-added columns. Add them here (existing rows get NULL). Only
    handles simple additive columns — anything more needs a real migration tool.
    """
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    user_cols = {c["name"] for c in inspector.get_columns("users")}
    if "wine_prefs" not in user_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN wine_prefs TEXT"))


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
