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
from sqlalchemy import create_engine
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


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
