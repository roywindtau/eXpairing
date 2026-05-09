"""
beta_updater.py
---------------
Daily batch job that updates each user's beta (waste aversion weight)
by comparing their stated preference with their revealed behaviour.

THE CORE IDEA
-------------
When a user onboards they choose a beta on a slider:
    0.0 = "I don't mind buying a few extra ingredients"
    1.0 = "I only want recipes I can cook with what I have"

But people lie — not maliciously, just aspirationally. Someone who
sets beta=0.9 (strict zero-waste) might consistently cook recipes
that required 2-3 missing ingredients. Their *revealed* beta is lower
than their stated one.

We observe this gap through UserEvents:
    event_type="cook"  with n_missing=0  -> revealed preference: high beta
    event_type="cook"  with n_missing=2  -> revealed preference: lower beta
    event_type="skip"  on a 100% match   -> revealed preference: lower beta
                                            (they wanted something fancier)

ALGORITHM
---------
For each user with >= MIN_EVENTS recent events:

1. Compute revealed_beta from cooking behavior:
       avg_missing = mean(n_missing) across recent "cook" events
       revealed_beta = 1.0 - (avg_missing / MAX_MISSING_NORMALIZER)
       clamped to [0.0, 1.0]

2. Drift the stored beta toward revealed_beta using a learning rate:
       new_beta = (1 - LEARNING_RATE) * current_beta
                + LEARNING_RATE       * revealed_beta

   This exponential moving average prevents wild swings from a single
   unusual cooking session — beta drifts gradually, not jumps.

3. Write new_beta back to User.beta in the DB.

RUNNING
-------
    # Once manually:
    python -m backend.services.beta_updater

    # In production: add to a cron job or APScheduler:
    # 0 3 * * * cd /app && python -m backend.services.beta_updater
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import User, UserEvent


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

# Only update users who have this many cook events in the lookback window.
# Too few events = noisy signal, not worth updating.
MIN_EVENTS: int = 3

# Only look at events from the last N days.
# Older events reflect past habits that may no longer apply.
LOOKBACK_DAYS: int = 30

# How fast beta drifts toward revealed_beta.
# 0.1 = 10% shift per day — slow, stable drift.
# 0.3 = 30% shift — faster adaptation.
LEARNING_RATE: float = 0.15

# Normalize n_missing: a recipe needing 3+ extra ingredients is treated
# as the "maximum" case (beyond this, it's all equally not-zero-waste).
MAX_MISSING_NORMALIZER: float = 4.0

# Beta is always kept within this range.
BETA_MIN: float = 0.05
BETA_MAX: float = 0.95


# ---------------------------------------------------------------------------
# Data class for a user's update record
# ---------------------------------------------------------------------------

@dataclass
class BetaUpdate:
    user_id:        int
    old_beta:       float
    revealed_beta:  float
    new_beta:       float
    n_cook_events:  int
    avg_missing:    float
    drift:          float    # new_beta - old_beta


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _compute_revealed_beta(cook_events: pd.DataFrame) -> tuple[float, float]:
    """
    Given a DataFrame of cook events for one user (must have n_missing column),
    return (revealed_beta, avg_missing).

    Events with n_missing=NULL are excluded (we don't know the missing count).
    """
    valid = cook_events[cook_events["n_missing"].notna()].copy()
    if valid.empty:
        return None, None

    avg_missing = valid["n_missing"].mean()

    # Normalize: 0 missing -> beta=1.0, MAX_MISSING missing -> beta~=0.0
    # Uses a soft sigmoid-like falloff rather than a hard floor so that
    # users who consistently cook with 5+ extra ingredients drift toward
    # BETA_MIN gradually rather than hitting it immediately.
    #
    # Soft formula: revealed = 1 / (1 + avg_missing / HALF_POINT)
    # where HALF_POINT = MAX_MISSING_NORMALIZER / 2
    # This gives:
    #   0 missing  -> 1.00  (zero-waste)
    #   2 missing  -> 0.50  (half-point)
    #   4 missing  -> 0.33
    #   8 missing  -> 0.20  (instead of hitting 0.0 hard)
    half_point = MAX_MISSING_NORMALIZER / 2
    revealed = 1.0 / (1.0 + avg_missing / half_point)
    revealed = round(float(revealed), 4)
    return revealed, round(avg_missing, 2)


def _drift_beta(current: float, revealed: float, rate: float) -> float:
    """Exponential moving average toward revealed_beta."""
    new = (1 - rate) * current + rate * revealed
    return round(max(BETA_MIN, min(BETA_MAX, new)), 4)


def compute_updates(db: Session, lookback_days: int = LOOKBACK_DAYS) -> list[BetaUpdate]:
    """
    Compute beta updates for all eligible users.
    Does NOT write to DB — returns a list of BetaUpdate records for inspection.
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)

    # Load all recent cook events with n_missing populated
    rows = (
        db.query(
            UserEvent.user_id,
            UserEvent.event_type,
            UserEvent.n_missing,
            UserEvent.created_at,
        )
        .filter(UserEvent.event_type == "cook")
        .filter(UserEvent.created_at >= cutoff)
        .all()
    )

    if not rows:
        return []

    events_df = pd.DataFrame(rows, columns=["user_id", "event_type",
                                              "n_missing", "created_at"])

    updates = []
    for user_id, group in events_df.groupby("user_id"):
        if len(group) < MIN_EVENTS:
            continue

        revealed_beta, avg_missing = _compute_revealed_beta(group)
        if revealed_beta is None:
            continue

        user = db.get(User, user_id)
        if user is None:
            continue

        new_beta = _drift_beta(user.beta, revealed_beta, LEARNING_RATE)

        updates.append(BetaUpdate(
            user_id=user_id,
            old_beta=round(user.beta, 4),
            revealed_beta=revealed_beta,
            new_beta=new_beta,
            n_cook_events=len(group),
            avg_missing=avg_missing,
            drift=round(new_beta - user.beta, 4),
        ))

    return updates


def apply_updates(db: Session, updates: list[BetaUpdate]) -> None:
    """Write the computed beta updates back to the User table."""
    for u in updates:
        user = db.get(User, u.user_id)
        if user:
            user.beta = u.new_beta
    db.commit()


def run(
    lookback_days: int = LOOKBACK_DAYS,
    dry_run: bool = False,
) -> list[BetaUpdate]:
    """
    Full update cycle. Call this from the scheduler or CLI.

    Args:
        lookback_days: how far back to look at events
        dry_run:       if True, compute but do NOT write to DB

    Returns:
        list of BetaUpdate records (for logging / inspection)
    """
    db = SessionLocal()
    try:
        updates = compute_updates(db, lookback_days=lookback_days)

        if not dry_run:
            apply_updates(db, updates)

        return updates
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_report(updates: list[BetaUpdate], dry_run: bool) -> None:
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Beta updater — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  {len(updates)} users updated\n")

    if not updates:
        print("  No users had enough recent cook events to update.")
        return

    print(f"  {'user_id':>8}  {'old_β':>6}  {'revealed_β':>10}  "
          f"{'new_β':>6}  {'drift':>7}  {'events':>7}  {'avg_missing':>11}")
    print("  " + "-" * 70)

    for u in sorted(updates, key=lambda x: abs(x.drift), reverse=True):
        arrow = "↑" if u.drift > 0 else "↓" if u.drift < 0 else "="
        print(f"  {u.user_id:>8}  {u.old_beta:>6.3f}  {u.revealed_beta:>10.3f}  "
              f"{u.new_beta:>6.3f}  {arrow}{abs(u.drift):>5.3f}  "
              f"{u.n_cook_events:>7}  {u.avg_missing:>11.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update user beta weights from behavior")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute updates but do not write to DB")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                        help=f"Days to look back (default {LOOKBACK_DAYS})")
    args = parser.parse_args()

    updates = run(lookback_days=args.lookback, dry_run=args.dry_run)
    _print_report(updates, dry_run=args.dry_run)
