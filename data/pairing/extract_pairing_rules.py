"""
extract_pairing_rules.py
========================
Extract the (wine_category x food_category) pairing-quality table from the
labeled pairing dataset, and save it as a JSON artifact the scorer can load.

WHY EXTRACT INSTEAD OF HAND-WRITE
---------------------------------
check_ingredient_signal.py showed the pairing CSV is RULE-GENERATED: each
(wine_category, food_category) cell has a characteristic base quality plus noise,
with extra "deliberately bad / idealized perfect" CONTRAST rows injected. So the
rules we want are literally the per-cell averages of the real (non-contrast) rows.
Rather than guess "acid cuts fat" by hand, we read the rulebook straight out of
the data:

    quality[wine_category][food_category] = mean pairing_quality (1-5)

We DROP the contrast rows first -- they are forced 1s and 5s that don't reflect
the underlying rule, only injected extremes.

OUTPUT
------
    models/pairing_rules.json
        {
          "scale": [1, 5],
          "wine_categories": [...],
          "food_categories": [...],
          "quality": { "<wine_cat>": { "<food_cat>": mean, ... }, ... },
          "global_mean": <float>   # fallback for unseen cells
        }

USED BY
-------
    serve_pairing.py (Module 4) blends this rule score with the cosine score.

Run:
    python -m data.pairing.extract_pairing_rules
"""

from __future__ import annotations

import collections
import csv
import json
import statistics
from pathlib import Path

CSV       = Path("data/pairing/wine_food_pairings.csv")
OUT       = Path("models/pairing_rules.json")
CONTRAST_MARK = "contrast"   # description substring marking synthetic extremes


def extract() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    real = [r for r in rows if CONTRAST_MARK not in r["description"].lower()]
    print(f"{len(rows):,} rows; {len(real):,} after dropping contrast rows.")

    cells: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for r in real:
        cells[(r["wine_category"], r["food_category"])].append(int(r["pairing_quality"]))

    wine_cats = sorted({w for w, _ in cells})
    food_cats = sorted({f for _, f in cells})

    quality: dict[str, dict[str, float]] = {wc: {} for wc in wine_cats}
    for (wc, fc), vals in cells.items():
        quality[wc][fc] = round(statistics.mean(vals), 3)

    global_mean = round(statistics.mean(int(r["pairing_quality"]) for r in real), 3)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "scale": [1, 5],
        "wine_categories": wine_cats,
        "food_categories": food_cats,
        "quality": quality,
        "global_mean": global_mean,
    }, indent=2), encoding="utf-8")

    print(f"Saved -> {OUT}  (global_mean={global_mean})")
    print("\n(wine_cat x food_cat) mean quality:")
    print("".ljust(11) + "".join(f"{f[:7]:>9s}" for f in food_cats))
    for wc in wine_cats:
        line = wc[:11].ljust(11)
        for fc in food_cats:
            v = quality[wc].get(fc)
            line += f"{v:9.2f}" if v is not None else f"{'-':>9s}"
        print(line)


if __name__ == "__main__":
    extract()
