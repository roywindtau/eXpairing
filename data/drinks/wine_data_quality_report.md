# Wine Data Quality Report
_Profiled: 2026-05-30_

## Files
| File | Rows | Size |
|---|---|---|
| XWines_Full_100K_wines.csv | 100,646 | 34 MB |
| XWines_Full_21M_ratings.csv | 21,013,536 | 1.0 GB |

## Catalog (wines)

**Clean columns:** `WineID`, `WineName`, `Type`, `Elaborate`, `Grapes`, `Harmonize`, `ABV`, `Body`, `Acidity`, `Country`, `RegionID`, `RegionName`, `WineryID`, `WineryName`, `Vintages` — all 100,646 non-null.

**Missing:** `Website` — 18K nulls. Irrelevant to recommendations, can drop.

**Needs parsing:** `Harmonize`, `Grapes`, `Vintages` are stored as strings that look like Python lists (`"['Beef', 'Lamb']"`). Must run `ast.literal_eval` before use.

**Categoricals are clean:**
- `Body`: 5 values (Very light-bodied → Very full-bodied), no nulls
- `Acidity`: 3 values (Low / Medium / High), no nulls

**Harmonize coverage:** 0 empty lists — every wine has food pairings. `wine → recipe` edge is fully supported.

## Ratings

| Metric | Value |
|---|---|
| Total ratings | 21,013,536 |
| Distinct users | 1,056,079 |
| Distinct wines | 100,646 |
| Avg rating | 3.88 (skewed high) |
| Median ratings/user | 11 |
| Mean ratings/user | 20 |
| Max ratings/user | 2,986 |
| Users with <5 ratings | 0 (pre-filtered) |
| Users with >50 ratings | 51,357 (~5%) |

**Sparsity:** 21M / (1M users × 100K wines) = **0.2%** — 99.8% of the matrix is empty. Normal for recsys.

## Edge Feasibility

| Edge | Feasible? | Notes |
|---|---|---|
| `wine → recipe` | ✅ | Harmonize fully populated, rich food lists |
| `wine → user` (CF) | ✅ with caveats | Median user has 11 ratings — thin. CB fallback is not optional. |
| `wine → beer` | ❓ | Can't assess yet — need to check user overlap with beer dataset |

## Cleaning Tasks Needed
1. Drop `Website` column
2. Parse `Harmonize`, `Grapes`, `Vintages` from string → list (`ast.literal_eval`)
3. Filter wines to only those present in ratings (verify join integrity)
4. Decide rating threshold for users in CF training (current min is 5 — may want higher)
