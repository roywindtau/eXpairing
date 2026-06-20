"""
region_rollup.py
----------------
Collapse X-Wines' 2,160 fine-grained appellations into a small set of broad
parent wine regions, so the content-based model's region feature actually fires.

Why
---
Raw region one-hot is near-useless: 2,160 distinct values, the top 50 cover only
~42% of wines, and two wines almost never share an *exact* appellation. Rolling
sub-regions up to their parent (Pauillac -> Bordeaux, Meursault -> Burgundy,
Napa Valley -> California) makes wines actually match on geography.

Resolution order for a (region, country) pair:
    1. explicit leaf -> parent in LEAF_TO_PARENT
    2. substring/keyword rules in PARENT_KEYWORDS (catches sub-appellations we
       didn't enumerate, e.g. "... Grand Cru" Bordeaux/Burgundy variants)
    3. fall back to the COUNTRY name (region implies country; this is the
       coarse-but-honest default for the obscure long tail)

This produces a static, committed artifact (region_rollup.json) consumed by the
CB encoder. Pure lookup, no ML, no network.

Run:
    python -m data.wine.region_rollup          # writes region_rollup.json
    python -m data.wine.region_rollup --stats  # + coverage report
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OUT = Path("models") / "region_rollup.json"

# ── Explicit leaf -> parent map for the well-known appellations ──────────────
# Keyed by the raw region string exactly as it appears in the catalog.
LEAF_TO_PARENT: dict[str, str] = {
    # ---- France: Burgundy (Bourgogne) ----
    "Bourgogne": "Burgundy", "Chablis": "Burgundy", "Côte de Beaune": "Burgundy",
    "Côte de Nuits": "Burgundy", "Meursault": "Burgundy", "Gevrey-Chambertin": "Burgundy",
    "Nuits-Saint-Georges": "Burgundy", "Vosne-Romanée": "Burgundy", "Pommard": "Burgundy",
    "Pouilly-Fuissé": "Burgundy", "Beaujolais": "Burgundy", "Mâcon": "Burgundy",
    "Chassagne-Montrachet": "Burgundy", "Puligny-Montrachet": "Burgundy",
    "Chambolle-Musigny": "Burgundy", "Volnay": "Burgundy", "Santenay": "Burgundy",
    "Aloxe-Corton": "Burgundy", "Savigny-lès-Beaune": "Burgundy", "Mercurey": "Burgundy",
    # ---- France: Bordeaux ----
    "Bordeaux": "Bordeaux", "Bordeaux Supérieur": "Bordeaux", "Médoc": "Bordeaux",
    "Haut-Médoc": "Bordeaux", "Margaux": "Bordeaux", "Pauillac": "Bordeaux",
    "Saint-Julien": "Bordeaux", "Saint-Estèphe": "Bordeaux", "Pessac-Léognan": "Bordeaux",
    "Saint-Émilion Grand Cru": "Bordeaux", "Saint-Émilion": "Bordeaux", "Pomerol": "Bordeaux",
    "Lalande-de-Pomerol": "Bordeaux", "Côtes de Bourg": "Bordeaux", "Sauternes": "Bordeaux",
    "Castillon-Côtes de Bordeaux": "Bordeaux", "Graves": "Bordeaux", "Fronsac": "Bordeaux",
    # ---- France: Rhône ----
    "Côtes-du-Rhône": "Rhône", "Southern Rhône": "Rhône", "Northern Rhône": "Rhône",
    "Rhone Valley": "Rhône", "Châteauneuf-du-Pape": "Rhône", "Gigondas": "Rhône",
    "Ventoux": "Rhône", "Luberon": "Rhône", "Hermitage": "Rhône", "Côte-Rôtie": "Rhône",
    "Crozes-Hermitage": "Rhône", "Vacqueyras": "Rhône",
    # ---- France: Loire ----
    "Loire Valley": "Loire", "Sancerre": "Loire", "Saumur": "Loire", "Touraine": "Loire",
    "Pouilly-Fumé": "Loire", "Vouvray": "Loire", "Muscadet": "Loire", "Chinon": "Loire",
    "Anjou": "Loire", "Bourgueil": "Loire",
    # ---- France: other majors ----
    "Champagne": "Champagne", "Champagne Premier Cru": "Champagne",
    "Champagne Grand Cru": "Champagne",
    "Alsace": "Alsace",
    "Languedoc-Roussillon": "Languedoc-Roussillon", "Languedoc": "Languedoc-Roussillon",
    "Pays d'Oc": "Languedoc-Roussillon", "Corbières": "Languedoc-Roussillon",
    "Côtes de Provence": "Provence", "Provence": "Provence", "Bandol": "Provence",
    "Cahors": "South West France", "Madiran": "South West France",
    # ---- Italy ----
    "Toscana": "Tuscany", "Chianti": "Tuscany", "Chianti Classico": "Tuscany",
    "Brunello di Montalcino": "Tuscany", "Bolgheri": "Tuscany",
    "Piemonte": "Piedmont", "Barolo": "Piedmont", "Barbaresco": "Piedmont",
    "Veneto": "Veneto", "Valpolicella": "Veneto", "Amarone della Valpolicella": "Veneto",
    "Soave": "Veneto", "Prosecco": "Veneto",
    "Friuli-Venezia Giulia": "Friuli", "Terre Siciliane": "Sicily", "Sicilia": "Sicily",
    "Puglia": "Puglia", "Abruzzo": "Abruzzo", "Lombardia": "Lombardy",
    # ---- Spain ----
    "Rioja": "Rioja", "Ribera del Duero": "Ribera del Duero", "Priorat": "Priorat",
    "Rías Baixas": "Rías Baixas", "Penedès": "Catalonia", "Jerez": "Jerez",
    # ---- Portugal ----
    "Douro": "Douro", "Porto": "Douro", "Dão": "Dão", "Alentejo": "Alentejo",
    "Vinho Verde": "Vinho Verde",
    # ---- USA ----
    "California": "California", "Napa Valley": "California", "Sonoma County": "California",
    "Paso Robles": "California", "Central Coast": "California", "Russian River Valley": "California",
    "Willamette Valley": "Oregon", "Columbia Valley": "Washington",
    "Finger Lakes": "New York",
    # ---- Argentina / Chile ----
    "Mendoza": "Mendoza", "Uco Valley": "Mendoza", "Patagonia": "Patagonia",
    "Central Valley (CL)": "Central Valley (CL)", "Colchagua Valley": "Central Valley (CL)",
    "Maipo Valley": "Central Valley (CL)", "Casablanca Valley": "Aconcagua (CL)",
    # ---- Germany ----
    "Mosel": "Mosel", "Pfalz": "Pfalz", "Rheinhessen": "Rheinhessen", "Baden": "Baden",
    "Rheingau": "Rheingau", "Nahe": "Nahe",
    # ---- Rest of New World ----
    "Stellenbosch": "Western Cape", "Western Cape": "Western Cape", "Swartland": "Western Cape",
    "Barossa Valley": "South Australia", "McLaren Vale": "South Australia",
    "Clare Valley": "South Australia", "Coonawarra": "South Australia", "Eden Valley": "South Australia",
    "Marlborough": "Marlborough", "Central Otago": "Central Otago",
    "Serra Gaúcha": "Serra Gaúcha",
}

# ── Keyword rules for sub-appellations we didn't enumerate ───────────────────
# Checked as case-insensitive substrings, in order. First hit wins.
PARENT_KEYWORDS: list[tuple[str, str]] = [
    ("médoc", "Bordeaux"), ("saint-émilion", "Bordeaux"), ("bordeaux", "Bordeaux"),
    ("rhône", "Rhône"), ("rhone", "Rhône"),
    ("champagne", "Champagne"),
    ("chablis", "Burgundy"), ("beaujolais", "Burgundy"), ("bourgogne", "Burgundy"),
    ("chianti", "Tuscany"), ("montalcino", "Tuscany"), ("toscana", "Tuscany"),
    ("barolo", "Piedmont"), ("barbaresco", "Piedmont"), ("piemonte", "Piedmont"),
    ("valpolicella", "Veneto"), ("prosecco", "Veneto"),
    ("rioja", "Rioja"), ("douro", "Douro"), ("porto", "Douro"),
    ("napa", "California"), ("sonoma", "California"),
]


def resolve(region: str, country: str) -> str:
    """Map one (region, country) to its parent. Country is the fallback."""
    if not region:
        return country or "Unknown"
    if region in LEAF_TO_PARENT:
        return LEAF_TO_PARENT[region]
    low = region.lower()
    for kw, parent in PARENT_KEYWORDS:
        if kw in low:
            return parent
    return country or region  # coarse fallback: region implies country


def build_mapping() -> dict[str, str]:
    """Return {region: parent} over every distinct region in the catalog."""
    from backend.db.database import SessionLocal
    from backend.db.models import Wine

    db = SessionLocal()
    try:
        pairs = {(w.region, w.country or "") for w in db.query(Wine).all() if w.region}
    finally:
        db.close()
    return {region: resolve(region, country) for region, country in pairs}


def main(stats: bool = False) -> None:
    mapping = build_mapping()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {len(mapping):,} region->parent entries -> {OUT}")

    if stats:
        from backend.db.database import SessionLocal
        from backend.db.models import Wine

        db = SessionLocal()
        try:
            wines = db.query(Wine).all()
            parent_counts = Counter(mapping.get(w.region, w.country or "Unknown")
                                    for w in wines if w.region)
        finally:
            db.close()
        n_leaf = len(mapping)
        n_parent = len(parent_counts)
        total = sum(parent_counts.values())
        print(f"\nConsolidation: {n_leaf:,} leaf regions -> {n_parent:,} parents")
        top = parent_counts.most_common(10)
        head = sum(v for _, v in top)
        print(f"Top 10 parents cover {100*head/total:.1f}% of wines:")
        for p, n in top:
            print(f"  {n:6,}  {p}")


if __name__ == "__main__":
    main(stats="--stats" in sys.argv)
