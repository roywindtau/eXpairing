"""
canonicalizer/ingredient_map.py
--------------------------------
Maps raw product names (from manual entry or vision scan) to canonical
ingredient tokens used throughout the recipe matching pipeline.

This is the bridge between the messy real world ("Tnuva 3% Fresh Milk
500ml", "Organic Free-Range Large Eggs x6") and the clean ingredient
vocabulary the TF-IDF and fuzzy matcher expect ("milk", "eggs").

TWO-STAGE PIPELINE
------------------
Stage 1 — Rule-based cleaning (fast, no ML):
    - Strip brand names from a known list
    - Remove quantity patterns (500ml, 1kg, 2 pack)
    - Remove noise adjectives (organic, fresh, free-range)
    - Lowercase, strip, collapse whitespace

Stage 2 — Fuzzy matching against recipe vocab (optional, more accurate):
    - Uses rapidfuzz WRatio scorer
    - Falls back to stage-1 result if no match above threshold
    - vocab built from Food.com ingredient corpus

USAGE
-----
    from backend.canonicalizer.ingredient_map import IngredientMapper

    mapper = IngredientMapper.from_db()   # loads vocab from DB
    mapper.map("Tnuva 3% Milk 500ml")     # → "milk"
    mapper.map("Heinz Tomato Ketchup")    # → "tomato ketchup"
    mapper.map("free range eggs 6 pack")  # → "eggs"
"""

from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz, process


# ── cleaning rules ──────────────────────────────────────────────────────────

# Known brand prefixes/names to strip. Kept lowercase.
BRAND_WORDS: frozenset[str] = frozenset({
    # Israeli brands
    "tnuva", "strauss", "osem", "telma", "tara", "yotvata", "elite",
    "angel", "diplomat", "adanim", "wissotzky",
    # International
    "heinz", "nestle", "danone", "yoplait", "kraft", "kelloggs", "kellogg",
    "nescafe", "lipton", "knorr", "hellmanns", "hellmann", "campbells",
    "del monte", "birds eye", "uncle bens", "pringles", "coca cola", "pepsi",
    "tropicana", "innocent", "lurpak", "anchor", "philadelphia", "president",
    "flora", "benecol", "ariel",
})

# Quantity/unit pattern: "500ml", "1.5 kg", "2 x 300g", "6 pack", etc.
_QTY_RE = re.compile(
    r"\b\d+(\.\d+)?\s*"
    r"(ml|l|cl|dl|fl\.?oz|g|kg|oz|lb|lbs|mg|"
    r"pack|packs|piece|pieces|pcs?|units?|count|ct|"
    r"x\s*\d+|×\s*\d+)\b",
    re.IGNORECASE,
)

# Standalone numbers (e.g. "3%" or "500")
_NUM_RE = re.compile(r"\b\d+%?\b")

# Noise adjectives that appear frequently but carry no ingredient meaning
NOISE_WORDS: frozenset[str] = frozenset({
    "free", "range", "organic", "natural", "fresh", "frozen", "chilled",
    "light", "lite", "low", "fat", "semi", "skimmed", "whole", "full",
    "extra", "premium", "classic", "original", "traditional", "homemade",
    "style", "flavour", "flavor", "brand", "new", "improved", "best",
    "finest", "select", "choice", "quality", "pure", "real", "genuine",
    "large", "medium", "small", "mini", "jumbo", "family", "economy",
    "sliced", "diced", "chopped", "grated", "crushed", "ground", "whole",
    "dried", "fresh", "tinned", "canned", "frozen",
})


def clean_product_name(raw: str) -> str:
    """
    Stage-1 cleaning: strip brands, quantities, and noise from a raw name.

    "Tnuva 3% Fresh Milk 500ml"  →  "milk"
    "Free Range Large Eggs 6pk"  →  "eggs"
    "Anchor Unsalted Butter"     →  "butter"
    "Heinz Baked Beans in Tom."  →  "baked beans"
    """
    text = raw.lower().strip()

    # Remove quantities
    text = _QTY_RE.sub(" ", text)

    # Remove standalone numbers and percentages
    text = _NUM_RE.sub(" ", text)

    # Tokenise, filter brands and noise
    tokens = text.split()
    tokens = [t for t in tokens if t not in BRAND_WORDS and t not in NOISE_WORDS]

    # Remove punctuation-only tokens
    tokens = [t for t in tokens if re.search(r"[a-z]", t)]

    result = " ".join(tokens).strip()

    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result)

    return result if result else raw.lower().strip()


# ── fuzzy matcher ───────────────────────────────────────────────────────────

class IngredientMapper:
    """
    Maps raw product names to canonical ingredient names.

    With a vocab (from the recipe DB), uses rapidfuzz WRatio matching.
    Without a vocab, falls back to stage-1 text cleaning only.
    """

    FUZZY_THRESHOLD = 72   # minimum WRatio score to accept a match

    def __init__(self, vocab: Optional[list[str]] = None):
        self._vocab = vocab or []

    @classmethod
    def from_db(cls) -> "IngredientMapper":
        """Build mapper with full ingredient vocab from the recipe DB."""
        try:
            from backend.db.database import SessionLocal
            from backend.db.models import Recipe

            db = SessionLocal()
            try:
                rows = db.query(Recipe.ingredients_csv).all()
            finally:
                db.close()

            vocab: set[str] = set()
            for (csv,) in rows:
                for ing in csv.split(","):
                    word = ing.strip().lower()
                    if word:
                        vocab.add(word)

            return cls(vocab=sorted(vocab))
        except Exception:
            return cls(vocab=[])

    @classmethod
    def from_vocab_list(cls, vocab: list[str]) -> "IngredientMapper":
        """Build from an explicit list of canonical ingredient names."""
        return cls(vocab=[v.strip().lower() for v in vocab if v.strip()])

    def map(self, raw_name: str) -> str:
        """
        Map a raw product name to a canonical ingredient.

        Returns the best fuzzy match from the vocab, or the cleaned name
        if no match exceeds FUZZY_THRESHOLD.
        """
        cleaned = clean_product_name(raw_name)

        if not self._vocab:
            return cleaned

        result = process.extractOne(
            cleaned,
            self._vocab,
            scorer=fuzz.WRatio,
            score_cutoff=self.FUZZY_THRESHOLD,
        )

        if result:
            return result[0]

        # Try matching the last two words of the cleaned name as a fallback
        # "full fat cream cheese" → try "cream cheese" if no full match
        words = cleaned.split()
        if len(words) > 2:
            short = " ".join(words[-2:])
            result2 = process.extractOne(
                short,
                self._vocab,
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_THRESHOLD,
            )
            if result2:
                return result2[0]

        return cleaned

    def map_batch(self, raw_names: list[str]) -> list[str]:
        """Map a list of raw names. Preserves order."""
        return [self.map(name) for name in raw_names]

    def vocab_size(self) -> int:
        return len(self._vocab)
