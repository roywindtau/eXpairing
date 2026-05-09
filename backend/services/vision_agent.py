"""
vision_agent.py
---------------
Sends a fridge/pantry photo to GPT-4o vision and returns a structured
list of detected products with estimated expiry dates.

PIPELINE
--------
  Photo (base64)
    → GPT-4o vision  (identify products + read visible expiry dates)
    → JSON list of {raw_name, expiry_date, quantity}
    → canonicalize raw_name → standard ingredient token
    → return list ready for POST /pantry/{user_id}/bulk

CANONICALIZATION
----------------
Raw names from vision ("Tnuva 3% Milk 500ml", "Heinz Tomato Ketchup")
need to be mapped to the canonical ingredient vocabulary the recipe
matcher uses ("milk", "tomato ketchup").

We do this in two steps:
  1. Strip brand names and quantities using a simple heuristic
  2. Fuzzy-match the cleaned name against the recipe ingredient vocab
     (built from the Food.com dataset via IngredientCanonicalizer)

If the Food.com vocab isn't available (dev mode), we fall back to
basic text cleaning only.

EXPIRY DATE HANDLING
--------------------
GPT-4o can read printed expiry dates from clear photos. When it
can't read one, it returns null and we prompt the user to enter it
manually in the UI. We never invent expiry dates.

PROMPT DESIGN
-------------
The system prompt is carefully constrained to:
  - Output ONLY valid JSON (no preamble, no markdown)
  - Use null for any field it cannot confidently determine
  - Not invent products that aren't visible in the image
  - Return an empty list [] if the image doesn't show food

This prevents hallucination creeping into the pantry.
"""

import base64
import json
import re
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

# OpenAI is optional — only needed if OPENAI_API_KEY is set
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


# ── system prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a pantry scanner. The user has sent a photo of their fridge or pantry.

Your job: identify every food product visible in the image.

Return ONLY a valid JSON array. No preamble, no markdown, no explanation.

Each element must have exactly these fields:
  "raw_name"    : string  - product name as it appears on the packaging (e.g. "Tnuva 3% Milk")
  "expiry_date" : string or null - expiry/best-before date in format "YYYY-MM-DD", null if not readable
  "quantity"    : string or null - quantity/volume if visible (e.g. "500ml", "6 eggs"), null if not visible

Rules:
- Only include items you can clearly see in the image
- Do NOT invent items that aren't visible
- Do NOT guess expiry dates — return null if you cannot read it
- If the image contains no food products, return []
- Return [] if the image is not a fridge or pantry photo

Example output:
[
  {"raw_name": "Tnuva 3% Milk", "expiry_date": "2025-05-10", "quantity": "500ml"},
  {"raw_name": "Free Range Eggs", "expiry_date": null, "quantity": "6"},
  {"raw_name": "Heinz Tomato Ketchup", "expiry_date": "2026-01-15", "quantity": "300ml"}
]"""


# ── ingredient canonicalizer ───────────────────────────────────────────────

# Common brand prefixes / suffixes to strip before fuzzy matching
_BRAND_WORDS = {
    'tnuva', 'heinz', 'nestle', 'danone', 'yoplait', 'kraft', 'kelloggs',
    'nescafe', 'lipton', 'knorr', 'hellmanns', 'campbells', 'del monte',
    'birds eye', 'uncle bens', 'pringles', 'coca cola', 'pepsi',
    'strauss', 'osem', 'telma', 'tara', 'yotvata',
}

# Quantity patterns to strip
_QTY_PATTERN = re.compile(
    r'\b\d+(\.\d+)?\s*(ml|l|g|kg|oz|lb|cl|dl|mg|units?|pcs?|pack|x\d+)\b',
    re.IGNORECASE,
)

# Common adjective noise words that don't help matching
_NOISE = {
    'free', 'range', 'organic', 'natural', 'fresh', 'frozen', 'light',
    'lite', 'low', 'fat', 'semi', 'skimmed', 'whole', 'full', 'extra',
    'premium', 'classic', 'original', 'traditional', 'homemade', 'style',
    'flavour', 'flavor', 'brand', 'new', 'improved',
}


def _clean_raw_name(raw: str) -> str:
    """
    Strip brand names, quantities, and noise adjectives from a raw product
    name to get the core ingredient.

    "Tnuva 3% Milk 500ml" → "milk"
    "Free Range Large Eggs"  → "eggs"
    "Heinz Tomato Ketchup"   → "tomato ketchup"
    """
    text = raw.lower().strip()

    # Remove quantity strings (500ml, 1kg, etc.)
    text = _QTY_PATTERN.sub('', text)

    # Remove brand words
    words = text.split()
    words = [w for w in words if w not in _BRAND_WORDS]

    # Remove noise adjectives
    words = [w for w in words if w not in _NOISE]

    # Remove standalone numbers and percentages
    words = [w for w in words if not re.fullmatch(r'\d+%?', w)]

    result = ' '.join(words).strip()
    return result if result else raw.lower().strip()


class IngredientCanonicalizer:
    """
    Fuzzy-matches cleaned product names against a vocabulary of canonical
    ingredient names built from the recipe corpus.

    Falls back to the cleaned name itself if vocab isn't available or
    no match exceeds the confidence threshold.
    """

    FUZZY_THRESHOLD = 70   # minimum rapidfuzz score to accept a match

    def __init__(self, vocab: Optional[list[str]] = None):
        """
        Args:
            vocab: list of canonical ingredient names, e.g. from
                   the set of unique ingredients across all recipes.
                   If None, canonicalization is best-effort text cleaning only.
        """
        self._vocab = vocab or []

    @classmethod
    def from_db(cls) -> 'IngredientCanonicalizer':
        """
        Build canonicalizer from the recipe DB vocabulary.
        Call this once at server startup for best matching.
        """
        try:
            from backend.db.database import SessionLocal
            from backend.db.models import Recipe
            db = SessionLocal()
            try:
                recipes = db.query(Recipe.ingredients_csv).all()
            finally:
                db.close()

            vocab: set[str] = set()
            for (csv,) in recipes:
                for ing in csv.split(','):
                    cleaned = ing.strip().lower()
                    if cleaned:
                        vocab.add(cleaned)
            return cls(vocab=sorted(vocab))
        except Exception:
            return cls(vocab=[])

    def canonicalize(self, raw_name: str) -> str:
        """
        Clean and match a raw product name to a canonical ingredient.
        Returns the best match, or the cleaned name if no match found.
        """
        cleaned = _clean_raw_name(raw_name)

        if not self._vocab:
            return cleaned

        # rapidfuzz.process.extractOne finds the best match in the vocab
        result = process.extractOne(
            cleaned,
            self._vocab,
            scorer=fuzz.WRatio,
            score_cutoff=self.FUZZY_THRESHOLD,
        )
        if result:
            return result[0]   # matched canonical name
        return cleaned


# ── GPT-4o vision call ─────────────────────────────────────────────────────

def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')


def _call_vision(client: 'OpenAI', image_bytes: bytes) -> list[dict]:
    """
    Send image to GPT-4o vision, parse and return the JSON list.
    Raises ValueError if the response is not valid JSON.
    """
    b64 = _encode_image(image_bytes)

    response = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=1000,
        messages=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {
                        'type':      'image_url',
                        'image_url': {
                            'url':    f'data:image/jpeg;base64,{b64}',
                            'detail': 'low',   # cheaper, sufficient for product labels
                        },
                    },
                    {'type': 'text', 'text': 'Scan this image for food products.'},
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content or '[]'

    # Strip any accidental markdown fences
    raw_text = re.sub(r'```(?:json)?', '', raw_text).strip('` \n')

    try:
        items = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f'GPT-4o returned invalid JSON: {e}\nRaw: {raw_text[:200]}')

    if not isinstance(items, list):
        return []

    return items


# ── main entry point ───────────────────────────────────────────────────────

def scan_image(
    image_bytes: bytes,
    api_key: Optional[str] = None,
    canonicalizer: Optional[IngredientCanonicalizer] = None,
) -> list[dict]:
    """
    Scan a fridge/pantry photo and return a list of pantry items ready
    for POST /pantry/{user_id}/bulk.

    Args:
        image_bytes:    raw bytes of a JPEG/PNG photo
        api_key:        OpenAI API key (falls back to OPENAI_API_KEY env var)
        canonicalizer:  pre-built canonicalizer (None = text cleaning only)

    Returns:
        list of dicts, each matching the PantryItemIn schema:
        {
          "ingredient":  str,          # canonical ingredient name
          "expiry_date": str | None,   # "YYYY-MM-DD" or None
          "raw_name":    str,          # original label text from vision
          "quantity":    str | None,
        }

    Raises:
        RuntimeError: if OpenAI is not installed or API key is missing
        ValueError:   if GPT-4o returns malformed JSON
    """
    if not _openai_available:
        raise RuntimeError(
            'openai package not installed. Run: pip install openai'
        )

    import os
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError(
            'OPENAI_API_KEY not set. Set it as an environment variable '
            'or pass api_key= to scan_image().'
        )

    client = OpenAI(api_key=key)
    canon  = canonicalizer or IngredientCanonicalizer()

    raw_items = _call_vision(client, image_bytes)

    results = []
    for item in raw_items:
        raw_name = item.get('raw_name', '').strip()
        if not raw_name:
            continue

        ingredient = canon.canonicalize(raw_name)

        # Validate expiry date format
        expiry = item.get('expiry_date')
        if expiry and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(expiry)):
            expiry = None   # reject malformed dates

        results.append({
            'ingredient':  ingredient,
            'expiry_date': expiry,
            'raw_name':    raw_name,
            'quantity':    item.get('quantity'),
        })

    return results


# ── dev/test helper: mock scan without OpenAI ─────────────────────────────

def mock_scan() -> list[dict]:
    """
    Returns a realistic fake scan result for development/testing
    without requiring an OpenAI API key.
    """
    from datetime import date, timedelta
    today = date.today()
    return [
        {'ingredient': 'milk',         'expiry_date': (today + timedelta(days=3)).isoformat(),  'raw_name': 'Tnuva 3% Milk',       'quantity': '500ml'},
        {'ingredient': 'eggs',         'expiry_date': (today + timedelta(days=7)).isoformat(),  'raw_name': 'Free Range Eggs',      'quantity': '6'},
        {'ingredient': 'butter',       'expiry_date': (today + timedelta(days=14)).isoformat(), 'raw_name': 'Anchor Butter',        'quantity': '200g'},
        {'ingredient': 'tomatoes',     'expiry_date': (today + timedelta(days=2)).isoformat(),  'raw_name': 'Cherry Tomatoes',      'quantity': '250g'},
        {'ingredient': 'cheddar cheese', 'expiry_date': (today + timedelta(days=10)).isoformat(),'raw_name': 'Mature Cheddar Cheese','quantity': '200g'},
        {'ingredient': 'yogurt',       'expiry_date': (today + timedelta(days=5)).isoformat(),  'raw_name': 'Danone Natural Yogurt','quantity': '150g'},
    ]
