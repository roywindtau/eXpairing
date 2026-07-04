"""
canonicalizer/openfoodfacts.py
-------------------------------
Looks up food products in the Open Food Facts database.

Open Food Facts is a free, open-source database of 3M+ food products.
We use it in two ways:

1. BARCODE LOOKUP  (most accurate)
   When the vision agent or user scans a barcode, we fetch the exact
   product entry and extract the primary ingredient.
   API: https://world.openfoodfacts.org/api/v0/product/{barcode}.json

2. PRODUCT NAME SEARCH  (fallback)
   When we have a product name but no barcode, we search the API.
   API: https://world.openfoodfacts.org/cgi/search.pl

Both return a canonical ingredient name by taking the first item in the
product's ingredient list after stripping quantities and additives.

CACHING
-------
Results are cached in-memory for the session. Barcode lookups are
also cached to disk (openfoodfacts_cache.json) to avoid re-hitting
the API on repeated runs.

RATE LIMITING
-------------
Open Food Facts asks for a User-Agent identifying your app.
We include one. No API key required — it's fully free/open.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

from backend.canonicalizer.ingredient_map import clean_product_name

# Open Food Facts endpoints
_BASE       = "https://world.openfoodfacts.org"
_PRODUCT_EP = _BASE + "/api/v0/product/{barcode}.json"
_SEARCH_EP  = _BASE + "/cgi/search.pl"

# Polite User-Agent as requested by OFF
_HEADERS = {
    "User-Agent": "eXpairing/1.0 (github.com/expairing; educational project)",
}

# Disk cache for barcode lookups
_CACHE_PATH   = Path("openfoodfacts_cache.json")
_memory_cache: dict[str, Optional[str]] = {}

# Rate limiting — OFF asks for max 100 req/min
_last_request = 0.0
_MIN_INTERVAL = 0.7   # seconds between requests


def _rate_limit():
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def _load_disk_cache() -> dict[str, Optional[str]]:
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_disk_cache(cache: dict) -> None:
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _extract_primary_ingredient(product_data: dict) -> Optional[str]:
    """
    Extract the primary ingredient name from an OFF product entry.

    Tries, in order:
      1. First item in ingredients_tags (most structured)
      2. First item in ingredients list
      3. product_name_en / product_name (fallback to name cleaning)
    """
    # Method 1: ingredients_tags e.g. ["en:wheat-flour", "en:sugar", ...]
    tags = product_data.get("ingredients_tags", [])
    if tags:
        # Strip language prefix and hyphens: "en:wheat-flour" → "wheat flour"
        first = tags[0]
        first = re.sub(r"^[a-z]{2}:", "", first)
        first = first.replace("-", " ").strip()
        if first and len(first) > 1:
            return clean_product_name(first)

    # Method 2: ingredients list
    ingredients = product_data.get("ingredients", [])
    if ingredients and isinstance(ingredients, list):
        first = ingredients[0].get("text", "")
        if first:
            return clean_product_name(str(first))

    # Method 3: fall back to product name
    name = (product_data.get("product_name_en")
            or product_data.get("product_name", ""))
    if name:
        return clean_product_name(name)

    return None


def lookup_barcode(barcode: str) -> Optional[str]:
    """
    Look up a product by barcode and return the primary ingredient name.

    Args:
        barcode: EAN-13, UPC-A, or similar barcode string

    Returns:
        canonical ingredient name, or None if not found

    Example:
        lookup_barcode("7290000066608")  # Tnuva milk → "milk"
    """
    # Check memory cache first
    if barcode in _memory_cache:
        return _memory_cache[barcode]

    # Check disk cache
    disk_cache = _load_disk_cache()
    if barcode in disk_cache:
        result = disk_cache[barcode]
        _memory_cache[barcode] = result
        return result

    # Hit the API
    _rate_limit()
    try:
        resp = requests.get(
            _PRODUCT_EP.format(barcode=barcode),
            headers=_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if data.get("status") != 1 or "product" not in data:
        result = None
    else:
        result = _extract_primary_ingredient(data["product"])

    # Cache result
    _memory_cache[barcode] = result
    disk_cache[barcode] = result
    _save_disk_cache(disk_cache)

    return result


def search_product(name: str, max_results: int = 3) -> Optional[str]:
    """
    Search OFF by product name and return the best-matching ingredient.

    Less reliable than barcode lookup. Used as a fallback when the
    vision agent can read a product name but not a barcode.

    Args:
        name:        product name to search for
        max_results: how many OFF results to consider

    Returns:
        canonical ingredient name from the top result, or None
    """
    cache_key = f"search:{name.lower()}"
    if cache_key in _memory_cache:
        return _memory_cache[cache_key]

    _rate_limit()
    try:
        resp = requests.get(
            _SEARCH_EP,
            params={
                "search_terms":   name,
                "search_simple":  1,
                "action":         "process",
                "json":           1,
                "page_size":      max_results,
                "fields":         "product_name,ingredients_tags,ingredients",
            },
            headers=_HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    products = data.get("products", [])
    if not products:
        _memory_cache[cache_key] = None
        return None

    # Take the first result
    result = _extract_primary_ingredient(products[0])
    _memory_cache[cache_key] = result
    return result


def enrich_pantry_item(
    raw_name: str,
    barcode: Optional[str] = None,
) -> Optional[str]:
    """
    Best-effort ingredient lookup. Tries barcode first, then name search.

    This is the main entry point called by the vision agent and the
    manual pantry add flow.

    Returns canonical ingredient name, or None if lookup fails.
    """
    if barcode:
        result = lookup_barcode(barcode)
        if result:
            return result

    return search_product(raw_name)
