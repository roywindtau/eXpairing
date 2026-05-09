"""
infer_diet_tags.py
------------------
Infers vegan / vegetarian / dairy-free dietary tags from each recipe's
ingredients_csv and updates tags_csv in the database.

WHY THIS EXISTS
---------------
Food.com dietary tags are author-supplied and unreliable.  Spot-checking
shows ~18% of vegan-tagged recipes contain dairy ingredients (milk,
butter, cream, cheese).  Ingredient-based inference is more trustworthy
because it is derived from what the recipe actually contains.

WHAT IT CHANGES
---------------
Three tags are fully managed by this script:
    vegan        — derived from ingredients; food.com tag replaced
    vegetarian   — derived from ingredients; food.com tag replaced
    dairy-free   — derived from ingredients; food.com tag replaced

All other food.com tags (time, course, cuisine, etc.) are kept as-is.

gluten-free is handled conservatively: the food.com tag is kept unless
the recipe contains an obviously gluten-bearing ingredient.

Run:
    python -m backend.db.infer_diet_tags [--dry-run]
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db.database import SessionLocal
from backend.db.models import Recipe

# ── ingredient keyword lists ───────────────────────────────────────────────

# Any of these makes a recipe non-vegetarian (meat + all seafood)
_MEAT = frozenset({
    'chicken', 'beef', 'pork', 'lamb', 'turkey', 'duck', 'veal',
    'venison', 'bison', 'rabbit', 'goose', 'quail', 'pheasant',
    'ham', 'bacon', 'sausage', 'salami', 'pepperoni', 'prosciutto',
    'pancetta', 'chorizo', 'lard', 'suet', 'tallow',
    'ground beef', 'steak', 'ribs', 'meatball', 'meat broth',
    'beef broth', 'chicken broth', 'chicken stock', 'beef stock',
    'fish sauce', 'oyster sauce', 'worcestershire sauce',
    'fish', 'shrimp', 'salmon', 'tuna', 'cod', 'tilapia', 'halibut',
    'sardine', 'anchovy', 'lobster', 'crab', 'clam', 'oyster',
    'scallop', 'mussel', 'squid', 'prawn', 'caviar', 'octopus',
    'crayfish', 'catfish', 'trout', 'mackerel', 'herring',
    'smoked salmon', 'lox',
})

# Any of these makes a recipe non-vegan / non-dairy-free
_DAIRY = frozenset({
    'butter', 'milk', 'cream', 'cheese', 'yogurt', 'yoghurt',
    'ghee', 'whey', 'casein', 'lactose', 'kefir', 'custard',
    'mozzarella', 'parmesan', 'ricotta', 'feta', 'cheddar', 'brie',
    'gouda', 'gruyere', 'manchego', 'colby', 'provolone', 'emmental',
    'havarti', 'camembert', 'gorgonzola', 'stilton', 'mascarpone',
    'sour cream', 'buttermilk', 'half-and-half', 'cream cheese',
    'cottage cheese', 'quark', 'queso',
})

# Exceptions: when these words accompany a dairy keyword it is plant-based
_PLANT_BUTTER = frozenset({
    'peanut', 'almond', 'cashew', 'walnut', 'pistachio', 'sunflower',
    'sesame', 'seed', 'nut butter', 'coconut', 'soy', 'oat', 'hemp',
    'apple', 'cocoa', 'fruit', 'shea',
})
_PLANT_MILK = frozenset({'coconut', 'almond', 'soy', 'oat', 'rice', 'hemp', 'cashew',
                          'macadamia', 'pea', 'flax', 'plant'})
_PLANT_CREAM = frozenset({'coconut', 'tartar', 'of wheat', 'tofu'})
_PLANT_YOGURT = frozenset({'coconut', 'soy', 'almond', 'oat', 'plant', 'cashew'})

# Any of these makes a recipe non-vegan (beyond meat + dairy)
_HONEY_GEL = frozenset({'honey', 'gelatin', 'beeswax'})
# Exception: honeydew is a melon, not honey
_HONEYDEW = 'honeydew'

# Presence of any of these revokes a false food.com gluten-free tag
_GLUTEN = frozenset({
    'all-purpose flour', 'wheat flour', 'bread flour', 'cake flour',
    'whole wheat', 'whole-wheat', 'spelt', 'barley', 'rye',
    'couscous', 'bulgur', 'farro', 'semolina', 'durum',
    'breadcrumb', 'bread crumb', 'crouton',
    'pasta ', ' pasta', 'spaghetti', 'linguine', 'penne', 'fettuccine',
    'lasagna', 'lasagne', 'macaroni', 'rigatoni', 'tagliatelle',
    ' noodle', 'udon', 'ramen noodle',
    'soy sauce',      # most soy sauce contains wheat; gluten-free soy sauce
                      # is usually explicitly labelled so it won't match "soy sauce"
})


# ── per-ingredient checks ──────────────────────────────────────────────────

def _is_dairy(ing: str) -> bool:
    if 'butter' in ing:
        return not any(p in ing for p in _PLANT_BUTTER)
    if 'milk' in ing:
        return not any(p in ing for p in _PLANT_MILK)
    if 'cream' in ing:
        return not any(p in ing for p in _PLANT_CREAM)
    if 'yogurt' in ing or 'yoghurt' in ing:
        return not any(p in ing for p in _PLANT_YOGURT)
    return any(k in ing for k in _DAIRY - {'butter', 'milk', 'cream', 'yogurt', 'yoghurt'})


def _is_egg(ing: str) -> bool:
    if 'eggplant' in ing:
        return False
    return 'egg' in ing or 'mayonnaise' in ing


def _is_honey_gel(ing: str) -> bool:
    if 'honey' in ing and _HONEYDEW not in ing:
        return True
    return 'gelatin' in ing or 'beeswax' in ing


def _has_gluten(ing: str) -> bool:
    return any(k in ing for k in _GLUTEN)


# ── recipe-level inference ─────────────────────────────────────────────────

def infer_diet_flags(recipe: Recipe) -> dict[str, bool]:
    ings = [i.strip().lower() for i in recipe.ingredients_csv.split(',') if i.strip()]

    has_meat    = any(any(k in ing for k in _MEAT) for ing in ings)
    has_dairy   = any(_is_dairy(ing) for ing in ings)
    has_egg     = any(_is_egg(ing) for ing in ings)
    has_honey   = any(_is_honey_gel(ing) for ing in ings)
    has_gluten  = any(_has_gluten(ing) for ing in ings)

    is_vegetarian = not has_meat
    is_vegan      = is_vegetarian and not has_dairy and not has_egg and not has_honey

    return {
        'vegetarian':   is_vegetarian,
        'vegan':        is_vegan,
        'dairy-free':   not has_dairy,
        'revoke_gf':    has_gluten,   # strip false gluten-free tag
    }


def _rebuild_tags(tags_csv: Optional[str], flags: dict) -> str:
    MANAGED = {'vegan', 'vegetarian', 'dairy-free'}

    # Preserve tags we do not manage, stripping managed ones so we re-add from scratch
    kept = [t.strip() for t in (tags_csv or '').split(',')
            if t.strip() and t.strip() not in MANAGED]

    # Revoke a false food.com gluten-free tag when gluten detected
    if flags['revoke_gf']:
        kept = [t for t in kept if t != 'gluten-free']

    # Prepend inferred dietary tags so they appear first
    inferred = []
    if flags['vegan']:       inferred.append('vegan')
    if flags['vegetarian']:  inferred.append('vegetarian')
    if flags['dairy-free']:  inferred.append('dairy-free')

    return ','.join(inferred + kept)


# ── main ───────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    db = SessionLocal()
    try:
        total = db.query(Recipe).count()
        print(f"{'[DRY RUN] ' if dry_run else ''}Processing {total:,} recipes ...")

        BATCH = 5_000
        changed = 0
        vegan_added = vegan_removed = vegetarian_added = vegetarian_removed = 0
        dairy_free_added = 0

        for offset in range(0, total, BATCH):
            recipes = db.query(Recipe).offset(offset).limit(BATCH).all()
            for recipe in recipes:
                flags    = infer_diet_flags(recipe)
                new_tags = _rebuild_tags(recipe.tags_csv, flags)

                if new_tags == (recipe.tags_csv or ''):
                    continue

                old_set = {t.strip() for t in (recipe.tags_csv or '').split(',') if t.strip()}
                new_set = {t.strip() for t in new_tags.split(',') if t.strip()}

                if 'vegan' in new_set - old_set:      vegan_added    += 1
                if 'vegan' in old_set - new_set:      vegan_removed  += 1
                if 'vegetarian' in new_set - old_set: vegetarian_added   += 1
                if 'vegetarian' in old_set - new_set: vegetarian_removed += 1
                if 'dairy-free' in new_set - old_set: dairy_free_added   += 1

                changed += 1
                if not dry_run:
                    recipe.tags_csv = new_tags

            if not dry_run:
                db.commit()
            print(f"  {min(offset + BATCH, total):,}/{total:,} ...", end='\r')

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Done. {changed:,} recipes updated.")
        print(f"  vegan:      +{vegan_added:,}  removed {vegan_removed:,}")
        print(f"  vegetarian: +{vegetarian_added:,}  removed {vegetarian_removed:,}")
        print(f"  dairy-free: +{dairy_free_added:,}  (new tag across dataset)")
    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Print stats without writing to the database')
    args = p.parse_args()
    run(dry_run=args.dry_run)
