"""
seed_dev.py
-----------
Seeds the DB with a demo user + pantry + 20 realistic recipes.
Run once before starting the server for the first time:

    python -m backend.db.seed_dev

This gives you a working demo without the full Food.com dataset.
For production, use seed_recipes.py + seed_ratings.py instead.
"""

import json
from datetime import date, timedelta
from backend.db.database import init_db, SessionLocal
from backend.db.models import User, PantryItem, Recipe


def today_plus(days: int) -> date:
    return date.today() + timedelta(days=days)


RECIPES = [
    {
        "name": "French toast",
        "ingredients": "eggs,milk,bread,butter,vanilla extract,cinnamon,maple syrup",
        "tags": "breakfast,quick,vegetarian",
        "minutes": 15,
        "rating": 4.6,
        "description": "Golden, custardy French toast made with thick-cut bread soaked in a vanilla-cinnamon egg mixture. Ready in 15 minutes.",
        "steps": [
            "Whisk together eggs, milk, vanilla extract, and cinnamon in a shallow bowl.",
            "Heat butter in a large skillet over medium heat until foamy.",
            "Dip each bread slice into the egg mixture, letting it soak for 10 seconds per side.",
            "Cook slices for 2–3 minutes per side until golden brown.",
            "Serve immediately with maple syrup.",
        ],
    },
    {
        "name": "Pasta carbonara",
        "ingredients": "pasta,eggs,pancetta,parmesan,black pepper,garlic",
        "tags": "dinner,italian",
        "minutes": 25,
        "rating": 4.7,
        "description": "Classic Roman carbonara with crispy pancetta and a silky egg-parmesan sauce. No cream needed.",
        "steps": [
            "Cook pasta in well-salted boiling water until al dente. Reserve 1 cup pasta water before draining.",
            "Fry pancetta in a large pan over medium heat until crispy. Add garlic for the last minute.",
            "Whisk eggs, parmesan, and plenty of black pepper in a bowl.",
            "Remove pan from heat. Add drained pasta and toss to coat in the pancetta fat.",
            "Pour in egg mixture, tossing rapidly while adding pasta water a splash at a time until creamy.",
            "Serve immediately with extra parmesan and black pepper.",
        ],
    },
    {
        "name": "Scrambled eggs on toast",
        "ingredients": "eggs,bread,butter,salt,black pepper,chives",
        "tags": "breakfast,quick,vegetarian",
        "minutes": 10,
        "rating": 4.2,
        "description": "Soft, creamy scrambled eggs on buttered toast. The key is low heat and patience.",
        "steps": [
            "Toast bread slices and butter them.",
            "Crack eggs into a cold non-stick pan with butter, salt, and pepper.",
            "Place over low heat and stir constantly with a rubber spatula.",
            "Remove from heat just before fully set — residual heat finishes cooking.",
            "Pile onto toast and garnish with chopped chives.",
        ],
    },
    {
        "name": "Banana pancakes",
        "ingredients": "banana,eggs,flour,milk,baking powder,butter,honey",
        "tags": "breakfast,vegetarian",
        "minutes": 20,
        "rating": 4.5,
        "description": "Fluffy banana pancakes with natural sweetness from ripe bananas. Great weekend breakfast.",
        "steps": [
            "Mash banana in a large bowl until smooth.",
            "Whisk in eggs, milk, and melted butter.",
            "Sift in flour and baking powder, stir until just combined — lumps are fine.",
            "Heat a non-stick pan over medium heat, lightly grease with butter.",
            "Pour 1/4 cup batter per pancake. Cook until bubbles form, then flip.",
            "Cook 1 minute more. Serve with honey.",
        ],
    },
    {
        "name": "Grilled cheese sandwich",
        "ingredients": "bread,cheddar cheese,butter,mustard",
        "tags": "lunch,quick,vegetarian",
        "minutes": 10,
        "rating": 4.3,
        "description": "Crispy buttered bread with melted cheddar. A thin layer of mustard inside elevates the classic.",
        "steps": [
            "Spread mustard on one side of each bread slice.",
            "Layer cheddar cheese between the mustard-side faces.",
            "Butter the outside of both bread slices generously.",
            "Cook in a pan over medium-low heat for 3–4 minutes per side until golden and cheese is melted.",
            "Cut diagonally and serve immediately.",
        ],
    },
    {
        "name": "Tomato omelette",
        "ingredients": "eggs,tomatoes,onion,olive oil,salt,black pepper,parsley",
        "tags": "breakfast,lunch,vegetarian,quick,gluten-free,dairy-free",
        "minutes": 12,
        "rating": 4.4,
        "description": "Light and fresh omelette with sautéed tomatoes and onion. Gluten-free and dairy-free.",
        "steps": [
            "Dice tomatoes and onion. Sauté onion in olive oil over medium heat for 3 minutes.",
            "Add tomatoes, season with salt and pepper, cook 2 minutes until softened.",
            "Beat eggs with a pinch of salt. Push tomato mixture to one side of the pan.",
            "Pour eggs into the empty side and swirl to cover the pan base.",
            "Once edges set, fold omelette over the filling. Cook 1 minute more.",
            "Slide onto a plate and garnish with fresh parsley.",
        ],
    },
    {
        "name": "Milk rice pudding",
        "ingredients": "milk,rice,sugar,vanilla extract,cinnamon,butter",
        "tags": "dessert,vegetarian,gluten-free",
        "minutes": 40,
        "rating": 4.0,
        "description": "Creamy, comforting rice pudding simmered slowly in milk with vanilla and cinnamon.",
        "steps": [
            "Combine milk, rice, sugar, and a pinch of salt in a heavy saucepan.",
            "Bring to a gentle simmer over medium heat, stirring frequently.",
            "Reduce heat to low, add vanilla extract, and cook 30–35 minutes stirring often until thick.",
            "Remove from heat, stir in butter.",
            "Serve warm or cold, dusted with cinnamon.",
        ],
    },
    {
        "name": "Butter pasta",
        "ingredients": "pasta,butter,parmesan,black pepper,salt,garlic",
        "tags": "dinner,quick,vegetarian",
        "minutes": 15,
        "rating": 4.1,
        "description": "Simple Italian pasta tossed with butter, garlic, and parmesan. Ready in 15 minutes.",
        "steps": [
            "Cook pasta in well-salted boiling water until al dente. Reserve 1/2 cup pasta water.",
            "Melt butter in a large pan over medium heat. Add minced garlic, cook 1 minute.",
            "Add drained pasta and a splash of pasta water. Toss to coat.",
            "Remove from heat, add parmesan and black pepper, toss vigorously until creamy.",
            "Serve immediately with extra parmesan.",
        ],
    },
    {
        "name": "Cheese omelette",
        "ingredients": "eggs,cheddar cheese,butter,salt,black pepper,chives",
        "tags": "breakfast,quick,vegetarian,gluten-free",
        "minutes": 8,
        "rating": 4.3,
        "description": "Classic folded omelette with melted cheddar cheese. Quick gluten-free breakfast.",
        "steps": [
            "Beat eggs with salt and pepper until uniform.",
            "Melt butter in a non-stick pan over medium-high heat.",
            "Pour in eggs and swirl to coat the pan. Cook, lifting edges to let raw egg flow underneath.",
            "When almost set, sprinkle cheddar over one half.",
            "Fold omelette over the cheese. Slide onto plate and garnish with chives.",
        ],
    },
    {
        "name": "Banana bread",
        "ingredients": "banana,flour,eggs,butter,sugar,baking soda,vanilla extract,salt",
        "tags": "baking,vegetarian,snack",
        "minutes": 65,
        "rating": 4.5,
        "description": "Moist, dense banana bread using very ripe bananas. Perfect for overripe bananas.",
        "steps": [
            "Preheat oven to 175°C (350°F). Grease a loaf pan.",
            "Mash 3 ripe bananas thoroughly in a large bowl.",
            "Mix in melted butter, sugar, egg, and vanilla extract.",
            "Stir in flour, baking soda, and salt until just combined.",
            "Pour batter into loaf pan and bake 55–60 minutes until a skewer comes out clean.",
            "Cool in pan 10 minutes before turning out.",
        ],
    },
    {
        "name": "Garlic bread",
        "ingredients": "bread,butter,garlic,parsley,olive oil",
        "tags": "side,quick,vegetarian",
        "minutes": 10,
        "rating": 4.2,
        "description": "Crispy garlic bread with herb butter. Perfect side for pasta and soups.",
        "steps": [
            "Preheat grill/broiler. Mix softened butter with minced garlic, chopped parsley, and a drizzle of olive oil.",
            "Slice bread in half lengthways or cut into thick slices.",
            "Spread garlic butter generously over cut surfaces.",
            "Grill for 3–4 minutes until golden and crispy at the edges.",
            "Slice and serve immediately.",
        ],
    },
    {
        "name": "Simple tomato pasta",
        "ingredients": "pasta,tomatoes,garlic,olive oil,basil,salt,parmesan",
        "tags": "dinner,vegetarian,quick",
        "minutes": 20,
        "rating": 4.4,
        "description": "Fresh tomato pasta with garlic and basil. A summer staple that highlights good tomatoes.",
        "steps": [
            "Cook pasta in well-salted boiling water until al dente.",
            "While pasta cooks, warm olive oil in a pan over medium heat. Add sliced garlic, cook 1 minute.",
            "Add diced tomatoes and a pinch of salt. Cook 5 minutes until saucy.",
            "Drain pasta, reserving some cooking water. Add pasta to the sauce and toss.",
            "Remove from heat, add fresh basil and parmesan. Toss and serve.",
        ],
    },
    {
        "name": "Egg fried rice",
        "ingredients": "rice,eggs,soy sauce,garlic,onion,sesame oil,frozen peas,carrot",
        "tags": "dinner,asian,quick,gluten-free,dairy-free,vegetarian",
        "minutes": 20,
        "rating": 4.6,
        "description": "Quick egg fried rice with vegetables. Best made with day-old cold rice.",
        "steps": [
            "Heat oil in a wok or large pan over high heat. Add diced onion and carrot, stir-fry 3 minutes.",
            "Add minced garlic and frozen peas, stir-fry 1 minute.",
            "Push vegetables to the side, scramble eggs in the empty space until just set.",
            "Add cold cooked rice and break up any clumps. Toss everything together.",
            "Drizzle soy sauce and sesame oil over the rice. Toss and taste for seasoning.",
            "Serve immediately.",
        ],
    },
    {
        "name": "Avocado toast",
        "ingredients": "bread,avocado,lemon,salt,chili flakes,olive oil,eggs",
        "tags": "breakfast,vegetarian,quick",
        "minutes": 8,
        "rating": 4.7,
        "description": "Creamy mashed avocado on toasted bread with lemon and chili. Top with a poached egg.",
        "steps": [
            "Toast bread until golden and crispy.",
            "Halve and pit the avocado. Scoop flesh into a bowl.",
            "Mash avocado with lemon juice, salt, and chili flakes to desired texture.",
            "Spread avocado mixture generously on toast.",
            "Optional: top with a fried or poached egg and a drizzle of olive oil.",
        ],
    },
    {
        "name": "Classic chicken soup",
        "ingredients": "chicken breast,carrot,celery,onion,garlic,chicken stock,salt,black pepper,parsley",
        "tags": "dinner,lunch,soup,gluten-free,dairy-free",
        "minutes": 60,
        "rating": 4.5,
        "description": "Nourishing chicken soup with vegetables simmered in rich stock. Gluten-free and dairy-free.",
        "steps": [
            "Dice onion, carrot, and celery. Mince garlic.",
            "Bring chicken stock to a boil in a large pot. Add chicken breasts whole.",
            "Add vegetables and garlic. Season with salt and pepper.",
            "Simmer on medium-low heat for 30 minutes.",
            "Remove chicken, shred with two forks, and return to pot.",
            "Simmer 15 more minutes. Taste and adjust seasoning. Serve with fresh parsley.",
        ],
    },
    {
        "name": "Vegetable stir fry",
        "ingredients": "carrot,broccoli,bell pepper,soy sauce,garlic,ginger,sesame oil,rice",
        "tags": "dinner,vegetarian,vegan,asian,quick,gluten-free,dairy-free",
        "minutes": 20,
        "rating": 4.3,
        "description": "Fast vegan stir fry with crispy vegetables in a ginger-garlic sauce. Serve over rice.",
        "steps": [
            "Cook rice according to package instructions.",
            "Cut carrot, broccoli, and bell pepper into bite-sized pieces.",
            "Heat sesame oil in a wok over high heat. Add carrots, stir-fry 2 minutes.",
            "Add broccoli and bell pepper, stir-fry 3 minutes.",
            "Add minced garlic and grated ginger, stir-fry 1 minute.",
            "Pour in soy sauce and toss to coat. Serve immediately over rice.",
        ],
    },
    {
        "name": "Cheese quesadilla",
        "ingredients": "tortilla,cheddar cheese,butter,sour cream,salsa",
        "tags": "lunch,quick,vegetarian",
        "minutes": 10,
        "rating": 4.1,
        "description": "Crispy flour tortilla filled with melted cheddar. Serve with sour cream and salsa.",
        "steps": [
            "Heat a dry pan over medium heat.",
            "Place one tortilla in the pan. Sprinkle grated cheddar over one half.",
            "Fold the tortilla over the cheese. Press lightly with a spatula.",
            "Cook 2–3 minutes per side until golden and cheese is fully melted.",
            "Cut into wedges and serve with sour cream and salsa.",
        ],
    },
    {
        "name": "Lobster bisque",
        "ingredients": "lobster,cream,shallots,brandy,tomato paste,fish stock,butter,tarragon",
        "tags": "dinner,seafood,fancy,gluten-free",
        "minutes": 75,
        "rating": 4.8,
        "description": "Luxurious creamy lobster bisque with brandy and fresh tarragon. Restaurant quality at home.",
        "steps": [
            "Cook lobster, remove meat, and reserve shells. Chop lobster meat and set aside.",
            "Sauté minced shallots in butter over medium heat until soft, about 5 minutes.",
            "Add tomato paste and cook 2 minutes. Deglaze with brandy, stirring up any bits.",
            "Add fish stock and lobster shells. Simmer 30 minutes.",
            "Strain stock through a fine sieve. Return liquid to pot.",
            "Stir in cream and chopped lobster meat. Simmer gently 10 minutes. Season and garnish with tarragon.",
        ],
    },
    {
        "name": "Lamb tagine",
        "ingredients": "lamb,chickpeas,apricots,onion,tomatoes,cumin,coriander,cinnamon,chicken stock",
        "tags": "dinner,moroccan,gluten-free,dairy-free",
        "minutes": 120,
        "rating": 4.7,
        "description": "Slow-cooked Moroccan lamb tagine with chickpeas and apricots. Warming and deeply flavourful.",
        "steps": [
            "Cut lamb into chunks. Brown in batches in a heavy pot over high heat. Set aside.",
            "Sauté diced onion in the same pot until soft. Add cumin, coriander, and cinnamon, cook 1 minute.",
            "Return lamb to pot. Add diced tomatoes, chicken stock, and apricots.",
            "Bring to a boil, then reduce heat to low. Cover and simmer 1 hour.",
            "Add drained chickpeas and cook a further 20 minutes until lamb is very tender.",
            "Taste and adjust seasoning. Serve with couscous or flatbread.",
        ],
    },
    {
        "name": "Classic pancakes",
        "ingredients": "flour,milk,eggs,butter,sugar,baking powder,salt,vanilla extract",
        "tags": "breakfast,vegetarian",
        "minutes": 20,
        "rating": 4.4,
        "description": "Fluffy American-style pancakes. Light inside, golden outside. Perfect stack every time.",
        "steps": [
            "Whisk together flour, baking powder, sugar, and salt in a bowl.",
            "In another bowl, whisk milk, eggs, melted butter, and vanilla.",
            "Pour wet ingredients into dry and stir until just combined — do not overmix.",
            "Heat a non-stick pan over medium heat, grease lightly with butter.",
            "Pour 1/4 cup batter per pancake. Cook until bubbles appear and edges look set, then flip.",
            "Cook 1–2 minutes more. Serve in a stack with butter and maple syrup.",
        ],
    },
]

DEMO_PANTRY = [
    {"ingredient": "milk",        "expiry_date": today_plus(2),  "quantity": "500ml"},
    {"ingredient": "eggs",        "expiry_date": today_plus(3),  "quantity": "6"},
    {"ingredient": "butter",      "expiry_date": today_plus(14), "quantity": "250g"},
    {"ingredient": "bread",       "expiry_date": today_plus(4),  "quantity": "1 loaf"},
    {"ingredient": "cheddar cheese", "expiry_date": today_plus(7), "quantity": "200g"},
    {"ingredient": "tomatoes",    "expiry_date": today_plus(2),  "quantity": "4"},
    {"ingredient": "pasta",       "expiry_date": today_plus(180),"quantity": "500g"},
    {"ingredient": "garlic",      "expiry_date": today_plus(30), "quantity": "1 bulb"},
    {"ingredient": "banana",      "expiry_date": today_plus(2),  "quantity": "3"},
    {"ingredient": "parmesan",    "expiry_date": today_plus(21), "quantity": "100g"},
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        # Idempotent: skip if already seeded
        if db.query(Recipe).count() > 0:
            print("Already seeded. Nothing to do.")
            return

        # Seed recipes
        for r in RECIPES:
            steps = r.get("steps", [])
            recipe = Recipe(
                name=r["name"],
                ingredients_csv=r["ingredients"],
                tags_csv=r["tags"],
                minutes=r["minutes"],
                n_steps=len(steps) if steps else None,
                avg_rating=r.get("rating", 4.2),
                n_ratings=100,
                description=r.get("description"),
                steps_json=json.dumps(steps) if steps else None,
            )
            db.add(recipe)
        db.commit()
        print(f"Seeded {len(RECIPES)} recipes.")

        # Create demo user
        user = User(name="Demo user", beta=0.35)
        db.add(user)
        db.commit()

        # Seed pantry
        for item in DEMO_PANTRY:
            db.add(PantryItem(
                user_id=user.id,
                ingredient=item["ingredient"],
                expiry_date=item["expiry_date"],
                quantity=item["quantity"],
            ))
        db.commit()
        print(f"Created user id={user.id} with {len(DEMO_PANTRY)} pantry items.")
        print("\nDemo pantry (sorted by expiry):")
        for item in sorted(DEMO_PANTRY, key=lambda x: x["expiry_date"]):
            days = (item["expiry_date"] - date.today()).days
            print(f"  {item['ingredient']:20s} expires in {days} days")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
