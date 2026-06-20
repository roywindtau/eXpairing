"""
main.py
-------
FastAPI application entry point.

Start the server:
    uvicorn backend.main:app --reload --port 8000

Then open:
    http://localhost:8000/docs   -- interactive Swagger UI
    http://localhost:8000/redoc -- ReDoc UI
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from a .env file at the project root (if present)
# so vision API keys (GEMINI_API_KEY / OPENAI_API_KEY) are available to the app.
load_dotenv()

from backend.db.database import init_db
from backend.routers import wine, pantry, recipes, users, shopping


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup if they don't exist yet."""
    init_db()
    yield


app = FastAPI(
    title="Fridge2Fork API",
    description="Recipe recommendation engine that minimizes food waste.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the React dev server (port 5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(pantry.router)
app.include_router(recipes.router)
app.include_router(shopping.router)
app.include_router(wine.router)


@app.get("/health")
def health():
    return {"status": "ok"}

# vision router (added separately so it's optional when openai not installed)
try:
    from backend.routers import vision as vision_router
    app.include_router(vision_router.router)
except ImportError:
    pass
