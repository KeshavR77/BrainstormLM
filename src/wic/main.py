"""
main.py — FastAPI application entry point for the WIC Engine.

Mounts the session router and manages DB lifecycle hooks.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from brainstormlm.wic.api import router as session_router
from brainstormlm.wic.database import close_pool, get_pool, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to DB & run migrations. Shutdown: close pool."""
    await get_pool()
    await init_db()
    yield
    await close_pool()


app = FastAPI(
    title="Recursive WIC Engine",
    description="Recursive What-If Cascade brainstorming engine",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)


@app.get("/")
async def root():
    return {"message": "WIC Engine is running 🚀"}
