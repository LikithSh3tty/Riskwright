"""Riskwright API.

A thin HTTP wrapper over src/. No business logic lives here: routers validate
input, call into src/, and shape the response. Keeping it thin is what lets the
UI be swapped without touching anything that computes.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import explain, health, predict
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Riskwright API",
    description=(
        "Credit risk scoring, explanation, derived rules, and natural-language "
        "querying over the Home Credit Default Risk dataset."
    ),
    version="0.1.0",
)

# The Streamlit UI is a separate container and therefore a separate origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)


@app.on_event("startup")
def log_startup() -> None:
    settings = get_settings()
    log.info("Riskwright API starting")
    log.info("database host: %s", settings.postgres_host)
    # Surfaced at startup so a missing key is visible immediately rather than
    # on the first chat request.
    log.info(
        "LLM: %s",
        settings.anthropic_model if settings.llm_configured else "NOT CONFIGURED",
    )


@app.get("/")
def root() -> dict:
    return {"service": "riskwright", "docs": "/docs", "health": "/health"}
