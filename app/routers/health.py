"""Health endpoint.

Reports each dependency separately rather than a single boolean, because the
usual startup failure is "Postgres is not ready yet" and a bare
{"status": "error"} tells nobody which piece is missing.
"""

from __future__ import annotations

import psycopg2
from fastapi import APIRouter

from src.utils.config import get_settings
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["health"])


def _database_ready() -> tuple[bool, str | None]:
    settings = get_settings()
    try:
        with psycopg2.connect(settings.dsn(), connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, None
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0]


def _model_loaded() -> bool:
    return (models_dir() / "lightgbm_model.joblib").exists()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    db_ok, db_error = _database_ready()
    model_ok = _model_loaded()

    return {
        "status": "ok" if db_ok else "degraded",
        "database": {"connected": db_ok, "error": db_error},
        "model_loaded": model_ok,
        "llm_configured": settings.llm_configured,
        "llm_model": settings.anthropic_model if settings.llm_configured else None,
    }
