"""Exploratory analysis endpoints.

Aggregates only. Bin arrays and group counts go to the client; rows never do.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.data import eda
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("/summary")
def summary() -> dict:
    return eda.dataset_summary()


@router.get("/columns")
def columns() -> dict:
    """What the distribution and group-by explorers can be pointed at."""
    return {
        "numeric": sorted(eda.NUMERIC_COLUMNS),
        "categorical": sorted(eda.CATEGORICAL_COLUMNS),
    }


@router.get("/distribution")
def distribution(
    column: str = Query(default="age_years"),
    bins: int = Query(default=30, ge=5, le=100),
) -> dict:
    try:
        return eda.distribution(column, bins)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/groupby")
def groupby(dimension: str = Query(default="name_education_type")) -> dict:
    try:
        return eda.group_default_rate(dimension)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/insights")
def insights() -> dict:
    """The five business insights.

    Served from one place so the notebook, the UI, and the README cannot quote
    different numbers for the same finding.
    """
    return eda.business_insights()
