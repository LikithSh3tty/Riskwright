"""Explainability and derived-rule endpoints.

Everything here returns JSON. Charts are the UI's job, which is what keeps the
frontend replaceable.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.routers.predict import PredictRequest, _require_model
from src.ml import explain as explanation
from src.ml.rules import load_rules
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["explainability"])


@router.post("/explain")
def explain(request: PredictRequest) -> dict:
    """Per-prediction SHAP contributions plus a plain-English narrative."""
    _require_model()
    try:
        if request.sk_id_curr is not None:
            return explanation.explain_applicant(request.sk_id_curr)
        return explanation.explain_features(request.features or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/explain/global")
def explain_global() -> dict:
    """Global feature importance, precomputed at training time.

    Sampled rather than exhaustive. Computing it per request would make the
    page slow for no benefit.
    """
    try:
        return explanation.load_global_importance()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/rules")
def rules() -> dict:
    """Business-readable rules derived from the model, sorted by lift."""
    try:
        return load_rules()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
