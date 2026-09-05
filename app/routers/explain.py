"""Explainability and derived-rule endpoints.

Everything here returns JSON. Charts are the UI's job, which is what keeps the
frontend replaceable.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.routers.predict import PredictRequest, _require_model
from src.ml import explain as explanation
from src.ml.rules import POLICY_RULES_FILE, RULES_FILE, load_rules
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
def rules(variant: str = Query(default="faithful", pattern="^(faithful|policy)$")) -> dict:
    """Business-readable rules derived from the model, sorted by lift.

    Two sets. "faithful" reflects what the model actually leans on, which on
    this dataset is the external bureau scores almost exclusively. "policy"
    excludes those scores: less agreement with the model, but rules a credit
    officer can act on and rules that still apply to an applicant with no
    bureau history.
    """
    filename = RULES_FILE if variant == "faithful" else POLICY_RULES_FILE
    try:
        payload = load_rules(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload["variant"] = variant
    return payload
