"""Risk scoring endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from src.data.loader import read_table
from src.ml import predict as scoring
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["prediction"])


class PredictRequest(BaseModel):
    """Either an existing applicant id, or an ad-hoc feature dict.

    Both paths run the same scoring function. The id path makes the demo and
    the screenshots effortless; the feature path shows the model responding to
    the inputs a credit officer would actually change.
    """

    sk_id_curr: int | None = Field(default=None, description="Existing applicant id")
    features: dict[str, Any] | None = Field(
        default=None, description="Partial applicant fields for what-if scoring"
    )

    @model_validator(mode="after")
    def exactly_one_input(self) -> "PredictRequest":
        if (self.sk_id_curr is None) == (self.features is None):
            raise ValueError("supply exactly one of sk_id_curr or features")
        return self


def _require_model() -> None:
    if not scoring.model_is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model on disk. Run `python -m src.ml.train` to "
                "produce the artifacts."
            ),
        )


@router.post("/predict")
def predict(request: PredictRequest) -> dict:
    _require_model()
    try:
        if request.sk_id_curr is not None:
            return scoring.predict_applicant(request.sk_id_curr)
        return scoring.predict_features(request.features or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/applicants")
def applicants(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Applicant ids for the UI selector.

    Returns ids and a few descriptive fields, never the full 122-column row.
    """
    frame = read_table(
        "application_train",
        columns=[
            "sk_id_curr",
            "amt_credit",
            "amt_income_total",
            "days_birth",
            "code_gender",
        ],
        limit=limit + offset,
    ).iloc[offset:]

    frame = frame.assign(age_years=(-frame["days_birth"] / 365.25).round(1))
    frame = frame.drop(columns=["days_birth"])

    return {
        "count": int(len(frame)),
        "applicants": frame.to_dict(orient="records"),
    }


@router.get("/model/metrics")
def model_metrics() -> dict:
    """Evaluation numbers, read from the artifact the training run wrote.

    The README, the UI, and this endpoint all read one file, so the reported
    numbers cannot drift apart from each other.
    """
    from src.ml.evaluate import load_bands
    from src.utils.docker_utils import models_dir
    import json

    metrics_path = models_dir() / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=503, detail="No metrics on disk. Train first.")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        payload["bands"] = load_bands()
    except FileNotFoundError:
        payload["bands"] = None
    return payload
