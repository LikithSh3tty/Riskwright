"""Adverse action reason codes.

A separate router rather than an extension of /explain. The explain response is
a published contract -- the frontend renders `{feature, value, shap_value}`
directly and the presentation screenshots show it -- so adding fields there
would risk a UI change for a disclosure concern that has nothing to do with the
waterfall chart. A new route costs one registration line and keeps both
contracts independent.

All logic is in src/ml/reason_codes.py. This validates, calls, and shapes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.routers.predict import PredictRequest, _require_model
from src.ml import reason_codes
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["adverse action"])


@router.post("/adverse-action")
def adverse_action(request: PredictRequest) -> dict:
    """Principal reasons for a decline, or an explicit statement that none apply.

    An applicant scoring below the decision threshold gets
    `adverse_action_required: false` and an empty reason list, not a 404. The
    absence of a notice is itself the answer a caller needs.
    """
    _require_model()
    try:
        if request.sk_id_curr is not None:
            return reason_codes.notice_for_applicant(request.sk_id_curr)
        return reason_codes.notice_for_features(request.features or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/adverse-action/reason-codes")
def reason_code_table() -> dict:
    """The disclosure vocabulary itself.

    Exposed so the mapping can be reviewed without reading the source. A
    regulated disclosure whose wording is only visible in a Python file is
    harder to sign off than one that can be listed.
    """
    return {
        "version": reason_codes.REASON_CODE_VERSION,
        "max_reasons": reason_codes.MAX_REASONS,
        "min_adverse_shap": reason_codes.MIN_ADVERSE_SHAP,
        "reasons": dict(sorted(reason_codes.REASON_CODES.items())),
        "not_disclosable": sorted(reason_codes.NOT_DISCLOSABLE),
        "fallback": reason_codes.FALLBACK_REASON,
        "note": reason_codes.DISCLOSURE_NOTE,
    }
