"""Inference and risk banding.

Two things happen here that are worth reading before trusting the numbers.

First, calibration. The model is fitted with scale_pos_weight, which is the
right way to handle an 8% positive rate but which multiplies the predicted odds
by that weight. Raw output is therefore a ranking score, not a probability: an
average applicant scores near 0.5 rather than near 0.08. Since the distortion
is a known constant, it is inverted exactly rather than fitted, and what the
API returns is a real probability.

Second, banding. The Medium/High boundary is the cost-optimal threshold. The
Low/Medium boundary is a business decision. They are not the same kind of
number and are not derived the same way. See evaluate.choose_bands.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data.loader import fetch_applicant
from src.data.preprocessor import load_feature_spec, prepare_features
from src.ml.evaluate import load_bands
from src.ml.train import LIGHTGBM_MODEL_FILE
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)


def calibrate(weighted_probability, scale_pos_weight: float):
    """Undo the odds shift introduced by scale_pos_weight.

    Training with weight w multiplies the odds of the positive class by w:

        odds_weighted = w * odds_true

    Inverting gives

        p_true = p_w / (p_w + w * (1 - p_w))

    A weighted output of exactly 0.5 maps back to 1 / (1 + w), which for
    w = 11.387 is 0.0807: the dataset's actual default rate. That identity is
    what makes this an exact correction rather than an approximation, and it is
    asserted in the tests.
    """
    p = np.asarray(weighted_probability, dtype=float)
    return p / (p + scale_pos_weight * (1.0 - p))


@lru_cache
def _load_model():
    path = models_dir() / LIGHTGBM_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Train the model first: python -m src.ml.train"
        )
    log.info("loading model from %s", path)
    return joblib.load(path)


@lru_cache
def _load_spec() -> dict:
    return load_feature_spec()


@lru_cache
def _load_bands() -> dict:
    return load_bands()


def model_is_available() -> bool:
    return (models_dir() / LIGHTGBM_MODEL_FILE).exists()


def band_for(probability: float, bands: dict) -> str:
    if probability >= bands["t_high"]:
        return "High"
    if probability >= bands["t_low"]:
        return "Medium"
    return "Low"


BAND_ACTIONS = {
    "Low": "Auto-approve without manual review",
    "Medium": "Route to a credit officer for manual review",
    "High": "Decline, or escalate for senior review",
}


def predict_frame(frame: pd.DataFrame) -> np.ndarray:
    """Calibrated default probabilities for a frame of raw applicant rows."""
    model = _load_model()
    spec = _load_spec()
    bands = _load_bands()

    features = prepare_features(frame, feature_spec=spec)
    weighted = model.predict_proba(features)[:, 1]
    return calibrate(weighted, bands["scale_pos_weight"])


def predict_applicant(sk_id_curr: int) -> dict:
    frame = fetch_applicant(sk_id_curr)
    if frame.empty:
        raise KeyError(f"no applicant with sk_id_curr {sk_id_curr}")
    return _result(frame, sk_id_curr=int(sk_id_curr))


def predict_features(features: dict) -> dict:
    """Score an ad-hoc applicant.

    Unspecified fields are left missing rather than guessed. LightGBM handles
    NaN natively, so a partial what-if request is scored on the fields actually
    supplied instead of on invented values.
    """
    frame = pd.DataFrame([features])
    return _result(frame, sk_id_curr=None)


def _result(frame: pd.DataFrame, sk_id_curr: int | None) -> dict:
    bands = _load_bands()
    probability = float(predict_frame(frame)[0])
    band = band_for(probability, bands)

    return {
        "sk_id_curr": sk_id_curr,
        "default_probability": probability,
        "risk_band": band,
        "recommended_action": BAND_ACTIONS[band],
        "thresholds": {
            "t_low": bands["t_low"],
            "t_high": bands["t_high"],
        },
        "population_default_rate": bands.get("population_default_rate"),
        "model": "lightgbm",
    }
