"""SHAP explanations.

Returns JSON, never a rendered figure, so the UI owns presentation and the
frontend stays replaceable.

Part 4 of the assignment asks for explanations a non-technical user can
understand. A list of signed floats is not that, so every explanation also
carries a plain-English narrative. The narrative is generated deterministically
from a label map rather than by an LLM call: it is free, fast, reproducible,
and cannot hallucinate a reason the model did not actually use.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from src.data.loader import fetch_applicant
from src.data.preprocessor import load_feature_spec, prepare_features
from src.ml.predict import _load_model, band_for, calibrate
from src.ml.evaluate import load_bands
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

GLOBAL_IMPORTANCE_FILE = "shap_global.json"

# How many contributions the API returns. Enough to explain the decision,
# few enough to render as a readable bar chart.
TOP_K = 15

# Rows sampled when computing global importance at training time. Exhaustive
# SHAP over 307k rows buys nothing for a summary chart and costs minutes.
GLOBAL_SAMPLE_ROWS = 2000

# Business-readable names. Home Credit column names are opaque to anyone who
# has not read the data dictionary, and the explanation is aimed at someone who
# has not.
FEATURE_LABELS: dict[str, str] = {
    "ext_source_1": "external credit score 1",
    "ext_source_2": "external credit score 2",
    "ext_source_3": "external credit score 3",
    "days_birth": "age",
    # age_years is derived from days_birth, so it needs its own label or a
    # narrative mentioning both reads "age and age".
    "age_years": "age in years",
    "days_employed": "length of employment",
    "days_employed_anomalous": "not currently employed",
    "days_registration": "time since registration",
    "days_id_publish": "time since ID was issued",
    "days_last_phone_change": "time since last phone change",
    "amt_credit": "loan amount",
    "amt_income_total": "annual income",
    "amt_annuity": "annual repayment",
    "amt_goods_price": "price of the goods financed",
    "credit_income_ratio": "loan size relative to income",
    "annuity_income_ratio": "repayment burden relative to income",
    "credit_term": "loan term",
    "employed_life_ratio": "share of life spent in current employment",
    "code_gender": "gender",
    "name_education_type": "education level",
    "name_family_status": "family status",
    "name_income_type": "income type",
    "name_contract_type": "contract type",
    "occupation_type": "occupation",
    "organization_type": "employer type",
    "region_population_relative": "region affluence",
    "region_rating_client": "region rating",
    "region_rating_client_w_city": "region rating including city",
    "cnt_children": "number of children",
    "cnt_fam_members": "family size",
    "own_car_age": "age of car owned",
    "flag_own_car": "owns a car",
    "flag_own_realty": "owns property",
}


def label_for(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    # Fall back to something readable rather than the raw column name.
    return feature.replace("_", " ")


@lru_cache
def _explainer() -> shap.TreeExplainer:
    log.info("building SHAP TreeExplainer")
    return shap.TreeExplainer(_load_model())


def _positive_class_shap(raw) -> np.ndarray:
    """Normalise SHAP output to a (n_rows, n_features) array.

    shap has changed this return shape for LightGBM binary classifiers across
    versions: it has been a two-element list, a 3D array, and a plain 2D array.
    Rather than pin a version and hope, every shape is handled.
    """
    if isinstance(raw, list):
        # [negative_class, positive_class]
        return np.asarray(raw[-1])
    array = np.asarray(raw)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def _base_value(explainer: shap.TreeExplainer) -> float:
    value = np.asarray(explainer.expected_value).ravel()
    return float(value[-1])


def _display_value(feature: str, value) -> object:
    """Render a feature value the way a reader expects to see it.

    Days columns are negative offsets from the application date, so a raw
    days_birth of -9461 under the label "age" looks like a defect. It is shown
    in years instead.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if feature.startswith("days_") and feature != "days_employed_anomalous":
            return round(abs(number) / 365.25, 1)
        return number
    return str(value)


def _narrative(contributions: list[dict], probability: float, band: str) -> str:
    """Plain-English summary of what drove the score.

    Positive SHAP values push toward default, negative pull away from it.
    """
    raising = [c for c in contributions if c["shap_value"] > 0][:5]
    lowering = [c for c in contributions if c["shap_value"] < 0][:5]

    def phrase(items: list[dict]) -> str:
        # Several features describe the same underlying quantity (days_birth
        # and age_years are the same fact in different units). Repeating the
        # label reads as a bug to the non-technical reader this is written for.
        names: list[str] = []
        for candidate in items:
            name = label_for(candidate["feature"])
            root = name.replace(" in years", "")
            if not any(root == existing.replace(" in years", "") for existing in names):
                names.append(name)
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        names = names[:3]
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    parts = [
        f"This applicant has a {probability:.1%} estimated chance of default, "
        f"which places them in the {band} risk band."
    ]
    if raising:
        parts.append(f"Risk is pushed up mainly by {phrase(raising)}.")
    if lowering:
        parts.append(f"It is pulled down by {phrase(lowering)}.")
    if not raising and not lowering:
        parts.append("No single feature moved the score materially.")
    return " ".join(parts)


def explain_frame(frame: pd.DataFrame) -> dict:
    spec = load_feature_spec()
    bands = load_bands()
    model = _load_model()
    explainer = _explainer()

    features = prepare_features(frame, feature_spec=spec)
    values = _positive_class_shap(explainer.shap_values(features))[0]

    weighted = float(model.predict_proba(features)[:, 1][0])
    probability = float(calibrate(weighted, bands["scale_pos_weight"]))
    band = band_for(probability, bands)

    raw_row = features.iloc[0]
    contributions = []
    for feature, shap_value in zip(features.columns, values):
        value = raw_row[feature]
        display = _display_value(feature, value)
        contributions.append(
            {
                "feature": feature,
                "label": label_for(feature),
                "value": display,
                "shap_value": float(shap_value),
                "direction": "raises risk" if shap_value > 0 else "lowers risk",
            }
        )

    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    top = contributions[:TOP_K]

    return {
        "base_value": _base_value(explainer),
        "default_probability": probability,
        "risk_band": band,
        "contributions": top,
        "narrative": _narrative(top, probability, band),
        "note": (
            "SHAP values are in log-odds on the model's weighted scale. The "
            "probability shown is calibrated back to the true base rate; the "
            "contributions describe direction and relative size, not "
            "percentage points."
        ),
    }


def explain_applicant(sk_id_curr: int) -> dict:
    frame = fetch_applicant(sk_id_curr)
    if frame.empty:
        raise KeyError(f"no applicant with sk_id_curr {sk_id_curr}")
    result = explain_frame(frame)
    result["sk_id_curr"] = int(sk_id_curr)
    return result


def explain_features(features: dict) -> dict:
    return explain_frame(pd.DataFrame([features]))


# --------------------------------------------------------------------------
# Global importance, precomputed at training time
# --------------------------------------------------------------------------

def compute_global_importance(
    features: pd.DataFrame,
    sample_rows: int = GLOBAL_SAMPLE_ROWS,
    random_state: int = 42,
) -> dict:
    """Mean absolute SHAP per feature over a sample.

    Sampled rather than exhaustive. That is a real limitation and is stated as
    one in the README rather than left for a reader to discover.
    """
    sample = features.sample(
        n=min(sample_rows, len(features)), random_state=random_state
    )
    values = _positive_class_shap(_explainer().shap_values(sample))
    mean_abs = np.abs(values).mean(axis=0)

    ranked = sorted(
        (
            {
                "feature": str(name),
                "label": label_for(str(name)),
                "mean_abs_shap": float(score),
            }
            for name, score in zip(sample.columns, mean_abs)
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )
    return {
        "sampled_rows": int(len(sample)),
        "total_rows": int(len(features)),
        "features": ranked,
    }


def save_global_importance(payload: dict) -> None:
    path = models_dir() / GLOBAL_IMPORTANCE_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def load_global_importance() -> dict:
    path = models_dir() / GLOBAL_IMPORTANCE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.ml.train` to produce it."
        )
    return json.loads(path.read_text(encoding="utf-8"))
