"""Preprocessing tests.

The determinism test is the important one. It is the guard against train/serve
skew, which produces no error and no warning, only quietly wrong predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    DAYS_EMPLOYED_SENTINEL,
    add_derived_features,
    build_feature_spec,
    clean,
    prepare_features,
)


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sk_id_curr": [1, 2, 3],
            "target": [0, 1, 0],
            "days_employed": [-1200, DAYS_EMPLOYED_SENTINEL, -300],
            "days_birth": [-14000, -20000, -11000],
            "amt_credit": [500000.0, 250000.0, 900000.0],
            "amt_income_total": [100000.0, 50000.0, 0.0],
            "amt_annuity": [25000.0, 12000.0, 45000.0],
            "code_gender": ["M", "XNA", "F"],
            "organization_type": ["Business Entity Type 3", "XNA", "School"],
        }
    )


def test_sentinel_becomes_null_and_is_flagged(raw):
    cleaned = clean(raw)
    assert np.isnan(cleaned.loc[1, "days_employed"])
    assert cleaned["days_employed_anomalous"].tolist() == [0, 1, 0]


def test_xna_gender_is_missing_but_organization_xna_is_kept(raw):
    cleaned = clean(raw)
    assert pd.isna(cleaned.loc[1, "code_gender"])
    # XNA here means "no organisation reported", which is a real group.
    assert cleaned.loc[1, "organization_type"] == "XNA"


def test_ratios_and_age(raw):
    derived = add_derived_features(clean(raw))
    assert derived.loc[0, "credit_income_ratio"] == pytest.approx(5.0)
    assert derived.loc[0, "credit_term"] == pytest.approx(20.0)
    assert derived.loc[0, "age_years"] == pytest.approx(14000 / 365.25)


def test_zero_income_does_not_produce_infinity(raw):
    # Row 3 has zero income. A ratio of inf would survive training and blow up
    # any scaler downstream, so zero denominators become NaN.
    derived = add_derived_features(clean(raw))
    assert pd.isna(derived.loc[2, "credit_income_ratio"])


def test_identifier_and_target_are_not_features(raw):
    features = prepare_features(raw)
    assert "sk_id_curr" not in features.columns
    assert "target" not in features.columns


def test_serving_path_reproduces_training_columns_exactly(raw):
    """The train/serve skew guard.

    A single row scored through the saved spec must yield the same columns, in
    the same order, with the same category levels, as the training matrix.
    """
    training = prepare_features(raw)
    spec = build_feature_spec(training)

    single_row = prepare_features(raw.iloc[[1]], feature_spec=spec)

    assert list(single_row.columns) == list(training.columns)
    for column in spec["categorical_columns"]:
        assert list(single_row[column].cat.categories) == spec["categories"][column]


def test_unseen_category_becomes_null_rather_than_shifting_codes(raw):
    training = prepare_features(raw)
    spec = build_feature_spec(training)

    unseen = raw.iloc[[0]].copy()
    unseen["organization_type"] = "Interplanetary Mining"

    scored = prepare_features(unseen, feature_spec=spec)
    assert pd.isna(scored.loc[0, "organization_type"])


def test_missing_column_at_inference_is_filled_not_dropped(raw):
    training = prepare_features(raw)
    spec = build_feature_spec(training)

    partial = raw.iloc[[0]].drop(columns=["amt_annuity"])
    scored = prepare_features(partial, feature_spec=spec)

    assert list(scored.columns) == spec["feature_names"]
    assert pd.isna(scored.loc[0, "amt_annuity"])


def test_all_null_numeric_column_stays_numeric(raw):
    """Regression: the bug that only appeared on the serving path.

    A single-applicant query where a numeric column is NULL comes back from
    pandas as object dtype. Training never sees this, because a full frame
    always has some non-null value to infer from, so LightGBM rejected the
    single-row request with "pandas dtypes must be int, float or bool".
    """
    training = prepare_features(raw)
    spec = build_feature_spec(training)

    single = raw.iloc[[0]].copy()
    single["amt_annuity"] = pd.Series([None], index=single.index, dtype=object)

    scored = prepare_features(single, feature_spec=spec)

    assert scored["amt_annuity"].dtype.kind == "f"
    for column in spec["numeric_columns"]:
        assert scored[column].dtype.kind in "fiub", f"{column} is {scored[column].dtype}"


def test_calibration_inverts_the_imbalance_weighting():
    """A weighted output of 0.5 must map back to the true base rate.

    Training with scale_pos_weight multiplies the predicted odds by that
    weight, so raw model output is a ranking score rather than a probability.
    The correction is exact, and this identity is what proves it.
    """
    from src.ml.predict import calibrate

    weight = 11.387
    assert calibrate(0.5, weight) == pytest.approx(1 / (1 + weight))
    assert calibrate(0.0, weight) == pytest.approx(0.0)
    assert calibrate(1.0, weight) == pytest.approx(1.0)

    # Calibration must not reorder anyone: ROC-AUC and PR-AUC are unchanged.
    raw_scores = np.array([0.1, 0.4, 0.5, 0.9])
    calibrated = calibrate(raw_scores, weight)
    assert np.all(np.diff(calibrated) > 0)
