"""Model training.

Two models on identical features: a logistic regression baseline and LightGBM.
Reporting both turns "we chose gradient boosting" from an assertion into a
demonstrated decision, and logistic regression is not a throwaway. It is what
banks actually deploy under regulatory pressure, so if the boosted model beats
it only slightly, that is a finding worth reporting rather than hiding.

Two modes:

  --quick   Single stratified split, LightGBM only. About 30 seconds. Produces
            real artifacts so the prediction endpoint can be exercised end to
            end before the full evaluation exists.
  (default) Stratified 5-fold cross-validation for both models, out-of-fold
            predictions retained, full metrics written.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import read_application_table
from src.data.preprocessor import (
    TARGET,
    build_feature_spec,
    prepare_features,
    save_feature_spec,
)
from src.utils.docker_utils import models_dir
from src.utils.helpers import human_count, timed
from src.utils.logger import get_logger

log = get_logger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5

LIGHTGBM_MODEL_FILE = "lightgbm_model.joblib"
LOGISTIC_MODEL_FILE = "logistic_model.joblib"
METRICS_FILE = "metrics.json"
OOF_FILE = "oof_predictions.npz"

# Deliberately untuned. Sensible defaults with correct imbalance handling score
# the same here as a searched model, and the time belongs to the chatbot.
LIGHTGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    with timed("read application_train from postgres"):
        frame = read_application_table()
    log.info("read %s rows, %d columns", human_count(len(frame)), frame.shape[1])

    target = frame[TARGET].astype(int)
    features = prepare_features(frame)
    log.info("prepared %d features", features.shape[1])
    return features, target


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Imbalance ratio, negatives over positives.

    Derived from whichever split is being fitted rather than hardcoded, so the
    value always describes the data the model actually saw. On a stratified
    split the per-fold values agree to within rounding, but deriving it keeps
    the code honest if the split ever changes.
    """
    y = np.asarray(y)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("no positive cases in the training split")
    weight = negatives / positives
    log.info(
        "imbalance: %s negative / %s positive, positive rate %.2f%%, "
        "scale_pos_weight %.3f",
        human_count(negatives),
        human_count(positives),
        100 * positives / len(y),
        weight,
    )
    return weight


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_lightgbm(scale_pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **LIGHTGBM_PARAMS)


def build_logistic(features: pd.DataFrame) -> Pipeline:
    """Baseline pipeline.

    LightGBM takes NaN and categoricals natively; logistic regression takes
    neither, so the imputation, encoding, and scaling it needs are attached to
    the model rather than applied to the shared feature matrix. Both models
    therefore consume the identical input frame.
    """
    categorical = sorted(
        features.select_dtypes(include=["category", "object"]).columns
    )
    numeric = [c for c in features.columns if c not in categorical]

    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # An unseen category at serving time must not shift the encoding.
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        ("numeric", numeric_steps, numeric),
                        ("categorical", categorical_steps, categorical),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _as_object_frame(features: pd.DataFrame) -> pd.DataFrame:
    """sklearn's imputer cannot handle pandas categorical dtype."""
    out = features.copy()
    for column in out.select_dtypes(include=["category"]).columns:
        out[column] = out[column].astype(object)
    return out


# --------------------------------------------------------------------------
# Training modes
# --------------------------------------------------------------------------

def quick_train() -> dict:
    """Single split, LightGBM only. Produces usable artifacts fast.

    Exists so the prediction endpoint can be wired and exercised end to end
    before the full evaluation runs, the same way session 1 ended on a verified
    docker-compose up rather than on code that looked correct.
    """
    features, target = load_training_data()

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    weight = compute_scale_pos_weight(y_train)
    model = build_lightgbm(weight)

    with timed("fit lightgbm (quick)"):
        model.fit(x_train, y_train)

    spec = build_feature_spec(features)
    save_feature_spec(spec)
    joblib.dump(model, models_dir() / LIGHTGBM_MODEL_FILE)
    log.info("saved %s", LIGHTGBM_MODEL_FILE)

    from src.ml.evaluate import choose_bands, save_bands, score_predictions

    probabilities = model.predict_proba(x_valid)[:, 1]
    metrics = score_predictions(y_valid, probabilities)
    log.info(
        "quick holdout: ROC-AUC %.4f, PR-AUC %.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
    )

    # Bands are derived on calibrated probabilities, not on raw model output.
    # Raw output carries the scale_pos_weight odds shift, so a threshold picked
    # on that scale would be measuring the weighting as much as the cost ratio.
    from src.ml.predict import calibrate

    calibrated = calibrate(probabilities, weight)
    bands = choose_bands(y_valid, calibrated)
    bands["scale_pos_weight"] = weight
    bands["population_default_rate"] = float(np.mean(target))
    bands["basis"] = "single holdout split (quick mode)"
    save_bands(bands)

    payload = {
        "mode": "quick",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "scale_pos_weight": weight,
        "models": {"lightgbm": {"holdout": metrics}},
    }
    _write_metrics(payload)
    return payload


def full_train() -> dict:
    """Stratified 5-fold cross-validation for both models."""
    from src.ml.evaluate import score_predictions, summarise_folds

    features, target = load_training_data()
    y = target.to_numpy()

    folds = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    oof = {
        "lightgbm": np.zeros(len(features)),
        "logistic": np.zeros(len(features)),
    }
    per_fold: dict[str, list[dict]] = {"lightgbm": [], "logistic": []}
    weights: list[float] = []

    features_objects = _as_object_frame(features)

    for fold, (train_idx, valid_idx) in enumerate(folds.split(features, y), start=1):
        log.info("fold %d of %d", fold, N_FOLDS)
        y_train, y_valid = y[train_idx], y[valid_idx]

        weight = compute_scale_pos_weight(y_train)
        weights.append(weight)

        with timed(f"fold {fold}: lightgbm"):
            booster = build_lightgbm(weight)
            booster.fit(features.iloc[train_idx], y_train)
            predictions = booster.predict_proba(features.iloc[valid_idx])[:, 1]
        oof["lightgbm"][valid_idx] = predictions
        per_fold["lightgbm"].append(score_predictions(y_valid, predictions))

        with timed(f"fold {fold}: logistic"):
            baseline = build_logistic(features)
            baseline.fit(features_objects.iloc[train_idx], y_train)
            predictions = baseline.predict_proba(features_objects.iloc[valid_idx])[:, 1]
        oof["logistic"][valid_idx] = predictions
        per_fold["logistic"].append(score_predictions(y_valid, predictions))

    # Refit on everything for serving. Cross-validation measures; the served
    # model is fitted on all available data.
    final_weight = compute_scale_pos_weight(y)
    with timed("refit lightgbm on full data"):
        final_booster = build_lightgbm(final_weight)
        final_booster.fit(features, y)

    with timed("refit logistic on full data"):
        final_baseline = build_logistic(features)
        final_baseline.fit(features_objects, y)

    spec = build_feature_spec(features)
    save_feature_spec(spec)
    joblib.dump(final_booster, models_dir() / LIGHTGBM_MODEL_FILE)
    joblib.dump(final_baseline, models_dir() / LOGISTIC_MODEL_FILE)
    log.info("saved both model artifacts")

    # Out-of-fold predictions are the basis for the cost threshold, so they are
    # kept rather than recomputed.
    np.savez_compressed(
        models_dir() / OOF_FILE,
        y=y,
        lightgbm=oof["lightgbm"],
        logistic=oof["logistic"],
    )

    # Bands come from out-of-fold predictions, calibrated first. Using the
    # in-sample refit here would pick a threshold on predictions the model has
    # already seen, which flatters every number that follows.
    from src.ml.evaluate import choose_bands, save_bands
    from src.ml.predict import calibrate

    bands = choose_bands(y, calibrate(oof["lightgbm"], final_weight))
    bands["scale_pos_weight"] = final_weight
    bands["population_default_rate"] = float(y.mean())
    bands["basis"] = f"out-of-fold predictions, {N_FOLDS}-fold stratified"
    save_bands(bands)

    payload = {
        "mode": "cross_validated",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "n_folds": N_FOLDS,
        "scale_pos_weight": float(np.mean(weights)),
        "positive_rate": float(y.mean()),
        "models": {
            name: {
                "per_fold": per_fold[name],
                "cross_validated": summarise_folds(per_fold[name]),
                "out_of_fold": score_predictions(y, oof[name]),
            }
            for name in ("lightgbm", "logistic")
        },
    }
    _write_metrics(payload)

    for name in ("lightgbm", "logistic"):
        summary = payload["models"][name]["cross_validated"]
        log.info(
            "%s: ROC-AUC %.4f +/- %.4f, PR-AUC %.4f +/- %.4f",
            name,
            summary["roc_auc_mean"],
            summary["roc_auc_std"],
            summary["pr_auc_mean"],
            summary["pr_auc_std"],
        )
    return payload


def build_derived_artifacts() -> dict:
    """Produce the SHAP global summary and the derived rules.

    Split out from training so it can be re-run against an existing model
    without paying for cross-validation again. full_train calls it at the end;
    `--artifacts` runs it on its own.
    """
    from src.ml.explain import (
        compute_global_importance,
        save_global_importance,
    )
    from src.ml.rules import (
        EXTERNAL_SCORE_PREFIX,
        POLICY_RULES_FILE,
        derive_rules,
        save_rules,
    )

    features, target = load_training_data()

    with timed("global SHAP importance"):
        importance = compute_global_importance(features)
    save_global_importance(importance)

    top_features = [row["feature"] for row in importance["features"]]

    def report(title: str, payload: dict) -> None:
        log.info("%s (fidelity %.3f)", title, payload["surrogate_fidelity"])
        for rule in payload["rules"][:3]:
            log.info(
                "  %s  lift %.2f  support %.1f%%  default rate %.1f%%  %s",
                rule["rule_id"],
                rule["lift"],
                rule["support_pct"],
                100 * rule["default_rate"],
                rule["readable"],
            )

    # Faithful set: whatever the model actually leans on.
    with timed("derive rules"):
        rules = derive_rules(features, target, top_features)
    save_rules(rules)
    report("rules (faithful)", rules)

    # Policy set: the same derivation with the external bureau scores removed.
    # Less faithful by construction, and more actionable, because a credit
    # officer cannot do anything with "the bureau score was low" and because
    # some applicants have no bureau coverage at all.
    with timed("derive policy rules"):
        policy = derive_rules(
            features, target, top_features, exclude=(EXTERNAL_SCORE_PREFIX,)
        )
    save_rules(policy, POLICY_RULES_FILE)
    report("rules (external scores excluded)", policy)

    return {"global_importance": importance, "rules": rules, "policy_rules": policy}


def _write_metrics(payload: dict) -> None:
    path = models_dir() / METRICS_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Riskwright models")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="single split, LightGBM only, for fast iteration",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="rebuild SHAP global importance and rules from the saved model",
    )
    args = parser.parse_args()

    if args.artifacts:
        build_derived_artifacts()
    elif args.quick:
        quick_train()
    else:
        full_train()
        build_derived_artifacts()


if __name__ == "__main__":
    main()
