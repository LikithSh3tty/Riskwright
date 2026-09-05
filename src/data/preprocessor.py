"""Cleaning, feature derivation, and encoding.

This module is the only place features are built. Training calls it and
inference calls it, on the same code path, because train/serve skew is silent:
nothing errors, the numbers are just quietly wrong.

Column names are lowercase throughout. The loader normalises CSV headers on the
way into Postgres and everything downstream reads from Postgres, so there is
one casing convention in the project.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

TARGET = "target"
ID_COLUMN = "sk_id_curr"

# DAYS_EMPLOYED uses 365243 (about 1000 years) as a sentinel for "not currently
# employed". Roughly a fifth of rows carry it. Left alone it is the single most
# damaging value in the dataset: it drags the mean, breaks every ratio built on
# it, and teaches the model a spurious cliff.
DAYS_EMPLOYED_SENTINEL = 365243

# XNA means "not available" in the Home Credit encoding, not a real category.
# ORGANIZATION_TYPE is excluded: there XNA is a meaningful group (the applicant
# reported no organisation) and is left as a category.
XNA_AS_MISSING = ("code_gender",)

DAYS_PER_YEAR = 365.25

# Dropped after the derived features are built. age_years is a strictly
# monotonic transform of days_birth, so keeping both gives a tree two ways to
# express one fact. It splits SHAP importance between them, makes the waterfall
# harder to read, and produced explanations that said "age and age". Nothing is
# being chased on AUC here that the duplicate would buy.
REDUNDANT_AFTER_DERIVATION = ("days_birth",)


# Attributes that are protected bases in credit decisions. ECOA names sex,
# marital status, and age explicitly; the Equal Credit Opportunity framework
# and Indian fair-lending expectations treat them the same way. Using them to
# price or refuse credit is a compliance failure, not a modelling tradeoff, so
# they are excluded from the feature matrix by default rather than left in with
# a caveat attached.
#
#   code_gender          sex
#   name_family_status   marital status
#   age_years            age (days_birth is already dropped as redundant)
#   cnt_children         familial status, a direct proxy for the above
#   cnt_fam_members      familial status, same
#
# Residual proxies that are NOT excluded, and why, are documented in the
# README's fair lending section. Excluding a protected attribute does not by
# itself make a model fair; it removes the most obvious defect.
PROTECTED_ATTRIBUTES: tuple[str, ...] = (
    "code_gender",
    "name_family_status",
    "age_years",
    "cnt_children",
    "cnt_fam_members",
)

# Written at training time, read at inference time. Guards against column-order
# drift and against a category the serving path has never seen.
FEATURE_SPEC_FILE = "feature_spec.json"


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the known data-quality defects. Returns a new frame."""
    out = df.copy()

    if "days_employed" in out.columns:
        anomalous = out["days_employed"] == DAYS_EMPLOYED_SENTINEL
        # Keep the fact that it was anomalous: "not employed" is predictive,
        # so the information is preserved as a flag rather than discarded.
        out["days_employed_anomalous"] = anomalous.astype(int)
        out.loc[anomalous, "days_employed"] = np.nan
        if anomalous.any():
            log.debug("days_employed sentinel on %.1f%% of rows",
                      100 * anomalous.mean())

    for column in XNA_AS_MISSING:
        if column in out.columns:
            out[column] = out[column].replace("XNA", np.nan)

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ratio features.

    Credit risk is about proportion, not magnitude: a 500k loan means something
    different at 100k income than at 1M. These five carry that.
    """
    out = df.copy()

    def ratio(numerator: str, denominator: str) -> pd.Series | None:
        if numerator not in out.columns or denominator not in out.columns:
            return None
        denom = out[denominator].replace(0, np.nan)
        return out[numerator] / denom

    derived = {
        "credit_income_ratio": ratio("amt_credit", "amt_income_total"),
        "annuity_income_ratio": ratio("amt_annuity", "amt_income_total"),
        "credit_term": ratio("amt_credit", "amt_annuity"),
        "employed_life_ratio": ratio("days_employed", "days_birth"),
    }
    for name, series in derived.items():
        if series is not None:
            out[name] = series

    if "days_birth" in out.columns:
        # Days columns are negative offsets from the application date. Age in
        # years is what appears in the UI and in the derived rules.
        out["age_years"] = -out["days_birth"] / DAYS_PER_YEAR

    return out


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------

def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return sorted(df.select_dtypes(include=["object", "category"]).columns)


def prepare_features(
    df: pd.DataFrame,
    feature_spec: dict | None = None,
    exclude_protected: bool = True,
) -> pd.DataFrame:
    """Turn raw rows into the model's feature matrix.

    feature_spec is None during training, where the spec is derived from the
    data. At inference it is the spec saved during training, which pins column
    order and category levels so a single-row request produces exactly the
    columns the model was fitted on.
    """
    frame = add_derived_features(clean(df))
    discard = (TARGET, ID_COLUMN, *REDUNDANT_AFTER_DERIVATION)
    if exclude_protected:
        discard = (*discard, *PROTECTED_ATTRIBUTES)
    frame = frame.drop(columns=[c for c in discard if c in frame.columns])

    if feature_spec is None:
        categorical = _categorical_columns(frame)
        for column in categorical:
            frame[column] = frame[column].astype("category")
        return frame

    categorical = feature_spec["categorical_columns"]
    categories = feature_spec["categories"]

    for column in categorical:
        if column not in frame.columns:
            frame[column] = np.nan
        # Fixing the category levels means an unseen value becomes NaN rather
        # than shifting LightGBM's internal codes.
        frame[column] = pd.Categorical(frame[column], categories=categories[column])

    for column in feature_spec["feature_names"]:
        if column not in frame.columns:
            frame[column] = np.nan

    # Numeric dtypes are pinned, not just column order and category levels.
    # A single-row query where a numeric column is NULL comes back from
    # pandas as object dtype, which LightGBM rejects outright. That only
    # shows up on the serving path, never in training, because a full frame
    # always has some non-null value to infer from.
    for column in feature_spec.get("numeric_columns", []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Column order is part of the contract with the fitted model.
    return frame[feature_spec["feature_names"]]


def build_feature_spec(features: pd.DataFrame) -> dict:
    """Describe a fitted feature matrix so inference can reproduce it."""
    categorical = _categorical_columns(features)
    numeric = [c for c in features.columns if c not in categorical]
    return {
        "feature_names": list(features.columns),
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "categories": {
            column: [str(v) for v in features[column].cat.categories]
            for column in categorical
        },
        "days_employed_sentinel": DAYS_EMPLOYED_SENTINEL,
        "protected_attributes_excluded": list(PROTECTED_ATTRIBUTES),
    }


def save_feature_spec(spec: dict) -> None:
    path = models_dir() / FEATURE_SPEC_FILE
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    log.info("wrote %s (%d features)", path, len(spec["feature_names"]))


def load_feature_spec() -> dict:
    path = models_dir() / FEATURE_SPEC_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Train the model first: python -m src.ml.train"
        )
    return json.loads(path.read_text(encoding="utf-8"))
