"""Evaluation metrics and the cost-based decision threshold.

PR-AUC is reported alongside ROC-AUC throughout. On a target where 8% of cases
are positive, ROC-AUC alone flatters a model badly, and reporting it on its own
is the clearest sign that imbalance was not thought about.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

THRESHOLD_FILE = "threshold.json"

# A false negative is an approved loan that defaults and is written off.
# A false positive is a creditworthy applicant refused, costing margin and
# goodwill. They are not remotely symmetric. The 10:1 ratio below is an
# assumption, stated so a reviewer can disagree with the number without
# disagreeing with the method.
FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0

# Where the Low/Medium boundary sits is a business decision, not an optimum.
# See choose_bands.
DEFAULT_AUTO_APPROVE_RATE = 0.40


def score_predictions(y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def summarise_folds(per_fold: list[dict]) -> dict:
    roc = np.array([f["roc_auc"] for f in per_fold])
    pr = np.array([f["pr_auc"] for f in per_fold])
    return {
        "roc_auc_mean": float(roc.mean()),
        "roc_auc_std": float(roc.std()),
        "pr_auc_mean": float(pr.mean()),
        "pr_auc_std": float(pr.std()),
    }


# --------------------------------------------------------------------------
# Cost-based threshold
# --------------------------------------------------------------------------

def expected_cost(y_true, y_score, threshold: float) -> float:
    """Expected cost of deciding at this threshold.

    Predicting positive means refusing the applicant. A false negative is an
    approved loan that defaulted; a false positive is a good customer turned
    away.
    """
    y_true = np.asarray(y_true)
    predicted_positive = np.asarray(y_score) >= threshold

    false_negatives = int(((y_true == 1) & ~predicted_positive).sum())
    false_positives = int(((y_true == 0) & predicted_positive).sum())

    return (
        FALSE_NEGATIVE_COST * false_negatives
        + FALSE_POSITIVE_COST * false_positives
    )


def find_cost_optimal_threshold(
    y_true,
    y_score,
    n_candidates: int = 500,
) -> tuple[float, list[dict]]:
    """Sweep thresholds and return the one minimising expected cost.

    Not 0.5. On an imbalanced target with asymmetric costs, 0.5 is an arbitrary
    number that happens to be the midpoint of a probability scale, and it has
    no connection to what a wrong decision costs.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    candidates = np.quantile(y_score, np.linspace(0.01, 0.99, n_candidates))
    candidates = np.unique(np.round(candidates, 6))

    curve = [
        {"threshold": float(t), "cost": float(expected_cost(y_true, y_score, t))}
        for t in candidates
    ]
    best = min(curve, key=lambda row: row["cost"])
    return best["threshold"], curve


def decision_report(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    predicted = (np.asarray(y_score) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "expected_cost": float(expected_cost(y_true, y_score, threshold)),
        "flagged_rate": float(predicted.mean()),
    }


def choose_bands(
    y_true,
    y_score,
    auto_approve_rate: float = DEFAULT_AUTO_APPROVE_RATE,
) -> dict:
    """Derive the Low / Medium / High boundaries.

    The two boundaries answer different questions and are set differently.

    t_high, the Medium/High boundary, is the cost-optimal threshold. Above it
    the expected cost of approving exceeds the expected cost of refusing, so
    the decision to decline or escalate follows from the cost assumption rather
    than from taste.

    t_low, the Low/Medium boundary, is a business decision and not an optimum.
    Nothing in the data says where automatic approval without review should
    stop; that depends on review capacity and risk appetite. It is set here so
    that the safest `auto_approve_rate` share of applicants falls in the Low
    band, and it is reported as a judgment call rather than a derived value.
    """
    y_score = np.asarray(y_score)

    t_high, curve = find_cost_optimal_threshold(y_true, y_score)
    t_low = float(np.quantile(y_score, auto_approve_rate))

    if t_low >= t_high:
        # Can happen if the cost ratio pushes the optimum very low. Keeping the
        # bands ordered matters more than hitting the requested share.
        t_low = float(t_high * 0.5)
        log.warning(
            "auto-approve quantile sat above the cost-optimal threshold; "
            "t_low reduced to %.4f to keep the bands ordered",
            t_low,
        )

    report = decision_report(y_true, y_score, t_high)
    baseline = decision_report(y_true, y_score, 0.5)

    bands = {
        "cost_assumption": {
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "ratio": FALSE_NEGATIVE_COST / FALSE_POSITIVE_COST,
            "note": (
                "A false negative is an approved loan that defaults; a false "
                "positive is a creditworthy applicant refused. The ratio is an "
                "assumption, not a measurement."
            ),
        },
        "t_high": float(t_high),
        "t_high_basis": (
            "cost-optimal: the threshold minimising expected cost under the "
            "stated ratio, computed on calibrated probabilities"
        ),
        "t_low": float(t_low),
        "t_low_basis": (
            f"business judgment: the safest {auto_approve_rate:.0%} of applicants "
            f"are auto-approved without review. Not an optimum."
        ),
        "auto_approve_rate": float(auto_approve_rate),
        "at_cost_optimal_threshold": report,
        "at_naive_half_threshold": baseline,
        "cost_saving_vs_half": float(
            baseline["expected_cost"] - report["expected_cost"]
        ),
        "sweep": curve,
    }

    log.info(
        "bands: low < %.4f <= medium < %.4f <= high", bands["t_low"], bands["t_high"]
    )
    log.info(
        "cost-optimal threshold %.4f flags %.1f%% of applicants, recall %.3f",
        t_high,
        100 * report["flagged_rate"],
        report["recall"],
    )
    return bands


def save_bands(bands: dict) -> None:
    path = models_dir() / THRESHOLD_FILE
    # The sweep is useful for a chart but bulky; it is kept out of the file the
    # API reads on every request.
    payload = {k: v for k, v in bands.items() if k != "sweep"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def load_bands() -> dict:
    path = models_dir() / THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.ml.train` to produce it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def band_for(probability: float, bands: dict | None = None) -> str:
    bands = bands or load_bands()
    if probability >= bands["t_high"]:
        return "High"
    if probability >= bands["t_low"]:
        return "Medium"
    return "Low"
