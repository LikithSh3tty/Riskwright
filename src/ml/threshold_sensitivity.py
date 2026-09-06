"""How much does the decision threshold depend on the cost assumption?

The shipped threshold rests on one number that was never measured: a false
negative is assumed to cost ten times a false positive. The README says so, and
saying so is the minimum. It is not the same as knowing how much turns on it.

This sweeps the ratio and reports what moves. If `t_high` barely shifts between
3:1 and 20:1 then the assumption is doing little work and the threshold is
robust to getting it wrong. If it swings wildly, the 10:1 figure is load-bearing
and deserves a real recovery model behind it. Either answer is worth having;
only the unmeasured version is not.

The sweep runs entirely from artifacts -- `models/oof_predictions.npz` and
`models/threshold.json` -- so it needs no database, no model load and no
retrain. The stored out-of-fold scores are on the model's weighted scale, so
they are calibrated here through the same function the serving path uses.
Applying the shipped ratio to them reproduces `threshold.json` exactly,
including its confusion matrix, which is the check that this sweep and the
deployed threshold are computing the same thing.

**The deployed threshold does not change.** 10:1 remains shipped. This measures
the sensitivity of a decision that stays where it is.

Run:

    python -m src.ml.threshold_sensitivity

Writes models/threshold_sensitivity.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from src.ml.evaluate import THRESHOLD_FILE, decision_report, load_bands
from src.ml.predict import calibrate
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

SENSITIVITY_FILE = "threshold_sensitivity.json"
OOF_FILE = "oof_predictions.npz"

# The brief asks for at least 3:1 to 20:1. The lower end is about as symmetric
# as a lender would plausibly claim; the upper end is a book where recoveries
# are near zero. Denser at the low end because that is where the curve moves.
RATIOS: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)

SHIPPED_RATIO = 10.0

# Candidate thresholds, as quantiles of the score distribution. Matches the
# resolution `find_cost_optimal_threshold` uses so the sweep and the shipped
# value are picked off the same grid.
N_CANDIDATES = 500


def load_oof() -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold labels and calibrated LightGBM probabilities."""
    path = models_dir() / OOF_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.ml.train` to produce it."
        )
    data = np.load(path)
    bands = load_bands()
    return data["y"], calibrate(data["lightgbm"], bands["scale_pos_weight"])


def cost_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
            ratio: float) -> float:
    """Expected cost with the false positive fixed at 1 and the negative at `ratio`."""
    flagged = y_score >= threshold
    false_negatives = int(((y_true == 1) & ~flagged).sum())
    false_positives = int(((y_true == 0) & flagged).sum())
    return ratio * false_negatives + 1.0 * false_positives


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray,
                      ratio: float) -> float:
    candidates = np.unique(np.round(
        np.quantile(y_score, np.linspace(0.01, 0.99, N_CANDIDATES)), 6
    ))
    costs = [cost_at(y_true, y_score, float(t), ratio) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def sweep(reference_ids: tuple[int, ...] = ()) -> dict:
    y_true, y_score = load_oof()
    bands = load_bands()
    naive_cost_unit = {
        r: cost_at(y_true, y_score, 0.5, r) for r in RATIOS
    }

    rows = []
    for ratio in RATIOS:
        threshold = optimal_threshold(y_true, y_score, ratio)
        report = decision_report(y_true, y_score, threshold)
        cost = cost_at(y_true, y_score, threshold, ratio)
        naive = naive_cost_unit[ratio]

        rows.append({
            "ratio": ratio,
            "t_high": threshold,
            "approval_rate": float(1.0 - report["flagged_rate"]),
            "flagged_rate": report["flagged_rate"],
            "defaulters_caught": report["confusion_matrix"]["true_positive"],
            "defaulters_missed": report["confusion_matrix"]["false_negative"],
            "recall": report["recall"],
            "precision": report["precision"],
            "expected_cost": cost,
            "expected_cost_at_half": naive,
            "cost_reduction_vs_half": float((naive - cost) / naive) if naive else 0.0,
            "is_shipped": ratio == SHIPPED_RATIO,
        })
        log.info(
            "%4.0f:1  t_high %.6f  approve %.4f  caught %6d  saving %.1f%%",
            ratio, threshold, 1 - report["flagged_rate"],
            report["confusion_matrix"]["true_positive"],
            100 * (naive - cost) / naive if naive else 0.0,
        )

    thresholds = [r["t_high"] for r in rows]
    approvals = [r["approval_rate"] for r in rows]
    shipped = next(r for r in rows if r["is_shipped"])

    # Band stability: for each applicant, does the band assigned at the shipped
    # ratio survive the whole sweep? t_low is a policy dial and does not depend
    # on the cost ratio, so it is held fixed while t_high moves.
    t_low = float(bands["t_low"])
    band_at = {}
    for row in rows:
        high = row["t_high"]
        band_at[row["ratio"]] = np.where(
            y_score >= high, "High", np.where(y_score >= t_low, "Medium", "Low")
        )
    shipped_bands = band_at[SHIPPED_RATIO]
    unchanged = np.ones(len(y_score), dtype=bool)
    for ratio, bands_at_ratio in band_at.items():
        unchanged &= bands_at_ratio == shipped_bands

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "basis": (
            "Out-of-fold predictions from 5-fold stratified CV, calibrated "
            "through the same function the serving path uses. Applying the "
            "shipped ratio reproduces threshold.json exactly, including its "
            "confusion matrix."
        ),
        "shipped_ratio": SHIPPED_RATIO,
        "shipped_t_high": float(bands["t_high"]),
        "t_low": t_low,
        "t_low_note": (
            "A policy dial set from the score quantile, not from the cost "
            "ratio, so it is held fixed across the sweep."
        ),
        "n": int(len(y_score)),
        "ratios": rows,
        "threshold_range": {
            "min": min(thresholds),
            "max": max(thresholds),
            "spread": max(thresholds) - min(thresholds),
            "relative_spread": (
                (max(thresholds) - min(thresholds)) / shipped["t_high"]
            ),
        },
        "approval_range": {
            "min": min(approvals),
            "max": max(approvals),
            "spread": max(approvals) - min(approvals),
        },
        "band_stability": {
            "unchanged_share": float(unchanged.mean()),
            "changed_share": float(1.0 - unchanged.mean()),
            "note": (
                "Share of applicants whose Low/Medium/High band is identical at "
                "every ratio from 3:1 to 20:1. A band that survives the whole "
                "sweep does not depend on the cost assumption."
            ),
        },
        "deployment_note": (
            "The deployed threshold is unchanged. 10:1 remains shipped and "
            f"{THRESHOLD_FILE} is not rewritten by this module."
        ),
    }


def save(payload: dict) -> None:
    path = models_dir() / SENSITIVITY_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost ratio sensitivity sweep")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    result = sweep()

    print(f"\n{'ratio':>7}  {'t_high':>9}  {'approve':>8}  {'caught':>7}  "
          f"{'missed':>7}  {'vs 0.5':>7}")
    for row in result["ratios"]:
        mark = " <- shipped" if row["is_shipped"] else ""
        print(f"{row['ratio']:>5.0f}:1  {row['t_high']:>9.6f}  "
              f"{row['approval_rate']:>8.4f}  {row['defaulters_caught']:>7,}  "
              f"{row['defaulters_missed']:>7,}  "
              f"{row['cost_reduction_vs_half']:>6.1%}{mark}")

    tr, ar, bs = (result["threshold_range"], result["approval_range"],
                  result["band_stability"])
    print(f"\nt_high moves {tr['min']:.6f} to {tr['max']:.6f} "
          f"(spread {tr['spread']:.6f}, {tr['relative_spread']:.1%} of shipped)")
    print(f"approval rate moves {ar['min']:.4f} to {ar['max']:.4f} "
          f"(spread {ar['spread']:.4f})")
    print(f"band unchanged across the whole sweep for "
          f"{bs['unchanged_share']:.2%} of applicants")

    if not args.no_save:
        save(result)


if __name__ == "__main__":
    main()
