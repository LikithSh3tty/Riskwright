"""Disparate impact measurement against the served model.

The model does not use protected attributes: `preprocessor.PROTECTED_ATTRIBUTES`
are dropped from the feature matrix, so they cannot reach the model, the SHAP
explanation, or the derived rules. Exclusion is the floor, not the bar. What it
does not tell you is whether the decisions the model produces fall unevenly
across the groups those attributes describe, and a model can carry a large
disparity through proxies while never seeing the attribute itself.

This module measures that. It scores the applicant population with the served
LightGBM artifact, applies the deployed decision threshold, and reports per
group: how many applicants, what share were approved, their mean predicted
probability, and their observed default rate from TARGET. Then, per attribute,
the four-fifths ratio: the lowest group's approval rate over the highest.

Two boundaries are worth stating plainly.

The protected columns are read from Postgres and used for *grouping only*. They
are never added to a feature path. `predict_frame` runs the same
`prepare_features` call the API does, which drops them, so the scores here are
the scores the served model produces.

And the four-fifths ratio is a screen, not a verdict. Disparate impact in the
legal sense requires a business-necessity analysis -- whether the practice is
job- or credit-related and consistent with necessity, and whether a
less-discriminatory alternative exists -- which this does not perform. A ratio
below 0.8 says a disparity is large enough to warrant that analysis. It does
not establish liability, and a ratio above 0.8 does not establish compliance.

Run standalone against a populated Postgres:

    POSTGRES_HOST=localhost python -m src.ml.fairness_audit

Writes models/fairness_audit.json so the README and the presentation read the
figures rather than having them transcribed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.loader import read_application_table
from src.data.preprocessor import DAYS_PER_YEAR, TARGET
from src.ml.evaluate import load_bands
from src.ml.predict import predict_frame
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

AUDIT_FILE = "fairness_audit.json"

# The four-fifths (80%) rule. A selection rate for any group below four fifths
# of the highest group's rate is the conventional screening threshold for
# adverse impact, from the Uniform Guidelines on Employee Selection Procedures
# and used as a starting screen in fair-lending review.
FOUR_FIFTHS = 0.8

# Groups smaller than this are reported in full but kept out of the ratio.
# A handful of applicants produces an approval rate with enormous variance,
# and letting a 4-row group set the minimum would make the ratio a statement
# about sampling noise rather than about the model. Every excluded group is
# listed in the artifact with its count and rate, so nothing is hidden by it.
MIN_GROUP_FOR_RATIO = 100

AGE_BAND_EDGES = (0.0, 30.0, 45.0, 60.0, np.inf)
AGE_BAND_LABELS = ("under 30", "30-45", "45-60", "60+")

# Read from Postgres for grouping only. None of these enters a feature path.
GENDER_COLUMN = "code_gender"
FAMILY_STATUS_COLUMN = "name_family_status"
BIRTH_COLUMN = "days_birth"


def age_bands(days_birth: pd.Series) -> pd.Series:
    """Age bands from days_birth.

    days_birth is a negative offset from the application date, which is why the
    sign is flipped before dividing. The bands are left-closed and right-open,
    so an applicant who is exactly 30 falls in "30-45" and never in both.
    """
    years = -pd.to_numeric(days_birth, errors="coerce") / DAYS_PER_YEAR
    return pd.cut(
        years,
        bins=list(AGE_BAND_EDGES),
        labels=list(AGE_BAND_LABELS),
        right=False,
        ordered=True,
    )


def _group_rows(
    groups: pd.Series,
    probability: np.ndarray,
    approved: np.ndarray,
    target: pd.Series,
) -> list[dict]:
    """Per-group counts and rates, ordered by group label."""
    rows: list[dict] = []
    labels = groups.cat.categories if isinstance(groups.dtype, pd.CategoricalDtype) \
        else sorted(groups.dropna().unique())

    for label in labels:
        mask = (groups == label).to_numpy(dtype=bool)
        count = int(mask.sum())
        if count == 0:
            continue
        observed = pd.to_numeric(target[mask], errors="coerce")
        rows.append({
            "group": str(label),
            "count": count,
            "share_of_population": float(mask.mean()),
            "approval_rate": float(approved[mask].mean()),
            "mean_predicted_probability": float(probability[mask].mean()),
            "observed_default_rate": float(observed.mean()),
        })

    missing = int(groups.isna().sum())
    if missing:
        mask = groups.isna().to_numpy(dtype=bool)
        observed = pd.to_numeric(target[mask], errors="coerce")
        rows.append({
            "group": "(missing)",
            "count": missing,
            "share_of_population": float(mask.mean()),
            "approval_rate": float(approved[mask].mean()),
            "mean_predicted_probability": float(probability[mask].mean()),
            "observed_default_rate": float(observed.mean()),
        })
    return rows


def four_fifths(rows: list[dict]) -> dict:
    """Lowest group approval rate over the highest, for one attribute.

    "(missing)" is excluded: absence of a recorded value is not a protected
    group, and including it would compare the model against a data-quality
    artefact. Groups below MIN_GROUP_FOR_RATIO are excluded for variance and
    named in the result.
    """
    eligible = [
        r for r in rows
        if r["group"] != "(missing)" and r["count"] >= MIN_GROUP_FOR_RATIO
    ]
    excluded = [
        {"group": r["group"], "count": r["count"], "approval_rate": r["approval_rate"]}
        for r in rows
        if r not in eligible
    ]

    if len(eligible) < 2:
        return {
            "ratio": None,
            "note": "fewer than two groups met the minimum size; no ratio computed",
            "excluded_from_ratio": excluded,
        }

    lowest = min(eligible, key=lambda r: r["approval_rate"])
    highest = max(eligible, key=lambda r: r["approval_rate"])
    ratio = (
        lowest["approval_rate"] / highest["approval_rate"]
        if highest["approval_rate"] > 0 else None
    )

    return {
        "ratio": ratio,
        "passes_four_fifths": None if ratio is None else bool(ratio >= FOUR_FIFTHS),
        "lowest_group": lowest["group"],
        "lowest_approval_rate": lowest["approval_rate"],
        "highest_group": highest["group"],
        "highest_approval_rate": highest["approval_rate"],
        "groups_compared": len(eligible),
        "excluded_from_ratio": excluded,
    }


def audit(limit: int | None = None) -> dict:
    """Score the population, band it, and measure disparity per attribute."""
    bands = load_bands()
    threshold = bands["t_high"]

    log.info("reading application_train%s", f" (limit {limit})" if limit else "")
    frame = read_application_table(limit=limit)
    log.info("scoring %d applicants with the served model", len(frame))

    probability = predict_frame(frame)
    # The deployed decision. t_high is the cost-optimal threshold and the
    # Medium/High boundary: at or above it an applicant is declined or
    # escalated, below it they are approved outright or sent to manual review.
    # Approval here means "not declined", which is the decision a disparate
    # impact screen is about.
    approved = probability < threshold
    target = frame[TARGET]

    attributes = {
        GENDER_COLUMN: frame[GENDER_COLUMN].astype("object").where(
            frame[GENDER_COLUMN].notna()
        ),
        "age_band": age_bands(frame[BIRTH_COLUMN]),
        FAMILY_STATUS_COLUMN: frame[FAMILY_STATUS_COLUMN].astype("object").where(
            frame[FAMILY_STATUS_COLUMN].notna()
        ),
    }

    results = {}
    for name, groups in attributes.items():
        rows = _group_rows(groups, probability, approved, target)
        results[name] = {"groups": rows, "four_fifths": four_fifths(rows)}
        ratio = results[name]["four_fifths"].get("ratio")
        log.info(
            "%s: %d groups, four-fifths ratio %s",
            name, len(rows), "n/a" if ratio is None else f"{ratio:.4f}",
        )

    failing = [
        name for name, r in results.items()
        if r["four_fifths"].get("passes_four_fifths") is False
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm",
        "n_scored": int(len(frame)),
        "decision": {
            "threshold": threshold,
            "threshold_basis": bands.get("t_high_basis"),
            "rule": (
                "approved = calibrated default probability < t_high. t_high is "
                "the Medium/High band boundary: at or above it an applicant is "
                "declined or escalated."
            ),
            "population_approval_rate": float(approved.mean()),
            "population_default_rate": float(pd.to_numeric(target).mean()),
        },
        "attributes": results,
        "four_fifths_rule": {
            "threshold": FOUR_FIFTHS,
            "min_group_for_ratio": MIN_GROUP_FOR_RATIO,
            "failing_attributes": failing,
            "note": (
                "A ratio below 0.8 is a screening flag, not a finding of "
                "disparate impact. Disparate impact in the legal sense requires "
                "a business-necessity analysis and a search for a "
                "less-discriminatory alternative, neither of which is performed "
                "here. Equally, a ratio above 0.8 is not a compliance result."
            ),
        },
        "scope": {
            "attributes_are_grouping_only": (
                "code_gender, days_birth and name_family_status are read from "
                "Postgres and used to group applicants. They are not added to "
                "any feature path: prepare_features drops them, so the scores "
                "measured here are the scores the served model produces."
            ),
            "population": (
                "application_train in full. These are the rows the model was "
                "fitted on, so the approval rates describe the training "
                "population rather than an out-of-sample one."
            ),
        },
    }


def save(result: dict) -> None:
    path = models_dir() / AUDIT_FILE
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disparate impact screen for the served model"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="score only the first N applicants (smoke test; skews every rate)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the summary without writing models/fairness_audit.json",
    )
    args = parser.parse_args()

    result = audit(limit=args.limit)

    for name, data in result["attributes"].items():
        print(f"\n{name}")
        print(f"  {'group':<24} {'n':>8} {'approve':>9} {'mean p':>9} {'default':>9}")
        for row in data["groups"]:
            print(
                f"  {row['group']:<24} {row['count']:>8,} "
                f"{row['approval_rate']:>9.4f} "
                f"{row['mean_predicted_probability']:>9.4f} "
                f"{row['observed_default_rate']:>9.4f}"
            )
        ff = data["four_fifths"]
        if ff.get("ratio") is None:
            print(f"  four-fifths ratio: n/a ({ff.get('note', '')})")
        else:
            verdict = "PASS" if ff["passes_four_fifths"] else "FAIL"
            print(
                f"  four-fifths ratio: {ff['ratio']:.4f} [{verdict}]  "
                f"lowest {ff['lowest_group']} ({ff['lowest_approval_rate']:.4f}) / "
                f"highest {ff['highest_group']} ({ff['highest_approval_rate']:.4f})"
            )

    failing = result["four_fifths_rule"]["failing_attributes"]
    print(
        f"\nthreshold {result['decision']['threshold']:.6f}, "
        f"{result['n_scored']:,} applicants, "
        f"population approval rate {result['decision']['population_approval_rate']:.4f}"
    )
    print(
        "attributes below 0.8: " + (", ".join(failing) if failing else "none")
    )

    if args.no_save:
        return
    save(result)


if __name__ == "__main__":
    main()
