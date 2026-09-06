"""Systematic proxy detection over the shipped feature matrix.

The fair-lending section used to say that `employed_life_ratio` is "unlikely to
be the only such path" and that no systematic pass had been run. This is that
pass. It settles the question rather than leaving it as a suspicion.

For each of the five excluded protected attributes, every one of the 120
features the model actually receives is scored for how strongly it predicts
that attribute. A feature that predicts a protected attribute well is a channel
through which the attribute reaches the model despite having been dropped from
the matrix -- which is the whole point: excluding a column removes the label,
not the information.

What is measured, and why three statistics rather than one
----------------------------------------------------------

Association between a feature and an attribute is a different computation
depending on the types involved, and using the wrong one silently understates
the relationship:

  numeric   vs numeric     Spearman rho. Rank-based, so it catches monotonic
                           relationships that are not linear. `days_birth`
                           against `age_years` is exactly linear; a ratio built
                           on it is not.
  numeric   vs categorical Correlation ratio, eta squared: the share of the
                           numeric column's variance explained by group
                           membership. Works in either direction -- a numeric
                           feature grouped by a categorical attribute, or a
                           numeric attribute grouped by a categorical feature.
  categorical vs categorical  Cramer's V, a chi-squared association normalised
                           to 0..1 so it is comparable to a correlation.

Those are not on the same scale as they stand: eta squared is a proportion of
variance while rho and V are correlation-like. Ranking them together would
compare a squared quantity against unsquared ones and understate every
numeric-vs-categorical pair. So each is also reported on a common 0..1
correlation-like scale -- `association` -- which is |rho|, sqrt(eta squared),
and V respectively. Ranking and thresholding happen on that column; the raw
statistic and its name travel alongside so nothing is hidden behind the
normalisation.

Thresholds, fixed before the first run
--------------------------------------

Choosing a cutoff after seeing the results would make it a description of the
output rather than a standard. These are set here, in advance:

  MATERIAL = 0.20  A feature explains about 4% of the attribute on its own
                   (0.20^2). That is the conventional floor for a small but
                   non-negligible effect, and it is the level at which a
                   feature starts to carry usable reconstructive signal.
  STRONG   = 0.50  About 25% of the attribute from one feature alone. At this
                   level the feature is a substantial stand-in for the
                   attribute rather than merely correlated with it.

Single-feature association is a floor, not a bound. A model combines 120 of
them, and features individually below MATERIAL can jointly reconstruct an
attribute that none of them predicts alone. A full treatment fits a model to
predict each attribute from the whole matrix and reports its accuracy. That is
named as remaining work rather than done here, and this pass is deliberately
the simpler, more interpretable half: it says which columns carry the signal,
which is what someone deciding whether to keep a feature needs to know.

Run against a populated Postgres:

    POSTGRES_HOST=localhost python -m src.ml.proxy_detection

Writes models/proxy_detection.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from src.data.loader import read_application_table
from src.data.preprocessor import (
    PROTECTED_ATTRIBUTES,
    add_derived_features,
    clean,
    load_feature_spec,
    prepare_features,
)
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

DETECTION_FILE = "proxy_detection.json"

# Fixed before the first run. See the module docstring.
MATERIAL = 0.20
STRONG = 0.50

# Categorical attributes need at least this many rows in a level before that
# level contributes to a chi-squared or a group mean. Postgres holds a
# four-applicant "XNA" gender and a two-applicant "Unknown" family status;
# a group that small produces an unstable statistic, not a finding.
MIN_LEVEL_SIZE = 100


def _is_categorical(series: pd.Series) -> bool:
    """Anything that is not a number is treated as a set of labels.

    Tested this way round rather than by listing the string dtypes: pandas 3
    reads text out of Postgres as an Arrow-backed `str` dtype, which is neither
    `object` nor a CategoricalDtype, so a whitelist silently mistakes
    `code_gender` for a numeric column and tries to correlate it.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    return not pd.api.types.is_numeric_dtype(series)


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Rank correlation between two numeric columns, NaNs dropped pairwise."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    a, b = x[mask], y[mask]
    if a.nunique() < 2 or b.nunique() < 2:
        return 0.0
    rho, _ = stats.spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else 0.0


def eta_squared(numeric: pd.Series, groups: pd.Series) -> float:
    """Share of the numeric column's variance explained by group membership.

    The between-group sum of squares over the total. 0 means the group means
    are identical; 1 means group membership determines the value exactly.
    """
    mask = numeric.notna() & groups.notna()
    if mask.sum() < 2:
        return float("nan")
    values, labels = numeric[mask].astype(float), groups[mask].astype(str)

    counts = labels.value_counts()
    keep = counts[counts >= MIN_LEVEL_SIZE].index
    if len(keep) < 2:
        return float("nan")
    level_mask = labels.isin(keep)
    values, labels = values[level_mask], labels[level_mask]

    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    if total == 0:
        return 0.0
    between = float(sum(
        len(group) * (group.mean() - grand) ** 2
        for _, group in values.groupby(labels, observed=True)
    ))
    return between / total


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Chi-squared association between two categoricals, normalised to 0..1.

    Bias-corrected: the uncorrected statistic rises with the number of levels
    even between independent columns, and `organization_type` has 58 of them.
    """
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    left, right = a[mask].astype(str), b[mask].astype(str)

    for series in (left, right):
        counts = series.value_counts()
        keep = counts[counts >= MIN_LEVEL_SIZE].index
        sub = series.isin(keep)
        left, right = left[sub], right[sub]

    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")

    table = pd.crosstab(left, right)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")

    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corrected / denominator))


def associate(feature: pd.Series, attribute: pd.Series) -> dict:
    """One feature against one protected attribute, on a common 0..1 scale."""
    feature_cat, attribute_cat = _is_categorical(feature), _is_categorical(attribute)

    if not feature_cat and not attribute_cat:
        rho = spearman(feature, attribute)
        return {"statistic": "spearman_rho", "value": rho, "association": abs(rho)}

    if feature_cat and attribute_cat:
        v = cramers_v(feature, attribute)
        return {"statistic": "cramers_v", "value": v, "association": v}

    numeric, groups = (attribute, feature) if feature_cat else (feature, attribute)
    eta2 = eta_squared(numeric, groups)
    return {
        "statistic": "eta_squared",
        "value": eta2,
        "association": float(np.sqrt(eta2)) if np.isfinite(eta2) else float("nan"),
    }


def build_frames(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The model's feature matrix, and the protected attributes it never sees.

    The features come from `prepare_features` on the real path, so what is
    scanned is exactly the 120 columns the model receives -- not a
    reconstruction of them.
    """
    raw = read_application_table(limit=limit)
    spec = load_feature_spec()
    features = prepare_features(raw, feature_spec=spec)

    # age_years is derived, not stored, and is dropped before the matrix is
    # built. Rebuilding it here from the same code path is what makes "age" a
    # measurable target rather than an idea.
    derived = add_derived_features(clean(raw))
    protected = derived[[c for c in PROTECTED_ATTRIBUTES if c in derived.columns]]

    return features, protected


def detect(limit: int | None = None) -> dict:
    features, protected = build_frames(limit=limit)
    log.info(
        "scanning %d features against %d protected attributes over %d rows",
        features.shape[1], protected.shape[1], len(features),
    )

    results: dict[str, dict] = {}
    for attribute in protected.columns:
        scored = []
        for feature in features.columns:
            outcome = associate(features[feature], protected[attribute])
            if not np.isfinite(outcome["association"]):
                continue
            scored.append({
                "feature": feature,
                "statistic": outcome["statistic"],
                "value": round(float(outcome["value"]), 6),
                "association": round(float(outcome["association"]), 6),
            })
        scored.sort(key=lambda r: r["association"], reverse=True)

        material = [r for r in scored if r["association"] >= MATERIAL]
        strong = [r for r in scored if r["association"] >= STRONG]
        results[attribute] = {
            "features_scored": len(scored),
            "material_count": len(material),
            "strong_count": len(strong),
            "material": material,
            "top_10": scored[:10],
        }
        log.info(
            "%s: %d of %d features at or above %.2f, %d at or above %.2f "
            "(strongest: %s at %.3f)",
            attribute, len(material), len(scored), MATERIAL, len(strong), STRONG,
            scored[0]["feature"] if scored else "-",
            scored[0]["association"] if scored else float("nan"),
        )

    every_material = sorted(
        {r["feature"] for a in results.values() for r in a["material"]}
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(features)),
        "features_scanned": int(features.shape[1]),
        "protected_attributes": list(protected.columns),
        "thresholds": {
            "material": MATERIAL,
            "strong": STRONG,
            "min_level_size": MIN_LEVEL_SIZE,
            "note": (
                "Fixed in the module before the first run. MATERIAL is the "
                "conventional floor for a small but non-negligible effect "
                "(about 4% of the attribute explained by one feature); STRONG "
                "is about 25%. Choosing a cutoff after seeing the ranking "
                "would make it a description of the output, not a standard."
            ),
        },
        "scale": (
            "association is a common 0..1 correlation-like scale: |Spearman "
            "rho| for numeric-numeric, sqrt(eta squared) for "
            "numeric-categorical, and Cramer's V (bias-corrected) for "
            "categorical-categorical. The raw statistic and its name are "
            "reported alongside."
        ),
        "attributes": results,
        "distinct_material_features": every_material,
        "distinct_material_count": len(every_material),
        "limitation": (
            "Single-feature association only. A model combines 120 features, "
            "and columns individually below the threshold can jointly "
            "reconstruct an attribute none of them predicts alone. Fitting a "
            "model to predict each attribute from the full matrix would bound "
            "that, and is not done here."
        ),
    }


def save(payload: dict) -> None:
    path = models_dir() / DETECTION_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic proxy detection over the shipped feature matrix"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="scan only the first N applicants (smoke test)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    result = detect(limit=args.limit)

    for attribute, data in result["attributes"].items():
        print(f"\n{attribute}  --  {data['material_count']} of "
              f"{data['features_scored']} features at or above {MATERIAL}")
        print(f"  {'feature':<32} {'assoc':>7}  statistic")
        for row in data["top_10"]:
            flag = "**" if row["association"] >= STRONG else (
                " *" if row["association"] >= MATERIAL else "  ")
            print(f"{flag}{row['feature']:<32} {row['association']:>7.3f}  "
                  f"{row['statistic']} = {row['value']:.4f}")

    print(f"\n{result['distinct_material_count']} distinct features are a "
          f"material proxy for at least one excluded attribute:")
    print("  " + ", ".join(result["distinct_material_features"]))

    if not args.no_save:
        save(result)


if __name__ == "__main__":
    main()
