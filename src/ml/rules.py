"""Business-readable decision rules derived from the model.

The assignment asks for rules that bridge ML output and credit policy. A
gradient-boosted ensemble of 400 trees is not something anyone can put in a
policy document, so a shallow surrogate tree is fitted and its root-to-leaf
paths are emitted as if-then rules with support and precision.

Two choices make the rules readable:

  Depth 3, so no rule has more than three conditions.
  Only the top features by SHAP importance, so conditions are on quantities a
  credit officer recognises rather than on obscure columns.

The rules describe the model's behaviour, not the model itself. They are a
summary, and the README says so.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.ml.explain import label_for
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

RULES_FILE = "rules.json"

MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 1000
TOP_FEATURES = 8
RANDOM_STATE = 42


def _readable_condition(feature: str, threshold: float, go_left: bool) -> str:
    label = label_for(feature)
    comparator = "at most" if go_left else "above"

    # Days columns are negative offsets from the application date, so a raw
    # threshold of -4000 means nothing to a reader. Convert to years and flip
    # the comparison, since more negative means longer ago.
    if feature.startswith("days_") and feature != "days_employed_anomalous":
        years = abs(threshold) / 365.25
        comparator = "at least" if go_left else "less than"
        return f"{label} is {comparator} {years:.1f} years"

    if abs(threshold) >= 1000:
        return f"{label} is {comparator} {threshold:,.0f}"
    return f"{label} is {comparator} {threshold:.3f}"


def _collect_paths(tree, feature_names: list[str]) -> dict[int, list[tuple]]:
    """Map each leaf id to the raw (feature, operator, threshold) triples."""
    paths: dict[int, list[tuple]] = {}

    def recurse(node: int, conditions: list[tuple]) -> None:
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:  # leaf
            paths[node] = list(conditions)
            return

        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        recurse(left, conditions + [(feature, "<=", threshold)])
        recurse(right, conditions + [(feature, ">", threshold)])

    recurse(0, [])
    return paths


def _simplify(conditions: list[tuple]) -> list[tuple]:
    """Collapse repeated splits on the same feature into one interval.

    A depth-3 tree frequently splits the same feature twice on a path, giving
    conditions like "score <= 0.536 and score <= 0.312". The tighter bound
    subsumes the looser one, and printing both makes a rule look muddled to
    exactly the reader it is written for.
    """
    upper: dict[str, float] = {}
    lower: dict[str, float] = {}
    order: list[str] = []

    for feature, operator, threshold in conditions:
        if feature not in order:
            order.append(feature)
        if operator == "<=":
            upper[feature] = min(upper.get(feature, threshold), threshold)
        else:
            lower[feature] = max(lower.get(feature, threshold), threshold)

    simplified: list[tuple] = []
    for feature in order:
        if feature in lower:
            simplified.append((feature, ">", lower[feature]))
        if feature in upper:
            simplified.append((feature, "<=", upper[feature]))
    return simplified


def derive_rules(
    features: pd.DataFrame,
    target: np.ndarray | pd.Series,
    top_features: list[str],
) -> dict:
    """Fit a shallow surrogate tree and emit readable rules."""
    selected = [f for f in top_features if f in features.columns][:TOP_FEATURES]
    log.info("deriving rules over %d features: %s", len(selected), ", ".join(selected))

    subset = features[selected].copy()
    # A decision tree needs numbers. Categoricals among the top features are
    # dropped rather than encoded, because "occupation_type <= 4.5" is not a
    # readable rule and encoding it would defeat the purpose.
    numeric = [c for c in subset.columns if subset[c].dtype.kind in "fiub"]
    dropped = [c for c in subset.columns if c not in numeric]
    if dropped:
        log.info("excluded non-numeric features from rules: %s", ", ".join(dropped))
    subset = subset[numeric]

    # The tree cannot take NaN. Median fill is acceptable here because the
    # output is a descriptive summary, not a scoring path.
    filled = subset.fillna(subset.median(numeric_only=True))

    y = np.asarray(target)
    base_rate = float(y.mean())

    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(filled, y)

    # Leaf statistics are computed from the data, never from tree_.value.
    # class_weight="balanced" makes that array hold reweighted proportions, so
    # reading a default rate off it reports something like 78% on a population
    # whose true rate is 8%. Assigning rows to leaves and counting gives the
    # real numbers.
    leaf_of_row = tree.apply(filled)
    paths = _collect_paths(tree.tree_, list(filled.columns))

    rules: list[dict] = []
    total = len(filled)
    for leaf_id, conditions in paths.items():
        mask = leaf_of_row == leaf_id
        support_n = int(mask.sum())
        if support_n == 0:
            continue
        default_rate = float(y[mask].mean())
        simplified = _simplify(conditions)
        rules.append(
            {
                "conditions": [f"{f} {op} {t:.4f}" for f, op, t in simplified],
                "readable": " and ".join(
                    _readable_condition(f, t, go_left=(op == "<="))
                    for f, op, t in simplified
                )
                or "all applicants",
                "support_n": support_n,
                "support_pct": round(100 * support_n / total, 2),
                "default_rate": default_rate,
                "lift": float(default_rate / base_rate) if base_rate else 0.0,
            }
        )

    ranked = sorted(rules, key=lambda r: r["lift"], reverse=True)
    for index, rule in enumerate(ranked, 1):
        rule["rule_id"] = f"R{index}"

    payload = {
        "base_default_rate": base_rate,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "features_used": numeric,
        "features_excluded": dropped,
        "surrogate_fidelity": float(tree.score(filled, y)),
        "note": (
            "These rules summarise the behaviour of the LightGBM model using a "
            "depth-3 surrogate tree. They are a readable approximation for "
            "credit policy, not the model itself."
        ),
        "rules": ranked,
    }

    log.info(
        "derived %d rules, lift range %.2f to %.2f",
        len(ranked),
        ranked[-1]["lift"] if ranked else 0,
        ranked[0]["lift"] if ranked else 0,
    )
    return payload


def save_rules(payload: dict) -> None:
    path = models_dir() / RULES_FILE
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def load_rules() -> dict:
    path = models_dir() / RULES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.ml.train` to produce it."
        )
    return json.loads(path.read_text(encoding="utf-8"))
