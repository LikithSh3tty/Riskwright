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


def _walk(tree, feature_names: list[str], base_rate: float) -> list[dict]:
    rules: list[dict] = []

    def recurse(node: int, conditions: list[str], readable: list[str]) -> None:
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:  # leaf
            counts = tree.value[node][0]
            # sklearn normalises value to class proportions for classifiers.
            n = int(tree.n_node_samples[node])
            default_rate = float(counts[1] / counts.sum()) if counts.sum() else 0.0
            rules.append(
                {
                    "conditions": list(conditions),
                    "readable": " and ".join(readable) if readable else "all applicants",
                    "support_n": n,
                    "default_rate": default_rate,
                    "lift": float(default_rate / base_rate) if base_rate else 0.0,
                }
            )
            return

        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])

        recurse(
            left,
            conditions + [f"{feature} <= {threshold:.4f}"],
            readable + [_readable_condition(feature, threshold, go_left=True)],
        )
        recurse(
            right,
            conditions + [f"{feature} > {threshold:.4f}"],
            readable + [_readable_condition(feature, threshold, go_left=False)],
        )

    recurse(0, [], [])
    return rules


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

    rules = _walk(tree.tree_, list(filled.columns), base_rate)
    total = len(filled)
    for index, rule in enumerate(sorted(rules, key=lambda r: r["lift"], reverse=True), 1):
        rule["rule_id"] = f"R{index}"
        rule["support_pct"] = round(100 * rule["support_n"] / total, 2)

    ranked = sorted(rules, key=lambda r: r["lift"], reverse=True)

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
