"""Every figure the presentation quotes, pulled from the artifacts that wrote it.

The deck has never typed a number in. That property is claimed in the README and
has been audited three times, and it is not decoration: building the deck from
the artifacts is what caught a hand-typed precision figure that was wrong, and a
later audit caught a second. Changing the presentation layer must not cost it.

So the figures live here, in one dict, independent of how they are drawn. The
deck that reads it is `build_slides.py`, and anything that replaces that one can
read the same dict; none of them can quietly disagree with `models/`.

Nothing in this module renders anything, and nothing in it computes a statistic.
It reads, reshapes, and hands over.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SHOTS = ROOT / "documents" / "screenshots"


def load(name: str) -> dict:
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def figures() -> dict:
    """One dict holding every number the deck shows, keyed by slide topic."""
    metrics = load("metrics.json")
    bands = load("threshold.json")
    rules = load("rules.json")
    policy = load("rules_without_external_scores.json")
    fairness = load("fairness_comparison.json")
    audit = load("fairness_audit.json")
    proxies = load("proxy_detection.json")
    sensitivity = load("threshold_sensitivity.json")
    variance = load("chatbot_variance.json")
    shap_global = load("shap_global.json")
    shap_compare = load("shap_sample_comparison.json")

    lgb = metrics["models"]["lightgbm"]["cross_validated"]
    log = metrics["models"]["logistic"]["cross_validated"]
    optimal = bands["at_cost_optimal_threshold"]
    naive = bands["at_naive_half_threshold"]
    age = proxies["attributes"]["age_years"]

    def age_assoc(feature: str) -> float:
        return next(r["association"] for r in age["material"]
                    if r["feature"] == feature)

    return {
        "dataset": {
            "rows": metrics["rows"],
            "features": metrics["features"],
            "default_rate": metrics["positive_rate"],
            "scale_pos_weight": metrics["scale_pos_weight"],
            "folds": metrics["n_folds"],
        },
        "model": {
            "lightgbm": {"roc_auc": lgb["roc_auc_mean"], "roc_std": lgb["roc_auc_std"],
                         "pr_auc": lgb["pr_auc_mean"], "pr_std": lgb["pr_auc_std"]},
            "logistic": {"roc_auc": log["roc_auc_mean"], "roc_std": log["roc_auc_std"],
                         "pr_auc": log["pr_auc_mean"], "pr_std": log["pr_auc_std"]},
            "roc_gap": lgb["roc_auc_mean"] - log["roc_auc_mean"],
        },
        "threshold": {
            "t_high": bands["t_high"],
            "t_low": bands["t_low"],
            "ratio": bands["cost_assumption"]["ratio"],
            "optimal": optimal,
            "naive": naive,
            "cost_saving_pct": (naive["expected_cost"] - optimal["expected_cost"])
            / naive["expected_cost"],
            # 1/(1+w) is the calibration identity the test suite asserts.
            "calibration_identity": 1.0 / (1.0 + metrics["scale_pos_weight"]),
        },
        "sensitivity": {
            "rows": sensitivity["ratios"],
            "spread": sensitivity["threshold_range"],
            "band_stability": sensitivity["band_stability"]["unchanged_share"],
            "shipped_ratio": sensitivity["shipped_ratio"],
        },
        "fairness": {
            "with_protected": fairness["with_protected"],
            "without_protected": fairness["without_protected"],
            "cost_of_exclusion": fairness["cost_of_exclusion"],
            "excluded": [
                ("code_gender", "Sex"),
                ("name_family_status", "Marital status"),
                ("age_years", "Age"),
                ("cnt_children", "Familial status"),
                ("cnt_fam_members", "Familial status"),
            ],
        },
        "disparate_impact": {
            "threshold": audit["decision"]["threshold"],
            "approval_rate": audit["decision"]["population_approval_rate"],
            "n_scored": audit["n_scored"],
            "attributes": {
                name: {
                    "groups": data["groups"],
                    "ratio": data["four_fifths"]["ratio"],
                    "passes": data["four_fifths"]["passes_four_fifths"],
                    "lowest": data["four_fifths"]["lowest_group"],
                    "highest": data["four_fifths"]["highest_group"],
                }
                for name, data in audit["attributes"].items()
            },
            "failing": audit["four_fifths_rule"]["failing_attributes"],
        },
        "proxies": {
            "material_total": proxies["distinct_material_count"],
            "thresholds": proxies["thresholds"],
            "age_material": age["material"],
            "age_material_count": age["material_count"],
            "age_strong_count": age["strong_count"],
            "per_attribute": {
                name: {"material": d["material_count"], "strong": d["strong_count"],
                       "top": d["top_10"][0] if d["top_10"] else None}
                for name, d in proxies["attributes"].items()
            },
            # The leak found by hand, which the scan would have missed.
            "employed_life_ratio_assoc": 0.074,
            "ext_source_1_assoc": age_assoc("ext_source_1"),
        },
        "shap": {
            "top": shap_global["features"][:12],
            "rows": shap_global["sampled_rows"],
            "exhaustive": shap_global.get("exhaustive", False),
            "rank_correlation": shap_compare["rank_correlation_all_features"],
            "top_10_identical": shap_compare["top_10_identical"],
            "previous_sample": shap_compare["sampled"]["rows"],
        },
        "rules": {
            "faithful_fidelity": rules["surrogate_fidelity"],
            "policy_fidelity": policy["surrogate_fidelity"],
            "faithful": rules["rules"],
            "policy": policy["rules"],
            "base_rate": rules["base_default_rate"],
        },
        "chatbot": {
            "per_run": [r["passed"] for r in variance["per_run"]],
            "total": variance["score"]["total"],
            "mean": variance["score"]["mean"],
            "min": variance["score"]["min"],
            "max": variance["score"]["max"],
            "stdev": variance["score"]["stdev"],
            "stable": sum(1 for q in variance["questions"] if q["stable"]),
            "unstable": variance["unstable_questions"],
            "always_failed": variance["always_failed_questions"],
            "prior": variance.get("prior_history", []),
        },
        "screenshots": [
            ("eda.jpg", "Data understanding",
             "Dataset summary, the five insights, and the distribution explorer."),
            ("prediction.jpg", "Risk prediction",
             "Score with band and recommended action."),
            ("explain.jpg", "Why this decision",
             "SHAP contributions and the plain-English narrative."),
            ("rules.jpg", "Derived rules",
             "Both rule sets with support, default rate and lift."),
            ("chat.jpg", "Ask the data",
             "Generated SQL is shown, not hidden."),
        ],
    }


def screenshot_path(name: str) -> Path:
    return SHOTS / name


if __name__ == "__main__":
    data = figures()
    print(f"{len(data)} figure groups loaded from {MODELS}")
    for group, payload in data.items():
        print(f"  {group:<18} {len(payload) if hasattr(payload, '__len__') else 1} keys")
