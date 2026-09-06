"""Cost ratio sensitivity sweep.

The load-bearing test here is the first one: applying the shipped ratio must
reproduce `models/threshold.json` exactly. If it does not, the sweep and the
deployed threshold are computing different things and every figure the sweep
reports is about a decision the system does not make.

Runs from committed artifacts, so no database and no model load.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ml import threshold_sensitivity as ts
from src.ml.evaluate import decision_report, load_bands
from src.utils.docker_utils import models_dir


@pytest.fixture(scope="module")
def oof():
    return ts.load_oof()


@pytest.fixture(scope="module")
def shipped() -> dict:
    return json.loads((models_dir() / "threshold.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def swept() -> dict:
    """One sweep for the whole module.

    Each call re-optimises nine thresholds over 307,511 out-of-fold scores,
    which is about eight seconds. Running it per test made the suite slower
    than everything else combined for no additional coverage.
    """
    return ts.sweep()


# --------------------------------------------------------------------------
# The sweep and the shipped threshold must agree
# --------------------------------------------------------------------------

def test_the_shipped_ratio_reproduces_the_shipped_threshold_report(oof, shipped):
    """The check that makes the rest of the sweep meaningful.

    The stored out-of-fold scores are on the model's weighted scale; the shipped
    threshold is on the calibrated one. If `load_oof` ever stops calibrating,
    every threshold in the sweep silently describes a different scale and
    nothing errors.
    """
    y_true, y_score = oof
    report = decision_report(y_true, y_score, shipped["t_high"])
    expected = shipped["at_cost_optimal_threshold"]

    assert report["confusion_matrix"] == expected["confusion_matrix"]
    assert report["precision"] == pytest.approx(expected["precision"])
    assert report["recall"] == pytest.approx(expected["recall"])
    assert report["flagged_rate"] == pytest.approx(expected["flagged_rate"])


def test_calibrated_scores_sit_on_the_probability_scale(oof):
    y_true, y_score = oof
    assert 0.0 <= y_score.min() and y_score.max() <= 1.0
    # A calibrated score's mean should land near the base rate, not near 0.5.
    assert y_score.mean() == pytest.approx(y_true.mean(), abs=0.02)


# --------------------------------------------------------------------------
# Cost arithmetic
# --------------------------------------------------------------------------

def test_cost_counts_only_the_two_kinds_of_mistake():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.9, 0.1])
    # At 0.5: one defaulter missed (0.1), one good customer refused (0.9).
    assert ts.cost_at(y_true, y_score, 0.5, ratio=10.0) == 10.0 + 1.0
    assert ts.cost_at(y_true, y_score, 0.5, ratio=3.0) == 3.0 + 1.0


def test_a_higher_ratio_prices_the_same_mistakes_higher():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.9, 0.1])
    assert (ts.cost_at(y_true, y_score, 0.5, 20.0)
            > ts.cost_at(y_true, y_score, 0.5, 3.0))


def test_refusing_everyone_costs_only_false_positives():
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    assert ts.cost_at(y_true, y_score, 0.0, 10.0) == 3.0


# --------------------------------------------------------------------------
# The sweep's shape
# --------------------------------------------------------------------------

def test_a_costlier_false_negative_lowers_the_threshold(oof):
    """The direction is the whole logic of the exercise.

    If missing a defaulter costs more, you refuse more people, which means a
    lower bar. A sweep that moved the other way would be a sign error.
    """
    y_true, y_score = oof
    low = ts.optimal_threshold(y_true, y_score, ratio=3.0)
    high = ts.optimal_threshold(y_true, y_score, ratio=20.0)
    assert high < low


def test_thresholds_are_monotonic_across_the_swept_range(swept):
    result = swept
    thresholds = [row["t_high"] for row in result["ratios"]]
    assert thresholds == sorted(thresholds, reverse=True)

    approvals = [row["approval_rate"] for row in result["ratios"]]
    assert approvals == sorted(approvals, reverse=True)


def test_every_ratio_beats_the_naive_half_threshold(swept):
    """The method survives even where the number does not: choosing by cost is
    better than 0.5 at every ratio in the range, which is the claim the README
    makes about robustness."""
    for row in swept["ratios"]:
        assert row["cost_reduction_vs_half"] > 0
        assert row["t_high"] < 0.5


def test_catching_more_defaulters_costs_more_approvals(swept):
    rows = swept["ratios"]
    for earlier, later in zip(rows, rows[1:]):
        assert later["defaulters_caught"] > earlier["defaulters_caught"]
        assert later["approval_rate"] < earlier["approval_rate"]


def test_the_range_covers_at_least_three_to_twenty(swept):
    ratios = [row["ratio"] for row in swept["ratios"]]
    assert min(ratios) <= 3.0
    assert max(ratios) >= 20.0
    assert ts.SHIPPED_RATIO in ratios


def test_exactly_one_row_is_marked_as_shipped(swept):
    rows = swept["ratios"]
    shipped_rows = [r for r in rows if r["is_shipped"]]
    assert len(shipped_rows) == 1
    assert shipped_rows[0]["ratio"] == 10.0


def test_the_shipped_row_matches_the_deployed_threshold(swept):
    shipped_row = next(r for r in swept["ratios"] if r["is_shipped"])
    assert shipped_row["t_high"] == pytest.approx(load_bands()["t_high"], abs=1e-6)


# --------------------------------------------------------------------------
# What the sweep concludes
# --------------------------------------------------------------------------

def test_band_stability_is_computed_against_the_shipped_ratio(swept):
    result = swept
    stability = result["band_stability"]
    assert 0.0 <= stability["unchanged_share"] <= 1.0
    assert stability["unchanged_share"] + stability["changed_share"] == pytest.approx(1.0)
    # An applicant whose band survives every ratio does not depend on the
    # assumption. That this is nowhere near 100% is the finding.
    assert stability["unchanged_share"] < 1.0


def test_the_threshold_spread_is_reported_relative_to_the_shipped_value(swept):
    spread = swept["threshold_range"]
    assert spread["max"] > spread["min"]
    assert spread["spread"] == pytest.approx(spread["max"] - spread["min"])
    assert spread["relative_spread"] > 0


def test_the_sweep_does_not_rewrite_the_deployed_threshold():
    """10:1 stays shipped. A sensitivity study that moved the threshold would
    invalidate every other figure in the README."""
    before = (models_dir() / "threshold.json").read_bytes()
    ts.sweep()
    assert (models_dir() / "threshold.json").read_bytes() == before


def test_t_low_is_held_fixed_because_it_is_not_a_cost_decision(swept):
    result = swept
    assert result["t_low"] == pytest.approx(load_bands()["t_low"])
    assert "policy dial" in result["t_low_note"]
