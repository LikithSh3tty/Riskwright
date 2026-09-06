"""Disparate impact screen: the arithmetic, and the judgement calls around it.

The four-fifths ratio is a number a reader will quote. Two things about how it
is computed are choices rather than facts -- the sub-100-row floor and the
exclusion of "(missing)" -- and both can move the headline figure. They are
tested here so the choice stays visible and cannot drift silently.

No database and no model. `audit` is driven with the loader and the scorer
mocked, which leaves the threshold arithmetic, the grouping and the ratio as the
only real things under test. Those are what the README quotes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import fairness_audit as fa


# --------------------------------------------------------------------------
# Age bands
# --------------------------------------------------------------------------

def test_age_bands_are_left_closed_so_no_applicant_lands_in_two():
    exactly = pd.Series([
        -30 * fa.DAYS_PER_YEAR,      # exactly 30
        -45 * fa.DAYS_PER_YEAR,      # exactly 45
        -60 * fa.DAYS_PER_YEAR,      # exactly 60
    ])
    assert list(fa.age_bands(exactly)) == ["30-45", "45-60", "60+"]


def test_age_bands_flip_the_sign_of_days_birth():
    """days_birth is a negative offset. Forgetting the sign puts everyone in one band."""
    banded = fa.age_bands(pd.Series([-9000, -14000, -20000, -25000]))
    assert list(banded) == ["under 30", "30-45", "45-60", "60+"]


def test_age_band_is_missing_rather_than_wrong_when_days_birth_is_null():
    assert pd.isna(fa.age_bands(pd.Series([None]))).all()


# --------------------------------------------------------------------------
# The four-fifths ratio
# --------------------------------------------------------------------------

def row(group, count, approval_rate):
    return {
        "group": group,
        "count": count,
        "share_of_population": 0.0,
        "approval_rate": approval_rate,
        "mean_predicted_probability": 0.0,
        "observed_default_rate": 0.0,
    }


def test_ratio_is_lowest_over_highest():
    result = fa.four_fifths([
        row("A", 1000, 0.90),
        row("B", 1000, 0.72),
        row("C", 1000, 0.81),
    ])
    assert result["ratio"] == pytest.approx(0.72 / 0.90)
    assert result["lowest_group"] == "B"
    assert result["highest_group"] == "A"
    assert result["groups_compared"] == 3


def test_exactly_four_fifths_passes():
    """0.8 is the boundary, and the rule is stated as "below four fifths"."""
    result = fa.four_fifths([row("A", 1000, 1.0), row("B", 1000, 0.8)])
    assert result["ratio"] == pytest.approx(0.8)
    assert result["passes_four_fifths"] is True


def test_just_below_four_fifths_fails():
    result = fa.four_fifths([row("A", 1000, 1.0), row("B", 1000, 0.79)])
    assert result["passes_four_fifths"] is False


def test_a_single_group_yields_no_ratio_rather_than_a_meaningless_one():
    result = fa.four_fifths([row("A", 1000, 0.7)])
    assert result["ratio"] is None
    assert "fewer than two groups" in result["note"]


def test_ratio_is_none_when_every_group_is_refused_outright():
    """Dividing by a zero approval rate is undefined, not 0."""
    result = fa.four_fifths([row("A", 1000, 0.0), row("B", 1000, 0.0)])
    assert result["ratio"] is None


# --------------------------------------------------------------------------
# The small-group floor
# --------------------------------------------------------------------------

def test_small_groups_are_excluded_from_the_ratio_but_still_reported():
    """An approval rate over four rows is a statement about sampling noise."""
    rows = [
        row("F", 202448, 0.7412),
        row("M", 105059, 0.6321),
        row("XNA", 4, 1.0),
    ]
    result = fa.four_fifths(rows)

    assert result["groups_compared"] == 2
    assert result["lowest_group"] == "M" and result["highest_group"] == "F"
    excluded = {e["group"] for e in result["excluded_from_ratio"]}
    assert excluded == {"XNA"}
    # Excluded, not hidden: the count and the rate travel with it.
    assert result["excluded_from_ratio"][0]["count"] == 4
    assert result["excluded_from_ratio"][0]["approval_rate"] == 1.0


def test_the_floor_actually_changes_the_answer():
    """The floor is load-bearing, not decoration.

    This is the real family-status case: a two-applicant "Unknown" group with a
    100% approval rate. Including it would take the attribute from 0.7885 down
    to 0.6376 -- so the floor makes the reported figure *better*, and that has
    to be visible rather than buried in a constant.
    """
    rows = [
        row("Civil marriage", 29775, 0.6376),
        row("Widow", 16088, 0.8086),
        row("Unknown", 2, 1.0),
    ]
    with_floor = fa.four_fifths(rows)["ratio"]

    without_floor = min(r["approval_rate"] for r in rows) / max(
        r["approval_rate"] for r in rows
    )

    assert with_floor == pytest.approx(0.6376 / 0.8086, rel=1e-6)
    assert without_floor == pytest.approx(0.6376)
    assert without_floor < with_floor < fa.FOUR_FIFTHS


def test_a_group_exactly_on_the_floor_is_included():
    rows = [
        row("A", 5000, 0.90),
        row("B", fa.MIN_GROUP_FOR_RATIO, 0.50),
    ]
    assert fa.four_fifths(rows)["groups_compared"] == 2


def test_missing_is_never_a_protected_group():
    """Absence of a recorded value is a data-quality artefact, not a class."""
    rows = [
        row("A", 5000, 0.90),
        row("B", 5000, 0.80),
        row("(missing)", 5000, 0.10),
    ]
    result = fa.four_fifths(rows)

    assert result["groups_compared"] == 2
    assert result["lowest_group"] == "B"
    assert "(missing)" in {e["group"] for e in result["excluded_from_ratio"]}


# --------------------------------------------------------------------------
# Group rows: approval rate at a threshold, and empty groups
# --------------------------------------------------------------------------

def test_group_rows_compute_rates_within_each_group():
    groups = pd.Series(["A", "A", "A", "A", "B", "B"])
    probability = np.array([0.01, 0.02, 0.50, 0.60, 0.90, 0.01])
    approved = probability < 0.5
    target = pd.Series([0, 0, 1, 1, 1, 0])

    rows = {r["group"]: r for r in fa._group_rows(groups, probability, approved, target)}

    assert rows["A"]["count"] == 4
    assert rows["A"]["approval_rate"] == pytest.approx(0.5)     # 2 of 4 below 0.5
    assert rows["A"]["mean_predicted_probability"] == pytest.approx(0.2825)
    assert rows["A"]["observed_default_rate"] == pytest.approx(0.5)
    assert rows["B"]["approval_rate"] == pytest.approx(0.5)
    assert rows["A"]["share_of_population"] == pytest.approx(4 / 6)


def test_an_empty_category_is_dropped_rather_than_reported_as_zero():
    """A band with no applicants has no approval rate. 0.0 would read as "all refused"."""
    groups = pd.Series(
        pd.Categorical(["under 30", "60+"], categories=list(fa.AGE_BAND_LABELS))
    )
    probability = np.array([0.01, 0.02])
    rows = fa._group_rows(groups, probability, probability < 0.5, pd.Series([0, 0]))

    reported = {r["group"] for r in rows}
    assert reported == {"under 30", "60+"}
    assert "30-45" not in reported and "45-60" not in reported


def test_missing_group_values_are_bucketed_and_labelled():
    groups = pd.Series(["A", "A", None, None])
    probability = np.array([0.01, 0.02, 0.90, 0.95])
    rows = {r["group"]: r
            for r in fa._group_rows(groups, probability, probability < 0.5,
                                    pd.Series([0, 0, 1, 1]))}

    assert rows["(missing)"]["count"] == 2
    assert rows["(missing)"]["approval_rate"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# audit(), end to end with the data layer mocked
# --------------------------------------------------------------------------

@pytest.fixture
def synthetic_population(monkeypatch):
    """Twelve applicants whose scores straddle the deployed threshold.

    Built so the expected approval rates are countable by hand: everyone under
    30 scores above t_high and is refused, everyone 60+ scores below it and is
    approved.
    """
    frame = pd.DataFrame({
        "sk_id_curr": range(12),
        "target": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "days_birth": (
            [-25 * fa.DAYS_PER_YEAR] * 3        # under 30
            + [-35 * fa.DAYS_PER_YEAR] * 3      # 30-45
            + [-50 * fa.DAYS_PER_YEAR] * 3      # 45-60
            + [-65 * fa.DAYS_PER_YEAR] * 3      # 60+
        ),
        "code_gender": ["M"] * 6 + ["F"] * 6,
        "name_family_status": ["Married"] * 6 + ["Widow"] * 6,
    })
    # Above t_high (0.0837) for the young, below it for the old.
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.01,
                       0.01, 0.01, 0.5, 0.01, 0.01, 0.01])

    monkeypatch.setattr(fa, "read_application_table", lambda limit=None: frame)
    monkeypatch.setattr(fa, "predict_frame", lambda f: scores)
    return frame, scores


def test_audit_applies_the_deployed_threshold(synthetic_population):
    _, scores = synthetic_population
    result = fa.audit()

    threshold = result["decision"]["threshold"]
    expected_approved = float((scores < threshold).mean())

    assert result["decision"]["population_approval_rate"] == pytest.approx(expected_approved)
    assert result["n_scored"] == 12
    assert "t_high" in result["decision"]["rule"]


def test_audit_approval_is_strictly_below_the_threshold(synthetic_population,
                                                        monkeypatch):
    """At the threshold an applicant is refused, not approved.

    t_high is the Medium/High boundary and High means decline, so a score
    landing exactly on it must fall on the refused side.
    """
    frame, _ = synthetic_population
    t_high = fa.load_bands()["t_high"]
    monkeypatch.setattr(fa, "predict_frame", lambda f: np.full(len(frame), t_high))

    result = fa.audit()
    assert result["decision"]["population_approval_rate"] == 0.0


def test_audit_covers_the_three_required_attributes(synthetic_population):
    result = fa.audit()
    assert set(result["attributes"]) == {
        "code_gender", "age_band", "name_family_status",
    }


def test_audit_flags_the_attributes_that_fall_below_four_fifths(synthetic_population):
    result = fa.audit()
    failing = set(result["four_fifths_rule"]["failing_attributes"])

    computed = {
        name for name, data in result["attributes"].items()
        if data["four_fifths"].get("passes_four_fifths") is False
    }
    assert failing == computed


def test_audit_records_that_the_screen_is_not_a_legal_finding(synthetic_population):
    """The caveat travels with the number, not only in the README."""
    note = fa.audit()["four_fifths_rule"]["note"].lower()

    assert "business-necessity" in note or "business necessity" in note
    assert "not" in note


def test_audit_states_that_the_protected_columns_never_reach_a_feature_path(
        synthetic_population):
    scope = fa.audit()["scope"]["attributes_are_grouping_only"]
    assert "not added to" in scope and "prepare_features" in scope


def test_audit_observed_default_rate_comes_from_target(synthetic_population):
    frame, _ = synthetic_population
    result = fa.audit()

    assert result["decision"]["population_default_rate"] == pytest.approx(
        frame["target"].mean()
    )
