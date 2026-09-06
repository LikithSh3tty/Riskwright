"""Proxy detection: the three statistics, and the constants that shape them.

The numbers this module produces are quoted in the README as evidence about
fair lending, so the arithmetic is worth pinning. Two constants are judgement
calls that move the answer -- the 0.20 material floor and the 100-row minimum
level size -- and each is tested by showing what changes when it binds, not
merely that the code runs with it.

No database. The frames are synthetic and small enough that the expected
statistics can be reasoned about by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import proxy_detection as pd_mod


# --------------------------------------------------------------------------
# Type dispatch
# --------------------------------------------------------------------------

def test_numeric_columns_are_not_treated_as_labels():
    assert not pd_mod._is_categorical(pd.Series([1, 2, 3]))
    assert not pd_mod._is_categorical(pd.Series([1.5, 2.5]))


def test_text_columns_are_labels_whatever_dtype_pandas_chose():
    """The dtype check is the reason this module works at all.

    pandas 3 reads text out of Postgres as an Arrow-backed `str` dtype, which is
    neither `object` nor a CategoricalDtype. A whitelist of string dtypes
    mistakes `code_gender` for numeric and tries to correlate 'M' with an age.
    """
    assert pd_mod._is_categorical(pd.Series(["M", "F"], dtype=object))
    assert pd_mod._is_categorical(pd.Series(["M", "F"]).astype("string"))
    assert pd_mod._is_categorical(pd.Series(["M", "F"]).astype("category"))


def test_association_picks_the_statistic_from_the_pair_of_types():
    numeric = pd.Series(np.arange(500.0))
    other_numeric = pd.Series(np.arange(500.0) * 2)
    labels = pd.Series(["a", "b"] * 250)
    other_labels = pd.Series(["x", "y"] * 250)

    assert pd_mod.associate(numeric, other_numeric)["statistic"] == "spearman_rho"
    assert pd_mod.associate(numeric, labels)["statistic"] == "eta_squared"
    assert pd_mod.associate(labels, numeric)["statistic"] == "eta_squared"
    assert pd_mod.associate(labels, other_labels)["statistic"] == "cramers_v"


# --------------------------------------------------------------------------
# Spearman
# --------------------------------------------------------------------------

def test_perfect_monotonic_relationship_is_one():
    x = pd.Series(np.arange(200.0))
    assert pd_mod.spearman(x, x ** 3) == pytest.approx(1.0)


def test_spearman_catches_monotonic_but_non_linear():
    """Rank-based on purpose: a ratio built on age is not linear in age."""
    x = pd.Series(np.arange(1.0, 201.0))
    assert pd_mod.spearman(x, np.log(x)) == pytest.approx(1.0)


def test_a_constant_column_has_no_association_rather_than_nan():
    x = pd.Series(np.arange(100.0))
    assert pd_mod.spearman(x, pd.Series([7.0] * 100)) == 0.0


def test_spearman_drops_pairs_not_columns():
    x = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pd_mod.spearman(x, y) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Correlation ratio
# --------------------------------------------------------------------------

def test_eta_squared_is_one_when_the_group_determines_the_value():
    values = pd.Series([10.0] * 200 + [20.0] * 200)
    groups = pd.Series(["a"] * 200 + ["b"] * 200)
    assert pd_mod.eta_squared(values, groups) == pytest.approx(1.0)


def test_eta_squared_is_zero_when_group_means_are_identical():
    values = pd.Series([1.0, 3.0] * 200)
    groups = pd.Series(["a", "a", "b", "b"] * 100)
    assert pd_mod.eta_squared(values, groups) == pytest.approx(0.0, abs=1e-9)


def test_eta_squared_is_a_proportion():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(size=1000))
    groups = pd.Series(rng.choice(["a", "b", "c"], size=1000))
    eta2 = pd_mod.eta_squared(values, groups)
    assert 0.0 <= eta2 <= 1.0


# --------------------------------------------------------------------------
# Cramer's V
# --------------------------------------------------------------------------

def test_cramers_v_is_one_for_a_perfect_correspondence():
    a = pd.Series(["x"] * 300 + ["y"] * 300)
    b = pd.Series(["p"] * 300 + ["q"] * 300)
    assert pd_mod.cramers_v(a, b) == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_is_near_zero_for_independent_columns():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.choice(["x", "y"], size=4000))
    b = pd.Series(rng.choice(["p", "q"], size=4000))
    assert pd_mod.cramers_v(a, b) < 0.1


def test_the_bias_correction_is_load_bearing_for_wide_columns():
    """Uncorrected V rises with the number of levels even between independent
    columns, and organization_type has 58 of them. Without the correction a
    high-cardinality column looks like a proxy purely for being wide.
    """
    rng = np.random.default_rng(2)
    n = 20000
    wide = pd.Series(rng.choice([f"org_{i}" for i in range(50)], size=n))
    binary = pd.Series(rng.choice(["M", "F"], size=n))

    corrected = pd_mod.cramers_v(wide, binary)

    table = pd.crosstab(wide, binary)
    from scipy import stats as scipy_stats
    chi2 = scipy_stats.chi2_contingency(table, correction=False)[0]
    uncorrected = float(np.sqrt((chi2 / n) / (min(table.shape) - 1)))

    assert uncorrected > corrected
    assert corrected < 0.05, "independent columns must not read as a proxy"


# --------------------------------------------------------------------------
# The minimum level size, and what it changes
# --------------------------------------------------------------------------

def test_a_tiny_level_is_ignored_rather_than_allowed_to_dominate():
    """A four-applicant XNA gender exists in this dataset.

    Its group mean is a statement about four people. Left in, it moves eta
    squared substantially; the floor is what stops a rounding error in the data
    reading as a strong proxy.
    """
    rng = np.random.default_rng(7)
    # Real within-group spread, so the two large groups on their own explain
    # only a modest share of the variance.
    values = pd.Series(
        list(rng.normal(10.0, 1.0, 500))
        + list(rng.normal(11.0, 1.0, 500))
        + [10000.0] * 4
    )
    groups = pd.Series(["a"] * 500 + ["b"] * 500 + ["tiny"] * 4)

    with_floor = pd_mod.eta_squared(values, groups)

    # Same data, floor lowered to admit the four-row level.
    original = pd_mod.MIN_LEVEL_SIZE
    try:
        pd_mod.MIN_LEVEL_SIZE = 1
        without_floor = pd_mod.eta_squared(values, groups)
    finally:
        pd_mod.MIN_LEVEL_SIZE = original

    # Four rows out of 1,004 take the statistic from "a modest relationship"
    # to "group membership explains almost everything". That is the floor
    # doing real work, not decorating the code.
    assert with_floor < 0.4
    assert without_floor > 0.9
    assert without_floor > 3 * with_floor


def test_fewer_than_two_usable_levels_is_not_a_measurement():
    values = pd.Series(np.arange(300.0))
    groups = pd.Series(["only_one"] * 300)
    assert np.isnan(pd_mod.eta_squared(values, groups))


def test_cramers_v_also_drops_levels_below_the_floor():
    a = pd.Series(["x"] * 500 + ["y"] * 500 + ["rare"] * 3)
    b = pd.Series(["p"] * 500 + ["q"] * 500 + ["p"] * 3)
    value = pd_mod.cramers_v(a, b)
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

def test_the_thresholds_are_fixed_constants_not_computed_from_results():
    """Declared in the module, so a reader can check them against the README."""
    assert pd_mod.MATERIAL == 0.20
    assert pd_mod.STRONG == 0.50
    assert pd_mod.MATERIAL < pd_mod.STRONG


def test_detect_partitions_features_at_the_material_threshold(monkeypatch):
    """A feature just below the floor is scored and reported, not counted."""
    n = 2000
    rng = np.random.default_rng(3)
    age = pd.Series(np.linspace(20.0, 70.0, n))

    features = pd.DataFrame({
        "strong_proxy": age * 3 + rng.normal(scale=0.1, size=n),
        "unrelated": pd.Series(rng.normal(size=n)),
    })
    protected = pd.DataFrame({"age_years": age})

    monkeypatch.setattr(pd_mod, "build_frames", lambda limit=None: (features, protected))
    result = pd_mod.detect()

    age_result = result["attributes"]["age_years"]
    names = {r["feature"] for r in age_result["material"]}

    assert "strong_proxy" in names
    assert "unrelated" not in names
    assert age_result["features_scored"] == 2
    assert age_result["material_count"] == 1
    assert age_result["strong_count"] == 1
    assert result["distinct_material_features"] == ["strong_proxy"]


def test_detect_records_the_thresholds_it_used(monkeypatch):
    frame = pd.DataFrame({"x": np.arange(500.0)})
    protected = pd.DataFrame({"age_years": np.arange(500.0)})
    monkeypatch.setattr(pd_mod, "build_frames", lambda limit=None: (frame, protected))

    result = pd_mod.detect()
    assert result["thresholds"]["material"] == pd_mod.MATERIAL
    assert result["thresholds"]["strong"] == pd_mod.STRONG
    assert "before the first run" in result["thresholds"]["note"]
    assert "Single-feature" in result["limitation"]


def test_features_that_cannot_be_scored_are_skipped_not_counted_as_zero(monkeypatch):
    """An all-null column has no association, which is not the same as none."""
    n = 500
    features = pd.DataFrame({
        "all_null": pd.Series([np.nan] * n),
        "real": pd.Series(np.arange(float(n))),
    })
    protected = pd.DataFrame({"age_years": pd.Series(np.arange(float(n)))})
    monkeypatch.setattr(pd_mod, "build_frames", lambda limit=None: (features, protected))

    result = pd_mod.detect()["attributes"]["age_years"]
    assert result["features_scored"] == 1
    assert {r["feature"] for r in result["top_10"]} == {"real"}
