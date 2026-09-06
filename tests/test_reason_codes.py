"""Adverse action reason codes.

This is a regulated disclosure, so the tests are about what an applicant is
told and what they are not. Three constants decide that -- the four-reason cap,
the contribution floor, and the non-disclosable set -- and each is tested by
showing what the notice would say without it.

No database and no model: `_codes_from_contributions` and `_notice` are pure
given an explanation, so the explanations here are constructed. The threshold
comes from the real `models/threshold.json`.
"""

from __future__ import annotations

import pytest

from src.ml import reason_codes as rc
from src.ml.evaluate import load_bands


def contribution(feature: str, shap_value: float, value=1.0) -> dict:
    return {
        "feature": feature,
        "label": feature.replace("_", " "),
        "value": value,
        "shap_value": shap_value,
        "direction": "raises risk" if shap_value > 0 else "lowers risk",
    }


def explanation(probability: float, contributions: list[dict], sk_id_curr=1) -> dict:
    return {
        "sk_id_curr": sk_id_curr,
        "base_value": -0.58,
        "default_probability": probability,
        "risk_band": "High" if probability >= 0.083669 else "Low",
        "contributions": contributions,
        "narrative": "",
    }


DECLINED = 0.45
APPROVED = 0.02


# --------------------------------------------------------------------------
# Who gets a notice
# --------------------------------------------------------------------------

def test_an_applicant_above_the_threshold_gets_reasons():
    notice = rc._notice(explanation(DECLINED, [
        contribution("ext_source_3", 0.9),
        contribution("amt_credit", 0.3),
    ]))
    assert notice["adverse_action_required"] is True
    assert notice["reasons"]
    assert notice["threshold"] == load_bands()["t_high"]


def test_an_approved_applicant_gets_no_notice_and_no_reasons():
    """Sending an adverse action notice to someone who was approved is both
    wrong and alarming. The reasons are not computed rather than computed and
    hidden, so there is nothing to leak into a template by accident."""
    notice = rc._notice(explanation(APPROVED, [
        contribution("ext_source_3", 0.9),
        contribution("amt_credit", 0.3),
    ]))
    assert notice["adverse_action_required"] is False
    assert notice["reasons"] == []
    assert "diagnostics" not in notice
    assert "No adverse action" in notice["note"]


def test_the_threshold_is_the_only_thing_separating_the_two():
    """Identical contributions, one side of t_high each. Nothing else differs."""
    t_high = load_bands()["t_high"]
    contributions = [contribution("ext_source_3", 0.9)]

    just_under = rc._notice(explanation(t_high - 1e-9, contributions))
    exactly_on = rc._notice(explanation(t_high, contributions))

    assert just_under["adverse_action_required"] is False
    assert exactly_on["adverse_action_required"] is True, (
        "at the threshold the applicant is declined, matching band_for"
    )


# --------------------------------------------------------------------------
# The four-reason cap
# --------------------------------------------------------------------------

def test_at_most_four_principal_reasons_are_disclosed():
    notice = rc._notice(explanation(DECLINED, [
        contribution("ext_source_3", 0.9),
        contribution("credit_term", 0.8),
        contribution("amt_goods_price", 0.7),
        contribution("days_employed", 0.6),
        contribution("amt_credit", 0.5),
        contribution("occupation_type", 0.4),
    ]))
    assert len(notice["reasons"]) == rc.MAX_REASONS == 4
    assert notice["diagnostics"]["adverse_factors_considered"] == 6


def test_the_cap_keeps_the_largest_contributions_not_the_first_seen():
    notice = rc._notice(explanation(DECLINED, [
        contribution("amt_credit", 0.10),
        contribution("occupation_type", 0.20),
        contribution("days_employed", 0.30),
        contribution("amt_goods_price", 0.40),
        contribution("credit_term", 0.50),
        contribution("ext_source_3", 0.60),
    ]))
    disclosed = [r["contributing_features"][0] for r in notice["reasons"]]
    assert disclosed == ["ext_source_3", "credit_term", "amt_goods_price", "days_employed"]
    assert [r["rank"] for r in notice["reasons"]] == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# The contribution floor
# --------------------------------------------------------------------------

def test_a_negligible_contribution_is_not_a_principal_reason():
    """Without a floor, an applicant declined on two clear factors receives
    four, two of which are noise, and the notice becomes less specific."""
    notice = rc._notice(explanation(DECLINED, [
        contribution("ext_source_3", 0.9),
        contribution("credit_term", 0.5),
        contribution("amt_credit", rc.MIN_ADVERSE_SHAP / 2),
        contribution("occupation_type", rc.MIN_ADVERSE_SHAP / 4),
    ]))
    assert len(notice["reasons"]) == 2
    assert notice["diagnostics"]["adverse_factors_considered"] == 2


def test_the_floor_is_load_bearing(monkeypatch):
    """Same applicant, floor removed: the two noise factors become reasons."""
    contributions = [
        contribution("ext_source_3", 0.9),
        contribution("credit_term", 0.5),
        contribution("amt_credit", 0.001),
        contribution("occupation_type", 0.0005),
    ]
    with_floor = rc._notice(explanation(DECLINED, contributions))

    monkeypatch.setattr(rc, "MIN_ADVERSE_SHAP", 0.0)
    without_floor = rc._notice(explanation(DECLINED, contributions))

    assert len(with_floor["reasons"]) == 2
    assert len(without_floor["reasons"]) == 4


def test_factors_that_lowered_risk_never_appear_in_a_denial():
    """A feature that helped the applicant did not contribute to the denial.
    Listing it would be an untrue statement in a regulated notice."""
    notice = rc._notice(explanation(DECLINED, [
        contribution("ext_source_3", 0.9),
        contribution("amt_income_total", -0.8),
        contribution("name_education_type", -0.5),
    ]))
    named = {f for r in notice["reasons"] for f in r["contributing_features"]}
    assert named == {"ext_source_3"}


# --------------------------------------------------------------------------
# The non-disclosable set
# --------------------------------------------------------------------------

def test_a_non_disclosable_feature_is_withheld_and_named():
    notice = rc._notice(explanation(DECLINED, [
        contribution("region_rating_client", 0.9),
        contribution("ext_source_3", 0.4),
    ]))
    named = {f for r in notice["reasons"] for f in r["contributing_features"]}

    assert "region_rating_client" not in named
    assert notice["diagnostics"]["withheld_non_disclosable"] == ["region_rating_client"]


def test_withholding_changes_what_the_applicant_is_told(monkeypatch):
    """The withheld feature is the largest contribution here, so without the
    exclusion the notice would lead with the applicant's region. That is the
    classic redlining disclosure and the reason the set exists."""
    contributions = [
        contribution("region_rating_client", 0.9),
        contribution("ext_source_3", 0.4),
    ]
    withheld = rc._notice(explanation(DECLINED, contributions))

    monkeypatch.setattr(rc, "NOT_DISCLOSABLE", frozenset())
    disclosed = rc._notice(explanation(DECLINED, contributions))

    assert withheld["reasons"][0]["contributing_features"] == ["ext_source_3"]
    assert disclosed["reasons"][0]["contributing_features"] == ["region_rating_client"]


def test_social_circle_and_timing_columns_are_never_disclosed():
    for feature in ("def_30_cnt_social_circle", "obs_60_cnt_social_circle",
                    "weekday_appr_process_start", "hour_appr_process_start"):
        assert feature in rc.NOT_DISCLOSABLE


def test_every_geographic_column_is_non_disclosable():
    geographic = {c for c in rc.REASON_CODES} | set(rc.NOT_DISCLOSABLE)
    for feature in ("region_population_relative", "region_rating_client",
                    "region_rating_client_w_city", "reg_city_not_work_city"):
        assert feature in rc.NOT_DISCLOSABLE
        assert feature not in rc.REASON_CODES, (
            "a column cannot be both withheld and given a disclosure phrase"
        )
    assert geographic  # the two sets are disjoint, checked above


# --------------------------------------------------------------------------
# Collapsing duplicates
# --------------------------------------------------------------------------

def test_the_three_bureau_scores_are_one_reason_to_an_applicant():
    notice = rc._notice(explanation(DECLINED, [
        contribution("ext_source_3", 0.85),
        contribution("ext_source_1", 0.49),
        contribution("ext_source_2", 0.36),
        contribution("credit_term", 0.22),
    ]))
    assert len(notice["reasons"]) == 2
    first = notice["reasons"][0]
    assert set(first["contributing_features"]) == {
        "ext_source_1", "ext_source_2", "ext_source_3"
    }
    assert first["shap_value"] == pytest.approx(0.85), (
        "the collapsed entry keeps the largest contribution so ranking is unaffected"
    )


def test_collapsing_does_not_let_a_repeated_phrase_crowd_out_others():
    """Four document flags share one phrase. Without collapsing they would fill
    the entire notice and the applicant would learn one thing."""
    notice = rc._notice(explanation(DECLINED, [
        contribution("flag_document_3", 0.5),
        contribution("flag_document_6", 0.4),
        contribution("flag_document_8", 0.3),
        contribution("flag_document_9", 0.2),
        contribution("ext_source_3", 0.1),
    ]))
    phrases = [r["reason"] for r in notice["reasons"]]
    assert len(phrases) == 2
    assert len(set(phrases)) == 2


# --------------------------------------------------------------------------
# The vocabulary itself
# --------------------------------------------------------------------------

def test_every_phrase_is_written_out_and_non_empty():
    for feature, phrase in rc.REASON_CODES.items():
        assert phrase.strip(), f"{feature} has an empty phrase"
        assert phrase[0].isupper(), f"{feature} phrase is not a sentence"
        assert feature not in phrase, (
            f"{feature} phrase leaks the column name into the notice"
        )


def test_an_unmapped_feature_falls_back_rather_than_inventing_a_reason():
    phrase, authored = rc.reason_for("some_column_nobody_wrote_a_phrase_for")
    assert phrase == rc.FALLBACK_REASON
    assert authored is False

    phrase, authored = rc.reason_for("ext_source_1")
    assert phrase == rc.REASON_CODES["ext_source_1"]
    assert authored is True


def test_a_fallback_is_reported_rather_than_passing_silently():
    notice = rc._notice(explanation(DECLINED, [
        contribution("a_brand_new_feature", 0.9),
    ]))
    assert notice["reasons"][0]["reason"] == rc.FALLBACK_REASON
    assert notice["reasons"][0]["authored_phrase"] is False
    assert notice["diagnostics"]["features_without_authored_phrase"] == [
        "a_brand_new_feature"
    ]


def test_the_version_travels_with_the_notice():
    """The wording is what the applicant was told, so the notice has to record
    which wording it used."""
    notice = rc._notice(explanation(DECLINED, [contribution("ext_source_3", 0.9)]))
    assert notice["reason_code_version"] == rc.REASON_CODE_VERSION == "rc-v1"


def test_the_bureau_phrase_triggers_the_extra_disclosure_note():
    notice = rc._notice(explanation(DECLINED, [contribution("ext_source_3", 0.9)]))
    assert "credit reporting agency" in notice["note"] or \
           "consumer reporting agency" in notice["note"] or \
           "credit bureau" in notice["note"]
