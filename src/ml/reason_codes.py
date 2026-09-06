"""Adverse action reason codes.

ECOA and Regulation B require a creditor who denies an application to give the
applicant the **specific principal reasons** for the denial. "Your credit score
was too low" does not satisfy it; neither does a list of everything the model
looked at. The reasons must be the ones that actually drove this decision, and
they must be intelligible to the applicant.

The SHAP contributions computed for every prediction are the right raw
material -- they already say which features pushed this applicant's risk up and
by how much -- but they are not reason codes. `ext_source_3 = 0.19,
shap_value = +0.847` is a model diagnostic. This module is the mapping between
the two.

Why the phrases are a fixed table
---------------------------------

Every phrase below is written out in source, versioned, and reviewed as text.
None is generated at request time. That is the point of the design, not an
implementation shortcut: an adverse action notice is a regulated disclosure,
the same input has to produce the same words every time, and the wording is
something a compliance function signs off once rather than something a language
model improvises per applicant. A generated notice cannot be reviewed before it
is sent, and two applicants denied for the same reason would receive different
letters.

REASON_CODES is therefore append-only in spirit: changing an existing phrase
changes what past applicants were told, so the version below moves when the
wording does.

What this is not
----------------

A production adverse action engine is rule-based and works from the credit
policy, not from a model attribution. It knows that "insufficient income for
the amount requested" is a reason a lender may give and that "your postcode"
is not, and it maps to a fixed enumerated set agreed with counsel. This derives
its reasons from SHAP instead, which means the reason given is the feature that
moved the model, and the model is not the policy. Two consequences worth
stating plainly:

  - A feature can be a legitimate driver of risk and still be an unusable
    reason to put in a letter. `weekday_appr_process_start` is in the model and
    is excluded from disclosure here for exactly that reason.
  - Ranking by SHAP magnitude ranks by influence on this prediction, not by
    what a reviewer would consider the principal cause. Those usually agree.
    They are not the same thing.
"""

from __future__ import annotations

from src.ml.evaluate import load_bands
from src.ml.explain import explain_applicant, explain_features
from src.utils.logger import get_logger

log = get_logger(__name__)

# Moves whenever a phrase below changes, because the phrase is what the
# applicant was told.
REASON_CODE_VERSION = "rc-v1"

# Number of principal reasons disclosed. Regulation B's model forms list up to
# four principal reasons, and the commentary treats disclosure of more than
# four as diluting rather than improving the notice.
MAX_REASONS = 4

# A contribution below this is not a principal reason. Without a floor an
# applicant declined on two clear factors receives four reasons, two of which
# are noise, and the notice becomes less specific rather than more.
MIN_ADVERSE_SHAP = 0.01


# Features that drive the model but must not appear in a notice. Excluded here
# rather than filtered by the caller, so the exclusion travels with the
# disclosure logic and cannot be forgotten at a call site.
NOT_DISCLOSABLE: frozenset[str] = frozenset({
    # No causal story an applicant can act on or would accept as a reason.
    "weekday_appr_process_start",
    "hour_appr_process_start",
    # Geographic. Naming a region as the reason for a denial is the classic
    # redlining disclosure, whatever the model's reason for using it.
    "region_population_relative",
    "region_rating_client",
    "region_rating_client_w_city",
    "reg_region_not_live_region",
    "reg_region_not_work_region",
    "live_region_not_work_region",
    "reg_city_not_live_city",
    "reg_city_not_work_city",
    "live_city_not_work_city",
    # Third-party behaviour. The applicant's social circle defaulting is not
    # the applicant's conduct and is not a reason to give them.
    "obs_30_cnt_social_circle",
    "def_30_cnt_social_circle",
    "obs_60_cnt_social_circle",
    "def_60_cnt_social_circle",
})


# The disclosure text. Hand-authored, reviewed as text, and stable.
#
# Each entry is what the applicant is told when that feature is among the
# principal reasons their application was declined. Phrases are written in the
# second person and describe the applicant's circumstances rather than the
# model's internals: the applicant is the audience, not an engineer.
REASON_CODES: dict[str, str] = {
    # External bureau scores. The dominant drivers, and the ones most
    # obviously requiring the "obtained from a consumer reporting agency"
    # disclosure that accompanies them.
    "ext_source_1": "Credit assessment obtained from an external credit bureau",
    "ext_source_2": "Credit assessment obtained from an external credit bureau",
    "ext_source_3": "Credit assessment obtained from an external credit bureau",

    # Loan structure relative to means.
    "amt_credit": "Amount of credit requested",
    "amt_goods_price": "Value of the goods being financed relative to the credit requested",
    "amt_annuity": "Size of the repayment relative to your income",
    "amt_income_total": "Income stated on the application",
    "credit_income_ratio": "Amount of credit requested relative to your income",
    "annuity_income_ratio": "Repayment burden relative to your income",
    "credit_term": "Length of the repayment term relative to the amount requested",
    "name_contract_type": "Type of credit applied for",

    # Employment and stability.
    "days_employed": "Length of time in your current employment",
    "days_employed_anomalous": "No current employment recorded on the application",
    "employed_life_ratio": "Length of employment relative to your age",
    "occupation_type": "Occupation recorded on the application",
    "organization_type": "Type of employer recorded on the application",
    "name_income_type": "Source of the income stated on the application",
    "flag_emp_phone": "No employer contact number provided",
    "flag_work_phone": "No work contact number provided",
    "flag_phone": "No contact telephone number provided",
    "flag_email": "No email address provided",
    "flag_cont_mobile": "Contact mobile number could not be reached",
    "flag_mobil": "No mobile number provided",

    # Education and assets.
    "name_education_type": "Level of education recorded on the application",
    "name_housing_type": "Housing situation recorded on the application",
    "flag_own_realty": "No property ownership recorded",
    "flag_own_car": "No vehicle ownership recorded",
    "own_car_age": "Age of the vehicle recorded on the application",

    # File age and identity records.
    "days_registration": "Length of time since your details were registered",
    "days_id_publish": "Length of time since your identity document was issued",
    "days_last_phone_change": "Recent change to your contact telephone number",
    "name_type_suite": "Circumstances recorded when the application was taken",

    # Credit bureau enquiry activity. Frequent recent enquiries are a
    # conventional and defensible reason.
    "amt_req_credit_bureau_hour": "Number of recent credit bureau enquiries",
    "amt_req_credit_bureau_day": "Number of recent credit bureau enquiries",
    "amt_req_credit_bureau_week": "Number of recent credit bureau enquiries",
    "amt_req_credit_bureau_mon": "Number of recent credit bureau enquiries",
    "amt_req_credit_bureau_qrt": "Number of recent credit bureau enquiries",
    "amt_req_credit_bureau_year": "Number of credit bureau enquiries in the past year",

    # Supporting documentation.
    "flag_document_2": "Supporting documentation not supplied with the application",
    "flag_document_3": "Supporting documentation not supplied with the application",
    "flag_document_4": "Supporting documentation not supplied with the application",
    "flag_document_5": "Supporting documentation not supplied with the application",
    "flag_document_6": "Supporting documentation not supplied with the application",
    "flag_document_7": "Supporting documentation not supplied with the application",
    "flag_document_8": "Supporting documentation not supplied with the application",
    "flag_document_9": "Supporting documentation not supplied with the application",
    "flag_document_10": "Supporting documentation not supplied with the application",
    "flag_document_11": "Supporting documentation not supplied with the application",
    "flag_document_12": "Supporting documentation not supplied with the application",
    "flag_document_13": "Supporting documentation not supplied with the application",
    "flag_document_14": "Supporting documentation not supplied with the application",
    "flag_document_15": "Supporting documentation not supplied with the application",
    "flag_document_16": "Supporting documentation not supplied with the application",
    "flag_document_17": "Supporting documentation not supplied with the application",
    "flag_document_18": "Supporting documentation not supplied with the application",
    "flag_document_19": "Supporting documentation not supplied with the application",
    "flag_document_20": "Supporting documentation not supplied with the application",
    "flag_document_21": "Supporting documentation not supplied with the application",
}

# Used when a feature drives a decline but has no authored phrase. It is
# deliberately vague, because a vague honest reason is better than a specific
# invented one -- and it is counted in the response so the gap is visible
# rather than silent.
FALLBACK_REASON = "Information recorded on the application"

DISCLOSURE_NOTE = (
    "These are the principal factors in the model's assessment, derived from "
    "the SHAP contributions for this applicant. Where a factor is a credit "
    "assessment obtained from an external credit bureau, a compliant notice "
    "must also identify that bureau and state the applicant's right to a free "
    "copy of the report."
)


def reason_for(feature: str) -> tuple[str, bool]:
    """The disclosure phrase for a feature, and whether it was authored."""
    if feature in REASON_CODES:
        return REASON_CODES[feature], True
    return FALLBACK_REASON, False


def _codes_from_contributions(contributions: list[dict]) -> tuple[list[dict], dict]:
    """Rank the adverse contributions and map them to disclosure phrases.

    Only positive SHAP values are adverse: a feature that lowered the applicant's
    risk did not contribute to the denial and has no place in the notice.
    Duplicate phrases collapse -- three external bureau scores are one reason to
    an applicant, not three -- and the collapsed entry keeps the largest
    contribution so ranking is unaffected.
    """
    withheld, unauthored = [], []
    ranked: list[dict] = []
    seen: dict[str, dict] = {}

    for contribution in contributions:
        shap_value = float(contribution["shap_value"])
        if shap_value <= MIN_ADVERSE_SHAP:
            continue

        feature = contribution["feature"]
        if feature in NOT_DISCLOSABLE:
            withheld.append(feature)
            continue

        phrase, authored = reason_for(feature)
        if not authored:
            unauthored.append(feature)

        if phrase in seen:
            seen[phrase]["contributing_features"].append(feature)
            continue

        entry = {
            "reason": phrase,
            "contributing_features": [feature],
            "shap_value": shap_value,
            "feature_value": contribution.get("value"),
            "authored_phrase": authored,
        }
        seen[phrase] = entry
        ranked.append(entry)

    ranked.sort(key=lambda r: r["shap_value"], reverse=True)
    principal = ranked[:MAX_REASONS]
    for rank, entry in enumerate(principal, start=1):
        entry["rank"] = rank

    diagnostics = {
        "withheld_non_disclosable": sorted(set(withheld)),
        "features_without_authored_phrase": sorted(set(unauthored)),
        "adverse_factors_considered": len(ranked),
    }
    return principal, diagnostics


def _notice(explanation: dict) -> dict:
    """Build the adverse action payload from a computed explanation."""
    bands = load_bands()
    probability = float(explanation["default_probability"])
    threshold = float(bands["t_high"])
    declined = probability >= threshold

    payload = {
        "sk_id_curr": explanation.get("sk_id_curr"),
        "default_probability": probability,
        "risk_band": explanation["risk_band"],
        "threshold": threshold,
        "adverse_action_required": declined,
        "reason_code_version": REASON_CODE_VERSION,
    }

    if not declined:
        # An approved applicant receives no adverse action notice. Sending one
        # would be both wrong and alarming, so the reasons are not computed
        # rather than computed and hidden.
        payload["reasons"] = []
        payload["note"] = (
            "No adverse action. The applicant scored below the decision "
            "threshold, so no notice is required and no reasons are given."
        )
        return payload

    reasons, diagnostics = _codes_from_contributions(explanation["contributions"])
    payload["reasons"] = reasons
    payload["diagnostics"] = diagnostics
    payload["note"] = DISCLOSURE_NOTE
    return payload


def notice_for_applicant(sk_id_curr: int) -> dict:
    return _notice(explain_applicant(sk_id_curr))


def notice_for_features(features: dict) -> dict:
    return _notice(explain_features(features))
