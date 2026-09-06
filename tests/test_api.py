"""API contract tests.

These assert the shape of what the endpoints return, not a reimplementation of
what src/ computes. The distinction matters: a test that recomputes the answer
passes when both copies of the logic are wrong together, which is the failure
mode it was supposed to catch.

No database. Postgres reads are mocked at the one seam where they happen --
`fetch_applicant` and `read_table` in the loader, and the aggregate functions in
`src.data.eda`. Everything downstream of that is the real thing: /predict and
/explain run the committed LightGBM artifact, the real preprocessor, and the
real SHAP explainer against a synthetic applicant row. That is deliberate. The
protected-attribute assertion below is only worth having if the model that
produced the contributions is the model that ships.

Model loading is cached, so the first test that scores pays about a second and
the rest are free.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data.preprocessor import PROTECTED_ATTRIBUTES
from src.utils.docker_utils import models_dir

BANDS = ("Low", "Medium", "High")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return json.loads((models_dir() / "threshold.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def feature_spec() -> dict:
    return json.loads((models_dir() / "feature_spec.json").read_text(encoding="utf-8"))


def applicant_row(sk_id_curr: int = 100002) -> pd.DataFrame:
    """One synthetic applicant.

    Deliberately partial. `prepare_features` fills anything absent with NaN and
    LightGBM takes NaN natively, so this exercises the real serving path
    without needing all 122 columns. The protected columns are present on
    purpose: they must survive the round trip into the preprocessor and be
    dropped there, which is what the exclusion claim actually rests on.
    """
    return pd.DataFrame([{
        "sk_id_curr": sk_id_curr,
        "target": 0,
        "days_birth": -14000,
        "days_employed": -1200,
        "amt_credit": 500000.0,
        "amt_income_total": 150000.0,
        "amt_annuity": 25000.0,
        "amt_goods_price": 450000.0,
        "ext_source_1": 0.5,
        "ext_source_2": 0.6,
        "ext_source_3": 0.4,
        "code_gender": "F",
        "name_family_status": "Married",
        "cnt_children": 0,
        "cnt_fam_members": 2.0,
        "name_education_type": "Higher education",
        "name_income_type": "Working",
        "name_contract_type": "Cash loans",
        "occupation_type": "Core staff",
        "organization_type": "Business Entity Type 3",
    }])


@pytest.fixture
def scored_applicant(monkeypatch):
    """Make the loader return a synthetic row instead of querying Postgres."""
    def fake_fetch(sk_id_curr: int) -> pd.DataFrame:
        if int(sk_id_curr) == 100002:
            return applicant_row(100002)
        return pd.DataFrame()          # the "no such applicant" case

    monkeypatch.setattr("src.ml.predict.fetch_applicant", fake_fetch)
    monkeypatch.setattr("src.ml.explain.fetch_applicant", fake_fetch)
    return fake_fetch


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------

def test_health_reports_each_dependency_separately(client, monkeypatch):
    monkeypatch.setattr("app.routers.health._database_ready", lambda: (True, None))
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert set(body) == {
        "status", "database", "model_loaded", "llm_configured", "llm_model",
    }
    assert body["database"] == {"connected": True, "error": None}
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["llm_configured"], bool)


def test_health_degrades_without_the_database_and_names_the_reason(client, monkeypatch):
    """A bare {"status": "error"} tells nobody which dependency is missing."""
    monkeypatch.setattr(
        "app.routers.health._database_ready",
        lambda: (False, "could not connect to server: Connection refused"),
    )
    response = client.get("/health")
    body = response.json()

    # Still 200: the service is up and answering, it is the dependency that is
    # not. A health endpoint that 500s cannot report which piece is down.
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["database"]["connected"] is False
    assert "Connection refused" in body["database"]["error"]


# --------------------------------------------------------------------------
# /applicants
# --------------------------------------------------------------------------

@pytest.fixture
def fake_applicant_table(monkeypatch):
    """Stand in for Postgres, honouring limit and order_by as the real one does."""
    def fake_read_table(table, columns=None, limit=None, order_by=None):
        assert table == "application_train"
        assert order_by == "sk_id_curr", "paging is only reproducible when ordered"
        ids = list(range(100002, 100002 + 400))
        frame = pd.DataFrame({
            "sk_id_curr": ids,
            "amt_credit": [100000.0 + i for i in range(len(ids))],
            "amt_income_total": [50000.0 + i for i in range(len(ids))],
            "days_birth": [-14000 - i for i in range(len(ids))],
            "code_gender": ["F" if i % 2 else "M" for i in range(len(ids))],
        })
        return frame.head(limit) if limit else frame

    monkeypatch.setattr("app.routers.predict.read_table", fake_read_table)


def test_applicants_are_ordered_by_id(client, fake_applicant_table):
    body = client.get("/applicants", params={"limit": 25}).json()
    ids = [a["sk_id_curr"] for a in body["applicants"]]

    assert body["count"] == 25 == len(ids)
    assert ids == sorted(ids)


def test_applicant_pages_are_contiguous_and_do_not_overlap(client, fake_applicant_table):
    first = client.get("/applicants", params={"limit": 10, "offset": 0}).json()
    second = client.get("/applicants", params={"limit": 10, "offset": 10}).json()

    a = [x["sk_id_curr"] for x in first["applicants"]]
    b = [x["sk_id_curr"] for x in second["applicants"]]

    assert len(a) == len(b) == 10
    assert not set(a) & set(b)
    assert b[0] == a[-1] + 1, "a gap here means rows are unreachable by paging"


def test_applicants_derive_age_and_never_ship_the_raw_row(client, fake_applicant_table):
    row = client.get("/applicants", params={"limit": 1}).json()["applicants"][0]

    assert set(row) == {
        "sk_id_curr", "amt_credit", "amt_income_total", "code_gender", "age_years",
    }
    assert "days_birth" not in row
    assert 0 < row["age_years"] < 120


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
def test_applicants_rejects_out_of_range_paging(client, fake_applicant_table, params):
    assert client.get("/applicants", params=params).status_code == 422


# --------------------------------------------------------------------------
# /predict
# --------------------------------------------------------------------------

def test_predict_response_shape(client, scored_applicant):
    body = client.post("/predict", json={"sk_id_curr": 100002}).json()

    assert set(body) >= {
        "sk_id_curr", "default_probability", "risk_band", "recommended_action",
        "thresholds", "model",
    }
    assert body["sk_id_curr"] == 100002
    assert body["model"] == "lightgbm"
    assert isinstance(body["recommended_action"], str) and body["recommended_action"]


def test_predicted_probability_is_a_probability(client, scored_applicant):
    p = client.post("/predict", json={"sk_id_curr": 100002}).json()["default_probability"]

    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_risk_band_is_one_of_the_three(client, scored_applicant):
    band = client.post("/predict", json={"sk_id_curr": 100002}).json()["risk_band"]
    assert band in BANDS


def test_band_agrees_with_the_thresholds_on_disk(client, scored_applicant, thresholds):
    """The band shown must follow from the thresholds the artifact publishes.

    If banding and thresholds ever disagree, the UI's band bar draws boundaries
    that do not match the band it labels, and nothing errors.
    """
    body = client.post("/predict", json={"sk_id_curr": 100002}).json()
    p, band = body["default_probability"], body["risk_band"]

    t_low, t_high = thresholds["t_low"], thresholds["t_high"]
    expected = "High" if p >= t_high else "Medium" if p >= t_low else "Low"

    assert band == expected
    assert body["thresholds"] == {"t_low": t_low, "t_high": t_high}


def test_unknown_applicant_is_404_not_500(client, scored_applicant):
    response = client.post("/predict", json={"sk_id_curr": 999999999})

    assert response.status_code == 404
    assert "999999999" in response.json()["detail"]


def test_predict_requires_exactly_one_of_id_or_features(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post(
        "/predict", json={"sk_id_curr": 100002, "features": {"amt_credit": 1.0}}
    ).status_code == 422


def test_what_if_scoring_takes_a_partial_applicant(client):
    """Unspecified fields stay missing rather than being guessed at."""
    body = client.post("/predict", json={"features": {"amt_credit": 500000.0}}).json()

    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in BANDS
    assert body["sk_id_curr"] is None


# --------------------------------------------------------------------------
# /explain
# --------------------------------------------------------------------------

def test_explain_contribution_contract(client, scored_applicant):
    body = client.post("/explain", json={"sk_id_curr": 100002}).json()

    assert "base_value" in body and isinstance(body["base_value"], float)
    assert body["risk_band"] in BANDS
    assert body["contributions"], "an explanation with no contributions explains nothing"

    for c in body["contributions"]:
        assert {"feature", "value", "shap_value"} <= set(c)
        assert isinstance(c["feature"], str) and c["feature"]
        assert isinstance(c["shap_value"], float)
        assert c["direction"] in ("raises risk", "lowers risk")


def test_contributions_are_sorted_by_absolute_influence(client, scored_applicant):
    body = client.post("/explain", json={"sk_id_curr": 100002}).json()
    magnitudes = [abs(c["shap_value"]) for c in body["contributions"]]

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_no_protected_attribute_appears_in_any_contribution(client, scored_applicant):
    """The fair-lending claim, expressed as a regression guard.

    The applicant row fed in *does* carry code_gender, name_family_status,
    cnt_children and cnt_fam_members. If the preprocessor ever stops dropping
    them, they arrive here and this fails. Nothing else in the suite would
    notice: the model would keep scoring, the API would keep answering, and the
    only symptom would be a protected attribute in a credit decision.
    """
    body = client.post("/explain", json={"sk_id_curr": 100002}).json()
    named = {c["feature"] for c in body["contributions"]}

    assert not named & set(PROTECTED_ATTRIBUTES)
    assert "days_birth" not in named, "dropped as redundant after derivation"


def test_feature_matrix_excludes_every_protected_attribute(feature_spec):
    """The same claim at the source, not just in the top-K slice of one row."""
    assert not set(feature_spec["feature_names"]) & set(PROTECTED_ATTRIBUTES)
    assert "days_birth" not in feature_spec["feature_names"]
    assert len(feature_spec["feature_names"]) == 120


def test_the_known_age_proxy_is_still_present_and_still_documented(feature_spec):
    """employed_life_ratio = days_employed / days_birth carries age into the model.

    This is the leak documented in the README's fair lending section. The
    assertion is deliberately the opposite of what you might expect: it pins
    the feature as *present*, so that removing it cannot happen quietly. Taking
    it out changes the feature set, requires a retrain, and invalidates every
    metric the README and the deck report -- so the removal and the
    documentation have to move together, and this test is what forces that.
    """
    assert "employed_life_ratio" in feature_spec["feature_names"]


def test_explain_unknown_applicant_is_404(client, scored_applicant):
    assert client.post("/explain", json={"sk_id_curr": 999999999}).status_code == 404


def test_global_importance_shape(client):
    body = client.get("/explain/global").json()

    # Global importance is exhaustive since v2.3: every applicant, not a sample.
    assert body["sampled_rows"] == body["total_rows"] == 307511
    assert len(body["features"]) == 120
    assert {"feature", "mean_abs_shap"} <= set(body["features"][0])


# --------------------------------------------------------------------------
# /eda
# --------------------------------------------------------------------------

def test_eda_summary_shape(client, monkeypatch):
    payload = {
        "table": "application_train", "rows": 307511, "columns": 122,
        "defaulted": 24825, "repaid": 282686, "default_rate": 0.0807288,
        "imbalance_ratio": 11.387, "columns_with_missing": 67,
        "missing_top": [{"column": "own_car_age", "missing": 202929}],
    }
    monkeypatch.setattr("src.data.eda.dataset_summary", lambda: payload)
    body = client.get("/eda/summary").json()

    assert body == payload
    assert {"rows", "columns", "defaulted", "repaid", "default_rate"} <= set(body)


def test_eda_insights_shape(client, monkeypatch):
    payload = {
        "base_default_rate": 0.0807288,
        "insights": [
            {"title": "A sentinel value hides in the employment column",
             "detail": "18.0% of applicants...", "so_what": "kept as a flag"}
            for _ in range(5)
        ],
    }
    monkeypatch.setattr("src.data.eda.business_insights", lambda: payload)
    body = client.get("/eda/insights").json()

    assert len(body["insights"]) == 5, "the assignment asks for five"
    assert {"title", "detail", "so_what"} <= set(body["insights"][0])


def test_eda_columns_lists_what_the_explorers_accept(client):
    body = client.get("/eda/columns").json()

    assert body["numeric"] and body["categorical"]
    assert body["numeric"] == sorted(body["numeric"])
    assert body["categorical"] == sorted(body["categorical"])


def test_eda_distribution_rejects_an_unknown_column(client, monkeypatch):
    def boom(column, bins):
        raise KeyError(f"{column} is not an exposed numeric column")

    monkeypatch.setattr("src.data.eda.distribution", boom)
    assert client.get("/eda/distribution", params={"column": "nope"}).status_code == 400


# --------------------------------------------------------------------------
# /chat
# --------------------------------------------------------------------------

class _Settings:
    def __init__(self, configured: bool = True):
        self.llm_configured = configured
        self.anthropic_model = "claude-haiku-4-5"


@pytest.fixture
def llm_configured(monkeypatch):
    monkeypatch.setattr("app.routers.chat.get_settings", lambda: _Settings(True))


def test_rejected_sql_comes_back_as_a_refusal_not_an_error(client, monkeypatch,
                                                           llm_configured):
    """A question the validator rejects must still be a 200 with a reason.

    Returning 500 here would be the difference between "the assistant explained
    why it cannot answer" and "the assistant is broken", and the UI renders
    those very differently.
    """
    refusal = {
        "question": "What is the applicant's credit score?",
        "action": "refuse",
        "answer": None,
        "sql": "SELECT credit_score FROM application_train",
        "reason": "Unknown column: credit_score.",
        "suggestion": "Try ext_source_1, ext_source_2 or ext_source_3 instead.",
        "columns": [], "rows": [], "row_count": 0, "truncated": False,
        "usage": {}, "prompt_version": "v4", "latency_ms": 12,
    }
    monkeypatch.setattr("app.routers.chat.answer", lambda *a, **k: dict(refusal))

    response = client.post("/chat", json={
        "question": "What is the applicant's credit score?",
        "session_id": "test-refusal",
    })
    body = response.json()

    assert response.status_code == 200
    assert body["action"] == "refuse"
    assert body["reason"]
    assert body["suggestion"], "a refusal without a suggestion is a dead end"
    assert body["rows"] == [] and body["row_count"] == 0


def test_chat_echoes_the_session_id_and_reports_memory_depth(client, monkeypatch,
                                                             llm_configured):
    """The client holds an id, never the history. The response has to confirm both."""
    monkeypatch.setattr("app.routers.chat.answer", lambda *a, **k: {
        "question": "q", "action": "sql", "answer": "8.07%", "sql": "SELECT 1",
        "reason": None, "suggestion": None, "columns": ["default_rate"],
        "rows": [[0.0807]], "row_count": 1, "truncated": False, "usage": {},
        "prompt_version": "v4", "latency_ms": 30,
    })

    body = client.post("/chat", json={
        "question": "What is the overall default rate?",
        "session_id": "test-session-echo",
    }).json()

    assert body["session_id"] == "test-session-echo"
    assert isinstance(body["turns_remembered"], int)


def test_chat_session_id_defaults_when_the_client_omits_it(client, monkeypatch,
                                                           llm_configured):
    monkeypatch.setattr("app.routers.chat.answer", lambda *a, **k: {
        "question": "q", "action": "sql", "answer": "a", "sql": "SELECT 1",
        "reason": None, "suggestion": None, "columns": [], "rows": [],
        "row_count": 0, "truncated": False, "usage": {},
        "prompt_version": "v4", "latency_ms": 1,
    })
    body = client.post("/chat", json={"question": "anything"}).json()

    assert body["session_id"] == "default"


def test_chat_rejects_an_empty_question(client, llm_configured):
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_chat_rejects_an_unknown_prompt_version(client, llm_configured):
    response = client.post(
        "/chat", json={"question": "hello", "prompt_version": "v99"}
    )
    assert response.status_code == 400
    assert "Available" in response.json()["detail"]


def test_chat_is_503_when_no_key_is_configured(client, monkeypatch):
    """Every other module works without a key, and the message has to say so."""
    monkeypatch.setattr("app.routers.chat.get_settings", lambda: _Settings(False))
    response = client.post("/chat", json={"question": "hello"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_clearing_a_session_is_idempotent(client):
    first = client.delete("/chat/never-used-session").json()

    assert first["session_id"] == "never-used-session"
    assert isinstance(first["cleared"], bool)


# --------------------------------------------------------------------------
# /rules and /model/metrics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["faithful", "policy"])
def test_rules_shape_for_both_variants(client, variant):
    body = client.get("/rules", params={"variant": variant}).json()

    assert body["variant"] == variant
    assert body["max_depth"] == 3
    assert 0.0 < body["surrogate_fidelity"] <= 1.0
    assert body["rules"], "a rule set with no rules is not a rule set"

    for rule in body["rules"]:
        assert {
            "rule_id", "conditions", "readable", "support_n", "support_pct",
            "default_rate", "lift",
        } <= set(rule)
        assert rule["conditions"] and rule["readable"]
        assert 0.0 <= rule["default_rate"] <= 1.0
        assert rule["lift"] > 0
        assert 0.0 < rule["support_pct"] <= 100.0

    # The leaves partition the population, so the shares have to add up.
    assert sum(r["support_pct"] for r in body["rules"]) == pytest.approx(100.0, abs=0.5)


def test_rules_rejects_an_unknown_variant(client):
    assert client.get("/rules", params={"variant": "made-up"}).status_code == 422


def test_metrics_reports_the_shipped_feature_count(client):
    body = client.get("/model/metrics").json()

    assert body["rows"] == 307511
    assert body["features"] == 120, "the shipped model excludes five protected columns"
    assert set(body["models"]) == {"lightgbm", "logistic"}
    assert body["n_folds"] == 5


def test_metrics_carries_the_bands_so_the_ui_reads_one_source(client, thresholds):
    body = client.get("/model/metrics").json()

    assert body["bands"]["t_high"] == thresholds["t_high"]
    assert body["bands"]["t_low"] == thresholds["t_low"]


def test_root_points_at_the_docs(client):
    body = client.get("/").json()
    assert body == {"service": "riskwright", "docs": "/docs", "health": "/health"}
