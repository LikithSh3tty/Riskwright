"""Validation gate tests.

These are the security tests, and they run without a database by stubbing the
schema. Everything an attacker or a confused model might produce should be
rejected by the parser before Postgres ever sees it, and the read-only role
exists so that a defect here is still not an incident.
"""

from __future__ import annotations

import pytest

from src.talk_to_data import query_runner
from src.talk_to_data.query_runner import SqlValidationError, strip_fences, validate

FAKE_SCHEMA = {
    "application_train": {
        "sk_id_curr", "target", "amt_credit", "amt_income_total", "amt_annuity",
        "code_gender", "flag_own_car", "name_education_type", "days_birth",
        "ext_source_1", "ext_source_2", "ext_source_3", "occupation_type",
    },
    "bureau": {"sk_id_curr", "credit_active", "amt_credit_sum_debt"},
    "previous_application": {"sk_id_curr", "name_contract_status", "amt_credit"},
}


@pytest.fixture(autouse=True)
def stub_schema(monkeypatch):
    monkeypatch.setattr(query_runner, "live_schema", lambda refresh=False: FAKE_SCHEMA)


# --------------------------------------------------------------------------
# Accepting valid SQL
# --------------------------------------------------------------------------

def test_simple_select_passes():
    result = validate("SELECT count(*) FROM application_train")
    assert "application_train" in result.tables
    assert "LIMIT" in result.sql.upper()


def test_join_across_two_whitelisted_tables_passes():
    result = validate(
        "SELECT count(*) FROM application_train a "
        "JOIN bureau b ON b.sk_id_curr = a.sk_id_curr "
        "WHERE b.credit_active = 'Active'"
    )
    assert result.tables == {"application_train", "bureau"}


def test_subquery_alias_is_not_treated_as_an_unknown_column():
    # Rejecting this would be a false positive: total_debt is defined by the
    # query itself, not by information_schema.
    result = validate(
        "SELECT avg(total_debt) FROM ("
        "  SELECT sk_id_curr, sum(amt_credit_sum_debt) AS total_debt"
        "  FROM bureau GROUP BY sk_id_curr"
        ") t"
    )
    assert result.tables == {"bureau"}


def test_cte_is_allowed_when_every_branch_reads():
    result = validate(
        "WITH active AS (SELECT sk_id_curr FROM bureau WHERE credit_active = 'Active') "
        "SELECT count(*) FROM active"
    )
    assert "bureau" in result.tables


# --------------------------------------------------------------------------
# Rejecting writes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM application_train",
        "UPDATE application_train SET target = 0",
        "INSERT INTO application_train (sk_id_curr) VALUES (1)",
        "DROP TABLE application_train",
        "CREATE TABLE evil (x int)",
        "ALTER TABLE application_train ADD COLUMN x int",
        "TRUNCATE application_train",
    ],
)
def test_write_statements_are_rejected(statement):
    with pytest.raises(SqlValidationError):
        validate(statement)


def test_second_statement_is_rejected():
    with pytest.raises(SqlValidationError, match="[Oo]nly one statement"):
        validate("SELECT 1 FROM application_train; DROP TABLE application_train")


def test_write_hidden_in_a_cte_is_rejected():
    with pytest.raises(SqlValidationError):
        validate(
            "WITH x AS (DELETE FROM application_train RETURNING sk_id_curr) "
            "SELECT count(*) FROM x"
        )


def test_the_word_delete_inside_a_string_literal_is_not_a_write():
    # String matching would reject this. Parsing does not, which is the point
    # of checking the tree rather than the text.
    result = validate(
        "SELECT count(*) FROM application_train WHERE occupation_type = 'Delete'"
    )
    assert result.tables == {"application_train"}


# --------------------------------------------------------------------------
# Anti-hallucination
# --------------------------------------------------------------------------

def test_invented_column_is_rejected_and_real_ones_suggested():
    with pytest.raises(SqlValidationError) as excinfo:
        validate("SELECT avg(credit_score) FROM application_train")
    message = str(excinfo.value)
    assert "credit_score" in message
    # The refusal is only useful if it points somewhere real.
    assert "ext_source" in message or "amt_credit" in message


def test_table_outside_the_whitelist_is_rejected():
    with pytest.raises(SqlValidationError, match="Unknown table"):
        validate("SELECT * FROM pg_shadow")


def test_query_reading_no_known_table_is_rejected():
    with pytest.raises(SqlValidationError):
        validate("SELECT 1")


# --------------------------------------------------------------------------
# Result bounding
# --------------------------------------------------------------------------

def test_missing_limit_is_added():
    result = validate("SELECT sk_id_curr FROM application_train")
    assert result.limit_applied
    assert "LIMIT 200" in result.sql.upper()


def test_oversized_limit_is_clamped():
    result = validate("SELECT sk_id_curr FROM application_train LIMIT 999999")
    assert result.limit_applied
    assert "LIMIT 200" in result.sql.upper()


def test_small_limit_is_left_alone():
    result = validate("SELECT sk_id_curr FROM application_train LIMIT 5")
    assert not result.limit_applied
    assert "LIMIT 5" in result.sql.upper()


# --------------------------------------------------------------------------
# Input shapes
# --------------------------------------------------------------------------

def test_markdown_fences_are_stripped():
    assert strip_fences("```sql\nSELECT 1\n```") == "SELECT 1"
    assert strip_fences("```\nSELECT 1\n```") == "SELECT 1"
    assert strip_fences("SELECT 1") == "SELECT 1"


def test_unparseable_text_is_rejected_cleanly():
    with pytest.raises(SqlValidationError, match="could not be parsed|SELECT"):
        validate("this is not sql at all")


def test_empty_input_is_rejected():
    with pytest.raises(SqlValidationError, match="No SQL"):
        validate("")
