"""The compact schema block sent to the model.

This is the prompt's largest fixed cost and its main defence against a wrong
column, so both the size and the content are worth pinning. Three properties
matter and all three can regress silently:

  - a column that is not in the database is never advertised, because the model
    will use anything it is shown;
  - the exact values of low-cardinality categoricals are inline, because
    without them the model writes 'Higher Ed', the query runs, and zero rows
    come back with no error anywhere;
  - the block stays inside its token budget, because the whole point of
    curating 61 columns out of 176 was the budget.

The database is mocked: `_column_types` and `_enum_values` are the only two
functions that touch Postgres. The column descriptions are real -- they come
from the data dictionary shipped in configs/ -- so the description assertions
below test the actual file the container ships.
"""

from __future__ import annotations

import pytest

from src.talk_to_data import schema_context as sc

# The module docstring claims "under 2,000" against roughly 6,000 for a full
# dump; the README quotes the measured 1,656. The ceiling is what a test should
# hold, since the exact figure moves with the data dictionary.
TOKEN_BUDGET = 2000

FAKE_ENUMS = {
    ("application_train", "code_gender"): ("F", "M"),
    ("application_train", "name_education_type"): (
        "Academic degree", "Higher education", "Incomplete higher",
        "Lower secondary", "Secondary / secondary special",
    ),
    ("application_train", "name_family_status"): (
        "Civil marriage", "Married", "Separated", "Single / not married", "Widow",
    ),
    ("bureau", "credit_active"): ("Active", "Bad debt", "Closed", "Sold"),
    ("previous_application", "name_contract_status"): (
        "Approved", "Canceled", "Refused", "Unused offer",
    ),
}


def all_columns_typed() -> dict[tuple[str, str], str]:
    """Pretend every exposed column exists, with a plausible type."""
    types = {}
    for table, columns in sc.EXPOSED_COLUMNS.items():
        for column in columns:
            if column.startswith(("amt_", "ext_source", "region_population")):
                kind = "float"
            elif column.startswith(("name_", "code_", "flag_own", "occupation_",
                                    "organization_", "credit_active", "credit_type")):
                kind = "text"
            else:
                kind = "int"
            types[(table, column)] = kind
    return types


@pytest.fixture
def schema(monkeypatch):
    """Build the schema block with the two database calls stubbed out."""
    monkeypatch.setattr(sc, "_column_types", lambda: all_columns_typed())
    monkeypatch.setattr(sc, "_enum_values", lambda: dict(FAKE_ENUMS))
    sc.build_schema_context.cache_clear()
    try:
        yield sc.build_schema_context()
    finally:
        sc.build_schema_context.cache_clear()


# --------------------------------------------------------------------------
# Tables and columns
# --------------------------------------------------------------------------

def test_all_three_loaded_tables_are_described(schema):
    for table in ("application_train", "bureau", "previous_application"):
        assert f"TABLE {table}" in schema


def test_no_table_outside_the_loaded_three_is_advertised(schema):
    """The validator whitelists three tables; offering a fourth guarantees a refusal."""
    declared = {line.split()[1] for line in schema.splitlines()
                if line.startswith("TABLE ")}
    assert declared == set(sc.EXPOSED_COLUMNS)


def test_every_exposed_column_reaches_the_prompt(schema):
    for table, columns in sc.EXPOSED_COLUMNS.items():
        for column in columns:
            assert f"  {column} " in schema, f"{table}.{column} missing from the schema"


def test_a_column_absent_from_the_database_is_never_advertised(monkeypatch):
    """The model uses whatever it is shown, so an unbacked column is a trap."""
    types = all_columns_typed()
    del types[("application_train", "occupation_type")]
    monkeypatch.setattr(sc, "_column_types", lambda: types)
    monkeypatch.setattr(sc, "_enum_values", lambda: dict(FAKE_ENUMS))

    sc.build_schema_context.cache_clear()
    try:
        built = sc.build_schema_context()
    finally:
        sc.build_schema_context.cache_clear()

    assert "  occupation_type " not in built
    assert "  organization_type " in built     # its neighbour still is


def test_the_curated_set_is_a_fraction_of_the_full_schema(schema):
    """122 + 17 + 37 columns exist. Sending all of them is the thing being avoided."""
    exposed = sum(len(c) for c in sc.EXPOSED_COLUMNS.values())
    assert exposed == 61
    assert exposed < 176 / 2


# --------------------------------------------------------------------------
# Descriptions, from the dataset's own data dictionary
# --------------------------------------------------------------------------

def test_descriptions_are_attached_where_the_dictionary_has_one(schema):
    """Real definitions beat guessed semantics."""
    line = next(l for l in schema.splitlines() if l.startswith("  days_birth "))
    assert "--" in line
    assert "age in days" in line.lower()


def test_descriptions_come_from_the_shipped_data_dictionary():
    descriptions = sc._descriptions()

    assert descriptions, "configs/columns_description.csv did not load"
    assert "days_birth" in descriptions
    assert descriptions["amt_credit"] == "Credit amount of the loan"


def test_long_descriptions_are_truncated_rather_than_sent_whole():
    for text in sc._descriptions().values():
        assert len(text) <= sc.MAX_DESCRIPTION_CHARS + 3   # + the ellipsis


def test_a_column_with_no_dictionary_entry_still_appears(schema):
    """Missing prose must not silently drop the column itself."""
    descriptions = sc._descriptions()
    undocumented = [
        column for columns in sc.EXPOSED_COLUMNS.values()
        for column in columns if column not in descriptions
    ]
    for column in undocumented:
        assert f"  {column} " in schema


# --------------------------------------------------------------------------
# Categorical values
# --------------------------------------------------------------------------

def test_exact_categorical_values_are_sent_inline(schema):
    line = next(l for l in schema.splitlines()
                if l.startswith("  name_education_type "))
    assert "[values:" in line
    assert "Secondary / secondary special" in line


def test_a_column_with_too_many_distinct_values_is_not_enumerated(schema):
    """occupation_type has ~18 values; listing them all costs more than it saves."""
    line = next(l for l in schema.splitlines() if l.startswith("  occupation_type "))
    assert "[values:" not in line


def test_semantic_notes_carry_the_facts_the_model_gets_wrong(schema):
    assert "1 means the applicant defaulted" in schema
    assert "NEGATIVE integers" in schema
    # The note wraps across lines, so normalise whitespace before matching.
    assert "no column called credit_score" in " ".join(schema.split())
    assert "avg(target::float)" in schema


# --------------------------------------------------------------------------
# Token budget
# --------------------------------------------------------------------------

def test_schema_block_stays_within_the_documented_budget(schema):
    tokens = sc.approximate_tokens(schema)

    assert tokens < TOKEN_BUDGET, (
        f"schema grew to ~{tokens} tokens against a documented budget of "
        f"{TOKEN_BUDGET}; the README's token accounting needs re-measuring"
    )
    # A block that collapsed to nothing would also pass a ceiling check.
    assert tokens > 800


def test_approximate_tokens_is_the_documented_four_chars_per_token():
    assert sc.approximate_tokens("a" * 400) == 100
    assert sc.approximate_tokens("") == 0
