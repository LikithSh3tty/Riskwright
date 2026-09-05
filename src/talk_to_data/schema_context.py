"""Compact schema description for the LLM.

Token optimization lives here, and it is the difference between a prompt that
costs 6,000 tokens per turn and one that costs under 2,000.

Three decisions:

1. Send a curated subset of columns, not all 176. application_train alone has
   122, most of which are normalised statistical aggregates that no business
   question ever asks about. Sending them all costs tokens and, worse, gives
   the model more chances to pick a column that technically exists but does not
   mean what the question asked.

2. Attach one short description per column, taken from the official Home Credit
   data dictionary shipped with the dataset. Real definitions beat guessed
   semantics, and they are what stops the model reading days_birth as a date.

3. Send the distinct values of low-cardinality categorical columns. This is the
   single highest-value item here. Without it the model writes
   `name_education_type = 'Higher Ed'` and gets zero rows back with no error.
   With it, it writes the exact string the database holds.

The whole block is stable across a conversation, so it sits at the front of the
prompt behind a cache breakpoint.
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache

import psycopg2

from src.utils.config import get_settings
from src.utils.docker_utils import config_file
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns worth exposing, per table. Chosen because a business user might
# plausibly ask about them.
EXPOSED_COLUMNS: dict[str, tuple[str, ...]] = {
    "application_train": (
        "sk_id_curr", "target", "name_contract_type", "code_gender",
        "flag_own_car", "flag_own_realty", "cnt_children", "cnt_fam_members",
        "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
        "name_income_type", "name_education_type", "name_family_status",
        "name_housing_type", "occupation_type", "organization_type",
        "days_birth", "days_employed", "days_registration", "days_id_publish",
        "days_last_phone_change", "ext_source_1", "ext_source_2", "ext_source_3",
        "region_rating_client", "region_population_relative",
        "reg_city_not_work_city", "amt_req_credit_bureau_year",
        "obs_30_cnt_social_circle", "def_30_cnt_social_circle",
        "name_type_suite", "flag_phone", "flag_email",
    ),
    "bureau": (
        "sk_id_curr", "sk_id_bureau", "credit_active", "credit_type",
        "days_credit", "credit_day_overdue", "days_credit_enddate",
        "amt_credit_sum", "amt_credit_sum_debt", "amt_credit_sum_overdue",
        "amt_credit_max_overdue", "cnt_credit_prolong",
    ),
    "previous_application": (
        "sk_id_prev", "sk_id_curr", "name_contract_type", "name_contract_status",
        "amt_application", "amt_credit", "amt_annuity", "amt_down_payment",
        "days_decision", "code_reject_reason", "name_cash_loan_purpose",
        "name_client_type", "name_yield_group", "cnt_payment",
    ),
}

# Categorical columns whose exact values the model must not guess. Anything
# with more distinct values than this is summarised rather than listed.
ENUMERATED_COLUMNS: dict[str, tuple[str, ...]] = {
    "application_train": (
        "name_contract_type", "code_gender", "flag_own_car", "flag_own_realty",
        "name_income_type", "name_education_type", "name_family_status",
        "name_housing_type",
    ),
    "bureau": ("credit_active", "credit_type"),
    "previous_application": ("name_contract_status", "name_client_type"),
}

MAX_ENUM_VALUES = 12
MAX_DESCRIPTION_CHARS = 90

# Facts the model cannot infer from column names and consistently gets wrong.
SEMANTIC_NOTES = """
Important semantics:
- target: 1 means the applicant defaulted, 0 means they repaid. Only
  application_train has it. The overall default rate is about 8%.
- All days_* columns are NEGATIVE integers counting backwards from the
  application date. days_birth = -10000 means the applicant is about 27 years
  old. To filter "under 30", use days_birth > -30 * 365.25.
- There are no calendar dates anywhere. No question about a specific year,
  month, or date range can be answered.
- ext_source_1, ext_source_2, ext_source_3 are normalised scores from external
  credit bureaus, between 0 and 1. Higher is safer. There is no column called
  credit_score.
- Money columns are in an unspecified currency and are not comparable to real
  world amounts.
- bureau and previous_application hold MANY rows per applicant. Join on
  sk_id_curr and aggregate, or you will multiply-count applicants.
- Rates such as default rate should be computed as avg(target::float).
""".strip()


@lru_cache
def _descriptions() -> dict[str, str]:
    """Column descriptions from the dataset's own data dictionary."""
    path = config_file("columns_description.csv")
    if not path.exists():
        log.warning("%s missing; schema will omit descriptions", path)
        return {}

    # The Kaggle file is not UTF-8. Reading it with errors="replace" silently
    # injects U+FFFD into the prompt, so the encoding is resolved properly.
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte sequence
        text = raw.decode("latin-1", errors="ignore")

    out: dict[str, str] = {}
    with io.StringIO(text, newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Row") or "").strip().lower()
            text = (row.get("Description") or "").strip()
            if not name or not text:
                continue
            text = " ".join(text.split())
            if len(text) > MAX_DESCRIPTION_CHARS:
                text = text[:MAX_DESCRIPTION_CHARS].rstrip() + "..."
            # The file repeats a column across tables; first definition wins.
            out.setdefault(name, text)
    return out


@lru_cache
def _column_types() -> dict[tuple[str, str], str]:
    settings = get_settings()
    types: dict[tuple[str, str], str] = {}
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s)
                """,
                (list(EXPOSED_COLUMNS),),
            )
            for table, column, data_type in cur.fetchall():
                short = {
                    "bigint": "int",
                    "double precision": "float",
                    "text": "text",
                }.get(data_type, data_type)
                types[(table, column)] = short
    return types


@lru_cache
def _enum_values() -> dict[tuple[str, str], tuple[str, ...]]:
    """Distinct values for the enumerated categorical columns.

    Read once at first use. Without these the model invents plausible category
    strings, the query runs, and zero rows come back with no error anywhere.
    """
    settings = get_settings()
    values: dict[tuple[str, str], tuple[str, ...]] = {}
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        with conn.cursor() as cur:
            for table, columns in ENUMERATED_COLUMNS.items():
                for column in columns:
                    cur.execute(
                        f"SELECT DISTINCT {column} FROM {table} "
                        f"WHERE {column} IS NOT NULL LIMIT {MAX_ENUM_VALUES + 1}"
                    )
                    found = sorted(str(r[0]) for r in cur.fetchall())
                    if len(found) <= MAX_ENUM_VALUES:
                        values[(table, column)] = tuple(found)
    return values


@lru_cache
def build_schema_context() -> str:
    """Render the compact schema block sent to the model."""
    descriptions = _descriptions()
    types = _column_types()
    enums = _enum_values()

    lines: list[str] = [
        "PostgreSQL database. Three tables. All identifiers are lowercase.",
        "",
    ]

    for table, columns in EXPOSED_COLUMNS.items():
        lines.append(f"TABLE {table}")
        for column in columns:
            data_type = types.get((table, column))
            if data_type is None:
                continue  # column not present; never advertise it
            parts = [f"  {column} {data_type}"]
            description = descriptions.get(column)
            if description:
                parts.append(f" -- {description}")
            values = enums.get((table, column))
            if values:
                parts.append(f" [values: {', '.join(values)}]")
            lines.append("".join(parts))
        lines.append("")

    lines.append(SEMANTIC_NOTES)
    return "\n".join(lines)


def approximate_tokens(text: str) -> int:
    """Rough size check, used in logging and in the README's token accounting."""
    return len(text) // 4
