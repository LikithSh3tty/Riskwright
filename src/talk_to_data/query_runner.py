"""SQL validation gate and execution.

Every statement the LLM produces passes through validate() before it reaches
Postgres. The gate serves two different purposes and it is worth separating
them, because they justify different design choices.

Security: only a single SELECT, only whitelisted tables, executed as a role
that holds SELECT and nothing else, inside a read-only transaction with a
statement timeout. Four independent layers, so a defect in the parser is not by
itself an incident.

Anti-hallucination: column references are checked against the live schema. A
model with no sanctioned way to say "I cannot answer that" will invent a
plausible column name, and the invented column is the thing that must never
reach the database. When the check fails, the error names the missing column
and suggests real ones, which feeds the single repair retry and, failing that,
becomes a useful refusal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import psycopg2
import sqlglot
from psycopg2 import sql as psql
from sqlglot import exp

from src.utils.config import (
    ALLOWED_TABLES,
    MAX_QUERY_ROWS,
    QUERY_TIMEOUT_SECONDS,
    get_settings,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

DIALECT = "postgres"

_FENCE = re.compile(r"^\s*```(?:sql)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

# Anything that is not a read. Presence of any of these is an immediate reject,
# checked on the parsed tree rather than by string matching, so a comment or a
# string literal containing the word "delete" is not a false positive.
FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Grant,
    exp.Command,  # TRUNCATE, VACUUM, COPY, SET, and friends parse to this
)


class SqlValidationError(Exception):
    """Raised when generated SQL fails the gate. The message is user-facing."""

    def __init__(self, message: str, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion


@dataclass
class ValidatedQuery:
    sql: str
    tables: set[str] = field(default_factory=set)
    limit_applied: bool = False


# --------------------------------------------------------------------------
# Schema, read live from the database
# --------------------------------------------------------------------------

_schema_cache: dict[str, set[str]] | None = None


def live_schema(refresh: bool = False) -> dict[str, set[str]]:
    """Column names per whitelisted table, straight from information_schema.

    Read from the database rather than from a hardcoded list so the validator
    cannot drift away from what actually exists.
    """
    global _schema_cache
    if _schema_cache is not None and not refresh:
        return _schema_cache

    settings = get_settings()
    schema: dict[str, set[str]] = {t: set() for t in ALLOWED_TABLES}
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = ANY(%s)
                """,
                (list(ALLOWED_TABLES),),
            )
            for table, column in cur.fetchall():
                schema[table].add(column)

    _schema_cache = schema
    return schema


def _all_columns(schema: dict[str, set[str]]) -> set[str]:
    return {column for columns in schema.values() for column in columns}


def suggest_columns(unknown: str, schema: dict[str, set[str]], limit: int = 4) -> list[str]:
    """Real columns that resemble the invented one.

    A refusal that says "there is no credit_score" is fine. One that adds
    "there are ext_source_1, ext_source_2, ext_source_3" is useful.
    """
    import difflib

    candidates = sorted(_all_columns(schema))
    close = difflib.get_close_matches(unknown, candidates, n=limit, cutoff=0.5)
    if close:
        return close

    # Fall back to substring overlap on the first token, which catches
    # credit_score against amt_credit where difflib does not.
    token = unknown.split("_")[0]
    return [c for c in candidates if token and token in c][:limit]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    match = _FENCE.match(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def _local_identifiers(tree: exp.Expression) -> set[str]:
    """Names the query defines for itself.

    Select aliases, CTE names, and subquery aliases are legitimate references
    that do not exist in information_schema. Treating them as unknown columns
    would reject correct SQL such as
    `SELECT avg(total_debt) FROM (SELECT sum(x) AS total_debt ...) t`.
    """
    names: set[str] = set()
    for alias in tree.find_all(exp.Alias):
        if alias.alias:
            names.add(alias.alias.lower())
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            names.add(cte.alias.lower())
    for table in tree.find_all(exp.Table):
        if table.alias:
            names.add(table.alias.lower())
    for subquery in tree.find_all(exp.Subquery):
        if subquery.alias:
            names.add(subquery.alias.lower())
    return names


def _referenced_tables(tree: exp.Expression) -> tuple[set[str], set[str]]:
    """Real tables referenced, and CTE names which are not real tables."""
    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
    tables = {
        table.name.lower()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    }
    return tables, cte_names


def _apply_limit(tree: exp.Expression) -> tuple[exp.Expression, bool]:
    """Force a bounded result set.

    A chatbot that returns 300,000 rows into a conversation is a denial of
    service against its own UI.
    """
    if not isinstance(tree, exp.Select):
        # A wrapped query (WITH ..., UNION) gets the limit on the outer node.
        limit = tree.args.get("limit")
        if limit is None:
            return tree.limit(MAX_QUERY_ROWS), True
        return tree, False

    existing = tree.args.get("limit")
    if existing is None:
        return tree.limit(MAX_QUERY_ROWS), True

    try:
        value = int(existing.expression.name)
    except (AttributeError, ValueError):
        return tree.limit(MAX_QUERY_ROWS), True

    if value > MAX_QUERY_ROWS:
        return tree.limit(MAX_QUERY_ROWS), True
    return tree, False


def validate(raw_sql: str) -> ValidatedQuery:
    """Run the gate. Raises SqlValidationError with a usable message."""
    text = strip_fences(raw_sql)
    if not text:
        raise SqlValidationError("No SQL was produced.")

    try:
        statements = sqlglot.parse(text, read=DIALECT)
    except Exception as exc:
        raise SqlValidationError(f"The SQL could not be parsed: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if not statements:
        raise SqlValidationError("No SQL was produced.")
    if len(statements) > 1:
        raise SqlValidationError(
            "Only one statement is allowed, but several were produced."
        )

    tree = statements[0]

    for node_type in FORBIDDEN_NODES:
        if isinstance(tree, node_type) or list(tree.find_all(node_type)):
            raise SqlValidationError(
                "Only read-only SELECT queries are allowed. "
                "This query attempts to modify the database."
            )

    # WITH ... SELECT parses with the CTE attached to the Select, so checking
    # for a Select anywhere at the root covers both shapes.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
        raise SqlValidationError("Only SELECT queries are allowed.")

    tables, _ = _referenced_tables(tree)
    unknown_tables = tables - set(ALLOWED_TABLES)
    if unknown_tables:
        raise SqlValidationError(
            f"Unknown table {', '.join(sorted(unknown_tables))}. "
            f"The available tables are {', '.join(ALLOWED_TABLES)}.",
            suggestion=f"Available tables: {', '.join(ALLOWED_TABLES)}",
        )
    if not tables:
        raise SqlValidationError(
            "The query does not read from any of the available tables."
        )

    schema = live_schema()
    permitted = set()
    for table in tables:
        permitted |= schema.get(table, set())
    permitted |= _local_identifiers(tree)

    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        if name == "*" or not name:
            continue
        if name not in permitted:
            suggestions = suggest_columns(name, schema)
            hint = (
                f" Columns that do exist and look similar: {', '.join(suggestions)}."
                if suggestions
                else ""
            )
            raise SqlValidationError(
                f"There is no column named '{column.name}' in the dataset.{hint}",
                suggestion=", ".join(suggestions) if suggestions else None,
            )

    tree, limit_applied = _apply_limit(tree)
    return ValidatedQuery(
        sql=tree.sql(dialect=DIALECT),
        tables=tables,
        limit_applied=limit_applied,
    )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def execute(validated: ValidatedQuery) -> dict:
    """Run validated SQL as the read-only role.

    The role holds SELECT on three tables and nothing else, the transaction is
    read-only, and the statement times out. Even a validator bypass cannot
    write, and a pathological query cannot hold the connection open.
    """
    settings = get_settings()
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute(
                psql.SQL("SET LOCAL statement_timeout = {}").format(
                    psql.Literal(f"{QUERY_TIMEOUT_SECONDS}s")
                )
            )
            cur.execute(validated.sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()

    return {
        "columns": columns,
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "truncated": validated.limit_applied and len(rows) >= MAX_QUERY_ROWS,
    }


def run(raw_sql: str) -> dict:
    """Validate then execute. Convenience wrapper used by the chat endpoint."""
    validated = validate(raw_sql)
    result = execute(validated)
    result["sql"] = validated.sql
    return result
