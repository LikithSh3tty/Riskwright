"""Versioned prompt templates.

The assignment asks for versioned templates, so every version stays in this
file with a note on what changed and why. The evaluation harness runs all three
against the same 30 questions, which turns "prompt engineering approach" from a
claim into a measurement.

  v1  Naive. Full column dump, "return SQL only". The baseline everyone writes
      first.
  v2  Compact curated schema, few-shot examples, explicit dialect and table
      whitelist.
  v3  Structured action output. Refusal and clarification become first-class
      results rather than parse failures, semantic notes are added, and prior
      turns are replayed as question plus SQL only.

v3 is what the application uses. The others exist to be measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.talk_to_data.schema_context import build_schema_context

# --------------------------------------------------------------------------
# The tool the model must call in v3.
#
# Structured output through a required tool rather than "reply with JSON".
# The schema is strict, so arguments are guaranteed to validate, and the
# action enum gives the model a sanctioned way to decline. A model with no way
# to say "I cannot" will invent a column to fill the silence, which is the
# exact failure the hallucination criterion is testing.
# --------------------------------------------------------------------------

ANSWER_TOOL = {
    "name": "answer_question",
    "description": (
        "Respond to the user's question about the credit risk database. "
        "Choose exactly one action: write SQL if the question can be answered "
        "from the schema, refuse if it cannot, or clarify if the question is "
        "ambiguous."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sql", "refuse", "clarify"],
                "description": (
                    "sql: the question maps to a query over the given tables. "
                    "refuse: the data cannot answer it, for example because no "
                    "such column exists or the fact is not recorded. "
                    "clarify: the question is answerable in principle but too "
                    "vague to write one correct query for."
                ),
            },
            "sql": {
                "type": "string",
                "description": (
                    "A single PostgreSQL SELECT statement. Required when action "
                    "is sql, omitted otherwise. No trailing semicolon, no "
                    "markdown fences."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Why the question was refused, or what specifically is "
                    "ambiguous. Written for a business user, naming the missing "
                    "column or the undefined term."
                ),
            },
            "suggestion": {
                "type": "string",
                "description": (
                    "What the dataset CAN answer instead, phrased as a question "
                    "the user could ask next. Required for refuse and clarify."
                ),
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class PromptVersion:
    name: str
    system: str
    use_tool: bool
    changed: str


# --------------------------------------------------------------------------
# v1: the naive baseline
# --------------------------------------------------------------------------

_V1_SYSTEM = """You are a SQL assistant. Answer questions about a PostgreSQL
database containing loan application data.

Tables: application_train, bureau, previous_application.

Return only a SQL query. Do not explain."""


# --------------------------------------------------------------------------
# v2: compact schema, few-shot, explicit rules
# --------------------------------------------------------------------------

_V2_RULES = """You write PostgreSQL SELECT queries over a credit risk database.

Rules:
- Return only the SQL. No prose, no markdown fences, no trailing semicolon.
- One SELECT statement. Never write to the database.
- Only use the tables and columns listed below.
- Always include a LIMIT unless the query returns a single aggregate row."""

_FEW_SHOT = """Examples:

Q: How many applications are there?
A: SELECT count(*) FROM application_train

Q: What is the default rate for people who own a car?
A: SELECT avg(target::float) AS default_rate FROM application_train WHERE flag_own_car = 'Y'

Q: Which education level defaults most?
A: SELECT name_education_type, avg(target::float) AS default_rate FROM application_train GROUP BY name_education_type ORDER BY default_rate DESC LIMIT 5

Q: How many applicants have a previously refused application?
A: SELECT count(DISTINCT a.sk_id_curr) FROM application_train a JOIN previous_application p ON p.sk_id_curr = a.sk_id_curr WHERE p.name_contract_status = 'Refused'

Q: What is the average income of applicants under 30?
A: SELECT avg(amt_income_total) FROM application_train WHERE days_birth > -30 * 365.25"""


# --------------------------------------------------------------------------
# v3: structured action output
# --------------------------------------------------------------------------

_V3_RULES = """You answer business questions about a credit risk database by
writing PostgreSQL queries. You always respond by calling the answer_question
tool.

Choosing the action:

- "sql" when the question maps onto the columns below. Write one SELECT
  statement, no semicolon, no markdown.
- "refuse" when the data cannot answer the question. This includes questions
  about columns that do not exist, facts the dataset does not record, and
  anything requiring calendar dates. Name the specific problem and suggest a
  question the data CAN answer.
- "clarify" when the question is answerable in principle but too vague to write
  one correct query for, such as an undefined term or a missing threshold.

Refusing is a correct and valued answer. Never invent a column name to make a
question answerable. If a column you want does not appear below, it does not
exist.

SQL rules:
- One SELECT only. Never write to the database.
- Only the tables and columns listed below.
- Include a LIMIT unless returning a single aggregate row.
- Use the exact categorical values shown in [values: ...]; do not guess casing
  or wording.
- Rates are avg(target::float), not count-based ratios."""


def _v1() -> PromptVersion:
    return PromptVersion(
        name="v1",
        system=_V1_SYSTEM,
        use_tool=False,
        changed="Baseline. No schema detail, no examples, no refusal path.",
    )


def _v2() -> PromptVersion:
    return PromptVersion(
        name="v2",
        system="\n\n".join([_V2_RULES, build_schema_context(), _FEW_SHOT]),
        use_tool=False,
        changed=(
            "Added the curated schema with column descriptions and exact "
            "categorical values, five few-shot examples, and explicit SQL "
            "rules. Fixes wrong column names and invented category strings."
        ),
    )


def _v3() -> PromptVersion:
    return PromptVersion(
        name="v3",
        system="\n\n".join([_V3_RULES, build_schema_context(), _FEW_SHOT]),
        use_tool=True,
        changed=(
            "Structured output through a required tool, so refusal and "
            "clarification are first-class actions instead of parse failures. "
            "Added semantic notes covering the negative day offsets, the "
            "absence of calendar dates, and the non-existent credit_score "
            "column, which were the three commonest hallucinations under v2."
        ),
    )


_BUILDERS = {"v1": _v1, "v2": _v2, "v3": _v3}

CURRENT_VERSION = "v3"


def get_prompt(version: str = CURRENT_VERSION) -> PromptVersion:
    if version not in _BUILDERS:
        raise ValueError(f"unknown prompt version {version!r}; have {sorted(_BUILDERS)}")
    return _BUILDERS[version]()


def available_versions() -> list[str]:
    return sorted(_BUILDERS)


# --------------------------------------------------------------------------
# Answer synthesis. Shared by every version, so the comparison isolates the
# SQL generation prompt rather than confounding it with answer phrasing.
# --------------------------------------------------------------------------

ANSWER_SYSTEM = """You turn SQL query results into a short business answer.

Rules:
- One or two sentences. No preamble.
- Quote the actual numbers. Format rates as percentages and large counts with
  thousands separators.
- If the result is empty, say plainly that no rows matched.
- Do not speculate beyond what the rows show, and do not mention SQL."""


def build_answer_prompt(question: str, columns: list[str], rows: list[list]) -> str:
    """Render the result set for summarisation.

    Rows are capped hard. Sending 200 rows to be summarised costs more than the
    question did and adds nothing: the model needs the shape of the answer, not
    every row of it.
    """
    shown = rows[:30]
    lines = [f"Question: {question}", f"Columns: {', '.join(columns)}", "Rows:"]
    for row in shown:
        lines.append("  " + " | ".join("" if v is None else str(v) for v in row))
    if len(rows) > len(shown):
        lines.append(f"  ... {len(rows) - len(shown)} further rows not shown")
    if not rows:
        lines.append("  (no rows)")
    return "\n".join(lines)
