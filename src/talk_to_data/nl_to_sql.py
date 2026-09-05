"""Natural language to SQL via Claude.

One model call per question, returning a structured action rather than bare
SQL. The action enum is what makes refusal a first-class outcome: a model with
no sanctioned way to decline will invent a column to fill the silence, and that
invented column is precisely what must never reach the database.

Flow for one turn:

  1. Build the prompt: cached system block, replayed memory, the question.
  2. One call. Action is sql, refuse, or clarify.
  3. refuse / clarify return immediately. One call, no query, cheap by design.
  4. sql goes through the validation gate. On failure the validator's specific
     error is fed back for exactly one retry, then it becomes a refusal.
  5. On success, execute and make a second call to phrase the answer.

Answered turns cost two calls, refused turns cost one.
"""

from __future__ import annotations

import time
from functools import lru_cache

import anthropic

from src.talk_to_data.memory import MEMORY
from src.talk_to_data.prompt_templates import (
    ANSWER_TOOL,
    ANSWER_SYSTEM,
    CURRENT_VERSION,
    build_answer_prompt,
    get_prompt,
)
from src.talk_to_data.query_runner import (
    SqlValidationError,
    execute,
    strip_fences,
    validate,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# SQL generation is short. A large ceiling here would only pay for runaway
# output on a malformed request.
MAX_TOKENS_SQL = 1024
MAX_TOKENS_ANSWER = 512

# Haiku 4.5 takes an explicit thinking budget; it does not support the adaptive
# thinking or the effort parameter used by the Opus and Sonnet families. SQL
# generation over a small schema does not need extended thinking at all, so it
# is left off and the tokens go to the schema instead.
MODEL_SUPPORTS_EFFORT = False


class LlmNotConfigured(RuntimeError):
    pass


@lru_cache
def _client() -> anthropic.Anthropic:
    settings = get_settings()
    if not settings.llm_configured:
        raise LlmNotConfigured(
            "ANTHROPIC_API_KEY is not set. The chatbot needs it; the rest of "
            "the platform runs without it."
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _system_blocks(version: str) -> list[dict]:
    """System prompt as a cacheable block.

    The breakpoint is correct and deliberately placed: the schema and rules are
    byte-identical on every turn, and everything volatile (memory, the
    question) sits after it in messages.

    Measured caveat: it does not currently engage. Haiku 4.5 will not cache a
    prefix below 4,096 tokens, and this prompt is about 3,000. Verified by
    padding the same prompt to 7,393 tokens, where cache_creation and then
    cache_read both fire, and by confirming zero on two identical calls at the
    real size.

    The breakpoint stays because it is free and starts working the moment the
    schema grows past the threshold. Padding the prompt to reach it would mean
    either filler, or re-adding the columns that were curated out to reduce
    wrong-column errors, to save roughly seven cents per full evaluation run.
    Compactness is the better trade at this volume.
    """
    prompt = get_prompt(version)
    return [
        {
            "type": "text",
            "text": prompt.system,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _parse_tool_response(response) -> dict:
    """Pull the structured action out of a tool_use block."""
    for block in response.content:
        if block.type == "tool_use" and block.name == ANSWER_TOOL["name"]:
            # Inputs are parsed JSON already; never string-match the raw input.
            data = dict(block.input)
            action = data.get("action")
            if action not in ("sql", "refuse", "clarify"):
                raise ValueError(f"unexpected action {action!r}")
            return data

    # The model answered in prose instead of calling the tool. Treat any SQL in
    # that text as a best effort rather than discarding the turn.
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if text.lower().startswith("select") or "```" in text:
        return {"action": "sql", "sql": strip_fences(text)}
    return {
        "action": "refuse",
        "reason": text or "The model did not produce a usable answer.",
    }


def _parse_plain_response(response) -> dict:
    """Versions v1 and v2 return bare SQL, so everything is an sql action.

    That is the point of the comparison: without a structured action they have
    no way to refuse, which is what the evaluation measures.
    """
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"action": "sql", "sql": strip_fences(text)}


def _call_model(
    version: str,
    messages: list[dict],
) -> tuple[dict, dict]:
    prompt = get_prompt(version)
    client = _client()
    settings = get_settings()

    request = {
        "model": settings.anthropic_model,
        "max_tokens": MAX_TOKENS_SQL,
        "system": _system_blocks(version),
        "messages": messages,
    }
    if prompt.use_tool:
        request["tools"] = [ANSWER_TOOL]
        # Forcing the tool guarantees a structured result. Supported on Haiku;
        # newer frontier models have removed forced tool choice.
        request["tool_choice"] = {"type": "tool", "name": ANSWER_TOOL["name"]}

    response = client.messages.create(**request)

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_input_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0
        ),
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0
        ),
    }

    parsed = _parse_tool_response(response) if prompt.use_tool else _parse_plain_response(response)
    return parsed, usage


def _summarise(question: str, columns: list[str], rows: list[list]) -> tuple[str, dict]:
    settings = get_settings()
    response = _client().messages.create(
        model=settings.anthropic_model,
        max_tokens=MAX_TOKENS_ANSWER,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": build_answer_prompt(question, columns, rows)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage


def _merge_usage(*usages: dict) -> dict:
    total: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            total[key] = total.get(key, 0) + int(value or 0)
    return total


def answer(
    question: str,
    session_id: str = "default",
    version: str = CURRENT_VERSION,
    use_memory: bool = True,
) -> dict:
    """Answer one question. Never raises for an ordinary failure."""
    started = time.perf_counter()
    history = MEMORY.history(session_id) if use_memory else []
    messages = history + [{"role": "user", "content": question}]

    try:
        parsed, usage = _call_model(version, messages)
    except LlmNotConfigured as exc:
        return _result(question, "error", reason=str(exc), started=started)
    except anthropic.APIError as exc:
        log.warning("model call failed: %s", exc)
        return _result(
            question, "error", reason=f"The language model call failed: {exc}",
            started=started,
        )

    action = parsed.get("action")

    if action in ("refuse", "clarify"):
        note = parsed.get("reason") or ""
        MEMORY.record(session_id, question, action, note=note)
        return _result(
            question,
            action,
            reason=note,
            suggestion=parsed.get("suggestion"),
            usage=usage,
            started=started,
            version=version,
        )

    sql = parsed.get("sql") or ""
    validated = None
    validation_error: SqlValidationError | None = None

    for attempt in (1, 2):
        try:
            validated = validate(sql)
            break
        except SqlValidationError as exc:
            validation_error = exc
            if attempt == 2:
                break
            # One repair attempt, with the validator's specific complaint fed
            # back. A second failure becomes a refusal rather than a third call.
            log.info("validation failed, retrying once: %s", exc.message)
            repair = messages + [
                {"role": "assistant", "content": f"SQL: {sql}"},
                {
                    "role": "user",
                    "content": (
                        f"That query was rejected: {exc.message} "
                        "Correct it using only columns that exist, or refuse "
                        "if the question cannot be answered."
                    ),
                },
            ]
            try:
                parsed, retry_usage = _call_model(version, repair)
            except anthropic.APIError as api_exc:
                return _result(
                    question, "error", reason=str(api_exc), usage=usage, started=started
                )
            usage = _merge_usage(usage, retry_usage)
            if parsed.get("action") in ("refuse", "clarify"):
                note = parsed.get("reason") or exc.message
                MEMORY.record(session_id, question, parsed["action"], note=note)
                return _result(
                    question,
                    parsed["action"],
                    reason=note,
                    suggestion=parsed.get("suggestion"),
                    usage=usage,
                    started=started,
                    version=version,
                )
            sql = parsed.get("sql") or ""

    if validated is None:
        message = validation_error.message if validation_error else "Invalid SQL."
        MEMORY.record(session_id, question, "refuse", note=message)
        return _result(
            question,
            "refuse",
            reason=message,
            suggestion=(
                validation_error.suggestion if validation_error else None
            ),
            sql=sql,
            usage=usage,
            started=started,
            version=version,
        )

    try:
        result = execute(validated)
    except Exception as exc:
        message = str(exc).strip().splitlines()[0]
        log.warning("query execution failed: %s", message)
        MEMORY.record(session_id, question, "refuse", note=message)
        return _result(
            question, "refuse", reason=f"The query could not be run: {message}",
            sql=validated.sql, usage=usage, started=started, version=version,
        )

    try:
        text, answer_usage = _summarise(question, result["columns"], result["rows"])
        usage = _merge_usage(usage, answer_usage)
    except anthropic.APIError as exc:
        log.warning("answer synthesis failed: %s", exc)
        text = "The query ran, but the summary could not be generated."

    MEMORY.record(session_id, question, "sql", sql=validated.sql)

    return _result(
        question,
        "sql",
        answer=text,
        sql=validated.sql,
        columns=result["columns"],
        rows=result["rows"],
        row_count=result["row_count"],
        truncated=result["truncated"],
        usage=usage,
        started=started,
        version=version,
    )


def _result(
    question: str,
    action: str,
    answer: str | None = None,
    sql: str | None = None,
    reason: str | None = None,
    suggestion: str | None = None,
    columns: list[str] | None = None,
    rows: list[list] | None = None,
    row_count: int = 0,
    truncated: bool = False,
    usage: dict | None = None,
    started: float | None = None,
    version: str = CURRENT_VERSION,
) -> dict:
    return {
        "question": question,
        "action": action,
        "answer": answer,
        "sql": sql,
        "reason": reason,
        "suggestion": suggestion,
        "columns": columns or [],
        "rows": rows or [],
        "row_count": row_count,
        "truncated": truncated,
        "usage": usage or {},
        "prompt_version": version,
        "latency_ms": int((time.perf_counter() - started) * 1000) if started else None,
    }
