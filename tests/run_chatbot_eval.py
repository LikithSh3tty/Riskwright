"""Chatbot evaluation harness.

Scores the chatbot the way a model gets scored, rather than demoing five
questions that happen to work.

For sql questions, the generated query and a hand-written reference query are
both executed and their results compared. Correct SQL written differently still
passes, which is the only way this measures capability rather than style.

For refuse and clarify questions, the action itself is the answer. A confident
answer to "what is the average credit score" is the failure that matters most,
because it means a column was invented.

Usage:
    python -m tests.run_chatbot_eval                    # current prompt version
    python -m tests.run_chatbot_eval --versions v1 v2 v3
    python -m tests.run_chatbot_eval --out docs/eval_results.md
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import psycopg2
import yaml

from src.talk_to_data.memory import MEMORY
from src.talk_to_data.nl_to_sql import answer
from src.talk_to_data.prompt_templates import CURRENT_VERSION
from src.utils.config import get_settings
from src.utils.docker_utils import project_root
from src.utils.logger import get_logger

log = get_logger(__name__)

EVAL_FILE = "tests/chatbot_eval.yaml"

# Relative tolerance when comparing numeric answers.
DEFAULT_TOLERANCE = 0.01


def load_questions() -> tuple[list[dict], dict]:
    path = project_root() / EVAL_FILE
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return spec["questions"], spec.get("defaults", {})


def run_reference(sql: str) -> list[tuple]:
    settings = get_settings()
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()


def _numbers(rows) -> list[float]:
    out: list[float] = []
    for row in rows:
        for value in row:
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out.append(float(value))
    return out


def _texts(rows) -> list[str]:
    out: list[str] = []
    for row in rows:
        for value in row:
            if isinstance(value, str):
                out.append(value.strip().lower())
    return out


def compare(expected: list[tuple], actual: list[list], mode: str, tolerance: float) -> tuple[bool, str]:
    """Compare a generated result against the reference result.

    Deliberately shape-tolerant. The reference returns one row with one column;
    a model that returns the same number alongside a label is still correct, and
    marking it wrong would measure formatting rather than capability.
    """
    if not expected and not actual:
        return True, "both empty"

    if mode == "text":
        want, got = _texts(expected), _texts(actual)
        if not want:
            return False, "reference produced no text to compare"
        return (want[0] in got, f"expected {want[0]!r} among {got[:5]}")

    if mode == "rowset":
        want, got = _numbers(expected), _numbers(actual)
        if len(want) != len(got):
            # Row counts differing is a real mismatch for a grouped query.
            return False, f"expected {len(want)} numbers, got {len(got)}"
        for a, b in zip(sorted(want), sorted(got)):
            if not math.isclose(a, b, rel_tol=tolerance, abs_tol=1e-9):
                return False, f"{a} != {b}"
        # Also require the group labels to line up when present.
        want_text, got_text = set(_texts(expected)), set(_texts(actual))
        if want_text and not want_text.issubset(got_text):
            return False, f"missing groups {sorted(want_text - got_text)[:3]}"
        return True, "rowset matches"

    # scalar
    want, got = _numbers(expected), _numbers(actual)
    if not want:
        return False, "reference produced no number"
    if not got:
        return False, "no number in the generated result"
    target = want[0]
    if any(math.isclose(target, g, rel_tol=tolerance, abs_tol=1e-9) for g in got):
        return True, f"{target:.6g} matched"
    return False, f"expected {target:.6g}, got {got[:4]}"


def evaluate_question(question: dict, version: str, defaults: dict) -> dict:
    expect = question["expect"]
    session = f"eval-{version}-{question['id']}"

    # A memory question must run after the turn it depends on, in the same
    # session, or it is not testing memory.
    if question.get("follows"):
        session = f"eval-{version}-{question['follows']}"

    started = time.perf_counter()
    result = answer(
        question["question"],
        session_id=session,
        version=version,
        use_memory=True,
    )
    elapsed = int((time.perf_counter() - started) * 1000)

    record = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected": expect,
        "actual": result["action"],
        "sql": result.get("sql"),
        "answer": result.get("answer"),
        "reason": result.get("reason"),
        "latency_ms": elapsed,
        "usage": result.get("usage", {}),
    }

    if expect in ("refuse", "clarify"):
        # A refusal and a clarification are both "declined to answer". Scoring
        # them interchangeably avoids penalising a defensible judgement call on
        # a question that is arguably either.
        declined = result["action"] in ("refuse", "clarify")
        exact = result["action"] == expect
        record["passed"] = declined
        record["exact_action"] = exact
        record["detail"] = (
            "declined as required" if declined else "answered when it should not have"
        )
        return record

    if result["action"] != "sql":
        record["passed"] = False
        record["detail"] = f"expected SQL, got {result['action']}: {result.get('reason')}"
        return record

    reference = question.get("reference_sql")
    if not reference:
        record["passed"] = None
        record["detail"] = "no reference SQL"
        return record

    try:
        expected_rows = run_reference(reference)
    except Exception as exc:
        record["passed"] = None
        record["detail"] = f"reference SQL failed: {exc}"
        return record

    mode = question.get("compare", "scalar")
    tolerance = question.get("tolerance", defaults.get("tolerance", DEFAULT_TOLERANCE))
    passed, detail = compare(expected_rows, result["rows"], mode, tolerance)
    record["passed"] = passed
    record["detail"] = detail
    return record


def run(version: str) -> list[dict]:
    questions, defaults = load_questions()
    records: list[dict] = []
    for question in questions:
        record = evaluate_question(question, version, defaults)
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[record["passed"]]
        log.info(
            "%s %s %-20s %s",
            status,
            record["id"],
            record["category"],
            record["detail"][:70],
        )
        records.append(record)

    # Sessions are per-question; clear so a rerun starts clean.
    for question in questions:
        MEMORY.clear(f"eval-{version}-{question['id']}")
    return records


def summarise(records: list[dict]) -> dict:
    scored = [r for r in records if r["passed"] is not None]
    by_category: dict[str, dict] = {}
    for record in scored:
        bucket = by_category.setdefault(
            record["category"], {"passed": 0, "total": 0}
        )
        bucket["total"] += 1
        bucket["passed"] += 1 if record["passed"] else 0

    tokens = sum(
        r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0)
        for r in records
    )
    return {
        "passed": sum(1 for r in scored if r["passed"]),
        "total": len(scored),
        "accuracy": (sum(1 for r in scored if r["passed"]) / len(scored)) if scored else 0.0,
        "by_category": by_category,
        "total_tokens": tokens,
        "median_latency_ms": sorted(r["latency_ms"] for r in records)[len(records) // 2]
        if records
        else 0,
    }


def render_markdown(results: dict[str, tuple[list[dict], dict]]) -> str:
    lines = ["# Chatbot evaluation results", ""]
    lines.append(
        "Generated by `python -m tests.run_chatbot_eval`. Each SQL question is "
        "scored by executing the generated query and a hand-written reference "
        "query and comparing results, so correct SQL written differently still "
        "passes. Refusal questions are scored on whether the bot declined."
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| Prompt version | Accuracy | Passed | Tokens | Median latency |")
    lines.append("|---|---|---|---|---|")
    for version, (_, summary) in results.items():
        lines.append(
            f"| {version} | {summary['accuracy']:.0%} | "
            f"{summary['passed']}/{summary['total']} | "
            f"{summary['total_tokens']:,} | {summary['median_latency_ms']}ms |"
        )
    lines.append("")

    categories = sorted(
        {c for _, summary in results.values() for c in summary["by_category"]}
    )
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | " + " | ".join(results) + " |")
    lines.append("|---" * (len(results) + 1) + "|")
    for category in categories:
        cells = []
        for _, summary in results.values():
            bucket = summary["by_category"].get(category)
            cells.append(f"{bucket['passed']}/{bucket['total']}" if bucket else "-")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("")

    final_version = list(results)[-1]
    records, _ = results[final_version]
    failures = [r for r in records if r["passed"] is False]
    lines.append(f"## Failures on {final_version}")
    lines.append("")
    if not failures:
        lines.append("None.")
    else:
        lines.append("| ID | Question | Expected | Got | Why |")
        lines.append("|---|---|---|---|---|")
        for record in failures:
            question = record["question"].replace("|", "/")
            detail = (record["detail"] or "").replace("|", "/")[:90]
            lines.append(
                f"| {record['id']} | {question} | {record['expected']} | "
                f"{record['actual']} | {detail} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the chatbot")
    parser.add_argument("--versions", nargs="+", default=[CURRENT_VERSION])
    parser.add_argument("--out", default=None, help="write a markdown report here")
    parser.add_argument("--json", default=None, help="write raw records here")
    args = parser.parse_args()

    results: dict[str, tuple[list[dict], dict]] = {}
    for version in args.versions:
        log.info("running evaluation for prompt %s", version)
        records = run(version)
        summary = summarise(records)
        results[version] = (records, summary)
        log.info(
            "%s: %d/%d correct (%.0f%%), %s tokens",
            version,
            summary["passed"],
            summary["total"],
            100 * summary["accuracy"],
            f"{summary['total_tokens']:,}",
        )

    report = render_markdown(results)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        log.info("wrote %s", path)
    else:
        print(report)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {v: {"records": r, "summary": s} for v, (r, s) in results.items()},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
