"""Repeat the held-out evaluation and report the spread.

A single held-out score is one sample from a distribution, and the temperature
is not zero. Reporting 19/20 as though it were a fixed property of the system
overstates what one run can tell you: the interesting question is not what the
bot scored once but how much that score moves, and which questions are doing
the moving.

This runs the held-out set N times against one prompt version and records, per
question, how many of those runs it passed. A question that passes 5/5 or 0/5
is stable -- correct or broken, but consistent. Anything in between is the
model exercising judgement, and those are the questions the README's claim
about probabilistic refusal is actually about.

It does not tune anything. The held-out set stops being held out the moment a
prompt is changed in response to what it says, so this reports and stops.

Usage, against a populated Postgres with ANTHROPIC_API_KEY set:

    POSTGRES_HOST=localhost python -m tests.run_chatbot_variance
    POSTGRES_HOST=localhost python -m tests.run_chatbot_variance --runs 3

Writes models/chatbot_variance.json, which the README reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from src.talk_to_data.prompt_templates import CURRENT_VERSION
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger
from tests.run_chatbot_eval import HELDOUT_FILE, run, summarise

log = get_logger(__name__)

VARIANCE_FILE = "chatbot_variance.json"
DEFAULT_RUNS = 5


def measure(runs: int, version: str, eval_file: str) -> dict:
    per_run: list[dict] = []
    outcomes: dict[str, list[bool]] = defaultdict(list)
    details: dict[str, dict] = {}

    for index in range(1, runs + 1):
        log.info("run %d of %d", index, runs)
        records = run(version, eval_file)
        summary = summarise(records)
        per_run.append({
            "run": index,
            "passed": summary["passed"],
            "total": summary["total"],
            "accuracy": summary["accuracy"],
            "total_tokens": summary["total_tokens"],
            "median_latency_ms": summary["median_latency_ms"],
        })
        log.info("run %d: %d/%d", index, summary["passed"], summary["total"])

        for record in records:
            if record["passed"] is None:
                continue
            outcomes[record["id"]].append(bool(record["passed"]))
            seen = details.setdefault(record["id"], {
                "category": record["category"],
                "question": record["question"],
                "expected": record["expected"],
                "actions": [],
            })
            seen["actions"].append(record["actual"])

    scores = [r["passed"] for r in per_run]
    questions = []
    for qid, passes in sorted(outcomes.items(), key=lambda kv: int(kv[0][1:])):
        n_pass = sum(passes)
        questions.append({
            "id": qid,
            "category": details[qid]["category"],
            "question": details[qid]["question"],
            "expected": details[qid]["expected"],
            "passed_runs": n_pass,
            "runs": len(passes),
            "stable": n_pass in (0, len(passes)),
            "always_failed": n_pass == 0,
            "actions_seen": sorted(set(details[qid]["actions"])),
        })

    unstable = [q for q in questions if not q["stable"]]
    always_failed = [q for q in questions if q["always_failed"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": version,
        "question_set": eval_file,
        "runs": runs,
        "per_run": per_run,
        "score": {
            "min": min(scores),
            "max": max(scores),
            "range": max(scores) - min(scores),
            "mean": statistics.fmean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "total": per_run[0]["total"],
            "mean_accuracy": statistics.fmean(r["accuracy"] for r in per_run),
        },
        "questions": questions,
        "unstable_questions": [q["id"] for q in unstable],
        "always_failed_questions": [q["id"] for q in always_failed],
        "note": (
            "Repeated runs of the same held-out set at the same prompt version. "
            "The prompt was not changed in response to these results: tuning "
            "against a held-out set is what stops it being one."
        ),
    }


def prior_history(path) -> list[dict]:
    """Condense whatever the artifact already holds into a history entry.

    The question set grows, so a later mean is not comparable to an earlier one
    and replacing the file outright would quietly erase the basis for the
    figure the README used to quote. Each run keeps its predecessors as
    summaries: enough to see how the measurement changed and why, without
    carrying every per-question record forward forever.
    """
    if not path.exists():
        return []
    previous = json.loads(path.read_text(encoding="utf-8"))
    history = list(previous.get("prior_history", []))
    history.append({
        "generated_at": previous.get("generated_at"),
        "prompt_version": previous.get("prompt_version"),
        "questions": previous.get("score", {}).get("total"),
        "runs": previous.get("runs"),
        "per_run": [r["passed"] for r in previous.get("per_run", [])],
        "mean": previous.get("score", {}).get("mean"),
        "min": previous.get("score", {}).get("min"),
        "max": previous.get("score", {}).get("max"),
        "stdev": previous.get("score", {}).get("stdev"),
        "unstable_questions": previous.get("unstable_questions", []),
        "always_failed_questions": previous.get("always_failed_questions", []),
    })
    return history


def save(payload: dict) -> None:
    path = models_dir() / VARIANCE_FILE
    payload = {**payload, "prior_history": prior_history(path)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote %s (%d prior measurements retained)",
             path, len(payload["prior_history"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out score variance")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--version", default=CURRENT_VERSION)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload = measure(args.runs, args.version, HELDOUT_FILE)
    score = payload["score"]

    print(f"\nheld-out set, {payload['prompt_version']}, {payload['runs']} runs")
    for entry in payload["per_run"]:
        print(f"  run {entry['run']}: {entry['passed']}/{entry['total']} "
              f"({entry['accuracy']:.0%})")
    print(f"  mean {score['mean']:.1f}/{score['total']}  "
          f"range {score['min']}-{score['max']}  sd {score['stdev']:.2f}")

    print("\nper question")
    for q in payload["questions"]:
        mark = "  " if q["stable"] else "* "
        print(f"  {mark}{q['id']:<4} {q['passed_runs']}/{q['runs']}  "
              f"{q['category']:<24} {'/'.join(q['actions_seen'])}")

    print("\nunstable across runs: "
          + (", ".join(payload["unstable_questions"]) or "none"))
    print("failed every run:     "
          + (", ".join(payload["always_failed_questions"]) or "none"))

    if not args.no_save:
        save(payload)


if __name__ == "__main__":
    main()
