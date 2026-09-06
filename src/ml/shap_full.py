"""Global SHAP importance over the whole population, computed in chunks.

`models/shap_global.json` was built from a 2,000-row sample of 307,511, which
the README listed as a limitation. This recomputes it exhaustively.

Why a separate module rather than raising `GLOBAL_SAMPLE_ROWS`
--------------------------------------------------------------

`compute_global_importance` materialises the whole SHAP matrix at once. At
2,000 rows that is 2MB and irrelevant. At 307,511 rows it is 307,511 x 120
float64 -- about 295MB -- and shap has historically returned a two-element list
for LightGBM binary classifiers, which doubles it before anything is reduced.
Add the feature matrix and the intermediate that `np.abs` allocates and the
peak is comfortably over a gigabyte on a machine that has run out of memory
three times.

So the population is walked in chunks and only a running sum of absolute SHAP
per feature is kept: 120 floats, regardless of how many rows have been
processed. Peak memory is set by the chunk, not by the dataset, and the result
is identical to the unchunked computation because a mean is a sum divided by a
count and both are exact here.

The output schema matches `compute_global_importance` exactly, so the artifact
is a drop-in replacement and `/explain/global` needs no change.

Run against a populated Postgres:

    POSTGRES_HOST=localhost python -m src.ml.shap_full

Writes models/shap_global.json, and models/shap_sample_comparison.json holding
the comparison against the sampled ranking it replaces.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.loader import read_application_table
from src.data.preprocessor import load_feature_spec, prepare_features
from src.ml.explain import (
    GLOBAL_IMPORTANCE_FILE,
    _explainer,
    _positive_class_shap,
    label_for,
    load_global_importance,
)
from src.utils.docker_utils import models_dir
from src.utils.logger import get_logger

log = get_logger(__name__)

COMPARISON_FILE = "shap_sample_comparison.json"

# 20,000 rows x 120 features x 8 bytes is about 19MB per array, so even if shap
# hands back both classes the chunk stays under 40MB. Small enough to be safe
# on a constrained machine, large enough that the per-chunk overhead of
# building the explanation does not dominate.
CHUNK_ROWS = 20_000

# How far down the ranking to compare. Below this the mean absolute SHAP values
# are close enough together that their order is noise in either computation,
# and reporting a rank change there would overstate the disagreement.
COMPARE_TOP_N = 20


def accumulate(features: pd.DataFrame, chunk_rows: int = CHUNK_ROWS) -> np.ndarray:
    """Sum of |SHAP| per feature over every row, one chunk at a time."""
    explainer = _explainer()
    totals = np.zeros(features.shape[1], dtype=np.float64)
    processed = 0

    for start in range(0, len(features), chunk_rows):
        chunk = features.iloc[start:start + chunk_rows]
        values = _positive_class_shap(explainer.shap_values(chunk))
        totals += np.abs(values).sum(axis=0)
        processed += len(chunk)
        del values
        log.info("%d / %d rows", processed, len(features))

    return totals / processed


def rank(features: list[str], mean_abs: np.ndarray) -> list[dict]:
    return sorted(
        (
            {
                "feature": str(name),
                "label": label_for(str(name)),
                "mean_abs_shap": float(score),
            }
            for name, score in zip(features, mean_abs)
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )


def compare(previous: dict, current: dict, top_n: int = COMPARE_TOP_N) -> dict:
    """Did widening the sample change what the model is said to rely on?

    The question the sample was ever asked is "which features matter, and in
    what order". So the comparison is of ranks, not of values: a mean computed
    over 2,000 rows will differ in the third decimal from one over 307,511
    whatever happens, and that difference is not interesting. A feature moving
    into or out of the top ten is.
    """
    old_order = [f["feature"] for f in previous["features"]]
    new_order = [f["feature"] for f in current["features"]]
    old_rank = {name: i + 1 for i, name in enumerate(old_order)}
    new_rank = {name: i + 1 for i, name in enumerate(new_order)}

    moves = []
    for name in new_order[:top_n]:
        before, after = old_rank.get(name), new_rank[name]
        moves.append({
            "feature": name,
            "sampled_rank": before,
            "full_rank": after,
            "rank_change": (before - after) if before else None,
        })

    old_values = {f["feature"]: f["mean_abs_shap"] for f in previous["features"]}
    new_values = {f["feature"]: f["mean_abs_shap"] for f in current["features"]}
    shared = [n for n in new_order if n in old_values]
    spearman = float(pd.Series([old_rank[n] for n in shared]).corr(
        pd.Series([new_rank[n] for n in shared]), method="spearman"
    ))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampled": {
            "rows": previous["sampled_rows"],
            "top_10": old_order[:10],
        },
        "full": {
            "rows": current["sampled_rows"],
            "top_10": new_order[:10],
        },
        "top_10_identical": old_order[:10] == new_order[:10],
        "top_10_same_set": set(old_order[:10]) == set(new_order[:10]),
        "top_n_compared": top_n,
        "rank_moves": moves,
        "max_abs_rank_change": max(
            (abs(m["rank_change"]) for m in moves if m["rank_change"] is not None),
            default=0,
        ),
        "rank_correlation_all_features": spearman,
        "largest_value_change": max(
            (
                {
                    "feature": n,
                    "sampled": old_values[n],
                    "full": new_values[n],
                    "absolute_change": abs(new_values[n] - old_values[n]),
                }
                for n in shared
            ),
            key=lambda row: row["absolute_change"],
        ),
        "note": (
            "Ranks are compared rather than values: a mean over 2,000 rows "
            "differs in the third decimal from one over 307,511 whatever "
            "happens. What matters is whether the ordering of the drivers "
            "changed."
        ),
    }


def compute(limit: int | None = None, chunk_rows: int = CHUNK_ROWS) -> tuple[dict, dict]:
    raw = read_application_table(limit=limit)
    features = prepare_features(raw, feature_spec=load_feature_spec())
    del raw

    log.info(
        "computing SHAP over %d rows in chunks of %d", len(features), chunk_rows
    )
    mean_abs = accumulate(features, chunk_rows=chunk_rows)

    payload = {
        # The existing schema calls this sampled_rows and /explain/global reads
        # it. Kept, so the artifact stays a drop-in replacement; it now equals
        # total_rows, which is the whole point.
        "sampled_rows": int(len(features)),
        "total_rows": int(len(features)),
        "exhaustive": True,
        "chunk_rows": int(chunk_rows),
        "features": rank(list(features.columns), mean_abs),
    }

    previous = load_global_importance()
    return payload, compare(previous, payload)


def save(payload: dict, comparison: dict) -> None:
    (models_dir() / GLOBAL_IMPORTANCE_FILE).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (models_dir() / COMPARISON_FILE).write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    log.info("wrote %s and %s", GLOBAL_IMPORTANCE_FILE, COMPARISON_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustive global SHAP importance, chunked"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N applicants (smoke test)")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    payload, comparison = compute(limit=args.limit, chunk_rows=args.chunk_rows)

    print(f"\nmean |SHAP| over {payload['sampled_rows']:,} rows "
          f"(previously {comparison['sampled']['rows']:,})\n")
    print(f"  {'rank':>4}  {'was':>4}  {'feature':<32} {'mean |SHAP|':>12}")
    for i, entry in enumerate(payload["features"][:COMPARE_TOP_N], start=1):
        move = next(m for m in comparison["rank_moves"] if m["feature"] == entry["feature"])
        was = move["sampled_rank"]
        print(f"  {i:>4}  {was if was else '-':>4}  {entry['feature']:<32} "
              f"{entry['mean_abs_shap']:>12.5f}")

    print(f"\ntop 10 identical: {comparison['top_10_identical']}")
    print(f"top 10 same set:  {comparison['top_10_same_set']}")
    print(f"largest rank move in the top {COMPARE_TOP_N}: "
          f"{comparison['max_abs_rank_change']}")
    print(f"rank correlation across all 120 features: "
          f"{comparison['rank_correlation_all_features']:.4f}")

    if not args.no_save:
        save(payload, comparison)


if __name__ == "__main__":
    main()
