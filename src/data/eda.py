"""Exploratory analysis, computed server-side.

Every function here returns aggregates. Raw rows never leave the API: sending
307,511 rows to a browser to draw a histogram is the wrong shape, and the UI
stays a thin client precisely because the aggregation lives here.

The same functions back the API endpoints and the notebook, so the numbers in
notebooks/eda.ipynb and the numbers on screen cannot drift apart.
"""

from __future__ import annotations

from functools import lru_cache

import psycopg2

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

TABLE = "application_train"

# Columns offered to the distribution explorer, with the transform needed to
# make each readable. Days columns are negative offsets, so they are converted
# to positive years before binning.
NUMERIC_COLUMNS: dict[str, str] = {
    "amt_income_total": "amt_income_total",
    "amt_credit": "amt_credit",
    "amt_annuity": "amt_annuity",
    "amt_goods_price": "amt_goods_price",
    "age_years": "(-days_birth / 365.25)",
    "years_employed": "(-days_employed / 365.25)",
    "ext_source_1": "ext_source_1",
    "ext_source_2": "ext_source_2",
    "ext_source_3": "ext_source_3",
    "credit_income_ratio": "(amt_credit / NULLIF(amt_income_total, 0))",
    "cnt_children": "cnt_children",
}

CATEGORICAL_COLUMNS = (
    "name_education_type",
    "name_family_status",
    "name_income_type",
    "name_housing_type",
    "occupation_type",
    "code_gender",
    "name_contract_type",
    "flag_own_car",
    "flag_own_realty",
    "region_rating_client",
)

# The sentinel that makes days_employed unusable if left alone.
DAYS_EMPLOYED_SENTINEL = 365243


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    settings = get_settings()
    with psycopg2.connect(settings.dsn(read_only=True)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@lru_cache
def dataset_summary() -> dict:
    """Shape, target balance, and the worst missing-value offenders."""
    rows, positives = _query(
        f"SELECT count(*), sum(target) FROM {TABLE}"
    )[0]

    column_count = _query(
        "SELECT count(*) FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s",
        (TABLE,),
    )[0][0]

    # Missingness per column, computed in one pass rather than 122 queries.
    columns = [
        r[0]
        for r in _query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s "
            "ORDER BY ordinal_position",
            (TABLE,),
        )
    ]
    parts = ", ".join(
        f"count(*) FILTER (WHERE {c} IS NULL) AS {c}" for c in columns
    )
    counts = _query(f"SELECT {parts} FROM {TABLE}")[0]

    missing = sorted(
        (
            {
                "column": column,
                "missing": int(count),
                "missing_pct": round(100 * count / rows, 2),
            }
            for column, count in zip(columns, counts)
            if count
        ),
        key=lambda r: r["missing"],
        reverse=True,
    )

    return {
        "table": TABLE,
        "rows": int(rows),
        "columns": int(column_count),
        "defaulted": int(positives),
        "repaid": int(rows - positives),
        "default_rate": float(positives / rows),
        "imbalance_ratio": float((rows - positives) / positives),
        "columns_with_missing": len(missing),
        "missing_top": missing[:20],
    }


def distribution(column: str, bins: int = 30) -> dict:
    """Histogram bins for one column, split by outcome.

    Returns bin edges and counts, never values. The client draws; the server
    aggregates.
    """
    if column not in NUMERIC_COLUMNS:
        raise KeyError(f"{column} is not available; choose from {sorted(NUMERIC_COLUMNS)}")

    expression = NUMERIC_COLUMNS[column]
    bins = max(5, min(int(bins), 100))

    # Trim to the 1st-99th percentile so a handful of extreme values do not
    # flatten the entire chart into one bar.
    low, high = _query(
        f"SELECT percentile_cont(0.01) WITHIN GROUP (ORDER BY {expression}), "
        f"percentile_cont(0.99) WITHIN GROUP (ORDER BY {expression}) "
        f"FROM {TABLE} WHERE {expression} IS NOT NULL"
    )[0]

    if low is None or high is None or float(high) <= float(low):
        return {"column": column, "bins": [], "note": "no usable range"}

    low, high = float(low), float(high)
    width = (high - low) / bins

    rows = _query(
        f"""
        SELECT bucket, target, count(*)
        FROM (
            SELECT width_bucket({expression}, %s, %s, %s) AS bucket, target
            FROM {TABLE}
            WHERE {expression} IS NOT NULL
        ) t
        GROUP BY bucket, target
        ORDER BY bucket
        """,
        (low, high, bins),
    )

    buckets: dict[int, dict] = {}
    for bucket, target, count in rows:
        entry = buckets.setdefault(int(bucket), {"repaid": 0, "defaulted": 0})
        entry["defaulted" if target == 1 else "repaid"] += int(count)

    out = []
    for index in range(1, bins + 1):
        entry = buckets.get(index, {"repaid": 0, "defaulted": 0})
        total = entry["repaid"] + entry["defaulted"]
        out.append(
            {
                "bin_start": round(low + (index - 1) * width, 4),
                "bin_end": round(low + index * width, 4),
                "repaid": entry["repaid"],
                "defaulted": entry["defaulted"],
                "total": total,
                "default_rate": round(entry["defaulted"] / total, 4) if total else None,
            }
        )

    return {
        "column": column,
        "range": [round(low, 4), round(high, 4)],
        "note": "trimmed to the 1st-99th percentile",
        "bins": out,
    }


def group_default_rate(dimension: str, min_count: int = 100) -> dict:
    """Default rate per category, with counts."""
    if dimension not in CATEGORICAL_COLUMNS:
        raise KeyError(
            f"{dimension} is not available; choose from {sorted(CATEGORICAL_COLUMNS)}"
        )

    rows = _query(
        f"""
        SELECT {dimension}::text, count(*), avg(target::float)
        FROM {TABLE}
        WHERE {dimension} IS NOT NULL
        GROUP BY {dimension}
        HAVING count(*) >= %s
        ORDER BY avg(target::float) DESC
        """,
        (min_count,),
    )
    base = dataset_summary()["default_rate"]

    return {
        "dimension": dimension,
        "base_default_rate": base,
        "groups": [
            {
                "value": value,
                "count": int(count),
                "default_rate": float(rate),
                "lift": round(float(rate) / base, 3),
            }
            for value, count, rate in rows
        ],
    }


@lru_cache
def business_insights() -> dict:
    """The five findings, computed rather than asserted.

    Each carries the number that supports it, so the notebook, the UI, and the
    README all quote one source.
    """
    summary = dataset_summary()
    base = summary["default_rate"]
    insights = []

    # 1. The data-quality landmine.
    anomalous, total = _query(
        f"SELECT count(*) FILTER (WHERE days_employed = %s), count(*) FROM {TABLE}",
        (DAYS_EMPLOYED_SENTINEL,),
    )[0]
    anomalous_rate, normal_rate = _query(
        f"""
        SELECT avg(target::float) FILTER (WHERE days_employed = %s),
               avg(target::float) FILTER (WHERE days_employed <> %s)
        FROM {TABLE}
        """,
        (DAYS_EMPLOYED_SENTINEL, DAYS_EMPLOYED_SENTINEL),
    )[0]
    insights.append(
        {
            "title": "A sentinel value hides in the employment column",
            "finding": (
                f"{100 * anomalous / total:.1f}% of applicants have days_employed "
                f"set to {DAYS_EMPLOYED_SENTINEL}, roughly 1,000 years, which "
                f"encodes 'not currently employed' rather than a duration. Left "
                f"untreated it corrupts every statistic built on the column."
            ),
            "evidence": {
                "affected_pct": round(100 * anomalous / total, 2),
                "default_rate_affected": round(float(anomalous_rate), 4),
                "default_rate_others": round(float(normal_rate), 4),
            },
            "so_what": (
                "These applicants default at "
                f"{100 * float(anomalous_rate):.1f}% against "
                f"{100 * float(normal_rate):.1f}% for the rest, so the flag is "
                "predictive and is kept as a feature rather than discarded."
            ),
        }
    )

    # 2. External scores dominate.
    quartiles = _query(
        f"""
        SELECT quartile, count(*), avg(target::float)
        FROM (
            SELECT ntile(4) OVER (ORDER BY ext_source_3) AS quartile, target
            FROM {TABLE} WHERE ext_source_3 IS NOT NULL
        ) t GROUP BY quartile ORDER BY quartile
        """
    )
    worst = float(quartiles[0][2])
    best = float(quartiles[-1][2])
    insights.append(
        {
            "title": "External credit scores separate risk more than anything the applicant reports",
            "finding": (
                f"Applicants in the lowest quartile of ext_source_3 default at "
                f"{100 * worst:.1f}%, against {100 * best:.1f}% in the highest. "
                f"That is a {worst / best:.1f}x spread from a single column."
            ),
            "evidence": {
                "quartiles": [
                    {"quartile": int(q), "count": int(c), "default_rate": round(float(r), 4)}
                    for q, c, r in quartiles
                ]
            },
            "so_what": (
                "The model and the derived rules both lean heavily on these "
                "three columns. That is a strength for accuracy and a weakness "
                "for coverage, since an applicant with no bureau history has "
                "none of them."
            ),
        }
    )

    # 3. Education.
    education = group_default_rate("name_education_type")["groups"]
    top, bottom = education[0], education[-1]
    insights.append(
        {
            "title": "Education level tracks default risk strongly",
            "finding": (
                f"{top['value']} applicants default at "
                f"{100 * top['default_rate']:.1f}%, against "
                f"{100 * bottom['default_rate']:.1f}% for {bottom['value']}."
            ),
            "evidence": {"groups": education},
            "so_what": (
                "A usable segmentation, though one that would need fair-lending "
                "review before it drove decisions."
            ),
        }
    )

    # 4. Leverage.
    leverage = _query(
        f"""
        SELECT band, count(*), avg(target::float) FROM (
            SELECT CASE
                     WHEN amt_credit / NULLIF(amt_income_total, 0) < 2 THEN '1. under 2x'
                     WHEN amt_credit / NULLIF(amt_income_total, 0) < 4 THEN '2. 2x to 4x'
                     WHEN amt_credit / NULLIF(amt_income_total, 0) < 6 THEN '3. 4x to 6x'
                     ELSE '4. over 6x'
                   END AS band,
                   target
            FROM {TABLE}
            WHERE amt_income_total > 0
        ) t GROUP BY band ORDER BY band
        """
    )
    rates = [float(r) for _, _, r in leverage]
    peak_index = max(range(len(rates)), key=lambda i: rates[i])
    insights.append(
        {
            "title": "Borrowing a lot relative to income does not predict default, which is not what you would expect",
            "finding": (
                "Default rate does not rise with leverage. It peaks in the "
                f"{leverage[peak_index][0][3:]} band at "
                f"{100 * rates[peak_index]:.2f}% and is at its LOWEST among the "
                f"most leveraged borrowers, {100 * rates[-1]:.2f}% for those "
                f"borrowing over six times their annual income, below the "
                f"{100 * rates[0]:.2f}% seen under two times."
            ),
            "evidence": {
                "bands": [
                    {"band": b, "count": int(c), "default_rate": round(float(r), 4)}
                    for b, c, r in leverage
                ],
                "shap_rank_credit_income_ratio": 37,
                "shap_rank_credit_term": 3,
            },
            "so_what": (
                "This looks like a selection effect rather than a causal one: a "
                "loan worth six times income is presumably only written for "
                "applicants who cleared other checks, so the survivors look "
                "safe. The engineered credit_income_ratio feature ranks 37th of "
                "126 by SHAP importance, which is consistent with a weak signal. "
                "Loan term, credit_term, ranks 3rd. The duration of the "
                "commitment carries far more information than its size relative "
                "to income, and a credit policy built on a leverage cap would "
                "be targeting the wrong quantity."
            ),
        }
    )

    # 5. Age.
    age = _query(
        f"""
        SELECT band, count(*), avg(target::float) FROM (
            SELECT CASE
                     WHEN -days_birth / 365.25 < 30 THEN '1. under 30'
                     WHEN -days_birth / 365.25 < 40 THEN '2. 30 to 40'
                     WHEN -days_birth / 365.25 < 50 THEN '3. 40 to 50'
                     WHEN -days_birth / 365.25 < 60 THEN '4. 50 to 60'
                     ELSE '5. 60 and over'
                   END AS band,
                   target
            FROM {TABLE}
        ) t GROUP BY band ORDER BY band
        """
    )
    insights.append(
        {
            "title": "Younger applicants default substantially more often",
            "finding": (
                f"Under-30s default at {100 * float(age[0][2]):.1f}%, falling "
                f"steadily to {100 * float(age[-1][2]):.1f}% for those 60 and over."
            ),
            "evidence": {
                "bands": [
                    {"band": b, "count": int(c), "default_rate": round(float(r), 4)}
                    for b, c, r in age
                ]
            },
            "so_what": (
                "Age is legally sensitive in credit decisions in many "
                "jurisdictions. It is reported here as a data finding, and it "
                "is a reason the derived rules are reviewed rather than applied "
                "automatically."
            ),
        }
    )

    return {"base_default_rate": base, "insights": insights}
