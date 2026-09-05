"""Exploratory data analysis for Riskwright.

Exported from eda.ipynb. Both files are kept because the assignment asks for
the notebook and its .py conversion.

Every figure here is computed by the same functions the API serves from
(`src.data.eda`), so the notebook, the UI, and the README cannot quote
different numbers for the same finding. That is the point of importing rather
than reimplementing: a notebook that recomputes its own statistics is a second
source of truth waiting to disagree with the first.

Run against a populated Postgres:

    POSTGRES_HOST=localhost python notebooks/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the file directly from the notebooks/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.eda import (  # noqa: E402
    business_insights,
    dataset_summary,
    distribution,
    group_default_rate,
)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Dataset summary
# ---------------------------------------------------------------------------

def summarise() -> dict:
    section("1. Dataset summary")
    summary = dataset_summary()

    print(f"Table            {summary['table']}")
    print(f"Rows             {summary['rows']:,}")
    print(f"Columns          {summary['columns']}")
    print(f"Defaulted        {summary['defaulted']:,}")
    print(f"Repaid           {summary['repaid']:,}")
    print(f"Default rate     {summary['default_rate']:.2%}")
    print(f"Imbalance ratio  {summary['imbalance_ratio']:.2f} negatives per positive")
    print()
    print(
        "The imbalance is the single most important property of this target. "
        "A model predicting 'never defaults' scores 92% accuracy, which is why "
        "accuracy is not reported anywhere in this project."
    )
    return summary


# ---------------------------------------------------------------------------
# 2. Data quality
# ---------------------------------------------------------------------------

def data_quality(summary: dict) -> None:
    section("2. Data quality and missing values")

    print(
        f"{summary['columns_with_missing']} of {summary['columns']} columns "
        f"contain at least one missing value.\n"
    )
    frame = pd.DataFrame(summary["missing_top"])
    print(frame.head(15).to_string(index=False))
    print()
    print(
        "The heavily missing columns are the building-level survey fields "
        "(commonarea, floorsmin, livingapartments and similar), all around 50 "
        "to 70% missing. They are left in: LightGBM handles missing values "
        "natively, and dropping them would discard signal on the applicants "
        "who do have them."
    )
    print()
    print(
        "The more dangerous problem is not missingness but a sentinel value "
        "masquerading as data. See insight 1 below."
    )


# ---------------------------------------------------------------------------
# 3. Feature categorization
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "Identity and target": ["sk_id_curr", "target"],
    "Demographics": [
        "code_gender", "days_birth", "cnt_children", "cnt_fam_members",
        "name_family_status", "name_education_type", "name_housing_type",
    ],
    "Employment": [
        "days_employed", "occupation_type", "organization_type",
        "name_income_type", "amt_income_total",
    ],
    "Loan terms": [
        "name_contract_type", "amt_credit", "amt_annuity", "amt_goods_price",
    ],
    "External credit signals": ["ext_source_1", "ext_source_2", "ext_source_3"],
    "Assets": ["flag_own_car", "own_car_age", "flag_own_realty"],
    "Contactability": [
        "flag_phone", "flag_email", "days_last_phone_change", "days_id_publish",
    ],
    "Region": [
        "region_rating_client", "region_population_relative",
        "reg_city_not_work_city",
    ],
    "Social circle": [
        "obs_30_cnt_social_circle", "def_30_cnt_social_circle",
    ],
    "Bureau enquiry counts": ["amt_req_credit_bureau_year"],
}


def categorize_features() -> None:
    section("3. Feature categorization")
    for group, columns in FEATURE_GROUPS.items():
        print(f"{group:28} {', '.join(columns)}")
    print()
    print(
        "The remaining roughly 60 columns are normalised building statistics "
        "and document flags. They are retained for the model but are not "
        "exposed to the chatbot, because no business user asks about "
        "nonlivingapartments_medi and every extra column is another chance for "
        "the model to choose a plausible but wrong one."
    )


# ---------------------------------------------------------------------------
# 4. Business insights
# ---------------------------------------------------------------------------

def insights() -> None:
    section("4. Five business insights")
    payload = business_insights()
    for index, insight in enumerate(payload["insights"], 1):
        print(f"\n--- {index}. {insight['title']} ---")
        print(insight["finding"])
        print(f"\nSo what: {insight['so_what']}")

        evidence = insight.get("evidence", {})
        for key in ("bands", "groups", "quartiles"):
            if key in evidence:
                print()
                print(pd.DataFrame(evidence[key]).to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Supporting charts
# ---------------------------------------------------------------------------

def charts() -> None:
    section("5. Supporting distributions")

    for column in ("age_years", "ext_source_3", "credit_income_ratio"):
        data = distribution(column, bins=10)
        frame = pd.DataFrame(data["bins"])[
            ["bin_start", "bin_end", "total", "default_rate"]
        ]
        print(f"\n{column}  (range {data['range'][0]} to {data['range'][1]})")
        print(frame.to_string(index=False))

    for dimension in ("name_education_type", "name_income_type"):
        grouped = group_default_rate(dimension)
        print(f"\nDefault rate by {dimension}")
        print(pd.DataFrame(grouped["groups"]).to_string(index=False))

    print()
    print(
        "In the notebook these render as Plotly figures. Printed here as "
        "tables so the .py export stays runnable in a terminal and produces "
        "the same numbers."
    )


def main() -> None:
    summary = summarise()
    data_quality(summary)
    categorize_features()
    insights()
    charts()

    section("Conclusions carried into the model")
    print(
        "1. days_employed needs the sentinel replaced before use, and the fact "
        "of the sentinel kept as a flag.\n"
        "2. The external credit scores dominate. They drive the model and the "
        "derived rules, and their absence is the main coverage risk.\n"
        "3. The target is imbalanced 11.4 to 1, so scale_pos_weight is used and "
        "PR-AUC is reported alongside ROC-AUC.\n"
        "4. Leverage does not behave as intuition suggests, so a leverage cap "
        "would be the wrong policy lever. Loan term carries more signal.\n"
        "5. Age and education both separate risk strongly and are both legally "
        "sensitive, which is why derived rules are presented for review rather "
        "than applied automatically."
    )


if __name__ == "__main__":
    main()
