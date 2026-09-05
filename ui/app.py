"""Riskwright UI.

Five sections covering the platform end to end. Every value on screen arrived
over HTTP from the API; nothing is computed here. That is what keeps this
layer replaceable and what stops business logic leaking into presentation.

Charts are Plotly, never st.pyplot, because a server-rendered figure would put
presentation logic behind the API.
"""

from __future__ import annotations

import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.api_client import ApiError, base_url, get, post

st.set_page_config(page_title="Riskwright", layout="wide", initial_sidebar_state="expanded")

# Restrained palette. Red reads as "risk" and is reserved for it.
RISK_COLOURS = {"Low": "#2e7d32", "Medium": "#ef6c00", "High": "#c62828"}
REPAID = "#4c78a8"
DEFAULTED = "#c62828"


st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .risk-banner {padding: 1rem 1.2rem; border-radius: 8px; color: #fff;
                    font-size: 1.05rem; margin-bottom: 0.8rem;}
      .muted {color: #6b6b6b; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600)
def cached_get(path: str, params: dict | None = None):
    return get(path, params)


def guard(fn, *args, **kwargs):
    """Render an API failure as a message rather than a traceback."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        st.error(str(exc))
        st.caption(f"The UI is configured to call {base_url()}")
        return None


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def section_overview() -> None:
    st.header("Data understanding")

    summary = guard(cached_get, "/eda/summary")
    if not summary:
        return

    a, b, c, d = st.columns(4)
    a.metric("Applications", f"{summary['rows']:,}")
    b.metric("Features", summary["columns"])
    c.metric("Default rate", f"{summary['default_rate']:.2%}")
    d.metric("Imbalance ratio", f"{summary['imbalance_ratio']:.1f} to 1")
    st.caption(
        f"{summary['defaulted']:,} defaulted, {summary['repaid']:,} repaid. "
        f"{summary['columns_with_missing']} of {summary['columns']} columns "
        "contain missing values."
    )

    st.subheader("Business insights")
    payload = guard(cached_get, "/eda/insights")
    if payload:
        for index, insight in enumerate(payload["insights"], 1):
            with st.expander(f"{index}. {insight['title']}", expanded=index == 1):
                st.write(insight["finding"])
                st.markdown(f"**So what:** {insight['so_what']}")
                evidence = insight.get("evidence", {})
                for key in ("bands", "groups", "quartiles"):
                    if key in evidence:
                        frame = pd.DataFrame(evidence[key])
                        label = frame.columns[0]
                        fig = px.bar(
                            frame,
                            x=label,
                            y="default_rate",
                            labels={"default_rate": "Default rate"},
                        )
                        fig.add_hline(
                            y=payload["base_default_rate"],
                            line_dash="dot",
                            annotation_text="portfolio average",
                        )
                        fig.update_layout(height=320, margin=dict(t=30, b=10))
                        fig.update_traces(marker_color=REPAID)
                        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore a distribution")
    options = guard(cached_get, "/eda/columns")
    if not options:
        return

    left, right = st.columns([2, 1])
    column = left.selectbox("Column", options["numeric"], index=options["numeric"].index("age_years") if "age_years" in options["numeric"] else 0)
    bins = right.slider("Bins", 10, 60, 30, step=5)

    data = guard(cached_get, "/eda/distribution", {"column": column, "bins": bins})
    if data and data.get("bins"):
        frame = pd.DataFrame(data["bins"])
        figure = go.Figure()
        figure.add_bar(x=frame["bin_start"], y=frame["repaid"], name="Repaid", marker_color=REPAID)
        figure.add_bar(x=frame["bin_start"], y=frame["defaulted"], name="Defaulted", marker_color=DEFAULTED)
        figure.add_scatter(
            x=frame["bin_start"],
            y=frame["default_rate"],
            name="Default rate",
            yaxis="y2",
            line=dict(color="#000", width=2, dash="dot"),
        )
        figure.update_layout(
            barmode="stack",
            height=420,
            yaxis=dict(title="Applicants"),
            yaxis2=dict(title="Default rate", overlaying="y", side="right", tickformat=".0%"),
            margin=dict(t=30),
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(data.get("note", ""))

    st.subheader("Default rate by category")
    dimension = st.selectbox("Dimension", options["categorical"])
    grouped = guard(cached_get, "/eda/groupby", {"dimension": dimension})
    if grouped:
        frame = pd.DataFrame(grouped["groups"])
        fig = px.bar(frame, x="value", y="default_rate", hover_data=["count", "lift"])
        fig.add_hline(y=grouped["base_default_rate"], line_dash="dot", annotation_text="portfolio average")
        fig.update_traces(marker_color=REPAID)
        fig.update_layout(height=380, yaxis_tickformat=".1%", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)


def _risk_banner(result: dict) -> None:
    band = result["risk_band"]
    st.markdown(
        f"<div class='risk-banner' style='background:{RISK_COLOURS[band]}'>"
        f"<strong>{band} risk</strong> &nbsp;|&nbsp; "
        f"{result['default_probability']:.2%} estimated probability of default"
        f"<br><span style='opacity:.9'>{result['recommended_action']}</span></div>",
        unsafe_allow_html=True,
    )
    thresholds = result["thresholds"]
    st.caption(
        f"Bands: Low below {thresholds['t_low']:.2%}, "
        f"High at or above {thresholds['t_high']:.2%}. "
        "The upper boundary is the cost-optimal threshold; the lower one is a "
        "business judgment."
    )


def section_prediction() -> None:
    st.header("Risk prediction")

    mode = st.radio(
        "Score",
        ["An existing applicant", "A what-if applicant"],
        horizontal=True,
    )

    if mode == "An existing applicant":
        listing = guard(cached_get, "/applicants", {"limit": 50})
        if not listing:
            return
        ids = [row["sk_id_curr"] for row in listing["applicants"]]
        chosen = st.selectbox("Applicant", ids)

        row = next(r for r in listing["applicants"] if r["sk_id_curr"] == chosen)
        a, b, c = st.columns(3)
        a.metric("Age", f"{row['age_years']:.0f}")
        b.metric("Income", f"{row['amt_income_total']:,.0f}")
        c.metric("Loan", f"{row['amt_credit']:,.0f}")

        result = guard(post, "/predict", {"sk_id_curr": int(chosen)})
        if result:
            _risk_banner(result)
            st.session_state["last_scored"] = {"sk_id_curr": int(chosen)}
    else:
        st.caption(
            "Unspecified fields are left missing rather than guessed. LightGBM "
            "handles missing values natively, so the score reflects only what "
            "was supplied."
        )
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Annual income", 25_000, 2_000_000, 150_000, step=10_000)
        credit = c2.number_input("Loan amount", 45_000, 4_000_000, 500_000, step=10_000)
        annuity = c3.number_input("Annual repayment", 2_000, 300_000, 25_000, step=1_000)
        c4, c5, c6 = st.columns(3)
        age = c4.slider("Age", 21, 70, 35)
        employed = c5.slider("Years employed", 0, 45, 5)
        ext = c6.slider("External credit score", 0.0, 1.0, 0.5, step=0.05)

        features = {
            "amt_income_total": income,
            "amt_credit": credit,
            "amt_annuity": annuity,
            "days_birth": -int(age * 365.25),
            "days_employed": -int(employed * 365.25),
            "ext_source_1": ext,
            "ext_source_2": ext,
            "ext_source_3": ext,
        }
        if st.button("Score this applicant", type="primary"):
            result = guard(post, "/predict", {"features": features})
            if result:
                _risk_banner(result)
                st.session_state["last_scored"] = {"features": features}


def section_explainability() -> None:
    st.header("Why this decision")

    payload = st.session_state.get("last_scored")
    if not payload:
        st.info("Score an applicant on the Risk prediction page first.")
        return

    result = guard(post, "/explain", payload)
    if not result:
        return

    st.markdown(f"### {result['risk_band']} risk, {result['default_probability']:.2%}")
    st.write(result["narrative"])

    frame = pd.DataFrame(result["contributions"])
    frame = frame.sort_values("shap_value")
    figure = go.Figure(
        go.Bar(
            x=frame["shap_value"],
            y=frame["label"],
            orientation="h",
            marker_color=[DEFAULTED if v > 0 else REPAID for v in frame["shap_value"]],
            hovertext=[f"value: {v}" for v in frame["value"]],
        )
    )
    figure.update_layout(
        height=520,
        xaxis_title="Pushes toward default  ->",
        margin=dict(l=10, t=30),
    )
    figure.add_vline(x=0, line_width=1, line_color="#333")
    st.plotly_chart(figure, use_container_width=True)
    st.caption(result["note"])

    with st.expander("Which features matter across all applicants"):
        overall = guard(cached_get, "/explain/global")
        if overall:
            frame = pd.DataFrame(overall["features"][:15]).sort_values("mean_abs_shap")
            fig = px.bar(frame, x="mean_abs_shap", y="label", orientation="h")
            fig.update_traces(marker_color=REPAID)
            fig.update_layout(height=460, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Mean absolute SHAP over a sample of {overall['sampled_rows']:,} "
                f"of {overall['total_rows']:,} applicants. Sampled, not exhaustive."
            )


def section_rules() -> None:
    st.header("Derived credit rules")
    st.caption(
        "A depth-3 surrogate tree fitted to the model's behaviour, so its "
        "logic can be read as policy. These approximate the model; they are "
        "not the model."
    )

    which = st.radio(
        "Rule set",
        ["Faithful to the model", "Policy usable, external scores excluded"],
        horizontal=True,
    )
    path = "/rules" if which.startswith("Faithful") else "/rules?variant=policy"

    payload = guard(cached_get, path)
    if not payload:
        return

    fidelity = payload.get("surrogate_fidelity")
    a, b = st.columns(2)
    a.metric("Agreement with the model", f"{fidelity:.1%}" if fidelity else "n/a")
    b.metric("Base default rate", f"{payload['base_default_rate']:.2%}")
    if not which.startswith("Faithful"):
        st.caption(
            "External bureau scores dominate the model, so the faithful rules "
            "key off them almost entirely and tell a credit officer little they "
            "can act on. This set excludes them, trading agreement with the "
            "model for rules that apply to applicants with no bureau history."
        )

    frame = pd.DataFrame(payload["rules"])
    display = frame[["rule_id", "readable", "support_pct", "default_rate", "lift"]].copy()
    display.columns = ["Rule", "Condition", "Share of applicants", "Default rate", "Lift"]
    st.dataframe(
        display.style.format(
            {"Share of applicants": "{:.1f}%", "Default rate": "{:.1%}", "Lift": "{:.2f}x"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def section_chat() -> None:
    st.header("Ask the data")

    if "session_id" not in st.session_state:
        # The identifier lives here; the history itself lives server-side, so
        # replacing this UI would not mean reimplementing memory.
        st.session_state["session_id"] = f"ui-{uuid.uuid4().hex[:12]}"
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    with st.sidebar:
        st.caption(f"Session {st.session_state['session_id'][-8:]}")
        if st.button("Clear conversation"):
            st.session_state["chat"] = []
            st.rerun()

    st.caption(
        "Questions become SQL, which is validated against the live schema "
        "before it runs. Questions the data cannot answer are declined rather "
        "than guessed at."
    )

    for entry in st.session_state["chat"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("sql"):
                with st.expander("SQL"):
                    st.code(entry["sql"], language="sql")
            if entry.get("table") is not None:
                st.dataframe(entry["table"], use_container_width=True, hide_index=True)

    question = st.chat_input("How does default rate vary by education level?")
    if not question:
        return

    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Writing and checking SQL"):
            try:
                result = post(
                    "/chat",
                    {"question": question, "session_id": st.session_state["session_id"]},
                    timeout=90,
                )
            except ApiError as exc:
                st.error(str(exc))
                return

        action = result["action"]
        if action == "sql":
            body = result.get("answer") or "The query ran."
            st.markdown(body)
            table = pd.DataFrame(result["rows"], columns=result["columns"]) if result["rows"] else None
            if table is not None and not table.empty:
                st.dataframe(table, use_container_width=True, hide_index=True)
            with st.expander("SQL"):
                st.code(result["sql"], language="sql")
            st.session_state["chat"].append(
                {"role": "assistant", "content": body, "sql": result["sql"], "table": table}
            )
        else:
            # Refusals are rendered distinctly. Making them visible is the
            # point: an honest decline is the correct answer to a question the
            # data cannot support.
            label = "Cannot answer this" if action == "refuse" else "Needs clarification"
            body = f"**{label}.** {result.get('reason') or ''}"
            if result.get("suggestion"):
                body += f"\n\n*Try instead:* {result['suggestion']}"
            st.warning(body)
            st.session_state["chat"].append({"role": "assistant", "content": body})


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------

SECTIONS = {
    "Data understanding": section_overview,
    "Risk prediction": section_prediction,
    "Why this decision": section_explainability,
    "Derived rules": section_rules,
    "Ask the data": section_chat,
}

st.sidebar.title("Riskwright")
st.sidebar.caption("Credit risk scoring, explained")
choice = st.sidebar.radio("Section", list(SECTIONS), label_visibility="collapsed")

try:
    status = get("/health", timeout=10)
    st.sidebar.divider()
    st.sidebar.caption(
        f"Database {'connected' if status['database']['connected'] else 'unavailable'}\n\n"
        f"Model {'loaded' if status['model_loaded'] else 'not trained'}\n\n"
        f"LLM {status.get('llm_model') or 'not configured'}"
    )
except ApiError as exc:
    st.sidebar.error("API unreachable")
    st.error(str(exc))
    st.stop()

SECTIONS[choice]()
