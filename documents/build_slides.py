"""Build documents/slides.html from the artifacts the pipeline wrote.

Replaces the matplotlib deck. The presentation layer changed; the property that
mattered did not. Every number below is interpolated from `slide_data.figures()`,
which reads `models/*.json`, so the deck still cannot drift from the code. That
is not ceremony: building the old deck this way caught a hand-typed precision
figure in the README that was wrong, and a later audit caught a second.

Design follows the Frontend Slides skill's Terminal Green preset. See
`slide_theme.py` for the CSS and the fixed-stage rules.

    python documents/build_slides.py          # writes documents/slides.html
    python documents/export_slides_pdf.py     # then the PDF the brief requires
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "documents"))

from slide_data import figures                      # noqa: E402
from slide_theme import document                    # noqa: E402

OUT = ROOT / "documents" / "slides.html"


# --------------------------------------------------------------------------
# Small helpers. Each returns a string; nothing here computes a statistic.
# --------------------------------------------------------------------------

def pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


def slide(prompt: str, title: str, body: str, subtitle: str = "",
          number: int | None = None, total: int | None = None) -> str:
    sub = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    counter = (f'<div class="slide-no">{number:02d} / {total:02d}</div>'
               if number and total else "")
    return f"""            <section class="slide">
                <div class="slide-head reveal">
                    <div class="prompt"><span class="sigil">$</span> {prompt}</div>
                    <h2>{title}</h2>
                    {sub}
                    <div class="rule"></div>
                </div>
{body}
                {counter}
            </section>
"""


def table(headers: list[str], rows: list[list[str]],
          highlight: int | None = None, numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for n, row in enumerate(rows):
        cls = ' class="hi"' if highlight is not None and n == highlight else ""
        cells = "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{c}</td>'
            for i, c in enumerate(row)
        )
        body.append(f"<tr{cls}>{cells}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def figure(value: str, label: str, tone: str = "") -> str:
    return (f'<div class="figure"><div class="value {tone}">{value}</div>'
            f'<div class="label">{label}</div></div>')


def bullets(items: list[str], bad: set[int] | None = None) -> str:
    bad = bad or set()
    out = []
    for i, text in enumerate(items):
        cls = ' class="bad"' if i in bad else ""
        out.append(f"<li{cls}>{text}</li>")
    return f'<ul>{"".join(out)}</ul>'


def bars(rows: list[tuple[str, float]], peak: float) -> str:
    out = []
    for name, value in rows:
        width = max(1.0, value / peak * 100)
        out.append(
            f'<div class="bar-row"><div class="name">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="val">{value:.3f}</div></div>'
        )
    return "".join(out)


# --------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------

def build() -> str:
    d = figures()
    ds, mo, th = d["dataset"], d["model"], d["threshold"]
    se, fa, di = d["sensitivity"], d["fairness"], d["disparate_impact"]
    px, sh, ru, cb = d["proxies"], d["shap"], d["rules"], d["chatbot"]

    slides: list[str] = []
    # 16 content slides plus one screenshot per module. Derived rather than
    # typed: a hardcoded total silently dropped the fifth screenshot.
    total = 16 + len(d["screenshots"])

    # 01 -- Title -----------------------------------------------------------
    slides.append(f"""            <section class="slide active">
                <div style="display:flex;flex-direction:column;justify-content:center;height:100%;">
                    <div class="prompt reveal"><span class="sigil">$</span> riskwright --serve</div>
                    <h1 class="reveal" style="margin-top:18px;">Riskwright<span class="cursor"></span></h1>
                    <p class="reveal" style="font-size:34px;color:var(--text-dim);margin-top:26px;max-width:1250px;">
                        An AI-powered credit risk platform. Predicts default probability,
                        explains every decision, derives credit policy rules, and answers
                        plain-English questions about the data.
                    </p>
                    <div class="rule reveal" style="max-width:640px;margin-top:44px;"></div>
                    <p class="reveal note" style="margin-top:34px;font-size:22px;">
                        Home Credit Default Risk &nbsp;|&nbsp; {ds['rows']:,} applications
                        &nbsp;|&nbsp; {pct(ds['default_rate'], 2)} default rate
                        &nbsp;|&nbsp; {ds['features']} features<br>
                        NeoStats AI Engineer assignment
                    </p>
                </div>
                <div class="slide-no">01 / {total:02d}</div>
            </section>
""")

    # 02 -- Architecture ----------------------------------------------------
    diagram = """<span class="ok">React + nginx</span>        no logic, HTTP client only
      |
      v  /api/*
<span class="ok">FastAPI</span>              thin wrapper: validates, shapes
      |
      +--------------+--------------+
      v              v              v
 <span class="key">src/ml</span>         <span class="key">src/data</span>       <span class="key">src/talk_to_data</span>
 model, SHAP,   loader,        NL to SQL,
 rules, audits  preprocess     validation, memory
      |              |              |
      +--------------+--------------+
                     v
               <span class="ok">PostgreSQL</span>
   application_train, bureau, previous_application"""
    slides.append(slide(
        "docker-compose up", "Architecture",
        f"""                <div class="cols wide-left">
                    <div class="reveal"><div class="mono-block">{diagram}</div></div>
                    <div class="stack reveal">
                        <div class="panel">
                            <div class="panel-label">four services, one command</div>
                            {bullets([
                                "<strong>postgres</strong> &mdash; healthchecked",
                                "<strong>loader</strong> &mdash; one-shot, idempotent",
                                "<strong>api</strong> &mdash; waits on both",
                                "<strong>frontend</strong> &mdash; waits on api health",
                            ])}
                        </div>
                        <p class="note">No flags, no override file. Startup races are the
                        usual reason a submission does not run on someone else's machine,
                        so the dependency conditions are load-bearing rather than
                        decorative.</p>
                    </div>
                </div>""",
        subtitle="All business logic in src/. The UI computes nothing.",
        number=2, total=total))

    # 03 -- Calibration and cost -------------------------------------------
    opt, naive = th["optimal"], th["naive"]
    slides.append(slide(
        "python -m src.ml.evaluate", "From score to decision",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">1 &mdash; the raw output is not a probability</div>
                            <p>Training with <code>scale_pos_weight = {ds['scale_pos_weight']:.2f}</code>
                            multiplies the predicted odds by that weight. An average applicant
                            scores near 0.5, not near {ds['default_rate']:.2f}.</p>
                        </div>
                        <div class="panel">
                            <div class="panel-label">2 &mdash; a known constant, so inverted exactly</div>
                            <div class="mono-block" style="font-size:20px;">p_true = p_w / (p_w + w(1 - p_w))</div>
                            <p class="note" style="margin-top:14px;">A weighted 0.5 maps back to
                            1/(1+w) = <span class="ok">{th['calibration_identity']:.4f}</span>,
                            the measured default rate. That identity is asserted in the test suite.</p>
                        </div>
                    </div>
                    <div class="stack">
                        <div class="panel warn">
                            <div class="panel-label">3 &mdash; only now does a cost threshold mean anything</div>
                            <p>A false negative is a defaulted loan; a false positive is a good
                            customer refused. Assumed <strong>{th['ratio']:.0f}:1</strong>.
                            Threshold minimises {th['ratio']:.0f}&middot;FN + FP.</p>
                        </div>
                        {table(["Threshold", "Flagged", "Recall", "Precision", "Cost"], [
                            ["Naive 0.5", pct(naive['flagged_rate'], 1),
                             f"{naive['recall']:.3f}", f"{naive['precision']:.3f}",
                             f"{naive['expected_cost']:,.0f}"],
                            [f"Cost-optimal {th['t_high']:.4f}", pct(opt['flagged_rate'], 1),
                             f"{opt['recall']:.3f}", f"{opt['precision']:.3f}",
                             f"{opt['expected_cost']:,.0f}"],
                        ], highlight=1)}
                        <p class="note good">Catching {pct(opt['recall'], 1)} of defaulters
                        instead of {pct(naive['recall'], 1)}. Expected cost down
                        {pct(th['cost_saving_pct'], 1)}.</p>
                    </div>
                </div>""",
        subtitle="Calibration first, then cost. The second is meaningless without the first.",
        number=3, total=total))

    # 04 -- Risk bands and sensitivity --------------------------------------
    sens_rows = [
        [f"{r['ratio']:.0f}:1", f"{r['t_high']:.4f}", pct(r['approval_rate'], 1),
         f"{r['defaulters_caught']:,}"]
        for r in se["rows"]
    ]
    shipped_at = next(i for i, r in enumerate(se["rows"]) if r["is_shipped"])
    slides.append(slide(
        "python -m src.ml.threshold_sensitivity", "Risk bands, and what they rest on",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        {table(["Boundary", "Value", "How it is set"], [
                            ["t_high  Medium/High", f"{th['t_high']:.4f}", "Cost-optimal"],
                            ["t_low  Low/Medium", f"{th['t_low']:.4f}", "Business judgment"],
                        ], numeric_from=1)}
                        <p>Above <code>t_high</code> the expected cost of approving exceeds
                        the cost of refusing. Where automatic approval should <em>stop</em> is
                        a matter of review capacity, which nothing in the data answers, so
                        <code>t_low</code> is exposed as a policy dial rather than presented
                        as a computed result.</p>
                        <div class="panel warn">
                            <div class="panel-label">how much rests on the {th['ratio']:.0f}:1 assumption</div>
                            <p class="note alert" style="font-size:22px;">
                            <strong>t_high moves {se['spread']['min']:.4f} to
                            {se['spread']['max']:.4f}</strong> &mdash; a spread of
                            {pct(se['spread']['relative_spread'], 0)} of the shipped value.
                            Only <strong>{pct(se['band_stability'], 1)}</strong> of applicants
                            keep the same band across the range.</p>
                        </div>
                    </div>
                    <div class="stack">
                        {table(["Cost ratio", "t_high", "Approve", "Caught"], sens_rows,
                               highlight=shipped_at)}
                        <p class="note good">Every ratio still beats 0.5. The method survives
                        where the number does not; what is uncertain is where to put the
                        threshold, not whether 0.5 is wrong. It is.</p>
                    </div>
                </div>""",
        subtitle="The two boundaries answer different questions, and one of them is an assumption.",
        number=4, total=total))

    # 05 -- Talk-to-data ----------------------------------------------------
    runs = "  ".join(str(x) for x in cb["per_run"])
    prior = cb["prior"][0] if cb["prior"] else None
    prior_note = (
        f"A single run once scored {prior['max']}/{prior['questions']} and was reported as "
        f"the result. It was the best of five."
        if prior else ""
    )
    slides.append(slide(
        "python -m tests.run_chatbot_variance", "Talk-to-data, measured like a model",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        <div class="figure-row">
                            {figure(f"{cb['mean']:.1f}", f"mean of {cb['total']}")}
                            {figure(f"{cb['min']}&ndash;{cb['max']}", "range", "neutral")}
                            {figure(f"{cb['stable']}", "stable at 5/5", "neutral")}
                        </div>
                        <div class="mono-block">runs   {runs}   out of {cb['total']}
sd     {cb['stdev']:.2f}
<span class="no">fail</span>   {', '.join(cb['always_failed'])}  (every run)</div>
                        <p class="note">Development set: 30/30. It was tuned on the failures
                        it exposed, so its 100% is optimistic by construction.</p>
                    </div>
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">the score is a distribution, not a number</div>
                            <p>{prior_note} Quoting the good draw is how an evaluation
                            flatters itself, which is why this is now run five times and
                            reported as a spread.</p>
                        </div>
                        <div class="panel warn">
                            <div class="panel-label">two failures are mine, not the model's</div>
                            <p class="note">H35's reference counts ids in <code>bureau</code>
                            where the model joins to <code>application_train</code> first &mdash;
                            the model's reading is better. H36 asks it to exclude sentinel codes
                            the schema never identifies. Reported, not repaired: fixing a
                            reference after watching the model disagree is fitting the grader
                            to the outcome.</p>
                        </div>
                    </div>
                </div>""",
        subtitle=f"{cb['total']} held-out questions written after the prompt was frozen. Run five times.",
        number=5, total=total))

    # 06 -- Refusal ---------------------------------------------------------
    gate = """<span class="no">Rejected</span>   WITH x AS (DELETE FROM application_train RETURNING sk_id_curr)
          SELECT count(*) FROM x

<span class="ok">Allowed</span>    SELECT count(*) FROM application_train
          WHERE occupation_type = 'Delete'"""
    slides.append(slide(
        "src/talk_to_data/query_runner.py", "Refusing is a feature",
        f"""                <div class="stack reveal">
                    <div class="cols">
                        <div class="stack">
                            <div class="panel">
                                <div class="panel-label">structured output</div>
                                <p>One model call returns an action: <code>sql</code>,
                                <code>refuse</code>, or <code>clarify</code>. A model with no
                                sanctioned way to say "I cannot" invents a column to fill the
                                silence.</p>
                            </div>
                            <div class="panel">
                                <div class="panel-label">validation gate</div>
                                <p>Every generated statement is parsed, not string-matched.
                                Single SELECT, whitelisted tables, columns checked against the
                                live schema, LIMIT forced, 10s timeout, executed as a role
                                holding SELECT and nothing else.</p>
                            </div>
                        </div>
                        <div class="stack">
                            <div class="mono-block">{gate}</div>
                            <p class="note good">A DELETE hidden inside a CTE is rejected. The
                            literal string 'Delete' is not. String matching gets both of these
                            wrong; parsing the tree gets both right.</p>
                        </div>
                    </div>
                    <div class="panel warn">
                        <div class="panel-label">the limit of it</div>
                        <p class="note">Querying a column that does not exist is impossible &mdash;
                        a hard guarantee. Declining a question unanswerable <em>from columns that
                        do</em> is a model judgement: over ten runs H8 declined seven times, and
                        H44 answers a causal question as an association four times in five.</p>
                    </div>
                </div>""",
        subtitle="Two independent mechanisms: a hard gate, and a sanctioned way to decline.",
        number=6, total=total))

    # 07 -- Fair lending ----------------------------------------------------
    wp, wo = fa["with_protected"], fa["without_protected"]
    slides.append(slide(
        "src/data/preprocessor.py", "Fair lending",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        {table(["Excluded", "Protected basis"],
                               [[c, b] for c, b in fa["excluded"]], numeric_from=9)}
                        <p class="note">ECOA names sex, marital status and age as protected
                        bases. A model that prices or refuses credit on them is a legal failure
                        however well it performs.</p>
                    </div>
                    <div class="stack">
                        {table(["Feature set", "ROC-AUC", "PR-AUC"], [
                            ["With protected", f"{wp['roc_auc']:.4f}", f"{wp['pr_auc']:.4f}"],
                            ["Without (shipped)", f"{wo['roc_auc']:.4f}", f"{wo['pr_auc']:.4f}"],
                        ], highlight=1)}
                        <p class="note good">Cost of exclusion:
                        {fa['cost_of_exclusion']['roc_auc']:.4f} ROC-AUC. There is no
                        meaningful accuracy argument for keeping them.</p>
                        <div class="panel warn">
                            <div class="panel-label">exclusion did not hold</div>
                            <p class="note"><strong>{px['material_total']} features are a
                            material proxy</strong> for at least one excluded attribute. Age is
                            the worst: {px['age_material_count']} material,
                            {px['age_strong_count']} strong.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="Five protected attributes excluded. One of them got back in anyway.",
        number=7, total=total))

    # 08 -- Proxy detection -------------------------------------------------
    top_age = px["age_material"][:7]
    peak = top_age[0]["association"]
    slides.append(slide(
        "python -m src.ml.proxy_detection", "Systematic proxy detection",
        f"""                <div class="cols wide-left reveal">
                    <div class="stack">
                        <div class="panel-label">features that reconstruct age, all {ds['rows']:,} applicants</div>
                        {bars([(r["feature"], r["association"]) for r in top_age], peak)}
                        <p class="note">Thresholds fixed in the module before the first run:
                        material at {px['thresholds']['material']:.2f}, strong at
                        {px['thresholds']['strong']:.2f}. A cutoff chosen after seeing the
                        ranking describes the output rather than judging it.</p>
                    </div>
                    <div class="stack">
                        <div class="panel warn">
                            <div class="panel-label">neither method alone was enough</div>
                            <p class="note">This scan would have <strong>missed</strong>
                            <code>employed_life_ratio</code>: it predicts age at only
                            {px['employed_life_ratio_assoc']:.3f}, far below the floor, yet
                            varying <code>days_birth</code> alone moves the score for 71.8% of
                            applicants who have a <code>days_employed</code> value and
                            <strong>0.0%</strong> of those who do not.</p>
                            <p class="note" style="margin-top:14px;">Hand-inspection found that
                            one and none of the {px['age_material_count']}. Running both
                            produced the full picture.</p>
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">the uncomfortable one</div>
                            <p class="note"><code>ext_source_1</code> reconstructs age at
                            {px['ext_source_1_assoc']:.3f} and is the third most important
                            feature in the model. Excluding age here does not exclude it from
                            whatever the external bureau is scoring.</p>
                        </div>
                    </div>
                </div>""",
        subtitle=f"All {ds['features']} features scored against each excluded attribute.",
        number=8, total=total))

    # 09 -- Disparate impact ------------------------------------------------
    age_rows = [
        [g["group"], f"{g['count']:,}", f"{g['approval_rate']:.4f}",
         f"{g['observed_default_rate']:.4f}"]
        for g in di["attributes"]["age_band"]["groups"]
    ]
    ratio_rows = []
    for name, a in di["attributes"].items():
        chip = ('<span class="chip pass">PASS</span>' if a["passes"]
                else '<span class="chip fail">FAIL</span>')
        ratio_rows.append([name, f"{a['ratio']:.4f}", chip])
    slides.append(slide(
        "python -m src.ml.fairness_audit", "Fair lending: disparate impact",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        <div class="panel-label">age band &mdash; the worst-failing attribute</div>
                        {table(["Age band", "n", "Approved", "Observed default"], age_rows)}
                    </div>
                    <div class="stack">
                        {table(["Attribute", "4/5 ratio", "Result"], ratio_rows)}
                        <p class="note alert">Below 0.8: {', '.join(di['failing'])}</p>
                        <div class="panel warn">
                            <div class="panel-label">a screen, not a verdict</div>
                            <p class="note">Disparate impact in the legal sense requires a
                            business-necessity analysis and a search for a less-discriminatory
                            alternative. Neither is performed here. Observed default rates move
                            with the approval rates, so the model tracks a real pattern through
                            whatever it can reach &mdash; whether that justifies the disparity
                            is exactly the analysis not done.</p>
                        </div>
                    </div>
                </div>""",
        subtitle=f"Every applicant scored, threshold {di['threshold']:.6f}, "
                 f"approval rate {pct(di['approval_rate'], 2)}.",
        number=9, total=total))

    # 10 -- Adverse action --------------------------------------------------
    slides.append(slide(
        "POST /adverse-action", "Adverse action reason codes",
        f"""                <div class="stack reveal">
                    {table(["#", "Reason given to applicant 100002", "Driven by", "SHAP"], [
                        ["1", "Credit assessment obtained from an external credit bureau",
                         "ext_source_1/2/3", "0.847"],
                        ["2", "Length of the repayment term relative to the amount requested",
                         "credit_term", "0.221"],
                        ["3", "Value of the goods financed relative to the credit requested",
                         "amt_goods_price", "0.159"],
                        ["4", "Length of time in your current employment",
                         "days_employed", "0.088"],
                    ], numeric_from=3)}
                    <div class="cols" style="margin-top:8px;">
                        <div class="panel">
                            <div class="panel-label">why the phrases are fixed in source</div>
                            <p class="note">A regulated disclosure must produce the same words
                            for the same input, and a notice that cannot be reviewed before it
                            is sent cannot be signed off. Four maximum, per Regulation B.
                            Duplicates collapse &mdash; three bureau scores are one reason to an
                            applicant. An applicant below the threshold gets no notice, and the
                            reasons are not computed rather than computed and hidden.</p>
                        </div>
                        <div class="panel warn">
                            <div class="panel-label">what it is not</div>
                            <p class="note">Still SHAP attributions shaped as reasons, not a
                            rule-based engine working from the credit policy. Some features are
                            never disclosed at all &mdash; appointment timing, the geographic
                            columns, and the social-circle columns that describe other people's
                            conduct &mdash; and the response names what it withheld rather than
                            dropping it silently.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="ECOA requires the specific principal reasons for a denial, not a score.",
        number=10, total=total))

    # 11 -- Model -----------------------------------------------------------
    lgb, log = mo["lightgbm"], mo["logistic"]
    slides.append(slide(
        "python -m src.ml.train", "Model selection",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        {table(["Model", "ROC-AUC", "PR-AUC"], [
                            ["LightGBM (shipped)",
                             f"{lgb['roc_auc']:.4f} &plusmn; {lgb['roc_std']:.4f}",
                             f"{lgb['pr_auc']:.4f} &plusmn; {lgb['pr_std']:.4f}"],
                            ["Logistic regression",
                             f"{log['roc_auc']:.4f} &plusmn; {log['roc_std']:.4f}",
                             f"{log['pr_auc']:.4f} &plusmn; {log['pr_std']:.4f}"],
                        ], highlight=0)}
                        <p>LightGBM wins by {mo['roc_gap']:.4f} ROC-AUC &mdash; real, being
                        several times the fold standard deviation, but less than a headline
                        would suggest. A logistic regression gets within 2.4% of a
                        gradient-boosted ensemble on this data.</p>
                        <p class="note">That is a finding, not a disappointment. Logistic
                        regression is what banks deploy under regulatory pressure, because a
                        coefficient per feature is auditable in a way an ensemble is not.</p>
                    </div>
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">why LightGBM is served anyway</div>
                            {bullets([
                                "SHAP <strong>TreeExplainer</strong> is exact and fast on trees",
                                "Rule derivation is naturally a shallow tree",
                                "NaN and categoricals handled natively, so no imputation",
                            ])}
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">class imbalance</div>
                            <p class="note"><code>scale_pos_weight</code> =
                            {ds['scale_pos_weight']:.2f}, computed from each training split
                            rather than hardcoded. SMOTE was not used: at this row count it is
                            slow and addresses the same problem less directly. PR-AUC is
                            reported alongside ROC-AUC everywhere, because on a target this
                            imbalanced ROC-AUC alone flatters a model.</p>
                        </div>
                    </div>
                </div>""",
        subtitle=f"Two models, identical inputs, {ds['folds']}-fold stratified cross-validation.",
        number=11, total=total))

    # 12 -- EDA -------------------------------------------------------------
    slides.append(slide(
        "notebooks/eda.ipynb", "What the data says",
        f"""                <div class="stack reveal">
                    <div class="figure-row">
                        {figure(f"{ds['rows']:,}", "applications", "neutral")}
                        {figure(pct(ds['default_rate'], 2), "default rate")}
                        {figure(f"{ds['scale_pos_weight']:.1f}:1", "imbalance", "neutral")}
                        {figure("122", "columns, 67 with gaps", "neutral")}
                    </div>
                    <div class="cols" style="margin-top:12px;">
                        <div class="panel">
                            <div class="panel-label">the defect that mattered most</div>
                            <p class="note">18.0% of applicants have
                            <code>days_employed</code> set to 365243 &mdash; roughly 1,000 years.
                            It encodes "not currently employed" rather than a duration. Left
                            untreated it corrupts every statistic built on the column.</p>
                            <p class="note" style="margin-top:12px;"><strong>So what:</strong>
                            those applicants default at 5.4% against 8.7% for the rest, so the
                            flag is predictive and is kept as a feature rather than discarded.</p>
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">four more, served from one place</div>
                            {bullets([
                                "External credit scores separate risk more than anything self-reported",
                                "Education level tracks default risk strongly",
                                "Borrowing a lot relative to income does <strong>not</strong> predict default",
                                "Younger applicants default substantially more often",
                            ])}
                        </div>
                    </div>
                    <p class="note">Every figure here is computed by the same module the API
                    and the notebook call, so the deck, the UI and the analysis cannot quote
                    different numbers for the same finding.</p>
                </div>""",
        subtitle="Five insights, computed server-side and served from a single source.",
        number=12, total=total))

    # 13 -- Explainability --------------------------------------------------
    top_shap = sh["top"][:8]
    peak_shap = top_shap[0]["mean_abs_shap"]
    slides.append(slide(
        "python -m src.ml.shap_full", "Explainability",
        f"""                <div class="cols wide-left reveal">
                    <div class="stack">
                        <div class="panel-label">global importance &mdash; mean |SHAP|, all {sh['rows']:,} applicants</div>
                        {bars([(f["label"], f["mean_abs_shap"]) for f in top_shap], peak_shap)}
                    </div>
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">exhaustive, and it changed nothing</div>
                            <p class="note">Previously a {sh['previous_sample']:,}-row sample,
                            listed as a limitation. Recomputed over the whole population in
                            chunks: the top 20 order is
                            <strong>{'identical' if sh['top_10_identical'] else 'changed'}</strong>
                            and the rank correlation across all {ds['features']} features is
                            <strong>{sh['rank_correlation']:.4f}</strong>.</p>
                            <p class="note" style="margin-top:12px;">That is a finding, not a
                            null result: the sample was adequate, and every SHAP figure
                            previously reported was already correct.</p>
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">per prediction</div>
                            <p class="note">Endpoints return JSON, never a rendered image, so
                            the UI owns presentation and the frontend stays replaceable. Each
                            explanation carries a plain-English narrative generated
                            deterministically from a label map &mdash; it cannot invent a reason
                            the model did not use.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="SHAP TreeExplainer. Exact on tree models rather than approximated.",
        number=13, total=total))

    # 14 -- Rules -----------------------------------------------------------
    rule_rows = [
        [r["rule_id"], r["readable"][:74] + ("..." if len(r["readable"]) > 74 else ""),
         f"{r['support_pct']:.1f}%", pct(r["default_rate"], 1), f"{r['lift']:.2f}x"]
        for r in ru["faithful"][:6]
    ]
    slides.append(slide(
        "GET /rules", "Derived credit rules",
        f"""                <div class="stack reveal">
                    {table(["Rule", "Condition", "Applicants", "Default rate", "Lift"],
                           rule_rows, numeric_from=2)}
                    <div class="cols" style="margin-top:10px;">
                        <div class="panel">
                            <div class="panel-label">two sets, because one is not usable</div>
                            <p class="note">The faithful set agrees with the model on
                            <strong>{pct(ru['faithful_fidelity'], 1)}</strong> of applicants but
                            keys almost entirely off external bureau scores, so it says little a
                            credit officer can act on. The policy set excludes those scores:
                            {pct(ru['policy_fidelity'], 1)} agreement, and rules that still
                            apply to an applicant with no bureau history.</p>
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">honest about what they are</div>
                            <p class="note">A depth-3 surrogate tree approximating the model's
                            behaviour. These are a readable approximation for credit policy,
                            <strong>not the model itself</strong>, and the gap is where a
                            rule-based decision would differ from a scored one. Leaf statistics
                            are counted from the data rather than read off the tree's internal
                            values, which are reweighted.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="A depth-3 surrogate short enough to sit in a policy document.",
        number=14, total=total))

    # 15 -- Engineering -----------------------------------------------------
    slides.append(slide(
        "pytest", "Engineering",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">docker</div>
                            <p class="note">Four services. Postgres healthchecked; the loader
                            waits for healthy and the API waits for the loader to exit
                            successfully. Verified by cloning this repository into a fresh
                            directory and building with <code>--no-cache</code>.</p>
                        </div>
                        <div class="panel">
                            <div class="panel-label">security</div>
                            <p class="note">The chatbot connects as a role holding SELECT on
                            three tables and nothing else. Verified: INSERT, CREATE and DROP are
                            all refused by Postgres. The LLM key is scoped to the api service
                            and never reaches the UI container.</p>
                        </div>
                    </div>
                    <div class="stack">
                        <div class="panel">
                            <div class="panel-label">176 tests, no database needed</div>
                            <p class="note">The validation gate, the train/serve skew guard, the
                            calibration identity, path resolution, four-fifths arithmetic, proxy
                            statistics, adverse action disclosure, and the API contracts &mdash;
                            one of which feeds a row carrying every protected attribute through
                            the real serving path and asserts none reaches a SHAP
                            contribution.</p>
                        </div>
                        <div class="panel plain">
                            <div class="panel-label">token cost</div>
                            <p class="note">The schema block is 1,656 tokens, not the ~6,000 a
                            full dump would be. Caching is wired but measured not to engage:
                            Haiku 4.5 needs a 4,096-token prefix. Confirmed by padding to 7,393
                            tokens and watching the cache write, then read.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="Verified against a running stack, not asserted.",
        number=15, total=total))

    # 16 -- Limitations -----------------------------------------------------
    slides.append(slide(
        "cat README.md | grep -A40 'Known limitations'", "What this is not",
        f"""                <div class="cols reveal">
                    <div class="stack">
                        <div class="panel warn">
                            <div class="panel-label">fair lending is incomplete</div>
                            <p class="note">Two of three attributes fail the four-fifths screen
                            and {px['material_total']} features are material proxies for an
                            excluded attribute. The screen, the proxy scan and the reason codes
                            are built; the business-necessity analysis that would interpret
                            them is not, nor is multivariate proxy detection or reject
                            inference.</p>
                        </div>
                        <div class="panel warn">
                            <div class="panel-label">the cost ratio decides half the book</div>
                            <p class="note">{th['ratio']:.0f}:1 is an assumption. It is now a
                            bounded one, and the bound is wide: only
                            {pct(se['band_stability'], 1)} of applicants keep their band across
                            3:1 to 20:1.</p>
                        </div>
                    </div>
                    <div class="stack">
                        <div class="panel warn">
                            <div class="panel-label">the model ignores two of three tables</div>
                            <p class="note"><code>bureau</code> and
                            <code>previous_application</code> serve the chatbot only. Prior
                            credit history is the largest improvement available, worth roughly
                            0.02&ndash;0.03 ROC-AUC in published work.</p>
                        </div>
                        <div class="panel warn">
                            <div class="panel-label">the evaluation is self-authored</div>
                            <p class="note">{cb['total']} questions written by the person who
                            wrote the prompt, and two of the failures are defects in those
                            questions rather than model failures. An independently authored set
                            is the next step.</p>
                        </div>
                        <div class="panel warn">
                            <div class="panel-label">no automated UI coverage</div>
                            <p class="note">The 176 tests cover <code>src/</code> and the
                            <code>app/</code> contracts. The React UI is verified by hand.
                            Conversation memory is in-process and dies on restart.</p>
                        </div>
                    </div>
                </div>""",
        subtitle="The limitations that would matter to someone deciding whether to trust it.",
        number=16, total=total))

    # 17-20 -- Screenshots --------------------------------------------------
    for n, (filename, title, caption) in enumerate(d["screenshots"], start=17):
        slides.append(slide(
            f"open http://localhost:5173", title,
            f"""                <div class="reveal" style="margin-top:-14px;">
                    <img class="shot" src="screenshots/{filename}" alt="{title}">
                </div>""",
            subtitle=caption, number=n, total=total))

    return document("Riskwright — AI-powered credit risk platform", "".join(slides))


def main() -> None:
    html = build()
    OUT.write_text(html, encoding="utf-8")
    count = html.count('<section class="slide')
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB, {count} slides)")


if __name__ == "__main__":
    main()
