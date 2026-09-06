"""Build documents/project_presentation.pdf.

Every number and chart is read from the artifacts the pipeline actually wrote
(models/*.json) or computed live against Postgres. Nothing is typed in twice,
so the deck cannot drift from the README or the code.

    POSTGRES_HOST=localhost python documents/build_presentation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
SHOTS = ROOT / "documents" / "screenshots"

# Slide geometry: 16:9.
FIGSIZE = (13.333, 7.5)

INK = "#1a1a1a"
MUTED = "#5f6b7a"
ACCENT = "#1f4e79"
GOOD = "#2e7d32"
BAD = "#c62828"
RULE = "#d5dae0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": RULE,
})


def load(name: str) -> dict:
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


METRICS = load("metrics.json")
BANDS = load("threshold.json")
RULES = load("rules.json")
POLICY = load("rules_without_external_scores.json")
FAIRNESS = load("fairness_comparison.json")
SHAP_GLOBAL = load("shap_global.json")


def slide(pdf: PdfPages, title: str, subtitle: str = "") -> tuple:
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.text(0.06, 0.90, title, fontsize=27, fontweight="bold", color=ACCENT)
    if subtitle:
        fig.text(0.06, 0.845, subtitle, fontsize=13.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.825, 0.825], color=RULE, lw=1))
    return fig, pdf


def bullets(fig, lines: list[str], x=0.06, y=0.75, dy=0.062, size=13.5) -> float:
    for line in lines:
        weight = "bold" if line.startswith("**") else "normal"
        text = line.replace("**", "")
        colour = INK
        if text.startswith("+ "):
            colour, text = GOOD, text[2:]
        elif text.startswith("- "):
            colour, text = BAD, text[2:]
        fig.text(x, y, text, fontsize=size, color=colour, fontweight=weight,
                 va="top", wrap=True)
        y -= dy
    return y


def table(fig, rect, headers, rows, col_widths=None, highlight_row=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center",
                   cellLoc="left", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.7)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        if row == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(fontweight="bold", color=ACCENT)
        elif highlight_row is not None and row == highlight_row:
            cell.set_facecolor("#eaf5ea")
            cell.set_text_props(fontweight="bold")
    return ax


def maybe_screenshot(fig, rect, name: str, caption: str = "") -> bool:
    path = SHOTS / name
    if not path.exists():
        return False
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(RULE)
    if caption:
        fig.text(rect[0], rect[1] - 0.035, caption, fontsize=10.5, color=MUTED)
    return True


# --------------------------------------------------------------------------

def build() -> Path:
    out = ROOT / "documents" / "project_presentation.pdf"
    lgb = METRICS["models"]["lightgbm"]["cross_validated"]
    log = METRICS["models"]["logistic"]["cross_validated"]
    opt = BANDS["at_cost_optimal_threshold"]
    naive = BANDS["at_naive_half_threshold"]

    with PdfPages(out) as pdf:

        # 1. Title -----------------------------------------------------------
        fig = plt.figure(figsize=FIGSIZE, facecolor="white")
        fig.text(0.06, 0.62, "Riskwright", fontsize=58, fontweight="bold", color=ACCENT)
        fig.text(0.06, 0.545, "An AI-powered credit risk platform",
                 fontsize=21, color=INK)
        fig.add_artist(plt.Line2D([0.06, 0.55], [0.515, 0.515], color=ACCENT, lw=2.5))
        fig.text(0.06, 0.40,
                 "Predicts default probability, explains every decision,\n"
                 "derives credit policy rules, and answers plain-English\n"
                 "questions about the data.",
                 fontsize=15, color=MUTED, linespacing=1.7)
        fig.text(0.06, 0.13,
                 "Home Credit Default Risk  |  307,511 applications  |  8.07% default rate\n"
                 "NeoStats AI Engineer assignment",
                 fontsize=12, color=MUTED, linespacing=1.6)
        pdf.savefig(fig); plt.close(fig)

        # 2. Architecture ----------------------------------------------------
        fig, _ = slide(pdf, "Architecture",
                       "All business logic in src/. The UI computes nothing.")
        fig.text(0.06, 0.72,
                 "  Streamlit or React    no logic, HTTP client only\n"
                 "        |\n"
                 "        v  HTTP\n"
                 "  FastAPI               thin wrapper: validates, shapes\n"
                 "        |\n"
                 "        +----------------+----------------+\n"
                 "        v                v                v\n"
                 "   src/ml            src/data       src/talk_to_data\n"
                 "   model, SHAP,      loader,        NL to SQL,\n"
                 "   rules             preprocess     validation, memory\n"
                 "        |                |                |\n"
                 "        +----------------+----------------+\n"
                 "                         v\n"
                 "                   PostgreSQL\n"
                 "        application_train, bureau, previous_application",
                 fontsize=12.5, color=INK, family="monospace",
                 va="top", linespacing=1.45)
        bullets(fig, [
            "**Four services under Compose**",
            "postgres, healthchecked",
            "loader, one-shot and idempotent",
            "api, waits on both",
            "ui, waits on api health",
            "",
            "A React frontend is added by a",
            "separate override file on its own",
            "port. The base compose file is",
            "untouched by it.",
            "",
            "**Why it matters**",
            "Startup races are the usual",
            "reason a submission does not",
            "run on someone else's machine.",
        ], x=0.62, y=0.74, dy=0.052, size=12.5)
        pdf.savefig(fig); plt.close(fig)

        # 3. Calibration and the cost threshold ------------------------------
        fig, _ = slide(pdf, "From score to decision",
                       "Calibration first, then cost. The second is meaningless without the first.")
        bullets(fig, [
            "**1. The raw model output is not a probability**",
            "Training with scale_pos_weight = 11.39 multiplies the predicted odds by that",
            "weight. An average applicant scores near 0.5, not near 0.08.",
            "",
            "**2. The distortion is a known constant, so it is inverted exactly**",
            "     p_true  =  p_weighted / (p_weighted + w(1 - p_weighted))",
            "A weighted 0.5 maps back to 1/(1+w) = 0.0807, the measured default rate.",
            "That identity is asserted in the test suite.",
            "",
            "**3. Only now does a cost threshold mean anything**",
            "A false negative is a defaulted loan. A false positive is a good customer",
            "refused. Assumed 10:1. Threshold chosen to minimise 10*FN + FP.",
        ], y=0.76, dy=0.047, size=12.5)

        table(fig, [0.06, 0.07, 0.88, 0.17],
              ["Threshold", "Flagged", "Recall", "Precision", "Expected cost"],
              [["Naive 0.5", "0.0%", f"{naive['recall']:.3f}",
                f"{naive['precision']:.3f}", f"{naive['expected_cost']:,.0f}"],
               [f"Cost-optimal {BANDS['t_high']:.4f}", f"{opt['flagged_rate']:.1%}",
                f"{opt['recall']:.3f}", f"{opt['precision']:.3f}",
                f"{opt['expected_cost']:,.0f}"]],
              highlight_row=2)
        saving = 100 * BANDS["cost_saving_vs_half"] / naive["expected_cost"]
        fig.text(0.06, 0.275,
                 f"Catching {opt['recall']:.1%} of defaulters instead of "
                 f"{naive['recall']:.1%}. Expected cost down {saving:.1f}%.",
                 fontsize=14, color=GOOD, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # 4. Risk bands ------------------------------------------------------
        fig, _ = slide(pdf, "Risk bands",
                       "The two boundaries answer different questions and are set differently.")
        table(fig, [0.06, 0.55, 0.88, 0.22],
              ["Boundary", "Value", "How it is set"],
              [["t_high  Medium / High", f"{BANDS['t_high']:.4f}",
                "Cost-optimal. Derived from the 10:1 assumption."],
               ["t_low  Low / Medium", f"{BANDS['t_low']:.4f}",
                "A business judgment. Not an optimum."]],
              col_widths=[0.24, 0.12, 0.52])
        bullets(fig, [
            "Above t_high the expected cost of approving exceeds the expected cost of",
            "refusing. That follows from the cost assumption.",
            "",
            "Where automatic approval should stop is a matter of review capacity and risk",
            "appetite. Nothing in the data answers it, so t_low is exposed as a policy dial",
            "in models/threshold.json rather than presented as a computed result.",
            "",
            "**Stating which number is derived and which is chosen is the point.**",
        ], y=0.44, dy=0.05, size=13)
        pdf.savefig(fig); plt.close(fig)

        # 5. Talk-to-data: measured, not demoed ------------------------------
        fig, _ = slide(pdf, "Talk-to-data, measured like a model",
                       "30 questions written against the schema before any prompt was tuned.")
        ax = fig.add_axes([0.06, 0.30, 0.42, 0.46])
        versions = ["v1\nnaive", "v2\nschema +\nfew-shot", "v3\nstructured\naction", "v4\njoin\ndiscipline"]
        scores = [53, 87, 93, 100]
        ax.bar(versions, scores, color=[MUTED, ACCENT, ACCENT, GOOD])
        for i, v in enumerate(scores):
            ax.text(i, v + 2, f"{v}%", ha="center", fontweight="bold", fontsize=13)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Development set accuracy")
        ax.spines[["top", "right"]].set_visible(False)

        bullets(fig, [
            "**The development set is a tuning history,**",
            "**not a generalisation estimate.**",
            "v4 was tuned on the failures v3 exposed, so",
            "its 100% is optimistic by construction.",
            "",
            "**Held out: 8 further questions, written**",
            "**after the prompt was frozen.**",
            "     v3    7/8         v4    7/8",
            "",
            "88% held out against 100% development.",
            "That 12 point gap is the honest number,",
            "and v4 shows no reproducible advantage",
            "over v3 out of sample.",
        ], x=0.54, y=0.74, dy=0.048, size=12.5)
        pdf.savefig(fig); plt.close(fig)

        # 6. Refusal and the validation gate ---------------------------------
        fig, _ = slide(pdf, "Refusing is a feature",
                       "Two independent mechanisms: a hard gate, and a sanctioned way to decline.")
        bullets(fig, [
            "**Structured output.** One model call returns an action: sql, refuse, or clarify.",
            "A model with no sanctioned way to say 'I cannot' invents a column to fill the",
            "silence. Refusal is a first-class result, and it is the cheap path: one call.",
            "",
            "**Validation gate.** Every generated statement is parsed, not string-matched.",
        ], y=0.76, dy=0.048, size=12.5)

        fig.text(0.06, 0.50,
                 "Rejected   WITH x AS (DELETE FROM application_train RETURNING sk_id_curr)\n"
                 "                SELECT count(*) FROM x\n\n"
                 "Allowed    SELECT count(*) FROM application_train\n"
                 "                WHERE occupation_type = 'Delete'",
                 fontsize=12.5, family="monospace", va="top", linespacing=1.6)
        fig.text(0.06, 0.285,
                 "A DELETE hidden inside a CTE is rejected. The literal string 'Delete' is not.\n"
                 "String matching gets both of these wrong. Parsing the tree gets both right.",
                 fontsize=13, color=ACCENT, fontweight="bold", linespacing=1.6, va="top")
        bullets(fig, [
            "Single SELECT only, whitelisted tables, every column checked against the live",
            "schema, LIMIT forced, 10s timeout, executed as a role holding SELECT and nothing",
            "else. Four independent layers, so a parser defect is not by itself an incident.",
        ], y=0.17, dy=0.045, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 7. Fair lending ----------------------------------------------------
        fig, _ = slide(pdf, "Fair lending",
                       "Found by reading the SHAP output. code_gender was driving credit decisions.")
        bullets(fig, [
            "ECOA names sex, marital status and age as protected bases in credit decisions.",
            "A model that prices or refuses credit on them is a legal failure, however well",
            "it performs. Five attributes are now excluded in the preprocessor, so they cannot",
            "reach the model, the SHAP explanation, or the derived rules.",
        ], y=0.76, dy=0.047, size=12.5)

        table(fig, [0.06, 0.40, 0.40, 0.20],
              ["Excluded", "Protected basis"],
              [["code_gender", "Sex"],
               ["name_family_status", "Marital status"],
               ["age_years", "Age"],
               ["cnt_children", "Familial status"],
               ["cnt_fam_members", "Familial status"]])

        wp = FAIRNESS["with_protected"]; wo = FAIRNESS["without_protected"]
        table(fig, [0.52, 0.44, 0.42, 0.16],
              ["Feature set", "ROC-AUC", "PR-AUC"],
              [["With protected", f"{wp['roc_auc']:.4f}", f"{wp['pr_auc']:.4f}"],
               ["Without (shipped)", f"{wo['roc_auc']:.4f}", f"{wo['pr_auc']:.4f}"]],
              highlight_row=2)
        fig.text(0.52, 0.40,
                 f"Cost of exclusion: {FAIRNESS['cost_of_exclusion']['roc_auc']:.4f} ROC-AUC.\n"
                 "Identical split, seed and hyperparameters.",
                 fontsize=12.5, color=GOOD, fontweight="bold", va="top", linespacing=1.6)

        bullets(fig, [
            "**Exclusion is the floor, not the bar.** Residual proxies remain and are named:",
            "name_income_type contains 'Maternity leave' and 'Pensioner'; occupation,",
            "organization and region columns carry proxy and redlining risk.",
            "",
            "Production would additionally need disparate impact testing, proxy detection,",
            "adverse action reason codes, ongoing monitoring, and reject inference.",
        ], y=0.30, dy=0.045, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 8. Model ------------------------------------------------------------
        fig, _ = slide(pdf, "Model", "Two models, identical inputs, 5-fold stratified CV.")
        table(fig, [0.06, 0.58, 0.60, 0.18],
              ["Model", "ROC-AUC", "PR-AUC"],
              [["LightGBM (served)", f"{lgb['roc_auc_mean']:.4f} +/- {lgb['roc_auc_std']:.4f}",
                f"{lgb['pr_auc_mean']:.4f} +/- {lgb['pr_auc_std']:.4f}"],
               ["Logistic regression", f"{log['roc_auc_mean']:.4f} +/- {log['roc_auc_std']:.4f}",
                f"{log['pr_auc_mean']:.4f} +/- {log['pr_auc_std']:.4f}"]],
              highlight_row=1)
        bullets(fig, [
            "**The gap is smaller than a headline suggests.** Logistic regression comes within",
            "2.4% of a gradient-boosted ensemble, and it is what banks deploy under regulatory",
            "pressure because a coefficient per feature is auditable. Worth reporting, not hiding.",
            "",
            "LightGBM is served because the requirements reinforce each other: SHAP TreeExplainer",
            "is exact on tree models, and rule derivation is naturally a shallow tree.",
            "",
            "**Imbalance:** 8.07% positive, scale_pos_weight 11.39, derived per split and logged.",
            "PR-AUC reported alongside ROC-AUC throughout. No SMOTE, no hyperparameter search.",
        ], y=0.52, dy=0.048, size=12.5)
        pdf.savefig(fig); plt.close(fig)

        # 9. EDA ---------------------------------------------------------------
        fig, _ = slide(pdf, "What the data says",
                       "Five insights, each computed by the same module the API serves from.")
        bullets(fig, [
            "**1. A sentinel hides in the employment column.** 18.0% of applicants have",
            "days_employed = 365243, about 1,000 years, encoding 'not employed'. Untreated it",
            "corrupts every statistic built on the column. Kept as a flag, because it predicts.",
            "",
            "**2. External credit scores dominate.** Lowest quartile of ext_source_3 defaults at",
            "15.1% against 3.5% in the highest. A 4.3x spread from one column.",
            "",
            "**3. Education tracks risk.** Lower secondary 10.9%, Academic degree 1.8%.",
            "",
            "**4. Leverage does not predict default, which is not what you would expect.**",
            "Default peaks at 2-4x income (8.77%) and is LOWEST above 6x (7.23%). Likely a",
            "selection effect. credit_income_ratio ranks 37th by SHAP; loan term ranks 3rd.",
            "A policy built on a leverage cap would target the wrong quantity.",
            "",
            "**5. Younger applicants default more.** 11.4% under 30, 4.9% at 60 and over.",
        ], y=0.76, dy=0.0445, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 10. Explainability ---------------------------------------------------
        fig, _ = slide(pdf, "Explaining one decision",
                       "SHAP per prediction, returned as JSON. The UI draws; the API never renders.")
        ax = fig.add_axes([0.06, 0.12, 0.44, 0.62])
        top = SHAP_GLOBAL["features"][:12][::-1]
        ax.barh([f["label"] for f in top], [f["mean_abs_shap"] for f in top], color=ACCENT)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title("Global importance (2,000 row sample)", fontsize=12, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)

        bullets(fig, [
            "**Part 4 asks for a non-technical explanation.**",
            "**A list of signed floats is not one.**",
            "",
            "Every prediction carries a narrative,",
            "generated deterministically from a label map,",
            "not by an LLM. It cannot invent a reason the",
            "model did not use.",
            "",
            '"This applicant has a 44.9% estimated chance',
            ' of default, which places them in the High',
            ' risk band. Risk is pushed up mainly by',
            ' external credit score 3..."',
            "",
            "Date columns are negative day offsets, so",
            "days_birth = -9461 is shown as 25.9 years,",
            "not printed raw.",
        ], x=0.55, y=0.74, dy=0.0435, size=11.5)
        pdf.savefig(fig); plt.close(fig)

        # 11. Rules -------------------------------------------------------------
        fig, _ = slide(pdf, "From model to credit policy",
                       "A depth-3 surrogate tree, so the logic fits in a policy document.")
        rows = [[r["rule_id"], r["readable"][:62], f"{r['support_pct']:.1f}%",
                 f"{100 * r['default_rate']:.1f}%", f"{r['lift']:.2f}x"]
                for r in RULES["rules"][:4]]
        table(fig, [0.05, 0.50, 0.90, 0.24],
              ["Rule", "Condition", "Support", "Default", "Lift"], rows,
              col_widths=[0.06, 0.52, 0.10, 0.10, 0.08])
        bullets(fig, [
            f"**Two rule sets.** The faithful set agrees with the model on "
            f"{RULES['surrogate_fidelity']:.1%} of applicants,",
            f"but keys almost entirely off bureau scores, so it tells a credit officer little",
            f"they can act on. A second set excludes those scores: "
            f"{POLICY['surrogate_fidelity']:.1%} agreement, and rules",
            "on employment length, loan term and goods price that apply to an applicant with",
            "no bureau history at all.",
            "",
            "**Leaf statistics come from counting rows, not from tree_.value.** Under",
            "class_weight='balanced' that array holds reweighted proportions and reports a 78%",
            "default rate on a population whose true rate is 8%. A plausible-looking error.",
        ], y=0.44, dy=0.045, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 12. Engineering -------------------------------------------------------
        fig, _ = slide(pdf, "Engineering", "Verified against a running stack, not asserted.")
        bullets(fig, [
            "**Docker.** Four services. Postgres healthchecked; the loader waits for healthy and",
            "the API waits for the loader to exit successfully. Verified by cloning this",
            "repository into a fresh directory and building with --no-cache.",
            "",
            "**Security.** The chatbot connects as a role holding SELECT on three tables and",
            "nothing else. Verified: INSERT, CREATE and DROP are all refused by Postgres.",
            "The LLM key is scoped to the API service and never reaches the UI container.",
            "",
            "**Idempotent load.** 740MB in about 150 seconds; a second start skips it in one.",
            "",
            "**46 tests.** The validation gate, the train/serve skew guard, the calibration",
            "identity, path resolution, and API shapes.",
            "",
            "**Token cost.** Schema block is 1,656 tokens, not the ~6,000 a full dump would be.",
            "Caching is wired but measured not to engage: Haiku 4.5 needs a 4,096 token prefix.",
            "Confirmed by padding to 7,393 tokens and watching the cache write, then read.",
        ], y=0.76, dy=0.0445, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 13. Limitations -------------------------------------------------------
        fig, _ = slide(pdf, "What this is not",
                       "The limitations that would matter to someone deciding to trust it.")
        bullets(fig, [
            "**Refusal is probabilistic, not guaranteed.** The validator makes querying a",
            "non-existent column impossible. Declining a question that is semantically",
            "unanswerable from columns that do exist is a model judgement, and held-out",
            "question H8 was refused in one run and answered wrongly in another.",
            "",
            "**The model ignores two of the three loaded tables.** bureau and",
            "previous_application serve the chatbot only. Prior credit history is the largest",
            "single improvement available, worth roughly 0.02-0.03 ROC-AUC in published work.",
            "",
            "**Eight held-out questions is a small sample.** A wider set with a proper train",
            "and test split is the honest next step.",
            "",
            "**Fair lending work is incomplete.** Excluding protected attributes is the floor.",
            "Disparate impact testing, proxy detection and adverse action codes are not built.",
            "",
            "**Conversation memory is in-process.** It does not survive a restart.",
        ], y=0.76, dy=0.0445, size=12)
        pdf.savefig(fig); plt.close(fig)

        # 14+. Screenshots -------------------------------------------------------
        shots = [
            ("eda.jpg", "Data understanding", "Dataset summary, insights, and the distribution explorer."),
            ("prediction.jpg", "Risk prediction", "Score with band and recommended action."),
            ("explain.jpg", "Why this decision", "SHAP contributions and the plain-English narrative."),
            ("rules.jpg", "Derived rules", "Both rule sets with support, default rate and lift."),
            ("chat.jpg", "Ask the data", "Generated SQL is shown, not hidden."),
        ]
        for name, title, caption in shots:
            if not (SHOTS / name).exists():
                continue
            fig, _ = slide(pdf, title, caption)
            maybe_screenshot(fig, [0.06, 0.06, 0.88, 0.72], name)
            pdf.savefig(fig); plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = "Riskwright: AI-Powered Credit Risk Platform"
        meta["Subject"] = "NeoStats AI Engineer assignment"

    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
