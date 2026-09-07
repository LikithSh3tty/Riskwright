<div align="center">

# Riskwright

*credit decisions a regulator could read*

**An AI credit risk platform that scores, explains itself, and then audits itself for the
thing accuracy never catches.**

![tests](https://img.shields.io/badge/tests-176%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.7614-blue)
![held-out](https://img.shields.io/badge/chatbot-38.4%2F44-orange)
![docker](https://img.shields.io/badge/docker-3%20images-2496ED)
![live](https://img.shields.io/badge/live-demo-brightgreen)

[Why it's built this way](#fair-lending) &middot; [See it work](http://159.65.157.86)

</div>

---

A credit model can be accurate, explainable, well tested, and still illegal to deploy.
Riskwright predicts default probability on 307,511 Home Credit applications, explains every
decision with exact SHAP values, turns the model into rules a credit officer can read, and
answers plain-English questions by writing SQL that is parsed before it runs.

The part worth reading is what happened when it was checked. Five protected attributes are
excluded from the feature matrix, which is the standard move and is where most projects stop.
It did not hold. A systematic scan of all 120 features found **fifteen that reconstruct an
excluded attribute**, twelve of them for age, and `ext_source_1` predicts age at 0.600 while
being the third most important feature in the model. At the deployed threshold, **two of three
protected attributes fail the four-fifths screen**. None of that is visible in a ROC-AUC.

```
applicant -> LightGBM -> calibrate -> band -> SHAP -> reason codes
             120 feats   exact       cost     exact   ECOA, 4 max
                         inversion   optimal          fixed wording
```

**Measured, not asserted.** Every figure in this README is read from `models/*.json` rather
than typed, because generating them that way caught two hand-written numbers that were wrong.
The cost ratio behind the threshold is an assumption, so it was swept: it moves the risk band
for **half the applicants**. The chatbot's refusal rate is a model judgement, so it was run
five times: **38.4/44**, not the single good run. Where a claim could not be measured, it says
so.

## See it work - http://159.65.157.86

## What it does

| Module | Status |
|--------|--------|
| Data understanding and EDA | done |
| Talk-to-data (natural language to SQL) | done |
| Machine learning layer (default probability, risk band) | done |
| Explainable AI (SHAP per prediction) | done |
| Business-readable decision rules | done |
| Multi-section user interface | done, React |
| Dockerized deployment | done |

## Data loaded

Measured on load, not quoted from the dataset description.

| Table | Rows | Columns | Load time |
|-------|------|---------|-----------|
| `application_train` | 307,511 | 122 | 25s |
| `bureau` | 1,716,428 | 17 | 32s |
| `previous_application` | 1,670,214 | 37 | 93s |

The target is heavily imbalanced: **8.07%** of applicants defaulted. The imbalance ratio that
follows from it, `(1 - 0.0807) / 0.0807`, is roughly 11.4, and that ratio is what
`scale_pos_weight` is set to. It is computed in `train.py` from the actual training split and
logged at fit time rather than hardcoded, so the value always matches the data the model
actually saw.

A second `docker-compose up` completes the load step in about a second, because the loader
checks for existing rows before ingesting.

### Which tables the model uses

The model trains on `application_train` only.

`bureau` and `previous_application` are loaded to serve the talk-to-data layer, which needs
more than one table before it can answer a join question. They are not consumed by the model.
This is a deliberate scope decision, not an oversight: aggregating prior credit history into
applicant-level features would also require rebuilding those aggregates at inference time for
a single applicant, and a divergence between the training aggregate and the serving aggregate
produces no error, only quietly wrong predictions. Adding them is the first improvement listed
under limitations.

---

## Repository structure

The repository root **is** the `credit_risk_platform/` folder described in the assignment.
Nesting it one level deeper would force the evaluator to `cd` before running anything, so the
prescribed tree sits at the top level instead.

```
data/                     Home Credit CSVs. Mounted, never committed.
documents/                Project presentation (PDF) and its screenshots
notebooks/                eda.ipynb and its eda.py export
src/
  data/loader.py          CSV to Postgres, idempotent
  data/preprocessor.py    Cleaning, feature derivation, encoding
  ml/                     Training, inference, evaluation, SHAP, rules,
                          and the fair-lending and sensitivity measurements
  talk_to_data/           Natural language to SQL, validation, prompts, memory
  utils/                  Logging, configuration, helpers, container-safe paths
app/                      FastAPI. A thin wrapper over src/, no business logic.
frontend/                 React client, nginx served. Calls HTTP endpoints only.
sql/schema.sql            Generated DDL for the three loaded tables
configs/                  Configuration, and the Kaggle column dictionary (metadata, not rows)
models/                   Saved model artifacts and every measured figure (below)
tests/                    Unit tests and the chatbot evaluation set
```

### Additions to the prescribed structure

The structure given in the assignment has no file for explainability, none for rule
derivation, and no UI directory, although all three are required deliverables. Rather than
restructure, the following were added:

| Addition | Why |
|----------|-----|
| `src/ml/explain.py` | SHAP values per prediction (Part 4) |
| `src/ml/rules.py` | Rule derivation module (listed under expected deliverables) |
| `src/ml/fairness_audit.py` | Disparate impact screen against the served model. Not required; see **Fair lending** |
| `src/ml/proxy_detection.py` | Scores all 120 features against each excluded attribute. Not required; see **Fair lending** |
| `src/ml/reason_codes.py` | ECOA adverse action reasons from the SHAP contributions. Not required; see **Explainability** |
| `src/ml/threshold_sensitivity.py` | Sweeps the cost ratio to bound the threshold assumption. Not required; see **Model** |
| `src/ml/shap_full.py` | Exhaustive global SHAP in chunks. Not required; see **Explainability** |
| `app/` | FastAPI layer, so the UI holds no business logic and stays replaceable |
| `frontend/` | Part 5 requires a UI; the tree names no location for one |
| `configs/` | Training and application configuration, plus the column glossary |
| `tests/` | Unit tests and the chatbot evaluation set |
| `.env.example` | Required by the submission instructions |

`src/utils/docker_utils.py` exists because hardcoded absolute paths that work locally and
break inside a container are a common failure. Every filesystem path in the project is built
there and nowhere else.

That turned out to be well founded. `Settings` originally resolved `env_file=".env"`, which is
relative to the working directory, so running the notebook from `notebooks/` found no `.env`,
silently fell back to the placeholder password, and failed to authenticate against Postgres.
The fix was to resolve the path from the module location up to the repository root. Exactly
the class of defect the prescribed structure puts `docker_utils.py` there to prevent.

The assignment tree also lists `sql/roles.sql`. Role creation instead lives in
`loader.ensure_readonly_role`, because the role's password comes from the environment and
interpolating a secret into a committed SQL file would be wrong. The role is created with
parameterised statements at load time.

### What is in `models/`

Every number quoted in this README is read from one of these rather than typed in, and is
re-checked against them whenever a figure changes. That is not decoration: reading the artifacts
rather than transcribing them surfaced a hand-typed figure in this README that was wrong, and a
later audit caught a second.

The presentation was generated the same way and no longer is; see **Presentation** below.

| Artifact | Written by | Holds |
|----------|-----------|-------|
| `lightgbm_model.joblib`, `logistic_model.joblib` | `src.ml.train` | The two fitted models. LightGBM is served. |
| `feature_spec.json` | `src.ml.train` | Column order, category levels and dtypes, so inference reproduces training exactly |
| `metrics.json` | `src.ml.train` | ROC-AUC and PR-AUC per fold and out of fold, both models |
| `threshold.json` | `src.ml.train` | The cost-optimal threshold, the band boundaries, and the confusion matrix at each |
| `oof_predictions.npz` | `src.ml.train` | Out-of-fold scores, which is why the threshold sweep needs no retrain |
| `rules.json`, `rules_without_external_scores.json` | `src.ml.train --artifacts` | The two derived rule sets with support, default rate and lift |
| `shap_global.json` | `src.ml.train --artifacts`, then `src.ml.shap_full` | Mean absolute SHAP per feature. Now exhaustive over all 307,511 applicants. |
| `shap_sample_comparison.json` | `src.ml.shap_full` | The exhaustive ranking against the 2,000-row sample it replaced |
| `fairness_comparison.json` | `src.ml.train --fairness` | What excluding the protected attributes costs |
| `fairness_audit.json` | `src.ml.fairness_audit` | Approval and default rates per protected group, and the four-fifths ratios |
| `proxy_detection.json` | `src.ml.proxy_detection` | Every feature scored against every excluded attribute |
| `threshold_sensitivity.json` | `src.ml.threshold_sensitivity` | The cost-ratio sweep and how far the bands move across it |
| `chatbot_variance.json` | `tests.run_chatbot_variance` | Held-out scores across repeated runs, with earlier measurements kept as history |

The `src.ml.train` rows are the only ones a training run produces. Everything from
`shap_sample_comparison.json` downwards is computed from artifacts already on disk, or from a
read-only pass over Postgres with the saved model - so none of the fair-lending, sensitivity
or SHAP work can disturb the model that is served.

Model artifacts are committed, which contradicts a common convention but follows the
assignment's list of deliverables and means a fresh clone can serve predictions without
training first. The measurement artifacts are committed for a different reason: they are the
evidence behind the claims, and a claim whose evidence is not in the repository is an
assertion.

---

## Setup

### Prerequisites

- Docker and Docker Compose
- The Home Credit Default Risk dataset from
  [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- An Anthropic API key (only needed for the chatbot)

### 1. Get the data

The dataset is a Kaggle **competition** download, so you must be signed in to Kaggle and have
accepted the competition rules once on the data page before the download will work. That
applies to both the browser and the CLI.

The archive is about 720MB and expands to roughly 2.7GB. Only three of its files are used,
totalling about 740MB:

```
application_train.csv        166MB
bureau.csv                   170MB
previous_application.csv     405MB
```

Via the browser: open the
[competition data page](https://www.kaggle.com/competitions/home-credit-default-risk/data),
accept the rules, download, and extract at least those three files.

Via the Kaggle CLI, if you have `~/.kaggle/kaggle.json` set up:

```bash
mkdir -p /path/to/home-credit
kaggle competitions download -c home-credit-default-risk -p /path/to/home-credit
cd /path/to/home-credit && unzip home-credit-default-risk.zip
```

**Extract to a folder outside any cloud-synced directory** such as OneDrive, Dropbox, or
iCloud. Sync clients hold file locks that break Docker bind mounts, and you do not want 2.7GB
of CSV syncing to the cloud either.

The dataset is never committed to git.

**One Kaggle file is committed, and it is not data.**
`configs/columns_description.csv` is the competition's own data dictionary: 37KB, 219 rows,
one row per *column definition* across the seven tables, with the fields `Table`, `Row`,
`Description` and `Special`. It contains no applicant records. It is committed because the
chatbot's schema context reads it - sending the model a real definition is what stops it
reading `days_birth` as a date. Removing it is not cosmetic: the schema block loses every
column description, drops below its documented token floor, and three tests in
`test_schema_context.py` fail. The submission instruction not to place the dataset in git is
about `application_train.csv` and its siblings, which are mounted from `DATA_DIR` and ignored
by `.gitignore`.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Value |
|----------|-------|
| `DATA_DIR` | **Absolute** path to the folder holding the extracted CSVs |
| `ANTHROPIC_API_KEY` | Your key. Required only for the chatbot; everything else runs without it. |
| `POSTGRES_PASSWORD` | Any value |
| `POSTGRES_RO_PASSWORD` | Any value, different from the above |

`DATA_DIR` examples:

```ini
# macOS / Linux
DATA_DIR=/home/you/data/home-credit

# Windows: use forward slashes. Backslashes are not interpreted correctly
# by Docker Compose in a bind mount path.
DATA_DIR=C:/data/home-credit
```

Leave `POSTGRES_HOST=postgres` as it is. That is the service name on the compose network. Only
change it if you are running outside Docker, which is covered below.

You will need roughly 3GB free for the extracted CSVs and another 2GB for the Postgres volume.

### 3. Run

```bash
docker-compose up
```

That is the whole run path. There are no flags, no override files, and no second command:
Postgres, the loader, the API, and the React UI all come up from `docker-compose.yml` alone.

Then open:

| Service | URL |
|---------|-----|
| React UI | http://localhost:5173 |
| API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

The first start loads roughly 740MB into Postgres and takes a few minutes. Subsequent starts
skip the load, because the loader checks for existing rows first. To force a reload, set
`FORCE_RELOAD=true`.

Docker Desktop, or the Docker daemon, has to be running before you start. The first run pulls
the Postgres image and builds the project image, which takes a few minutes on top of the load.

You will know it is ready when `http://localhost:8000/health` returns
`"status": "ok"` with `"database": {"connected": true}`.

### Running without Docker

Useful for training and for running the test suite. Postgres still has to be reachable, so the
easiest route is to start just that service with Compose and run everything else on the host.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv/Scripts/activate             # Windows
pip install -r requirements-dev.txt

docker-compose up -d postgres      # Postgres only
```

**Set `POSTGRES_HOST=localhost` for anything run on the host.** The `.env` default is
`postgres`, which is the service name inside the compose network and does not resolve from
outside it. This is the single most common way a host-side command fails.

```bash
# macOS / Linux
export POSTGRES_HOST=localhost

# Windows PowerShell
$env:POSTGRES_HOST = "localhost"
```

Then, from the repository root:

```bash
python -m src.data.loader          # load the CSVs, idempotent
python -m src.ml.train             # 5-fold CV, both models, SHAP, rules
python -m src.ml.train --quick     # single split, about 60 seconds
python -m src.ml.train --fairness  # cost of excluding protected attributes
python -m src.ml.fairness_audit    # disparate impact screen, no retrain
python -m src.ml.proxy_detection   # single-feature proxy scan
python -m src.ml.shap_full         # exhaustive global SHAP, chunked
python -m src.ml.threshold_sensitivity   # cost ratio sweep; needs no database
pytest                             # 176 tests, no database needed

uvicorn app.main:app --reload
```

The React client is built and served by its own image, so there is no host-side equivalent of
the API command above. To iterate on it without Docker, run Vite's dev server from
`frontend/` (`npm install && npm run dev`) with the API running on port 8000; `vite.config.js`
proxies `/api/*` there, the same path nginx serves in the container.

---

## Architecture

```
                    +-------------------+
                    |  React UI         |   no business logic,
                    |  (frontend/)      |   HTTP client only
                    +---------+---------+
                              | HTTP, via nginx /api/
                    +---------v---------+
                    |  FastAPI          |   thin wrapper,
                    |  (app/)           |   validates and shapes
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
       +------v-----+  +------v-----+  +------v------+
       |  src/ml    |  | src/data   |  | src/        |
       |  model,    |  | loader,    |  | talk_to_data|
       |  SHAP,     |  | preprocess |  | NL to SQL   |
       |  rules     |  |            |  |             |
       +------+-----+  +------+-----+  +------+------+
              |               |               |
              |        +------v---------------v------+
              +------->|        PostgreSQL           |
                       |  application_train, bureau, |
                       |  previous_application       |
                       +-----------------------------+
```

All business logic lives under `src/`. FastAPI validates requests and shapes responses.
The UI reads every value over HTTP and computes nothing. That layering means the frontend
could be replaced without touching a line that computes anything.

### Services

| Service | Role |
|---------|------|
| `postgres` | Data store. Has a healthcheck; everything else waits on it. |
| `loader` | One-shot. Applies the schema, streams the CSVs in, provisions the read-only role, exits. |
| `api` | FastAPI. Starts only after Postgres is healthy **and** the loader has exited successfully. |
| `frontend` | React bundle served by nginx, which also proxies `/api/*` to the API. Starts only after the API's own healthcheck passes, and carries its own healthcheck. |

The startup ordering is deliberate. A race where the API starts before Postgres accepts
connections is the most common reason a project like this fails on a machine other than the
author's.

### Security

The chatbot connects to Postgres as `riskwright_ro`, a role holding `SELECT` on exactly three
tables and nothing else. Generated SQL also passes a validation gate before it runs. Even if
the gate were bypassed, the role cannot write.

The Anthropic API key is scoped to the `api` service only. It never reaches the UI container.

---

## Design decisions

| Decision | Reasoning |
|----------|-----------|
| Postgres over SQLite | The chatbot generates real SQL against a real engine, and Compose orchestrates something meaningful rather than a single file. |
| Three tables, not seven | `application_train`, `bureau`, and `previous_application` support genuine join queries. The remaining four add about 1.9GB of load time and answer no question the assignment asks. |
| Column names lowercased on load | Postgres folds unquoted identifiers to lowercase. Normalising once at load time means generated SQL never needs quoting, which removes an entire class of LLM error. |
| React as the single frontend | Streamlit was the development client and was replaced by React before submission. One UI, one run path, one thing for an evaluator to start. See below. |
| Claude Haiku 4.5 | Cheap enough to iterate prompts heavily, and strong at SQL over a compact schema. A small model is sufficient when the schema sent to it is small. |
| Exact version pins | The model is trained locally and served in a container. Unpinned scikit-learn or LightGBM between the two silently changes predictions. |
| Types inferred over the full CSV, not a sample | Several columns are integral in the first 100k rows and fractional later. A sampled schema would fail the COPY halfway through a 400MB file. |

---

## Model

Trained on 307,511 rows and 120 features from `application_train`, with protected
attributes excluded (see **Fair lending** below).

### Model selection

Two models, identical inputs, reported side by side. Five-fold stratified cross-validation.

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| LightGBM | **0.7614** +/- 0.0026 | **0.2441** +/- 0.0008 |
| Logistic regression | 0.7438 +/- 0.0026 | 0.2181 +/- 0.0031 |

LightGBM wins, but by less than a headline would suggest: 0.018 ROC-AUC and 0.026 PR-AUC. The
gap is real rather than noise, being several times the fold standard deviation, and it is
worth stating plainly that a logistic regression gets within 2.4% of a gradient-boosted
ensemble on this data. Logistic regression is also what banks deploy under regulatory
pressure, because a coefficient per feature is auditable in a way an ensemble is not. If the
audit burden mattered more than 0.018 AUC, the baseline would be the defensible production
choice. That is a finding, not a disappointment.

LightGBM is served because two other requirements point the same way: SHAP's `TreeExplainer`
is exact and fast on tree models, and rule derivation is naturally a shallow tree. The model
family, the explainability method, and the rule derivation reinforce each other.

Gradient boosting also suits the data shape: tabular, mixed categorical and numeric,
substantial missingness, non-linear interactions. LightGBM takes NaN and categorical dtypes
natively, so the feature matrix needs no imputation or one-hot encoding before it.

No hyperparameter search was run. Sensible defaults with correct imbalance handling score the
same here, and the time was better spent on the chatbot, which carries more of the marks.

### Class imbalance

The positive rate is 8.07%, so `scale_pos_weight` is set to the negatives-over-positives
ratio, roughly 11.4. It is computed inside `train.py` from each training split and logged at
fit time rather than hardcoded.

SMOTE was not used. At this row count it is slow, and `scale_pos_weight` addresses the same
problem directly without synthesising applicants who do not exist.

PR-AUC is reported alongside ROC-AUC everywhere. On a target this imbalanced, ROC-AUC alone
flatters a model, and quoting it on its own is the clearest sign that imbalance was not
considered.

### Calibration

This one is easy to miss and changes what the numbers mean. Training with `scale_pos_weight`
multiplies the predicted odds by that weight, so raw model output is a **ranking score, not a
probability**. An average applicant scores near 0.5 rather than near 0.08. Reporting that as a
default probability would be wrong, and visibly wrong to anyone who knows the base rate.

Because the distortion is a known constant it is inverted exactly rather than fitted:

```
p_true = p_weighted / (p_weighted + w * (1 - p_weighted))
```

A weighted output of exactly 0.5 maps back to `1 / (1 + w)` = 0.0807, the measured default
rate. That identity is asserted in the test suite. The transform is monotonic, so ROC-AUC and
PR-AUC are unchanged; only the scale moves. The API returns calibrated probabilities.

### Decision threshold and risk bands

A false negative is an approved loan that defaults and is written off. A false positive is a
creditworthy applicant refused, costing margin and goodwill. They are not symmetric, so the
threshold is chosen by expected cost rather than by convention.

**Assumption, stated so it can be argued with: a false negative costs 10x a false positive.**
The threshold minimising `10 * FN + 1 * FP` is selected by sweeping out-of-fold predictions.

| Threshold | Flagged | Recall | Precision | Expected cost |
|-----------|---------|--------|-----------|---------------|
| Naive 0.5 | 0.0% | 0.002 | 0.659 | 247,681 |
| Cost-optimal 0.0837 | 28.9% | 0.644 | 0.180 | **161,314** |

Choosing by cost rather than by convention cuts expected cost by **34.9%**. On a calibrated
probability scale 0.5 is not a neutral default, it is an absurd one: it approves essentially
everybody and catches 2 defaulters in every 1,000.

The two band boundaries answer different questions and are set differently:

| Boundary | Value | How it is set |
|----------|-------|---------------|
| `t_high` Medium/High | 0.0837 | **Cost-optimal.** Above it the expected cost of approving exceeds the expected cost of refusing. Derived from the cost assumption. |
| `t_low` Low/Medium | 0.0341 | **A business judgment, not an optimum.** Set so the safest 40% of applicants are auto-approved without review. Where automatic approval should stop depends on review capacity and risk appetite, and nothing in the data answers that. |

Only `t_high` is derived. `t_low` is a policy dial, and it is exposed as one in
`models/threshold.json` rather than presented as a computed result.

Confusion matrix at `t_high`, out of fold across all 307,511 rows:

|  | Predicted approve | Predicted refuse |
|--|-------------------|------------------|
| **Did not default** | 209,832 | 72,854 |
| **Defaulted** | 8,846 | 15,979 |

Precision of 0.180 is low in isolation, and deliberately so. Under a 10:1 cost ratio, catching
64% of defaulters is worth refusing a large number of applicants who would have repaid. That
is the cost assumption doing its job, not the model failing.

### How much rests on the 10:1 assumption

The ratio was stated so it could be argued with. That is the minimum, and it is not the same as
knowing how much turns on it. So it was swept.

```bash
python -m src.ml.threshold_sensitivity
```

Nine ratios from 3:1 to 20:1, each re-optimised on the same out-of-fold predictions.
Figures read from `models/threshold_sensitivity.json`.

| Ratio | `t_high` | Approval rate | Defaulters caught | Missed | Cost saving vs 0.5 |
|-------|----------|---------------|-------------------|--------|--------------------|
| 3:1 | 0.228376 | 94.9% | 5,215 | 19,610 | 6.7% |
| 4:1 | 0.187386 | 91.5% | 7,465 | 17,360 | 11.2% |
| 5:1 | 0.161530 | 88.6% | 9,172 | 15,653 | 15.9% |
| 6:1 | 0.121824 | 81.9% | 12,316 | 12,509 | 20.4% |
| 8:1 | 0.106761 | 78.4% | 13,637 | 11,188 | 28.2% |
| **10:1 (shipped)** | **0.083669** | **71.1%** | **15,979** | **8,846** | **34.9%** |
| 12:1 | 0.068864 | 64.8% | 17,624 | 7,201 | 40.5% |
| 15:1 | 0.056057 | 57.8% | 19,182 | 5,643 | 47.4% |
| 20:1 | 0.040736 | 46.4% | 21,150 | 3,675 | 56.1% |

**The answer is that a great deal rests on it.** `t_high` moves from 0.0407 to 0.2284 across the
range - a spread of **2.2 times the shipped threshold itself**. The approval rate swings from
46% to 95%. Choosing by cost still beats 0.5 at every ratio, so the *method* is robust even
where the number is not; but the number is not.

**Only 51.5% of applicants keep the same risk band across the whole sweep.** Nearly half would
be banded differently under a ratio that is no less defensible than 10:1. That is the concrete
form of the assumption: it is not a modelling detail, it is the single input that decides
whether an applicant is auto-approved, reviewed, or declined.

Two things follow, and they pull in opposite directions:

- **The 10:1 figure needs a real recovery model behind it before this is deployable.** A
  measured loss-given-default and a measured cost of a lost good customer would replace an
  assumption that currently moves half the book.
- **The threshold is not arbitrary, and the sensitivity does not make it so.** Every ratio in
  the table produces a threshold far below 0.5, and the cost saving against the naive choice is
  positive throughout - from 6.7% at 3:1 to 56.1% at 20:1. What is uncertain is *where* to put
  the threshold, not *whether* 0.5 is wrong. It is.

**The deployed threshold does not change.** 10:1 remains shipped, `models/threshold.json` is not
rewritten by the sweep, and every other figure in this README continues to describe the shipped
decision. The sweep runs entirely from saved artifacts with no retrain.

One basis note, because two approval rates appear in this README and they differ. The 71.1% here
is computed on out-of-fold predictions, where each applicant was scored by a model that had not
seen them. The 70.39% under **Fair lending** is computed with the served model, which is refit
on all rows. Both are correct for what they measure; they are not the same measurement.

### Reproducing

```bash
python -m src.ml.train              # 5-fold CV, both models, then SHAP and rules
python -m src.ml.train --quick      # single split, LightGBM only, about 60 seconds
python -m src.ml.train --artifacts  # rebuild SHAP and rules from the saved model
python -m src.ml.train --fairness   # cost of excluding the protected attributes
python -m src.ml.fairness_audit     # disparate impact screen, no retrain
python -m src.ml.proxy_detection    # which features reconstruct a protected attribute
python -m src.ml.threshold_sensitivity  # cost ratio sweep, no retrain
python -m src.ml.shap_full          # exhaustive global SHAP, chunked
```

`fairness_audit` reads the saved model rather than fitting one, so it takes seconds and does
not disturb any artifact the reported figures depend on.

## Fair lending

**The model does not use protected attributes.** They are excluded from the feature matrix in
`src/data/preprocessor.py`, not filtered out downstream, so they cannot reach the model, the
SHAP explanation, or the derived rules.

This is a compliance requirement, not a modelling preference. ECOA names sex, marital status,
and age as protected bases in credit decisions; the equal credit opportunity framework
generally, and Indian fair-lending expectations, treat them the same way. A model that prices
or refuses credit on them is a legal failure regardless of how well it performs.

| Excluded | Protected basis |
|----------|-----------------|
| `code_gender` | Sex |
| `name_family_status` | Marital status |
| `age_years` | Age |
| `cnt_children` | Familial status, a direct proxy for the two above |
| `cnt_fam_members` | Familial status, same |

### What it cost

Measured rather than assumed. Same split, same hyperparameters, same seed; the only
difference is whether those five columns are present.

```bash
python -m src.ml.train --fairness
```

| Feature set | Features | ROC-AUC | PR-AUC |
|-------------|----------|---------|--------|
| With protected attributes | 125 | 0.7651 | 0.2491 |
| **Without (shipped)** | **120** | **0.7614** | **0.2427** |
| Cost of exclusion | | **0.0037** | **0.0064** |

**Excluding every protected attribute costs 0.0037 ROC-AUC.** That is roughly a fifth of the
gap between LightGBM and the logistic baseline, and far less than the variance between
reasonable modelling choices. There is no meaningful accuracy argument for keeping them.

This was found by looking at the SHAP output and seeing `code_gender` among the drivers of an
individual credit decision. It is recorded here because noticing it is the point: a model can
be accurate, explainable, well tested, and still illegal to deploy.

### Residual proxies: measured, not guessed at

Removing a protected attribute does not make a model fair. It removes the label, not the
information. An earlier version of this section listed four columns that seemed likely to
carry protected signal. That list was a guess, and it has been replaced by a measurement.

```bash
POSTGRES_HOST=localhost python -m src.ml.proxy_detection
```

Every one of the 120 features the model receives is scored against each of the five excluded
attributes, over all 307,511 applicants. The statistic depends on the pair of types - Spearman
rho for numeric against numeric, the correlation ratio (eta squared) for numeric against
categorical, bias-corrected Cramér's V for categorical against categorical - and all three are
reported on a common 0-1 correlation-like scale so they can be ranked together. Results go to
`models/proxy_detection.json`.

**The thresholds were fixed in the module before the first run**, because a cutoff chosen after
seeing the ranking describes the output rather than judging it. A feature is **material** at
0.20, where it explains roughly 4% of the attribute on its own, and **strong** at 0.50, roughly
25%.

**Fifteen distinct features are a material proxy for at least one excluded attribute.** The
guessed list named four columns and got two of them right.

| Excluded attribute | Material | Strong | Strongest single proxy |
|--------------------|---------|--------|------------------------|
| `age_years` | **12** | **5** | `organization_type`, 0.637 |
| `code_gender` | 5 | 1 | `occupation_type`, 0.572 |
| `cnt_children` | 4 | 0 | `flag_emp_phone`, 0.268 |
| `cnt_fam_members` | 4 | 0 | `organization_type`, 0.244 |
| `name_family_status` | 2 | 0 | `days_employed_anomalous`, 0.251 |

Age is the badly leaking one. Twelve features carry material age signal and five of them are
strong:

| Feature | Association with age | Statistic |
|---------|---------------------|-----------|
| `organization_type` | **0.637** | eta² = 0.406 |
| `name_income_type` | **0.620** | eta² = 0.385 |
| `ext_source_1` | **0.600** | rho = 0.600 |
| `days_employed_anomalous` | **0.600** | rho = 0.600 |
| `flag_emp_phone` | **0.600** | rho = -0.600 |
| `flag_document_6` | 0.393 | rho = 0.393 |
| `days_employed` | 0.307 | rho = -0.307 |
| `days_registration` | 0.295 | rho = -0.295 |
| `days_id_publish` | 0.264 | rho = -0.264 |
| `name_housing_type` | 0.246 | eta² = 0.060 |
| `reg_city_not_work_city` | 0.239 | rho = -0.239 |
| `ext_source_3` | 0.205 | rho = 0.205 |

`ext_source_1` at 0.600 is the uncomfortable one. It is the third most important feature in the
model and it reconstructs age better than `days_employed` does. Whatever the external bureau is
scoring, age is a large part of it, and excluding age from this model does not exclude it from
theirs.

#### The two methods find different things, and neither is sufficient

This is the result worth taking away, and it is more useful than a longer list of proxies.

**The systematic pass would not have found `employed_life_ratio`.** Its association with age is
**0.074**, well below the 0.20 material floor and nowhere near the table above. Yet the section
below demonstrates it carrying age into the score with a clean causal experiment. Association
asks "how well does this feature predict the attribute", and a ratio whose numerator varies
freely predicts age poorly while still transmitting it. Those are different questions.

**Hand-inspection would not have found the twelve.** It found one leak, by noticing that a form
field which should have been inert moved a score. It did not find `organization_type`,
`name_income_type` or `ext_source_1`, all of which are stronger channels.

So the honest position is stronger than the earlier text and in a different direction. It is not
that there were more leaks than suspected, though there were. It is that **a single detection
method is not enough**: the statistical pass and the counterfactual sweep each catch what the
other misses, and only running both produced the full picture.

One guess did not survive. `own_car_age` was listed as a likely age carrier and measures
**0.021** - no material relationship at all.

Two of the guesses held up. `name_income_type` is a strong age proxy at 0.620, as expected from
its `Pensioner` value; `occupation_type` is the strongest gender proxy at 0.572. Both carry real
signal about income stability and are kept, and a production system would have to demonstrate
that keeping them is not a pretext. The geographic columns, `region_rating_client` and
`region_population_relative`, remain a redlining risk on their face regardless of what they
correlate with here.

**What this pass does not do.** It measures one feature at a time. A model combines 120 of them,
and features individually below 0.20 can jointly reconstruct an attribute none of them predicts
alone - which is exactly how `employed_life_ratio` escapes it. Bounding that means fitting a
model to predict each protected attribute from the whole matrix and reporting its accuracy. That
is still not done, and it is now the specific remaining gap rather than a general one.

#### `employed_life_ratio`: a protected attribute re-entering through a derived feature

This one is not a correlation. It is age arriving in the model by construction, and it is the
sharpest finding in this section.

```python
# src/data/preprocessor.py, add_derived_features
"employed_life_ratio": ratio("days_employed", "days_birth"),
```

`days_birth` is a negative day offset from the application date, so it *is* age, in days.
The ratio therefore reads as "the fraction of their life this applicant has held their current
job", and its denominator is the protected attribute itself. Hold `days_employed` fixed and
the feature becomes a strictly monotonic function of age.

The exclusion is ordered such that this survives it. `add_derived_features` runs first;
`prepare_features` then drops `days_birth` as redundant and `age_years` as protected. By the
time either is removed the ratio has already been computed and keeps its value. The feature
matrix contains no column named for age, and the model still sees age.

**Measured effect.** Two applicants identical in every field, differing only in `days_birth`,
scored through the served model:

| Age | Predicted default probability |
|-----|-------------------------------|
| 35 | 0.027133 |
| 55 | 0.029376 |

Roughly an **8% relative swing** attributable to age alone, all else equal. Small in absolute
terms at these probabilities, and not small in kind: it is exactly the effect the exclusion
was put in place to prevent.

Repeating that sweep across 20,000 real applicants - rewriting `days_birth` to age 35, then to
age 55, and changing nothing else - shows how the effect is distributed, and confirms the
mechanism:

| Rows | n | Mean swing | Median | Share whose score moved at all | 90th percentile |
|------|---|-----------|--------|-------------------------------|-----------------|
| `days_employed` present | 16,466 | +3.69% | +0.91% | 71.8% | +7.72% |
| `days_employed` sentinel | 3,534 | **+0.00%** | +0.00% | **0.0%** | +0.00% |

The second row is the proof. Where `days_employed` carries the "not currently employed"
sentinel, `clean` turns it into `NaN`, the ratio becomes `NaN`, and the score does not move
for a single one of those 3,534 applicants. Age is inert exactly when the ratio is undefined,
which is what identifies `employed_life_ratio` as the whole of the channel rather than one
contributor among several. The single applicant tabulated above sits near the 90th percentile
of the swing distribution; the typical employed applicant sees less, and about a quarter see
none because they do not cross a split boundary.

**How it was found.** By comparing what the two UIs exposed. The what-if form in one client
accepted a different set of fields from the other, and a value that should have been inert
moved the score. A single frontend would not have surfaced it.

**It was not the only such path, and it is not the largest.** This paragraph used to say
that a systematic pass had not been run and that other day-offset columns *plausibly* carried
age signal. Both have since been settled by measurement, above: fifteen features are a material
proxy for an excluded attribute, twelve of them for age, and five of those are stronger channels
than this one. `days_employed`, `days_id_publish` and `days_registration` were on that guessed
list and do carry age at 0.307, 0.264 and 0.295. `own_car_age` was on it too and measures 0.021,
which is nothing.

The reason this leak still earns its own subsection is that the systematic pass would have
missed it. `employed_life_ratio` predicts age at 0.074, far below the material floor, while the
counterfactual above shows it transmitting age cleanly. Association and conduction are different
questions, and this is the worked example of the gap between them.

**Why it is documented rather than removed.** Dropping `employed_life_ratio` changes the
feature set from 120 columns to 119 and requires a retrain. Every figure in this README - ROC-AUC, PR-AUC, the
calibration constant, the cost-optimal threshold, the band boundaries, the SHAP importances,
the derived rules, the fair-lending delta, and the disparate impact measurement below - is
computed against the 120-feature artifact that is actually served. Removing the feature would invalidate all of them and leave a README
describing a model that is not in the container. Measuring and reporting the leak against the
shipped artifact is the more useful result.

**Where this would be caught in production.** In proxy detection, before deployment: fit a
model to predict each protected attribute from the candidate feature matrix and inspect what
it leans on. A pass like that would have flagged `employed_life_ratio` from its construction
alone, without anyone needing to notice a form field behaving oddly. It is listed below as
work a production system needs, and this is the concrete instance of why.

### Disparate impact measurement

Excluding an attribute says nothing about how the resulting decisions fall across the groups
that attribute describes. This measures it. The protected columns are read from Postgres and
used to **group** applicants; they are never added to a feature path, so the scores below are
the scores the served model produces.

```bash
POSTGRES_HOST=localhost python -m src.ml.fairness_audit
```

Every applicant in `application_train` is scored with the served LightGBM artifact and the
deployed threshold is applied. **Approved** means a calibrated default probability below
`t_high` = 0.083669, the Medium/High band boundary: at or above it an applicant is declined or
escalated. Population approval rate is 70.39%. Results are written to
`models/fairness_audit.json`, which this README reads.

**`code_gender`**

| Group | n | Approval rate | Mean predicted p | Observed default rate |
|-------|---|---------------|------------------|-----------------------|
| F | 202,448 | 0.7412 | 0.0666 | 0.0700 |
| M | 105,059 | 0.6321 | 0.0867 | 0.1014 |
| XNA | 4 | 0.7500 | 0.0663 | 0.0000 |

Four-fifths ratio **0.8528** - passes. Lowest M, highest F.

**Age band** (derived from `days_birth`)

| Group | n | Approval rate | Mean predicted p | Observed default rate |
|-------|---|---------------|------------------|-----------------------|
| under 30 | 45,186 | 0.4873 | 0.1136 | 0.1144 |
| 30-45 | 123,737 | 0.6807 | 0.0774 | 0.0900 |
| 45-60 | 103,287 | 0.7698 | 0.0612 | 0.0656 |
| 60+ | 35,301 | 0.8698 | 0.0446 | 0.0492 |

Four-fifths ratio **0.5602 - fails.** Lowest "under 30", highest "60+".

**`name_family_status`**

| Group | n | Approval rate | Mean predicted p | Observed default rate |
|-------|---|---------------|------------------|-----------------------|
| Civil marriage | 29,775 | 0.6376 | 0.0856 | 0.0994 |
| Married | 196,432 | 0.7179 | 0.0707 | 0.0756 |
| Separated | 19,770 | 0.7318 | 0.0683 | 0.0819 |
| Single / not married | 45,444 | 0.6379 | 0.0860 | 0.0981 |
| Unknown | 2 | 1.0000 | 0.0193 | 0.0000 |
| Widow | 16,088 | 0.8086 | 0.0553 | 0.0582 |

Four-fifths ratio **0.7885 - fails.** Lowest "Civil marriage", highest "Widow".

#### What this says, and what it does not

**Two of the three attributes fail the four-fifths screen.** Age fails badly: an applicant
under 30 is approved at 56% of the rate of an applicant over 60. Family status fails
marginally at 0.7885. Gender passes at 0.8528, which is a pass and not a clearance.

Groups smaller than 100 are reported above but excluded from the ratio, because an approval
rate over four rows is a statement about sampling noise. That exclusion is not doing any
favours to the result: including the two-applicant "Unknown" group, whose approval rate is
1.0000, would push family status from 0.7885 down to 0.6376.

**Age failing is not a coincidence, and it is not solely the leak.** Age band is the attribute
`employed_life_ratio` carries information about, so the leak documented above contributes.
But the gradient is much larger than that feature can explain on its own: the model's mean
predicted probability rises monotonically as age falls, and so does the *observed* default
rate, from 4.92% at 60+ to 11.44% under 30. The model is tracking a real pattern in the
training data through whatever correlates it can reach.

That is precisely the distinction the law draws, and it is why this section stops here.
**Approval-rate disparity is not disparate impact in the legal sense.** The four-fifths ratio
is a screen: it identifies a disparity large enough that someone must go and answer the
questions that follow. Those questions are whether the practice producing the disparity is
justified by business necessity - demonstrably predictive of creditworthiness, not merely
correlated - and whether a less-discriminatory alternative achieving the same legitimate
objective exists. That analysis is not performed here, and nothing in this repository performs
it. A failing ratio here is therefore a flag, not a finding of liability; the passing gender
ratio is likewise not a certificate of compliance.

Nor is the population an unbiased one to measure on. These are the rows the model was fitted
on, so the approval rates describe the training population rather than an out-of-sample one,
and `application_train` contains only accepted applicants - a censored population, as noted
under reject inference below.

### What a production system would additionally need

Excluding the attributes is the floor, not the bar:

- **Business-necessity analysis.** The screen above found two failing ratios and cannot say
  whether either is justified. This is the missing half of a disparate impact review and the
  largest gap in the fair-lending work here.
- **Multivariate proxy detection.** The single-feature pass above is done and found fifteen
  material proxies. What remains is the joint version: fit a model to predict each protected
  attribute from the whole feature matrix and report its accuracy. Features individually below
  the threshold can reconstruct an attribute together, and `employed_life_ratio` at 0.074 is
  the proof that they do.
- **A rule-based adverse action engine.** The reason codes under **Explainability** below are
  built and serve a fixed, reviewed vocabulary rather than free-generated text, which closes
  the mechanical half. What remains is deriving the reasons from the credit policy rather than
  from SHAP attributions, and agreeing the enumerated set with counsel.
- **Ongoing monitoring.** Fairness is not established once at training time. Population drift
  can reintroduce disparate impact from an unchanged model.
- **Reject inference.** The training data contains only accepted applicants, so the model
  learns from a censored population. This biases the model in ways that interact with fairness
  testing.

Three of that list are implemented: the disparate impact screen, the single-feature proxy
scan, and the reason codes. What they have in common is that each is the half of its problem
that does not require judgement - counting approval rates, measuring associations, mapping
attributions to reviewed text. The halves that remain are the ones needing a business
decision, a credit policy, and counsel.

This is a technical demonstration. The honest position is that excluding the protected
attributes, measuring what the remaining decisions look like, finding the fifteen features
that carry those attributes anyway, and saying which of them the model leans on hardest makes
it defensible to discuss. It does not make it deployable.

## Explainability

SHAP `TreeExplainer` against the LightGBM model. Exact on tree models rather than approximated.

Endpoints return JSON, never a rendered image, so the UI owns presentation and the frontend
stays replaceable. `/explain` returns the base value, the calibrated probability, the risk
band, and the top 15 contributions sorted by absolute SHAP value.

Part 4 asks for explanations a **non-technical** user can understand, and a list of signed
floats is not that. Every explanation therefore carries a plain-English narrative:

> This applicant has a 40.2% estimated chance of default, which places them in the High risk
> band. Risk is pushed up mainly by external credit score 3, external credit score 1 and
> external credit score 2. It is pulled down by age.

The narrative is generated deterministically from a feature label map, not by an LLM. It costs
nothing, is reproducible, and cannot invent a reason the model did not use.

Feature values are shown as a reader expects them. Date columns in this dataset are negative
day offsets from the application date, so `days_birth = -9461` is rendered as an age of 25.9
years rather than printed raw.

Global importance is `mean(|SHAP|)` over **all 307,511 applicants**, served from
`models/shap_global.json`. The strongest drivers are the three external credit scores, loan
term, and the price of the goods financed.

It was previously computed over a 2,000-row sample, and the README listed that as a limitation.

```bash
POSTGRES_HOST=localhost python -m src.ml.shap_full
```

`compute_global_importance` materialises the whole SHAP matrix at once, which is 2MB at 2,000
rows and about 295MB at 307,511 - doubled while shap holds both classes, before anything is
reduced. So `src/ml/shap_full.py` walks the population in 20,000-row chunks and keeps only a
running sum of absolute SHAP per feature: 120 floats, whatever the row count. Peak memory is set
by the chunk rather than by the dataset, and the result is exact rather than approximate,
because a mean is a sum divided by a count.

**The ranking did not change.** The top 20 features are in the identical order, the largest rank
move anywhere in that range is zero, and the rank correlation across all 120 features is
**0.9990**. The largest change in any single value is `organization_type`, from 0.12446 to
0.13016 - under 0.006.

That is a finding, not a null result: **the 2,000-row sample was adequate**, and the limitation
the README carried was more cautious than it needed to be. Every SHAP figure previously quoted
here was already correct. Comparison written to `models/shap_sample_comparison.json`.

### Adverse action reason codes

ECOA and Regulation B require a creditor who denies an application to give the applicant the
**specific principal reasons** for the denial. A score, or a list of everything the model
looked at, does not satisfy that. The reasons have to be the ones that drove this decision and
they have to be intelligible to the person receiving them.

The SHAP contributions are the right raw material and are not themselves reason codes.
`ext_source_3 = 0.19, shap_value = +0.847` is a model diagnostic.
`src/ml/reason_codes.py` is the mapping between the two.

```bash
curl -X POST localhost:8000/adverse-action -H 'content-type: application/json' \
  -d '{"sk_id_curr": 100002}'
curl localhost:8000/adverse-action/reason-codes    # the vocabulary itself
```

A separate endpoint rather than extra fields on `/explain`: that response is a published
contract the frontend renders directly, and a disclosure concern should not risk changing a
chart.

**Worked example.** Applicant 100002 scores 0.4491, above `t_high` = 0.083669, so a notice is
required:

| Rank | Reason given | Driven by | SHAP |
|------|--------------|-----------|------|
| 1 | Credit assessment obtained from an external credit bureau | `ext_source_3`, `ext_source_1`, `ext_source_2` | 0.847 |
| 2 | Length of the repayment term relative to the amount requested | `credit_term` | 0.221 |
| 3 | Value of the goods being financed relative to the credit requested | `amt_goods_price` | 0.159 |
| 4 | Length of time in your current employment | `days_employed` | 0.088 |

Applicant 100003 scores 0.0188 and receives `adverse_action_required: false` with an empty
reason list. An approved applicant gets no notice, and the reasons are not computed rather than
computed and withheld.

Four design decisions carry the weight:

- **The phrases are a fixed table in source, versioned as `rc-v1`.** None is generated at
  request time. An adverse action notice is a regulated disclosure: the same input must produce
  the same words, and the wording is something a compliance function signs off once. A
  generated notice cannot be reviewed before it is sent, and two applicants declined for the
  same reason would receive different letters.
- **Four reasons maximum.** Regulation B's model forms list up to four principal reasons and
  treat more as diluting the notice. A floor on contribution size stops an applicant declined
  on two clear factors receiving four, two of which are noise.
- **Duplicate reasons collapse.** The three external bureau scores are three features and one
  reason. The example above shows all three folding into rank 1, which keeps its combined
  ranking weight.
- **Some features are never disclosed.** `weekday_appr_process_start` has no causal story an
  applicant could act on. The geographic columns would make a notice read as redlining whatever
  the model's reason for using them. The social-circle columns describe other people's conduct,
  not the applicant's - both were withheld from applicant 100002's notice above, and the
  response says so in `diagnostics.withheld_non_disclosable` rather than dropping them
  silently.

**The limitation, stated plainly: these are SHAP attributions dressed as reasons, not a
rule-based adverse action engine.** A production system derives reasons from the credit policy.
It knows that "insufficient income for the amount requested" is a reason a lender may give and
that a postcode is not, and it maps to a fixed enumerated set agreed with counsel. Two things
follow. A feature can be a legitimate driver of risk and still be unusable in a letter, which
is why the exclusion list above exists and why it is a judgement call rather than a
calculation. And ranking by SHAP magnitude ranks by influence on this prediction, not by what
a reviewer would call the principal cause; those usually agree, but they are not the same
thing. What is here is the raw material correctly shaped, not the finished compliance control.

## Rule derivation

A 400-tree ensemble cannot go into a credit policy document. A depth-3 surrogate tree is
fitted over the top 8 features by SHAP importance, and its root-to-leaf paths are emitted as
if-then rules. Depth 3 caps a rule at three conditions; restricting the feature set keeps the
conditions on quantities a credit officer recognises.

Two details that matter for correctness:

- Leaf statistics are computed by assigning rows to leaves and counting, **not** read from
  `tree_.value`. With `class_weight="balanced"` that array holds reweighted proportions, and
  reading a default rate off it reports around 78% on a population whose true rate is 8%.
- Repeated splits on the same feature along a path are collapsed into one interval, so a rule
  reads `score is above 0.248 and at most 0.460` rather than listing both bounds separately.

Sample output, against a base default rate of 8.07%:

| Rule | Condition | Support | Default rate | Lift |
|------|-----------|---------|--------------|------|
| R1 | external credit score 2 at most 0.460 and score 3 at most 0.312 | 5.5% (17,005) | 24.06% | 2.98 |
| R2 | external credit score 2 above 0.460 and score 3 at most 0.216 | 4.4% (13,457) | 15.58% | 1.93 |
| R3 | score 2 at most 0.460 and score 3 above 0.312 and at most 0.536 | 15.5% (47,551) | 13.70% | 1.70 |
| R8 | external credit score 2 above 0.460 and score 3 above 0.545 | 27.9% (85,637) | 3.05% | 0.38 |

The eight leaves partition the population, so supports sum to 100%.

An honest observation: every derived rule keys off the external bureau scores. That is faithful
to the model rather than a defect in the derivation, since those three columns dominate SHAP
importance, but it does mean the rules say little about applicant-supplied fields. A second
rule set fitted with the external scores excluded would be more useful as fallback policy for
applicants without bureau coverage, and is listed under improvements.

Surrogate fidelity against the model is 0.798 for the faithful set and 0.714 for the policy set. The rules are a readable approximation of the
model's behaviour, not the model itself.

## Talk-to-data

Ask a question in English, get a business answer backed by SQL that actually ran.

### Control flow

One model call per question, returning a structured **action** rather than bare SQL:

| Action | Meaning | Cost |
|--------|---------|------|
| `sql` | The question maps onto the schema. Query is validated, executed, then summarised. | 2 calls |
| `refuse` | The data cannot answer it. Names the problem, suggests what it can answer. | 1 call |
| `clarify` | Answerable in principle, too vague to write one correct query for. | 1 call |

The action is produced through a required tool call with a strict schema, so the response is
guaranteed to parse. This is the central design decision: **a model with no sanctioned way to
say "I cannot" will invent a column to fill the silence.** Making refusal a first-class output
rather than a parse failure is what turns hallucination control from a hope into a mechanism.
Declining is also the cheap path, which is the right incentive.

### SQL validation gate

Every generated statement passes `src/talk_to_data/query_runner.py` before Postgres sees it:

1. Strip markdown fences.
2. Parse with `sqlglot`. Unparseable is rejected.
3. Exactly one statement.
4. Root node must be a `SELECT`. Writes are detected on the parse tree, not by string
   matching, so `WHERE occupation_type = 'Delete'` is correctly allowed while a `DELETE`
   hidden inside a CTE is correctly rejected.
5. Every table must be one of the three whitelisted.
6. Every column must exist in live `information_schema`. This is the anti-hallucination
   backstop: an invented column cannot reach the database.
7. `LIMIT` forced and clamped to 200.
8. Executed as `riskwright_ro` in a read-only transaction with a 10 second statement timeout.

Layers 4 through 8 are independent, so a defect in the parser is not by itself an incident.
The read-only role is verified to reject `INSERT`, `CREATE`, and `DROP`.

When validation fails, the specific error is fed back for exactly **one** repair attempt. A
second failure becomes a refusal rather than a third API call.

### Prompt engineering

Four versions, all retained in `prompt_templates.py`, all measured against the same 30
questions:

| Version | What changed |
|---------|--------------|
| v1 | Naive baseline. Table names only, "return SQL". |
| v2 | Curated schema with column descriptions and exact categorical values, five few-shot examples, explicit SQL rules. |
| v3 | Structured action output, so refusal and clarification become first-class. Semantic notes on negative day offsets, absent calendar dates, and the non-existent `credit_score`. |
| v4 | Join discipline: use `EXISTS` rather than join-then-average, and no `GROUP BY` unless a breakdown was asked for. |

### Token optimization

The schema block is **1,656 tokens**, not the roughly 6,000 a full dump of all 176 columns
would cost. Three decisions get it there:

- **Curated columns.** Roughly 60 business-relevant columns out of 176. Fewer tokens, and
  fewer chances to pick a column that exists but does not mean what was asked.
- **Real descriptions.** One short definition per column, taken from the dataset's own data
  dictionary. This is what stops the model reading `days_birth` as a date.
- **Exact categorical values.** The distinct values of low-cardinality columns are sent
  inline. Without them the model writes `name_education_type = 'Higher Ed'`, the query runs,
  and zero rows come back with no error anywhere. This is the highest-value item in the
  prompt.

Conversation memory replays only the prior question and the SQL that answered it, never the
result rows. A remembered turn costs about 40 tokens instead of several hundred.

**Prompt caching is wired but does not currently engage, and that is a deliberate trade.**
The cache breakpoint sits correctly after the stable system block. Haiku 4.5 will not cache a
prefix below 4,096 tokens, and this prompt is about 3,000. Verified rather than assumed: two
identical calls at the real size both report zero cache activity, while the same prompt padded
to 7,393 tokens shows a cache write followed by a cache read. Reaching the threshold would
mean adding filler, or re-adding the columns that were curated out to reduce wrong-column
errors, to save roughly seven cents per full evaluation run. Compactness wins at this volume.
The breakpoint stays because it costs nothing and starts working if the schema ever grows.

### Conversation memory

Server-side, keyed by a `session_id` the client sends, held in-process with a one hour TTL and
a six turn replay window. Deliberately not in client state: if history lived in the UI,
replacing the frontend would mean reimplementing memory and the API could not answer a
follow-up on its own. That was not hypothetical - the frontend was in fact replaced, and
memory did not move.

Refusals are replayed too. Without that, the model re-attempts a question it has already
correctly refused.

Working example over HTTP:

> **Q:** What is the default rate for applicants with higher education?
> **A:** The default rate for applicants with higher education is 5.36%.
>
> **Q:** And for those with only secondary education?
> **A:** The default rate for those with only secondary education is 8.94%.

The second question names no table, no column, and no metric.

### Evaluation

The chatbot is scored the way a model is scored, not demoed on five questions that happen to
work. `tests/chatbot_eval.yaml` holds 30 questions across eight categories, **written against
the schema before any prompt was tuned**, so the harness measures rather than mirrors.

Each SQL question carries a hand-written **reference query**. The harness executes both the
generated and the reference query and compares results, so correct SQL written differently
still passes. Refusal questions are scored on whether the bot declined.

```bash
python -m tests.run_chatbot_eval --versions v1 v2 v3 v4   # development set
python -m tests.run_chatbot_eval --heldout --versions v4  # held-out set
python -m tests.run_chatbot_variance --runs 5             # the same set, five times
```

**Headline: a mean of 38.4/44 on held-out questions written after the prompt was frozen,
ranging from 37 to 40 across five runs, against 100% on the development set.** The held-out
number is the one to trust, and the range matters more than the mean: this is a sampled system
and a single score is one draw from a distribution, not a property of it.

#### Held-out set

The 30 development questions stopped being a clean measurement the moment v4 was tuned on the
failures they exposed. At that point they are a training set, and a score on your own training
set says nothing about generalisation.

`tests/chatbot_heldout.yaml` therefore holds **44 questions written after the prompt was
frozen**, in three batches, none derived from any observed failure. They deliberately use
shapes absent from the development set: percentiles, multi-condition filters,
column-to-column ratios, distinct counts, per-applicant averaging over a child table,
conditional aggregates, NULL semantics, negated existence, aggregate ratios, unit conversion,
and thresholded group counts. Thirty-seven categories across 33 SQL questions, 6 refusals and 5
clarifications. The third batch of 24 adds dispersion, correlation, HAVING, anti-joins,
three-table joins, nested aggregates, weighted averages, mode, share-of-total, a legitimately
empty result, and four further ways for a question to be unanswerable.

| Set | Questions | v4 |
|-----|-----------|-----|
| Held out, written after freeze | 44 | **38.4/44 mean (87%), range 37-40, sd 1.34** |
| Development, used for tuning | 30 | 30/30, 100% |

#### The score is a distribution, not a number

The temperature is not zero, so the same questions do not produce the same answers twice. Five
consecutive runs at v4 over the full 44-question set, no prompt changes between them:

| Run | Score |
|-----|-------|
| 1 | 39/44 |
| 2 | 37/44 |
| 3 | 40/44 |
| 4 | 39/44 |
| 5 | 37/44 |

Mean **38.4/44 (87%)**, range 37-40, standard deviation 1.34. Figures are read from
`models/chatbot_variance.json`, written by `python -m tests.run_chatbot_variance`. The earlier
20-question measurement - mean 17.8/20, range 16-19 - is retained in that file under
`prior_history`, because a mean over 44 questions is not comparable to a mean over 20 and
overwriting it would erase the basis for the figure this README used to quote.

**This section has now corrected the same claim twice, which is the point of measuring it.** A
single run once scored 19/20 and was reported as the held-out result; it was the best of five.
Quoting the good draw is how an evaluation flatters itself.

Thirty-four of the forty-four questions are perfectly stable at 5/5. Eight move, and two fail
every run.

| ID | Category | Passed | What varies |
|----|----------|--------|-------------|
| H5 | join | 1/5 | "On average, how many previous applications does an applicant have?" Returns 4.929 or 4.597 depending on whether the denominator is applicants who *have* prior applications or all applicants. Both are legitimate; the reference picks one. |
| H2 | multi-filter | 2/5 | "How many unemployed applicants own property?" Alternates between answering and asking what "unemployed" means. A defensible hesitation. |
| H9 | conditional aggregate | 3/5 | Returns the share as 0.2353 or as 23.53. Identical quantity, the other conventional unit. |
| H29 | weighted average | 3/5 | Sometimes asks what "weighted by family members" should mean rather than computing it. |
| H23 | HAVING clause | 4/5 | Occasionally counts occupations including the null group. |
| H30 | cross-group difference | 4/5 | Occasionally returns both rates rather than the difference asked for. |
| H38 | ratio of ratios | 4/5 | Occasionally asks whether "repayment relative to income" means the annuity or the total. |
| H44 | causal overreach | 1/5 | "Does having more children cause applicants to default?" Answers the association directly in four runs of five instead of distinguishing correlation from causation. |

**H44 is a genuine finding and a new one.** The refusal machinery handles a question the data
*cannot* answer. It does not handle a question the data can answer but not in the sense asked.
The model computes the association and presents it as the answer, which is a subtler failure
than inventing a column and one no schema check can catch.

#### The two questions that fail every run, and why one of them is my fault

Reported rather than fixed. Repairing a reference query after watching the model disagree with
it is fitting the grader to the outcome, which is what a held-out set exists to prevent.

**H35 - the reference is arguably wrong and the model is arguably right.** "How many applicants
have at least one active bureau credit?" The reference counts distinct ids in `bureau`, giving
251,815. The model joins to `application_train` first and gets 217,150. `bureau` contains ids
that are not applicants in `application_train`, so the model's reading of the word "applicants"
is the better one. It is scored as a failure regardless.

**H36 - the reference is right and the model ignored the question.** "Excluding the
not-applicable codes, which cash loan purpose is most frequent?" The answer is `Repairs`. The
model returned `XAP`, which *is* the not-applicable code the question told it to exclude. In
mitigation, nothing in the schema block tells the model that `XAP` and `XNA` are sentinels for
that column, so it could not know which codes to drop - but it did not ask, either. That is a
fair criticism of the question and a real failure to honour an explicit constraint.

**Two of the 5.6 average failures are therefore attributable to how I wrote the questions, not
to the model.** The honest reading of 38.4/44 is that the model-attributable score is nearer
40/44, and that a set authored by one person has authoring defects in roughly 5% of its
questions. That is the argument for an independent set, made concrete.

An earlier, narrower held-out batch of 8 questions scored v3 at 7/8 twice and v4 at 6/8 then
7/8, which showed **v4 had no reproducible advantage over v3 out of sample** even though it
gained two questions on the development set. Those 8 are the first batch of the 20 above. The
variance measured above is the same effect seen properly: two runs were enough to suggest the
scores moved, and five are enough to say by how much.

#### Development set, as tuning history

| Version | Accuracy | Passed |
|---------|----------|--------|
| v1 naive | 53% | 16/30 |
| v2 compact schema, few-shot | 87% | 26/30 |
| v3 structured action output | 93% | 28/30 |
| v4 join discipline | 100% | 30/30 |

Run-to-run variance is one to two questions: across two runs v1 scored 57% then 53% and v2
scored 83% then 87%. Only the v1 to v2 step, worth ten questions, is clearly beyond noise. The
v2 to v3 step is smaller but consistent and concentrated in the refusal and ambiguity
categories, which is what it was designed to move.

#### The failure that matters most

Held-out question H8 asks "How many applicants were approved for the loan they applied for?"
The dataset cannot answer this: `application_train` records whether an applicant later
defaulted, not whether their application was approved, and `name_contract_status` in
`previous_application` describes *prior* applications.

**Over ten runs across two evaluations it declined correctly seven times and answered
three.** The split is uneven and instructive: in the first five-run block it declined twice of
five, in the second it declined five of five. Same question, same prompt, same model. A
per-question rate estimated from five observations has a confidence interval wide enough to
contain both, which is a caution about every per-question figure in this section, including
the ones that look stable.

When it does answer, it answers like this:

> 290,065 applicants were approved for the loan they applied for.

That number is real, the SQL was valid, and it answers a different question than the one
asked. It is the most dangerous output this system can produce, because nothing about it looks
wrong.

**Refusal is therefore probabilistic, not guaranteed: 7 of 10 on the question designed to
test it.** The schema validator makes it impossible to query a column that does not exist,
which is a hard guarantee. Declining a question that is semantically unanswerable from columns
that *do* exist is a judgement the model makes, and it does not make it identically every time.

The expanded set added three more unanswerable shapes and they behave better: a question about
an unloaded table (H40), one about calendar dates that do not exist in the schema (H41), and a
subjective question with no defined quantity (H42) were each handled correctly 5/5. What the
expansion also found is a worse case than H8 - H44, where the data can answer the question
asked but not in the causal sense intended, and the model answers anyway 4 times in 5.

Note what the failure is not. In the runs where it answered, it did not invent a column: it
used real columns to compute a real number that answers a neighbouring question. Schema
validation cannot catch that, because there is nothing invalid about the SQL. A production
system would need a second check on whether the query actually answers what was asked, and
that check is not built here.

Four questions across two evaluations is still a thin basis for a rate. The honest reading is
"this fails often enough to matter", not a percentage anyone should quote.

#### The v3 failures that produced v4

Both were the same root cause:

| ID | Question | What went wrong |
|----|----------|-----------------|
| Q17 | Default rate for applicants with a prior refused application | Added a spurious `GROUP BY a.sk_id_curr`, returning one row per applicant instead of one overall rate. The model's own answer text gave it away: "ranges from 0.0% to 100.0%". |
| Q19 | Do applicants with active bureau credits default more often | `SELECT DISTINCT` over a join placed applicants holding both an active and a closed bureau credit into **both** sides of the comparison, contaminating the "without" group. |

Both are many-to-one join fan-out, which the schema notes already warned about in prose. v4
replaced the warning with a specific instruction to use `EXISTS`, and both now pass.

## Exploratory analysis

`notebooks/eda.ipynb` and its `notebooks/eda.py` export cover the dataset summary, data
quality, feature categorization, five business insights, and supporting charts. Both import
from `src/data/eda.py`, the same module the API serves from, so the notebook, the UI, and this
README cannot quote different numbers for the same finding.

Five findings, each computed rather than asserted:

1. **A sentinel value hides in the employment column.** 18.0% of applicants have
   `days_employed` set to 365243, roughly 1,000 years, encoding "not currently employed"
   rather than a duration. Left untreated it corrupts every statistic built on the column.
   Those applicants default at 5.4% against 8.7% for everyone else, so the fact is kept as a
   flag rather than thrown away with the value.
2. **External credit scores separate risk more than anything the applicant reports.** The
   lowest quartile of `ext_source_3` defaults at 15.1% against 3.5% in the highest, a 4.3x
   spread from one column.
3. **Education tracks default strongly.** Lower secondary 10.9%, Academic degree 1.8%.
4. **Borrowing heavily relative to income does not predict default, which is not what you
   would expect.** Default rate peaks in the 2x to 4x band at 8.77% and is *lowest* among the
   most leveraged borrowers, 7.23% above 6x, below the 7.48% seen under 2x. This looks like a
   selection effect: a loan worth six times income is presumably only written for applicants
   who cleared other checks. The engineered `credit_income_ratio` ranks 37th of 126 by SHAP
   importance, consistent with a weak signal, while loan term ranks 3rd. A credit policy built
   on a leverage cap would be targeting the wrong quantity.
5. **Younger applicants default substantially more often.** 11.4% under 30, falling to 4.9%
   at 60 and over.

Insight 4 is the one worth pausing on. It is a negative result that contradicts the obvious
hypothesis, and it is reported because it changes what a policy built on this data should do.

## User interface

Five sections, each a pure client of the API. No business logic, no database access and no
model loading in the UI layer: every value on screen arrived over HTTP.

The client is React, built with Vite and served by nginx. It is the only UI in the stack and
starts with `docker-compose up`.

| Section | Contents |
|---------|----------|
| Data understanding | Dataset metrics, missing values, the five insights with charts, a distribution explorer, and default rate by category |
| Risk prediction | Score an existing applicant or a what-if applicant; probability, band, and recommended action |
| Why this decision | Diverging SHAP contribution chart plus the plain-English narrative, and global importance |
| Derived rules | Both rule sets with support, default rate, and lift |
| Ask the data | Chat with generated SQL shown in an expander, results as a table, refusals rendered distinctly |

Charts are drawn with Recharts in the browser. The API never renders a figure: a
server-rendered image would put presentation logic behind the API and make the frontend hard
to replace. Aggregation happens server-side; raw rows never reach the browser.

### A note on Streamlit

Streamlit was the development client. It was built first, against the same endpoints, because
it gets a working multi-section UI up in an afternoon and let the API contract be exercised
early. It was replaced by React before submission and is no longer part of the project.

The swap cost one compose service rather than a rewrite, which is the point of the layering:
the UI holds no business logic, reads every value over HTTP, and keeps conversation state
server-side behind a session id, so nothing that computes anything had to move.

The React client is Vite plus React, built in a multi-stage Dockerfile and served by nginx,
which also proxies `/api/*` to the API service. The browser therefore never learns the API's
host, and no `VITE_*` variable is used: Vite inlines those into the client bundle at build
time, so the Anthropic key stays in the `api` service where it is set. The built bundle is
checked for key material as part of the release routine.

## Presentation

`documents/project_presentation.pdf` covers the use case with output screenshots, per the
submission instructions. Twenty-one slides: architecture, the calibration and cost-threshold
reasoning, fair lending across three slides, the model comparison, EDA, explainability, rules,
engineering, limitations, and one screenshot per module.

The screenshots it embeds are kept alongside it in `documents/screenshots/`, since the PDF
holds them only at export quality.

**The deck is a build artifact without its build.** Its figures were generated from
`models/*.json` rather than transcribed, but the scripts that did so are no longer in the
repository, so the PDF cannot be regenerated from the artifacts. If a figure in `models/`
changes, the deck will not follow it. Every other number in this README is still checked
against the artifacts, and the deck agreed with them when it was exported.

## Deployment

The platform is packaged to run hosted from three published images, with the
database seeded from a local run of the same loader. The images are built and
pushed from this commit, so a hosted instance runs exactly what the repository
describes.

**Live: [http://159.65.157.86](http://159.65.157.86)**

A DigitalOcean droplet in Bangalore (2 vCPU, 4GB, Ubuntu 24.04) running the
three published images, provisioned and deployed by the procedure in
[deploy/README.md](deploy/README.md).

Verified against that host rather than assumed: all three containers healthy,
**307,511 / 1,716,428 / 1,670,214** rows restored across the three tables,
`riskwright_ro` holding exactly three SELECT grants and refused on INSERT,
UPDATE, DELETE, CREATE and DROP, fifteen endpoints answering through the nginx
`/api/` proxy, a live chatbot question returning 8.07% from generated SQL
against the deployed database, all five UI sections driven in Chrome with no
console errors and no 4xx/5xx, none of the sixty served files containing a key
or a `VITE_` variable, and **ports 8000 and 5432 closed from the public
internet** while 80 is open.

| Image | Contents |
|-------|----------|
| `likithsh3tty/riskwright-api:v2.3` | Built from the repository's `Dockerfile`, unchanged |
| `likithsh3tty/riskwright-frontend:v2.3` | Built from `frontend/Dockerfile`, unchanged |
| `likithsh3tty/riskwright-postgres:v2.3` | `postgres:16-alpine` plus a dump of the loaded database and the read-only role |

`deploy/docker-compose.deploy.yml` differs from this repository's
`docker-compose.yml` in exactly two ways: it **pulls** the images rather than
building them, and it has **no loader service**, because the data ships inside
the postgres image. The healthchecks, dependency conditions, restart policy and
the rule that only the frontend is published are identical.

The loader does not run on the VM. It streams 740MB of CSV through pandas, and
doing that on a small host would also mean putting Kaggle credentials on a
public box. Seeding from a dump means what is deployed is the *output* of the
verified loader rather than a second implementation of it. `pg_dump` does not
carry roles, so the image reprovisions `riskwright_ro` on first boot and checks
its grants - the SELECT-on-three-tables property is a documented security claim
and had to survive the move.

The Anthropic key is supplied on the host at run time. It is in no image and in
no file in this repository.

Host setup, sizing and troubleshooting: **[deploy/README.md](deploy/README.md)**.

## Known limitations

Specific rather than vague, because the vague version is useless to anyone deciding whether to
trust this.

**Model**

- Trained on `application_train` only. `bureau` and `previous_application` are loaded for the
  chatbot but contribute no features. Published solutions gain roughly 0.02 to 0.03 ROC-AUC
  from prior credit history; that is the largest single improvement available.
- No hyperparameter search. Deliberate, but it means the reported 0.7614 is a floor.
- Protected attributes are excluded from the model (see **Fair lending** above), at a measured
  cost of 0.0037 ROC-AUC. **Exclusion did not hold, and the systematic pass says how badly:
  fifteen features are a material proxy for at least one excluded attribute**, twelve of them
  for age and five of those strongly. `ext_source_1` reconstructs age at 0.600 and is the third
  most important feature in the model. Documented rather than removed, because removing any of
  them requires a retrain and invalidates every figure reported here.
- **Two of three protected attributes fail the four-fifths screen** at the deployed threshold:
  age band at 0.5602 and `name_family_status` at 0.7885. `code_gender` passes at 0.8528. That
  is a screening flag, not a finding of disparate impact - the business-necessity analysis the
  legal test requires is not performed.
- Still not done, now specifically rather than generally: the **business-necessity analysis**
  that would say whether either failing ratio is justified; **multivariate** proxy detection,
  since the pass above measures one feature at a time and `employed_life_ratio` at 0.074 proves
  features below the floor can still carry an attribute; a **rule-based adverse action engine**
  working from the credit policy rather than from SHAP attributions; and **reject inference**.
  Not deployable without that work.
- **The 10:1 cost ratio is still an assumption, but a bounded one.** Sweeping 3:1 to 20:1 moves
  `t_high` from 0.0407 to 0.2284 - 2.2 times the shipped threshold - and the approval rate from
  46% to 95%. Only 51.5% of applicants keep the same risk band across that range, so the
  assumption decides the outcome for nearly half the book. Choosing by cost beats 0.5 at every
  ratio, so the method survives; the number needs a real recovery model. Measured in
  `models/threshold_sensitivity.json`.

**Explainability**

- Global SHAP importance is now exhaustive: all 307,511 applicants, computed in chunks.
  The previous 2,000-row sample produced an identical top-20 ordering and a rank
  correlation of 0.9990 across all 120 features, so the sampling was never distorting
  anything. Per-prediction SHAP was always exact.
- SHAP values are in log-odds on the model's weighted scale. The probability is calibrated;
  the contributions describe direction and relative size, not percentage points.
- The narrative is template-generated. It is reliable and cheap, and it will not phrase an
  unusual combination of drivers as fluently as a model would.

**Rules**

- The faithful rule set agrees with the model on 79.8% of applicants, the policy set on 71.4%.
  Neither is the model, and the gap is where a rule-based decision would differ from a scored
  one.
- The faithful set keys almost entirely off external bureau scores, so it says little a credit
  officer can act on. That is what the policy set exists for.

**Chatbot**

- **Refusal is probabilistic, not guaranteed.** The schema validator makes querying a
  non-existent column impossible, which is a hard guarantee. Declining a question that is
  semantically unanswerable from columns that *do* exist is a model judgement: across five
  runs, held-out question H8 was declined twice and answered wrongly three times. When it
  answers it uses real columns to compute a real number for a neighbouring question, which
  no schema check can catch.
- **Held-out accuracy is a distribution, not a number: mean 38.4/44, range 37-40 over five
  runs**, against 100% on the development set. The development set was used for tuning and its
  score is not a generalisation estimate. Thirty-four of forty-four questions are stable at
  5/5; the eight that move and the two that fail every run are listed under the evaluation
  section.
- **Two of the average 5.6 failures are defects in questions I wrote, not model failures.** H35
  has a contestable reference query and H36 assumes knowledge of sentinel codes the schema
  block never supplies. They are reported rather than repaired, because fixing a reference
  after seeing the model disagree is fitting the grader to the outcome. The model-attributable
  score is nearer 40/44.
- **Forty-four questions written by the person who wrote the prompt is still the weakest part
  of this evaluation.** The second batch was written to cover shapes the first lacked, but its
  author had already seen which of H1-H20 were unstable; that is disclosed at the top of
  `tests/chatbot_heldout.yaml`. An independently authored set remains the right next step, and
  the 5% authoring-defect rate found above is the concrete argument for it.
- Five runs is a small sample for a variance estimate, and per-question rates rest on five
  observations each. H8 declined 2/5 in the first evaluation and 5/5 in the second with nothing
  changed between them, which is how wide those intervals are.
- Conversation memory is in-process: it does not survive an API restart and does not scale
  beyond one container. Redis is the obvious next step and is not warranted at this size.
- Prompt caching does not engage, because the prompt is smaller than Haiku 4.5's 4,096 token
  minimum. Measured, not assumed, and a deliberate trade against prompt compactness.
- Only three tables are exposed. Questions needing the other four are refused correctly but
  are refused nonetheless.

**Demonstration versus measurement**

- **Applicants shown in the UI are in the training set.** `full_train` refits both models on
  all 307,511 rows before saving them, which is the right production choice but means no row
  of `application_train` is out-of-sample for the served model. The cross-validation folds
  were a measurement device and are not retained.
- The reported metrics are unaffected by this. ROC-AUC, PR-AUC, the calibration identity, the
  cost threshold and the fair-lending delta all come from 5-fold cross-validation, where every
  scored row was out of fold. What the UI demonstrates is the interface, the SHAP
  decomposition and the banding, none of which is a performance claim.
- An individual prediction shown in the UI should therefore not be read as evidence of
  generalisation. Serving a model deliberately fitted on a subset, purely so the demo could
  be out-of-sample, would mean the published metrics no longer described the deployed
  artifact. That is a worse trade than this caveat.

**Engineering**

- Model artifacts are committed to git. This contradicts a common convention but follows the
  assignment, which lists saved model artifacts as a repository deliverable, and it means a
  fresh clone can serve predictions without training first.
- No authentication on the API. Appropriate for an assignment, not for anything else.
- The 176 tests cover `src/` and the `app/` request/response contracts. **The React UI has
  no automated coverage** and is verified by hand against a running stack. Nor is anything
  that needs a live database or a live LLM covered: the API tests mock the data layer, so
  what is asserted is the contract, not the SQL underneath it. The chatbot is measured by
  the evaluation harness instead of by pytest, which is a different kind of evidence.
- The `employed_life_ratio` leak was found by hand, comparing what two frontends exposed,
  not by a test. That is a fair indication of what hand-verification catches and what it
  does not. There is now a test pinning the feature as *present*, so it cannot be removed
  without the documentation moving with it, but nothing would have found it in the first
  place.

---
## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

176 tests, no database required.

| File | Tests | What it covers |
|------|-------|----------------|
| `test_api.py` | 39 | Endpoint contracts. `/predict` and `/explain` run the committed model and the real SHAP explainer against a synthetic applicant; only the Postgres reads are mocked. |
| `test_query_runner.py` | 23 | The SQL validation gate: single-SELECT enforcement, table and column whitelisting, forced LIMIT. |
| `test_fairness_audit.py` | 22 | Four-fifths arithmetic, the small-group floor, approval rate at the deployed threshold, empty groups. |
| `test_proxy_detection.py` | 20 | Spearman, correlation ratio and Cramér's V arithmetic; the bias correction; the minimum level size; threshold partitioning. |
| `test_reason_codes.py` | 19 | Who gets a notice, the four-reason cap, the contribution floor, the non-disclosable set, phrase collapsing. |
| `test_threshold_sensitivity.py` | 16 | That the sweep reproduces the shipped threshold exactly, sweep monotonicity, and that it never rewrites `threshold.json`. |
| `test_schema_context.py` | 14 | The compact schema block: tables and columns advertised, descriptions attached, token budget. |
| `test_docker_utils.py` | 13 | Container-safe path resolution. |
| `test_preprocessor.py` | 10 | Cleaning, feature derivation, the train/serve skew guard, the calibration identity. |

Where a constant is a judgement call, the test asserts the counterfactual rather than the
happy path: the proxy-detection floor is tested by showing that four rows out of a thousand
take a statistic from 0.26 to 0.99 when it is lifted, the reason-code exclusion list by showing
that without it a notice would lead with the applicant's region, and the four-reason cap by
showing which two reasons get dropped. A constant nobody can break is not being tested.

Two are worth calling out. `test_no_protected_attribute_appears_in_any_contribution` feeds an
applicant row that *does* carry `code_gender`, `name_family_status`, `cnt_children` and
`cnt_fam_members` through the real serving path and asserts none of them reaches a SHAP
contribution: the fair-lending claim as a regression guard rather than a paragraph. And
`test_the_known_age_proxy_is_still_present_and_still_documented` asserts the opposite of what
you would expect, pinning `employed_life_ratio` as present so that removing it cannot happen
without the retrain and the documentation moving too.

The React UI has no automated coverage; that gap is listed under limitations.
