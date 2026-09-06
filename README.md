# Riskwright

An AI-powered credit risk platform built on the Home Credit Default Risk dataset. It predicts
loan default probability, explains every prediction, derives business-readable credit rules,
and lets a non-technical user query the data in plain English.

Built for the NeoStats AI Engineer assignment.

> **Status:** all six modules complete and verified against a running stack.

---

## What it does

| Module | Status |
|--------|--------|
| Data understanding and EDA | done |
| Talk-to-data (natural language to SQL) | done |
| Machine learning layer (default probability, risk band) | done |
| Explainable AI (SHAP per prediction) | done |
| Business-readable decision rules | done |
| Multi-section user interface | done, Streamlit and React |
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
documents/                Project presentation (PDF)
notebooks/                eda.ipynb and its eda.py export
src/
  data/loader.py          CSV to Postgres, idempotent
  data/preprocessor.py    Cleaning, feature derivation, encoding
  ml/                     Training, inference, evaluation, SHAP, rule derivation
  talk_to_data/           Natural language to SQL, validation, prompts, memory
  utils/                  Logging, configuration, helpers, container-safe paths
app/                      FastAPI. A thin wrapper over src/, no business logic.
ui/                       Streamlit client. Calls HTTP endpoints only.
frontend/                 Optional React client, nginx served. Same endpoints.
sql/schema.sql            Generated DDL for the three loaded tables
configs/                  Configuration and the official column glossary
models/                   Saved model artifacts
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
| `app/` | FastAPI layer, so the UI holds no business logic and stays replaceable |
| `ui/` | Part 5 requires a UI; the tree names no location for one |
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

To also start the optional React frontend, add the override file:

```bash
docker-compose -f docker-compose.yml -f docker-compose.react.yml up
```

Then open:

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| React UI (only with the override) | http://localhost:5173 |
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
pytest                             # 46 tests, no database needed

uvicorn app.main:app --reload
streamlit run ui/app.py            # needs API_URL=http://localhost:8000
```

---

## Architecture

```
                    +-------------------+
                    |  Streamlit UI     |   no business logic,
                    |  (ui/)            |   HTTP client only
                    +---------+---------+
                              | HTTP
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
| `ui` | Streamlit. Starts only after the API's own healthcheck passes. |

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
| Streamlit first, React added after | Streamlit carried the submission; React was added later as an optional upgrade that changes nothing about the API. See below. |
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

### Reproducing

```bash
python -m src.ml.train              # 5-fold CV, both models, then SHAP and rules
python -m src.ml.train --quick      # single split, LightGBM only, about 60 seconds
python -m src.ml.train --artifacts  # rebuild SHAP and rules from the saved model
```

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

### Residual proxies, not excluded

Removing a protected attribute does not make a model fair. It removes the most obvious defect.
These remain and are genuine risks:

- **`name_income_type`** contains the values `Maternity leave` and `Pensioner`, which are
  proxies for sex and age respectively. The column carries real, legitimate signal about
  income stability, so it is kept, but a production system would need to demonstrate that its
  use is not a pretext.
- **`occupation_type` and `organization_type`** correlate with sex and national origin in most
  labour markets.
- **`region_rating_client` and `region_population_relative`** are geographic and therefore
  carry redlining risk, the classic proxy failure in credit.
- **`days_employed`** correlates with age, though it measures something a lender may
  legitimately consider.

### What a production system would additionally need

Excluding the attributes is the floor, not the bar:

- **Disparate impact testing.** Compare approval and default rates across protected groups at
  the chosen threshold, using the attributes for *testing* while keeping them out of the
  *model*. The four-fifths rule is the usual starting screen.
- **Proxy detection.** Fit a model to predict each protected attribute from the remaining
  features. High accuracy means the model can reconstruct it regardless of exclusion.
- **Adverse action reason codes.** ECOA requires a declined applicant to be told why. The SHAP
  contributions are the right raw material, but they need mapping to a fixed, reviewed set of
  reasons, not free-generated text.
- **Ongoing monitoring.** Fairness is not established once at training time. Population drift
  can reintroduce disparate impact from an unchanged model.
- **Reject inference.** The training data contains only accepted applicants, so the model
  learns from a censored population. This biases the model in ways that interact with fairness
  testing.

None of that is implemented here. This is a technical demonstration, and the honest position
is that removing the protected attributes makes it defensible to discuss, not deployable.

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

Global importance is `mean(|SHAP|)` over a 2,000-row sample, computed once at training time
and served from `models/shap_global.json`. The strongest drivers are the three external credit
scores, loan term, and the price of the goods financed.

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
a six turn replay window. Deliberately not in Streamlit session state: if history lived in the
UI, replacing the frontend would mean reimplementing memory and the API could not answer a
follow-up on its own.

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
```

**Headline: 19/20 on held-out questions written after the prompt was frozen, against 100% on
the development set.** The held-out number is the one to trust.

#### Held-out set

The 30 development questions stopped being a clean measurement the moment v4 was tuned on the
failures they exposed. At that point they are a training set, and a score on your own training
set says nothing about generalisation.

`tests/chatbot_heldout.yaml` therefore holds **20 questions written after the prompt was
frozen**, in two batches, neither derived from any observed failure. They deliberately use
shapes absent from the development set: percentiles, multi-condition filters,
column-to-column ratios, distinct counts, per-applicant averaging over a child table,
conditional aggregates, NULL semantics, negated existence, aggregate ratios, unit conversion,
and thresholded group counts. Sixteen categories across 14 SQL questions, 3 refusals and 3
clarifications.

| Set | Questions | v4 |
|-----|-----------|-----|
| Held out, written after freeze | 20 | **19/20, 95%** |
| Development, used for tuning | 30 | 30/30, 100% |

Every held-out category passed except one. By category on the held-out set: aggregate ratio
1/1, ambiguous 3/3, column ratio 1/1, conditional aggregate 0/1, distinct count 1/1, filtered
extremum 1/1, group-by ranking 1/1, join 2/2, multi-filter 1/1, negated existence 1/1,
non-existent column 2/2, null semantics 1/1, percentile 1/1, thresholded group count 1/1,
unanswerable 1/1, unit conversion 1/1.

**The one failure is worth reading, because it is arguably the harness being wrong rather than
the model.** H9 asks "What share of applicants own both a car and a property?". The reference
returns a fraction, 0.235309. The model returned 23.53% and phrased it as "23.53% of applicants
own both a car and a property". It computed the identical quantity and expressed it in the
other conventional unit for the word "share". It is scored as a failure here regardless:
relaxing the comparison after seeing the result would be fitting the grader to the outcome,
which is exactly what a held-out set exists to prevent.

An earlier, narrower held-out batch of 8 questions scored v3 at 7/8 twice and v4 at 6/8 then
7/8, which showed **v4 had no reproducible advantage over v3 out of sample** even though it
gained two questions on the development set. Those 8 are the first batch of the 20 above.

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

In one run v4 refused it correctly. In another it answered:

> 290,065 applicants were approved for the loan they applied for.

That number is real, the SQL was valid, and it answers a different question than the one
asked. It is the most dangerous output this system can produce, because nothing about it looks
wrong.

**Refusal is therefore probabilistic, not guaranteed.** The schema validator makes it
impossible to query a column that does not exist, which is a hard guarantee. Declining a
question that is semantically unanswerable from columns that *do* exist is a judgement the
model makes, and it does not make it identically every time. That distinction is stated here
rather than buried, because a reader is entitled to know which safety properties are enforced
and which are merely likely.

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

There are two implementations of these same five sections, Streamlit and React, described
below. Both consume the same endpoints. The screenshots in the presentation are from the React
client.

| Section | Contents |
|---------|----------|
| Data understanding | Dataset metrics, missing values, the five insights with charts, a distribution explorer, and default rate by category |
| Risk prediction | Score an existing applicant or a what-if applicant; probability, band, and recommended action |
| Why this decision | Diverging SHAP contribution chart plus the plain-English narrative, and global importance |
| Derived rules | Both rule sets with support, default rate, and lift |
| Ask the data | Chat with generated SQL shown in an expander, results as a table, refusals rendered distinctly |

The Streamlit client draws with Plotly and the React client with Recharts. Neither renders a
figure on the server: a server-rendered image would put presentation logic behind the API and
make the frontend hard to replace. Aggregation happens server-side; raw rows never reach the
browser.

### Two frontends, and why

Streamlit came first and is what the submission was built and verified around. The evaluation
criteria contain no line for frontend quality, so the time that a React build would have taken
went into the chatbot evaluation harness, the held-out question set, and the fair lending work
instead, all of which sit against criteria that are scored.

A React frontend was then added afterwards as a strict upgrade, once the Streamlit build was
complete and tagged. It consumes exactly the same endpoints, adds no API surface, and lives
behind a separate compose file so the original stack is untouched:

```bash
docker-compose up                    # Postgres, API, Streamlit on 8501

docker-compose -f docker-compose.yml -f docker-compose.react.yml up
                                     # the same, plus React on 5173
```

Both UIs can run at once because they bind different ports. `docker-compose.yml` is byte
identical to the version tagged `v1.2-submission`; the React service is added purely by the
override file.

The React client is Vite plus React, built in a multi-stage Dockerfile and served by nginx,
which also proxies `/api/*` to the API service. The browser therefore never learns the API's
host, and no `VITE_*` variable is used: Vite inlines those into the client bundle at build
time, so the Anthropic key stays in the `api` service where it is set. The built bundle is
checked for key material as part of the release routine.

That this was cheap to add is the point of the layering. The UI holds no business logic, reads
every value over HTTP, and keeps conversation state server side behind a session id, so a
second frontend was a new compose service rather than a rewrite.

## Presentation

`documents/project_presentation.pdf` covers the use case with output screenshots, per the
submission instructions. It is generated by `documents/build_presentation.py`:

```bash
POSTGRES_HOST=localhost python documents/build_presentation.py
```

**Every figure in the deck is read from `models/*.json` at build time rather than transcribed.**
That is not decoration. Building it immediately surfaced a number in this README that had been
typed by hand and was wrong: the naive-threshold precision read 0.545 where the artifact said
0.659. Generating from the source of truth catches transcription drift; typing numbers twice
invites it.

## Known limitations

Specific rather than vague, because the vague version is useless to anyone deciding whether to
trust this.

**Model**

- Trained on `application_train` only. `bureau` and `previous_application` are loaded for the
  chatbot but contribute no features. Published solutions gain roughly 0.02 to 0.03 ROC-AUC
  from prior credit history; that is the largest single improvement available.
- No hyperparameter search. Deliberate, but it means the reported 0.7653 is a floor.
- Protected attributes are excluded from the model (see **Fair lending** above), at a measured
  cost of 0.0037 ROC-AUC. What is *not* done: disparate impact testing, proxy detection,
  adverse action reason codes, and reject inference. Several residual proxies remain and are
  listed in that section. Not deployable without that work.
- The 10:1 cost ratio is an assumption, not a measurement. Every threshold and band moves if a
  real recovery model replaces it.

**Explainability**

- Global SHAP importance is sampled at 2,000 rows, not exhaustive.
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
  semantically unanswerable from columns that *do* exist is a model judgement, and held-out
  question H8 was refused in one run and answered wrongly in another.
- Held-out accuracy is 19/20 against 100% on the development set. The development set was used
  for tuning and its score is not a generalisation estimate.
- Twenty held-out questions is still a modest sample, and they were written by the same person
  who wrote the prompt. An independently authored set would be a stronger test.
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

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
