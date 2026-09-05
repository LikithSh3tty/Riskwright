# Riskwright

An AI-powered credit risk platform built on the Home Credit Default Risk dataset. It predicts
loan default probability, explains every prediction, derives business-readable credit rules,
and lets a non-technical user query the data in plain English.

Built for the NeoStats AI Engineer assignment.

> **Status:** in progress. Sections marked *pending* land as each module is built.

---

## What it does

| Module | Status |
|--------|--------|
| Data understanding and EDA | pending |
| Talk-to-data (natural language to SQL) | pending |
| Machine learning layer (default probability, risk band) | pending |
| Explainable AI (SHAP per prediction) | pending |
| Multi-section user interface | pending |
| Dockerized deployment | done |

## Data loaded

Measured on load, not quoted from the dataset description.

| Table | Rows | Columns | Load time |
|-------|------|---------|-----------|
| `application_train` | 307,511 | 122 | 25s |
| `bureau` | 1,716,428 | 17 | 32s |
| `previous_application` | 1,670,214 | 37 | 93s |

The target is heavily imbalanced: **8.07%** of applicants defaulted. That figure sets
`scale_pos_weight` for the model and the boundary of the Low risk band.

A second `docker-compose up` completes the load step in about a second, because the loader
checks for existing rows before ingesting.

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

Download and extract the dataset. Only three files are used:

```
application_train.csv
bureau.csv
previous_application.csv
```

Extract them to a folder **outside** any cloud-synced directory such as OneDrive or Dropbox.
Sync clients hold file locks that break Docker bind mounts.

The dataset is never committed to git.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Value |
|----------|-------|
| `DATA_DIR` | Absolute path to the folder holding the extracted CSVs |
| `ANTHROPIC_API_KEY` | Your key. Required only for the chatbot. |
| `POSTGRES_PASSWORD` | Any value |
| `POSTGRES_RO_PASSWORD` | Any value, different from the above |

### 3. Run

```bash
docker-compose up
```

Then open:

| Service | URL |
|---------|-----|
| UI | http://localhost:8501 |
| API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

The first start loads roughly 740MB into Postgres and takes a few minutes. Subsequent starts
skip the load, because the loader checks for existing rows first. To force a reload, set
`FORCE_RELOAD=true`.

### Running without Docker

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements-dev.txt

python -m src.data.loader       # needs a reachable Postgres
uvicorn app.main:app --reload
streamlit run ui/app.py
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
| Streamlit, not React | The evaluation criteria contain no line for frontend polish. The time saved went into the chatbot evaluation harness. |
| Claude Haiku 4.5 | Cheap enough to iterate prompts heavily, and strong at SQL over a compact schema. A small model is sufficient when the schema sent to it is small. |
| Exact version pins | The model is trained locally and served in a container. Unpinned scikit-learn or LightGBM between the two silently changes predictions. |
| Types inferred over the full CSV, not a sample | Several columns are integral in the first 100k rows and fractional later. A sampled schema would fail the COPY halfway through a 400MB file. |

---

## Model

*Pending. Covers model selection rationale, class imbalance strategy, evaluation metrics,
the cost-based decision threshold, and how the Low/Medium/High bands are derived from it.*

## Explainability

*Pending.*

## Rule derivation

*Pending. Covers the derivation logic and sample outputs.*

## Talk-to-data

*Pending. Covers prompt engineering, token optimization, conversation memory, the SQL
validation gate, and chatbot evaluation results.*

## Known limitations

*Pending.*

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
