# 🏦 Credit Risk Scoring System (MLOps Portfolio Project)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end machine learning system that scores loan applications for probability
of default, trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset (~307k applications, ~8% default rate).

The XGBoost model is served through a FastAPI microservice, consumed by a Streamlit
dashboard for loan officers, containerized with Docker Compose, and covered by an
automated test suite with CI.## Results

Produced by the pipeline in `scripts/train.py`: the eight applicant-facing features
(bureau score, amounts, age, three financial ratios), SMOTE balancing, and XGBoost
(100 trees, depth 5). Fully reproducible via `make train`.

| Metric | Score |
|---|---|
| ROC AUC | 0.721 |
| Accuracy (0.5 threshold) | 0.767 |

Deployment metrics — the serving threshold is **calibrated at training time** so the
riskiest ~25% of applicants are flagged for underwriter review (a business input,
not a magic constant):

| Business metric (test set) | Value |
|---|---|
| Applications flagged for review | 24.8% |
| Defaults caught by the flag | 53.5% |
| Precision of the flag | 17.4% (2.1× the 8.1% base rate) |

A note on honesty: a fixed 0.20 cutoff is not portable across models because SMOTE
compresses predicted probabilities — this model's riskiest quartile sits near 0.49.
So training carves out a calibration split, picks the cutoff matching the target
review rate there, and serves that threshold; the API reads it from
`models/metrics.json`.

Context: ROC AUC of 0.72 with only eight features is in the typical range for
tabular-only models on this dataset; public-leaderboard solutions reach ~0.80 only
by joining the supplementary tables (bureau, previous applications), which this
project deliberately keeps out of scope.

## Architecture

```text
                    ┌────────────────────────┐
                    │  Streamlit Dashboard   │
                    │  (loan officers)       │
                    └──────────┬─────────────┘
                               │ POST /predict (JSON)
                    ┌──────────▼─────────────┐
                    │  FastAPI Service       │
                    │  · pydantic validation │
                    │  · feature engineering │
                    │  · XGBoost inference   │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Model artifacts       │
                    │  model + scaler + cols │
                    └────────────────────────┘
```

- **`src/features.py`** — the single source of truth for feature engineering, defining
  the eight features the product can actually collect (train/serve parity by
  construction). The same code runs in `scripts/train.py` and inside the API,
  eliminating the train/serve skew that plagues notebook-first projects.
- **`src/model.py`** — model loading and scoring, independent of the web layer so
  it can be unit-tested without FastAPI.
- **`src/app.py`** — the API: input validation, `/predict`, `/health` (used by the
  Docker healthcheck and CI), and `/model-info` (exposes training metrics).
- **`src/dashboard.py`** — the Streamlit UI, with a live API health indicator and
  graceful handling of API and validation errors.
- **`scripts/train.py`** — reproducible training CLI with optional MLflow tracking.
- **`tests/`** — 25 pytest tests over feature engineering, schema validation, model
  serving, and the API, using synthetic model artifacts (no dataset required).

## Quickstart

### Option A: Docker (recommended)

Prereq: train the model first so `models/` has artifacts (see Option B, steps 1–2).

```bash
docker compose up --build
# API docs:    http://localhost:80/docs
# Dashboard:   http://localhost:8501
```

### Option B: Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt

# 1. Get the data (~2.5 GB). Place application_train.csv in data/.
python scripts/download_data.py            # or download manually from Kaggle

# 2. Train (SMOTE + XGBoost; ~10 min on a laptop, ~4 GB RAM)
python scripts/train.py --data-path data/application_train.csv

# 3. Serve the API (terminal 1)
uvicorn src.app:app --port 8000

# 4. Run the dashboard (terminal 2)
streamlit run src/dashboard.py
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Score a loan application. Returns `default_probability`, `risk_level`, `threshold`. |
| `/health` | GET | Liveness probe used by Docker healthchecks and CI. |
| `/model-info` | GET | Model version, training timestamp, and test metrics. |
| `/docs` | GET | Interactive OpenAPI docs. |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 202500,
    "AMT_CREDIT": 406597,
    "AMT_ANNUITY": 24700,
    "DAYS_BIRTH": -9461,
    "CODE_GENDER": "F",
    "EXT_SOURCE_3": 0.14
  }'
```

```json
{
  "default_probability": 0.711,
  "risk_level": "High",
  "threshold": 0.489
}
```

## Modeling decisions

- **Class imbalance:** the 8% default rate is handled with SMOTE oversampling on
  the training split only — the test split stays untouched to keep evaluation honest.
- **Feature parity:** financial ratios are computed by the same function at train
  and serve time (`src/features.py`).
- **Serving assumptions (documented, not hidden):** the form collects one external
  score, which is propagated to all three `EXT_SOURCE_*` fields; employment length
  is fixed at ~5.5 years; ownership flags are not collected. These simplifications
  trade some accuracy for a realistic single-form UX, and they are explicit in code.
- **Threshold calibration:** training holds out a calibration split and picks the
  cutoff that flags the target review rate (25% by default, `--target-flag-rate`).
  The API serves that threshold from `metrics.json` — the review-queue size is a
  business decision, not a hardcoded constant.
- **Median-fill at training:** a handful of rows lack bureau scores or annuity
  amounts; gaps are filled with the median. The serving form always sends complete
  input, so this path is training-only.

## Walkthrough notebook

[`notebooks/03_data_and_backend_walkthrough.ipynb`](notebooks/03_data_and_backend_walkthrough.ipynb)
is an executed, self-contained tour of the whole system with 11 charts: the 8%
default imbalance, bureau-score distributions by outcome, financial-strain ratios,
feature correlation with default, SMOTE before/after, ROC curve, feature
importance, the calibrated threshold against the score distribution, the
precision-recall trade-off, and a **live sweep of the running API** showing the
risk verdict flip with bureau score. Rebuild it with `make notebook` (needs the
dataset and the API running).

## Testing & CI

```bash
ruff check src scripts tests
pytest
```

GitHub Actions runs lint + tests on every push and PR, then builds the API image.
Tests use synthetic artifacts, so the full suite runs in seconds with no dataset.

## Project structure

```text
├── src/
│   ├── app.py            # FastAPI service
│   ├── dashboard.py      # Streamlit UI
│   ├── features.py       # shared feature engineering
│   ├── model.py          # artifact loading + scoring
│   ├── schemas.py        # pydantic request/response models
│   └── config.py         # paths and constants
├── notebooks/
│   └── 03_data_and_backend_walkthrough.ipynb  # executed walkthrough w/ charts
├── scripts/
│   ├── train.py                       # reproducible training CLI
│   ├── download_data.py               # dataset download helper
│   └── build_walkthrough_notebook.py  # notebook generator/executor
├── tests/                # pytest suite (synthetic artifacts)
├── requirements/         # base.txt (runtime), dev.txt (all-in)
├── Dockerfile            # API image (non-root, healthcheck)
├── Dockerfile.dashboard  # Streamlit image
└── docker-compose.yml    # two-service stack with healthchecks
```

## Roadmap

- [ ] Join supplementary tables (bureau, previous applications) to push AUC toward 0.75+
- [ ] Isotonic/Platt calibration of probabilities instead of rank-based thresholding
- [ ] SHAP-based per-prediction explanations in the API and dashboard
- [ ] Serve batch predictions from the raw CSV
