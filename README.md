# Research Layer — Phase 1

Standalone backend scaffold for the Quant Research Platform: Strategy,
Feature, Model, and Backtest CRUD, plus the plugin architecture that later
phases (Feature Engine, Model Training, Signal Engine, Backtest Engine,
Validation, MLflow, AI Agents) build on top of.

## What's implemented

### Phase 1
- **Domain models** (`app/domain/*/orm.py`): `Strategy`, `Feature`,
  `MLModel`, `Experiment`, `Backtest` — async SQLAlchemy 2.0, UUID PKs,
  timestamps, versioning.
- **Pydantic schemas** (`app/domain/*/schemas.py`): Create/Update/Read
  models per resource.
- **CRUD API** (`app/api/*/router.py`): full REST CRUD for strategies,
  features, models, backtests, built on a generic `CRUDRepository`
  (`app/db/crud_base.py`) to avoid per-resource boilerplate.
- **Plugin architecture** (`app/plugins/`): `BaseFeature`, `BaseModel`,
  `BaseSignalGenerator`, `BaseBacktestEngine` abstract interfaces plus a
  `PluginRegistry` so new features/models/signals/engines can be added by
  dropping a module in the relevant package — no core code changes.
  Example plugins included: RSI/ATR features, an XGBoost model, a
  threshold signal generator, and vectorbt/Backtrader backtest adapters.
- **Alembic** migration scaffolding wired to the async engine and the
  model registry (`app/db/models_registry.py`).
- **FastAPI app** (`app/main.py`) wiring it all together with a `/health`
  endpoint and `/api/v1` routers.

### Phase 2 — Feature Engine + Feature Store
- **`FeatureDataset` model** (`app/domain/feature/dataset_orm.py`): tracks
  each *generated instance* of a feature definition (per symbol/timeframe/
  date range), separate from the `Feature` definition row itself.
- **Versioning** (`app/engines/feature_engine/versioning.py`): SHA-256
  content hash over `(plugin_key, params, symbol, timeframe, date range,
  source_fingerprint)`. Identical inputs → identical hash → automatic
  reuse. Changed source data (revisions, late-arriving bars) → new hash →
  new version, without overwriting history.
- **Feature Store** (`app/engines/feature_engine/store.py`): durable
  storage on S3/MinIO (parquet), metadata/lineage in Postgres, fronted by
  a Redis cache (`app/core/cache.py`) for repeated reads within a session.
  Supports `list_versions()` for historical regeneration/audit.
- **Object storage client** (`app/core/storage.py`): boto3/MinIO wrapper
  shared by the Feature Store and, later, model/backtest artifact storage.
- **`FeaturePipeline`** (`app/engines/feature_engine/pipeline.py`):
  orchestrates running one or many feature plugins against market data,
  persisting through the Feature Store, and joining results into a wide
  DataFrame. `regenerate()` forces recomputation with the same
  reproducibility guarantees as a fresh run.
- **Market Data Layer client** (`app/engines/feature_engine/market_data_client.py`):
  the single integration point for OHLCV/news from the external Market
  Data Layer (HTTP, configurable via `MARKET_DATA_URL`).
- **API endpoints** (`app/api/features/generation_router.py`):
  - `POST /api/v1/features/{id}/generate` — compute or reuse a feature
  - `POST /api/v1/features/{id}/regenerate` — force recomputation
  - `GET /api/v1/features/{id}/versions` — list historical dataset versions
  - `GET /api/v1/features/plugins/available` — list registered feature plugins

### Phase 3 — Model Training Engine
- **Time-series CV** (`cross_validation.py`): rolling and expanding window
  splitters — strictly ordered, no shuffle, no leakage. The heavier
  purged/embargoed CPCV for strategy *validation* lives in Phase 7/8.
- **Dataset assembler** (`dataset_assembler.py`): joins Feature Store
  outputs into a wide X matrix, derives a forward-return y target aligned
  to the same index, drops NaN rows from indicator warm-up and target
  look-ahead.
- **CV evaluator** (`evaluation.py`): shared scoring util (MSE, MAE,
  directional accuracy per fold) used identically by the trainer and the
  Optuna tuner so both measure performance the same way.
- **Optuna tuner** (`tuning.py`): declarative param-space specs
  (`float`/`int`/`categorical`) matching the Model Builder UI form,
  configurable metric/direction, returns best params + per-trial history.
- **AutoML** (`automl.py`): `run_automl()` evaluates fixed-param
  candidates; `tune_candidates()` runs a per-candidate Optuna study —
  both rank by the same CV metric and return a sorted leaderboard.
- **Model trainer** (`trainer.py`): full-dataset final fit, CV metrics
  report, artifact persistence to S3/MinIO, model row update.
- **Default search spaces** (`search_spaces.py`): Optuna param specs for
  all ML/DL plugins, served from `GET /models/plugins/search-spaces` so
  the frontend and tuner share a single source of truth.
- **Model plugins**: LightGBM, CatBoost, Random Forest, LSTM (PyTorch)
  added alongside the existing XGBoost plugin.
- **API endpoints** (`training_router.py`):
  - `POST /models/{id}/train` — synchronous train + CV report
  - `POST /models/{id}/train/async` — dispatch to Celery worker
  - `POST /models/tune` — synchronous Optuna study
  - `POST /models/tune/async` — dispatch to Celery worker
  - `POST /models/automl` — leaderboard across candidate plugins
  - `GET  /models/plugins/available` — registered plugin keys
  - `GET  /models/plugins/search-spaces` — default param spaces
- **Experiments CRUD** (`api/experiments/router.py`): create/read/list/
  update/delete experiments, plus `POST /experiments/compare` which diffs
  metrics across up to 10 runs and highlights the best per metric.
- **Celery workers** (`workers/`): `training_tasks.py` (train + tune),
  `feature_tasks.py` (generate), shared `celery_app.py` instance.
- **`GET /api/v1/tasks/{task_id}`** — generic task-status polling endpoint.

### Phase 5 — vectorbt + Backtrader Integration
- **Metrics engine** (`engines/backtest_engine/metrics.py`): single
  source of truth for all metrics — CAGR, Sharpe, Sortino, Calmar,
  Max Drawdown + duration, VaR 95/99, CVaR 95/99, annualised vol,
  Win Rate, Profit Factor, Avg Win/Loss, Expectancy, Turnover.
  Both adapters call this so the Experiment Tracker always sees
  identical metric definitions regardless of engine.
- **`BacktestResult`** (`result.py`): normalized container (equity
  curve, trade list, drawdown series, metrics, raw engine stats)
  returned by every adapter.
- **VectorBTAdapter** (full implementation): handles discrete
  BUY/SELL/HOLD signals, numeric ±1 signals, and signal-weight
  position sizing. Extracts per-trade P&L from `vbt.Portfolio.trades`.
- **BacktraderAdapter** (full implementation): dynamically builds a
  `bt.Strategy` subclass from the signal series, wires Cerebro with
  commission/slippage, extracts TradeAnalyzer + TimeReturn data.
- **Engine registry** (`engine_registry.py`): maps key strings
  (`"vectorbt"`, `"backtrader"`) to adapter classes; factory
  instantiates the right one from a config dict.
- **Result storage** (`storage.py`): persists equity curve, trades,
  and drawdowns to S3/MinIO as parquet; writes flat metrics dict into
  `Backtest.metrics` JSONB; exposes `load_equity_curve` /
  `load_trades` for the download endpoints.
- **`BacktestPipeline`** (`pipeline.py`): full orchestration chain —
  resolves symbol/features/model/signals from the strategy config,
  calls the engine, persists results, handles RUNNING → COMPLETED /
  FAILED status transitions.
- **API endpoints** (`api/backtests/router.py`):
  - `POST   /backtests`               — create config row
  - `GET    /backtests`               — list (filterable by strategy)
  - `GET    /backtests/{id}`          — get with structured metrics
  - `PATCH  /backtests/{id}`          — update
  - `DELETE /backtests/{id}`          — delete
  - `POST   /backtests/{id}/execute`  — sync or async execution
  - `POST   /backtests/{id}/execute/async` — always-async
  - `GET    /backtests/{id}/equity-curve`         — JSON
  - `GET    /backtests/{id}/equity-curve/parquet` — download
  - `GET    /backtests/{id}/trades`               — JSON
  - `GET    /backtests/{id}/trades/parquet`       — download
  - `POST   /backtests/compare`       — metric diff, best-run highlighting
  - `GET    /backtests/engines/available`
- **Parameter sweep** (`api/backtests/sweep_router.py` +
  `workers/sweep_tasks.py`): `POST /backtests/sweep` accepts a
  `base_config` + `param_grid` list, creates and executes N backtest
  rows in the Celery worker pool, returns a ranked leaderboard.
- **Celery task** (`workers/backtest_tasks.py`): `backtests.execute`
  wraps the full pipeline for async dispatch.

## Not yet implemented (later phases)

MLflow experiment tracking integration (Phase 6), Walk-Forward + CPCV
validation (Phase 7/8), News Sentiment/FinBERT pipeline (Phase 9),
AI Research Agents (Phase 10), Portfolio Layer promotion (Phase 11),
auth middleware.

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Postgres/Redis/MinIO/MLflow via your own docker-compose (not included yet)

# Run migrations
alembic revision --autogenerate -m "init schema"
alembic upgrade head

# Start the API
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Project layout

```
backend/app/
  api/            REST routers per resource
  core/           settings/config
  db/             session, base mixins, generic CRUD, model registry
  domain/         ORM models + Pydantic schemas per resource
  engines/        feature/signal/backtest/validation engine implementations
  plugins/        BaseFeature/BaseModel/BaseSignalGenerator/BaseBacktestEngine
                  + registries + example plugins
  workers/        Celery/RQ task definitions (Phase 3+)
  events/         event publishing/consuming (Phase 9+)
```
