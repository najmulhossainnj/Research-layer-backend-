# Unified Quant Research Platform

A production-grade backend for quantitative research combining the **Research Layer** and **Data Layer** into a single, unified service.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Unified Quant Research Platform                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Research Layer                          │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │   │
│   │  │Strategy  │ │Feature   │ │Model     │ │Signal    │      │   │
│   │  │Builder   │ │Engine    │ │Training  │ │Generator │      │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │   │
│   │  │Backtest  │ │Validation│ │MLflow    │ │AI Agents │      │   │
│   │  │Engine    │ │Center    │ │Tracking  │ │          │      │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      Data Layer                              │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│   │  │Yahoo     │ │News      │ │FRED      │ │Internal  │       │   │
│   │  │Finance   │ │API       │ │Macro     │ │Cache     │       │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │LOCAL (SQLite + DiskCache + APScheduler)│         │  LOCAL (SQLite + DiskCache + APScheduler)   │         │  MinIO   │
   │ (Meta)   │         │ (Cache)  │         │ (S3)     │
   └──────────┘         └──────────┘         └──────────┘
```

## 🚀 Quick Start

### Docker Compose (Recommended)

```bash
# Clone and start all services
git clone https://github.com/najmulhossainnj/Hedge-fund-backend.git
cd Hedge-fund-backend

# Start all services (API + LOCAL (SQLite + DiskCache + APScheduler) worker + LOCAL (SQLite + DiskCache + APScheduler) + LOCAL (SQLite + DiskCache + APScheduler) + MinIO + MLflow)
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# Run database migrations
docker-compose exec api alembic upgrade head

# Create S3 buckets
docker-compose exec api python -c "
from app.core.storage import get_storage_client
s = get_storage_client()
s.ensure_bucket('research-artifacts')
s.ensure_bucket('feature-store')
print('Buckets ready')
"

# Watch API logs
docker-compose logs -f api

# Watch LOCAL (SQLite + DiskCache + APScheduler) worker logs
# Background tasks run in-process (no separate worker needed)
```

### Local Development

```bash
# Clone
git clone https://github.com/najmulhossainnj/Hedge-fund-backend.git
cd Hedge-fund-backend

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment
cp .env.example .env
# Edit .env with your settings

# Run migrations
alembic revision --autogenerate -m "init"
alembic upgrade head

# Start the server
uvicorn app.main:app --reload

# In a separate terminal, start LOCAL (SQLite + DiskCache + APScheduler) worker
# Background tasks run in-process via APScheduler --loglevel=info
```

## 📁 Project Structure

```
Hedge-fund-backend/
├── app/
│   ├── api/                    # REST API routers
│   │   ├── strategies/         # Strategy CRUD + promotion
│   │   ├── features/          # Feature CRUD + generation
│   │   ├── models/            # Model CRUD + training
│   │   ├── signals/           # Signal rule tree
│   │   ├── backtests/         # Backtest execution + sweep
│   │   ├── experiments/       # Experiment tracking
│   │   ├── validation/        # Walk-forward + CPCV
│   │   ├── tracking/          # MLflow integration
│   │   ├── news/              # News sentiment
│   │   ├── agents/            # AI research agents
│   │   └── data/              # Data inspection proxy
│   │
│   ├── core/                  # Configuration + utilities
│   ├── db/                    # SQLAlchemy setup + CRUD base
│   ├── domain/                # ORM models + Pydantic schemas
│   ├── engines/               # Feature/Signal/Backtest engines
│   ├── plugins/               # Plugin system (Base classes + examples)
│   ├── workers/               # LOCAL (SQLite + DiskCache + APScheduler) task definitions
│   └── data/                  # Merged Data Layer
│       ├── delivery/          # FastAPI endpoints
│       ├── ingestion/         # Provider implementations
│       └── shared/            # Shared utilities
│
├── docker-compose.yml          # Full stack deployment
├── Dockerfile.unified         # Unified container
├── requirements.txt          # All dependencies (merged)
└── alembic/                  # Database migrations
```

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
  storage on S3/MinIO (parquet), metadata/lineage in SQLite, fronted by
  a LOCAL (SQLite + DiskCache + APScheduler) cache (`app/core/cache.py`) for repeated reads within a session.
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
  - `POST /models/{id}/train/async` — dispatch to LOCAL (SQLite + DiskCache + APScheduler) worker
  - `POST /models/tune` — synchronous Optuna study
  - `POST /models/tune/async` — dispatch to LOCAL (SQLite + DiskCache + APScheduler) worker
  - `POST /models/automl` — leaderboard across candidate plugins
  - `GET  /models/plugins/available` — registered plugin keys
  - `GET  /models/plugins/search-spaces` — default param spaces
- **Experiments CRUD** (`api/experiments/router.py`): create/read/list/
  update/delete experiments, plus `POST /experiments/compare` which diffs
  metrics across up to 10 runs and highlights the best per metric.
- **LOCAL (SQLite + DiskCache + APScheduler) workers** (`workers/`): `training_tasks.py` (train + tune),
  `feature_tasks.py` (generate), shared `task_app.py` instance (no external broker needed).
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
  rows in the LOCAL (SQLite + DiskCache + APScheduler) worker pool, returns a ranked leaderboard.
- **LOCAL (SQLite + DiskCache + APScheduler) task** (`workers/backtest_tasks.py`): `backtests.execute`
  wraps the full pipeline for async dispatch.

## Not yet implemented (later phases)

MLflow experiment tracking integration (Phase 6), Walk-Forward + CPCV
validation (Phase 7/8), News Sentiment/FinBERT pipeline (Phase 9),
AI Research Agents (Phase 10), Portfolio Layer promotion (Phase 11),
auth middleware.

## Deployment

### Railway Deployment (Recommended for Cloud)

On Railway, deploy **two separate services** that share environment variables:

#### Service 1: API (using Dockerfile.unified)
1. Create new Railway project
2. Add a **Nix** service pointing to your GitHub repo
3. Use the `Dockerfile.unified`
4. Set environment variables:
   ```
   DATABASE_URL=<Railway LOCAL (SQLite + DiskCache + APScheduler) connection string>
   REDIS_URL=# Not needed - using LOCAL (SQLite + DiskCache + APScheduler) URL>
   CELERY_BROKER_URL=# Not needed - using LOCAL (SQLite + DiskCache + APScheduler) URL>
   CELERY_RESULT_BACKEND=# Not needed - using LOCAL (SQLite + DiskCache + APScheduler) URL>
   S3_ENDPOINT_URL=<Cloudflare R2 endpoint>
   S3_ACCESS_KEY=<R2 access key>
   S3_SECRET_KEY=<R2 secret key>
   S3_BUCKET_ARTIFACTS=research-artifacts
   S3_BUCKET_FEATURES=feature-store
   DATA_SERVICE_URL=<your-data-layer.railway.app>
   DATA_SERVICE_API_KEY=<your-api-key>
   APP_ENV=production
   SECRET_KEY=<generate-secure-key>
   ```
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

#### Service 2: LOCAL (SQLite + DiskCache + APScheduler) Worker (using Dockerfile.worker)
1. Add another **Nix** service to the same Railway project
2. Use the `Dockerfile.worker`
3. **Same environment variables** as API (Railway auto-shares them)
4. Set start command: `# Background tasks run in-process via APScheduler --loglevel=info --concurrency=2`

#### Key: How services communicate
| Service | Reaches |
|---------|---------|
| API | LOCAL (SQLite + DiskCache + APScheduler), LOCAL (SQLite + DiskCache + APScheduler) via Railway private networking |
| LOCAL (SQLite + DiskCache + APScheduler) Worker | Same LOCAL (SQLite + DiskCache + APScheduler), LOCAL (SQLite + DiskCache + APScheduler) via Railway private networking |
| API → Data Layer | `DATA_SERVICE_URL` external URL |

### Docker Compose (Local Development)

See Docker Compose section above for full local stack.

### Environment Variables

See `.env.example` for all configurable options.

**Required for async backtest execution:**
- `CELERY_BROKER_URL` - NOT NEEDED (local backend)
- `CELERY_RESULT_BACKEND` - NOT NEEDED (local backend)

When these are configured, async execution works automatically. The backend gracefully falls back to sync execution if LOCAL (SQLite + DiskCache + APScheduler) is unavailable.

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
  workers/        LOCAL (SQLite + DiskCache + APScheduler)/RQ task definitions (Phase 3+)
  events/         event publishing/consuming (Phase 9+)
```
