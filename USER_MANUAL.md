# Hedge Fund Backend - Developer Manual

## Overview

The **Hedge Fund Backend** is a FastAPI-based backend service for the BLACKWOOD CAPITAL QUANT quantitative research platform. It provides REST APIs for strategy management, feature engineering, model training, backtesting, and validation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────┬─────────────┬─────────────┬─────────────┬────────┤
│  Strategies │  Features  │   Models   │  Backtests  │ Agents │
│   Router    │   Router   │   Router   │   Router    │ Router │
├─────────────┴─────────────┴─────────────┴─────────────┴────────┤
│                      CRUD Repository Layer                       │
├─────────────────────────────────────────────────────────────────┤
│                     SQLAlchemy Async ORM                         │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│   LOCAL (SQLite + DiskCache + APScheduler)  │     LOCAL (SQLite + DiskCache + APScheduler)     │   MinIO/S3    │    LOCAL (SQLite + DiskCache + APScheduler)       │
│  (Metadata)   │   (Cache)     │  (Artifacts)  │   (Workers)     │
└───────────────┴───────────────┴───────────────┴─────────────────┘
```

## Project Structure

```
app/
├── api/                    # REST API routers
│   ├── strategies/
│   ├── features/
│   ├── models/
│   ├── signals/
│   ├── backtests/
│   ├── experiments/
│   ├── validation/
│   └── agents/
├── core/                   # Settings, config, utilities
├── db/                    # Session, models registry, CRUD base
├── domain/                 # ORM models and Pydantic schemas
│   ├── strategy/
│   ├── feature/
│   ├── model/
│   ├── signal/
│   ├── backtest/
│   └── experiment/
├── engines/               # Feature/Signal/Backtest engines
│   ├── feature_engine/
│   ├── backtest_engine/
│   └── validation_engine/
├── plugins/               # Plugin base classes and registries
│   ├── base_feature.py
│   ├── base_model.py
│   ├── base_signal.py
│   └── base_backtest.py
└── workers/               # LOCAL (SQLite + DiskCache + APScheduler) task definitions
    ├── feature_tasks.py
    ├── training_tasks.py
    └── backtest_tasks.py
```

---

## API Endpoints Reference

### Strategies

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/strategies` | List all strategies |
| POST | `/api/v1/strategies` | Create new strategy |
| GET | `/api/v1/strategies/{id}` | Get strategy by ID |
| PATCH | `/api/v1/strategies/{id}` | Update strategy |
| DELETE | `/api/v1/strategies/{id}` | Delete strategy |
| POST | `/api/v1/strategies/{id}/promote` | Promote to production |
| GET | `/api/v1/strategies/{id}/status` | Get strategy status |

### Features

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/features` | List all features |
| POST | `/api/v1/features` | Create new feature |
| GET | `/api/v1/features/{id}` | Get feature by ID |
| POST | `/api/v1/features/{id}/generate` | Generate feature data |
| POST | `/api/v1/features/{id}/regenerate` | Force regenerate |
| GET | `/api/v1/features/{id}/versions` | List version history |
| GET | `/api/v1/features/plugins/available` | List available plugins |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List all models |
| POST | `/api/v1/models` | Create new model |
| GET | `/api/v1/models/{id}` | Get model by ID |
| POST | `/api/v1/models/{id}/train` | Train model (sync) |
| POST | `/api/v1/models/{id}/train/async` | Train model (async) |
| POST | `/api/v1/models/tune` | Tune hyperparameters (sync) |
| POST | `/api/v1/models/tune/async` | Tune hyperparameters (async) |
| POST | `/api/v1/models/automl` | Run AutoML |
| GET | `/api/v1/models/plugins/available` | List available plugins |
| GET | `/api/v1/models/plugins/search-spaces` | Get default param spaces |

### Signals

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/signals` | List all signals |
| POST | `/api/v1/signals` | Create new signal |
| GET | `/api/v1/signals/{id}` | Get signal by ID |
| PATCH | `/api/v1/signals/{id}` | Update signal |
| POST | `/api/v1/signals/validate-rule-tree` | Validate rule tree |
| POST | `/api/v1/signals/generate` | Generate signals preview |

### Backtests

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/backtests` | List all backtests |
| POST | `/api/v1/backtests` | Create new backtest |
| GET | `/api/v1/backtests/{id}` | Get backtest by ID |
| PATCH | `/api/v1/backtests/{id}` | Update backtest |
| DELETE | `/api/v1/backtests/{id}` | Delete backtest |
| POST | `/api/v1/backtests/{id}/execute` | Execute backtest |
| POST | `/api/v1/backtests/{id}/execute/async` | Execute async |
| GET | `/api/v1/backtests/{id}/equity-curve` | Get equity curve |
| GET | `/api/v1/backtests/{id}/equity-curve/parquet` | Download equity curve |
| GET | `/api/v1/backtests/{id}/trades` | Get trade list |
| GET | `/api/v1/backtests/{id}/trades/parquet` | Download trades |
| POST | `/api/v1/backtests/compare` | Compare backtests |
| GET | `/api/v1/backtests/engines/available` | List engines |
| POST | `/api/v1/backtests/sweep` | Parameter sweep |

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/experiments` | List all experiments |
| POST | `/api/v1/experiments` | Create experiment |
| GET | `/api/v1/experiments/{id}` | Get experiment by ID |
| PATCH | `/api/v1/experiments/{id}` | Update experiment |
| DELETE | `/api/v1/experiments/{id}` | Delete experiment |
| POST | `/api/v1/experiments/compare` | Compare experiments |

### Validation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/validation/strategies/{id}` | Get validation results |
| POST | `/api/v1/validation/walk-forward/async` | Run walk-forward |
| POST | `/api/v1/validation/cpcv/async` | Run CPCV analysis |

### AI Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents/research` | Submit research query |
| POST | `/api/v1/agents/chat` | Chat with AI agent |
| GET | `/api/v1/agents/sessions` | List sessions |
| GET | `/api/v1/agents/sessions/{id}` | Get session context |

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tasks/{task_id}` | Get task status |

---

## Plugin Architecture

### Feature Plugins

Create a new feature plugin by extending `BaseFeature`:

```python
from app.plugins.base_feature import BaseFeature, register_feature

@register_feature("technical.rsi")
class RSIFeature(BaseFeature):
    name = "Relative Strength Index"
    
    def compute(self, data: pd.DataFrame, params: dict) -> pd.Series:
        period = params.get("period", 14)
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

### Model Plugins

Create a new model plugin by extending `BaseModel`:

```python
from app.plugins.base_model import BaseModel, register_model

@register_model("ml.xgboost")
class XGBoostModel(BaseModel):
    name = "XGBoost Regressor"
    
    def train(self, X: pd.DataFrame, y: pd.Series, params: dict):
        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        return model
    
    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)
```

### Signal Generator Plugins

Create a new signal plugin by extending `BaseSignalGenerator`:

```python
from app.plugins.base_signal import BaseSignalGenerator, register_signal

@register_signal("threshold")
class ThresholdSignal(BaseSignalGenerator):
    name = "Threshold Signal Generator"
    
    def generate(self, predictions: pd.Series, params: dict) -> pd.Series:
        buy_threshold = params.get("buy_threshold", 0.01)
        sell_threshold = params.get("sell_threshold", -0.01)
        
        signals = pd.Series(0, index=predictions.index)  # HOLD
        signals[predictions >= buy_threshold] = 1         # BUY
        signals[predictions <= sell_threshold] = -1      # SELL
        return signals
```

### Backtest Engine Plugins

Create a new backtest engine by extending `BaseBacktestEngine`:

```python
from app.plugins.base_backtest import BaseBacktestEngine, register_engine

@register_engine("custom")
class CustomEngine(BaseBacktestEngine):
    name = "Custom Backtest Engine"
    
    def run(self, signals: pd.Series, prices: pd.DataFrame, config: dict):
        # Implement backtest logic
        ...
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- LOCAL (SQLite + DiskCache + APScheduler) 15+
- LOCAL (SQLite + DiskCache + APScheduler) 7+
- MinIO (or AWS S3)
- Docker & Docker Compose (optional)

### Setup

```bash
# Clone and enter directory
cd hedge-fund-backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings (all local by default)

# Run migrations
alembic revision --autogenerate -m "init schema"
alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f app

# Run migrations in container
docker-compose exec app alembic upgrade head
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | sqlite+aiosqlite://user:pass@localhost/db | LOCAL (SQLite + DiskCache + APScheduler) connection |
| `REDIS_URL` | NOT NEEDED (SQLite + DiskCache + APScheduler) connection |
| `MINIO_ENDPOINT` | localhost:9000 | MinIO/S3 endpoint |
| `MINIO_ACCESS_KEY` | minioadmin | MinIO access key |
| `MINIO_SECRET_KEY` | minioadmin | MinIO secret key |
| `MINIO_BUCKET` | hedge-fund-artifacts | S3 bucket name |
| `DATA_SERVICE_URL` | http://localhost:8001 | Data layer URL |
| `CELERY_BROKER_URL` | NOT NEEDED (SQLite + DiskCache + APScheduler) broker |
| `GEMINI_API_KEY` | - | Gemini API key for AI agents |

---

## Database Schema

### Strategy

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| name | VARCHAR(255) | Strategy name |
| description | TEXT | Strategy description |
| universe | JSONB | Asset universe |
| timeframe | VARCHAR(10) | Data timeframe |
| status | ENUM | draft/backtested/validated/promoted/archived |
| version | INTEGER | Version number |
| pipeline_config | JSONB | Pipeline configuration |
| feature_ids | JSONB | List of feature IDs |
| model_id | UUID | Associated model |
| signal_logic_id | UUID | Associated signal |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Last update |

### Feature

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| name | VARCHAR(255) | Feature name |
| type | VARCHAR(50) | Feature type |
| plugin_key | VARCHAR(100) | Plugin identifier |
| version | VARCHAR(20) | Version string |
| params | JSONB | Parameters |
| created_at | TIMESTAMP | Creation timestamp |

### Model

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| name | VARCHAR(255) | Model name |
| family | ENUM | ML/DL/Statistical |
| plugin_key | VARCHAR(100) | Plugin identifier |
| version | VARCHAR(20) | Version string |
| params | JSONB | Hyperparameters |
| cv_results | JSONB | Cross-validation results |
| created_at | TIMESTAMP | Creation timestamp |

---

## LOCAL (SQLite + DiskCache + APScheduler) Workers

### Start Workers

```bash
# Feature generation worker
# Background tasks run in-process - no celery worker needed -l info -Q feature_queue -n feature_worker

# Model training worker
# Background tasks run in-process - no celery worker needed -l info -Q training_queue -n training_worker

# Backtest worker
# Background tasks run in-process - no celery worker needed -l info -Q backtest_queue -n backtest_worker
```

### Task Endpoints

All async tasks return a task ID that can be polled:

```bash
# Check task status
curl http://localhost:8000/api/v1/tasks/{task_id}
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/api/test_strategies.py -v

# Run with marker
pytest -m integration -v
```

---

## Error Handling

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Invalid request parameters |
| 401 | Authentication failed |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (upstream) |

---

## Performance Tips

1. **Use async endpoints** - All database operations use async SQLAlchemy
2. **Enable caching** - LOCAL (SQLite + DiskCache + APScheduler) caching for repeated queries
3. **Use LOCAL (SQLite + DiskCache + APScheduler)** - Long-running tasks (training, backtests) should be async
4. **Connection pooling** - LOCAL (SQLite + DiskCache + APScheduler) and LOCAL (SQLite + DiskCache + APScheduler) use connection pools
5. **S3 artifacts** - Large files (models, backtest results) stored in MinIO/S3

---

## Troubleshooting

### Database Connection Issues

```bash
# Test database connection
psql $DATABASE_URL -c "SELECT 1"

# Check migrations status
alembic current
alembic history
```

### LOCAL (SQLite + DiskCache + APScheduler) Connection Issues

```bash
# Test LOCAL (SQLite + DiskCache + APScheduler) connection
redis-cli ping
# Should return: PONG

# Check LOCAL (SQLite + DiskCache + APScheduler)
celery -A app.workers.celery_app inspect active
```

### MinIO/S3 Issues

```bash
# List buckets
mc ls local/

# Check bucket exists
mc ls local/hedge-fund-artifacts
```

---

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
