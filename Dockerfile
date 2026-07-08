# Unified Quant Research Platform - Dockerfile
# Local Backend: SQLite + DiskCache (no PostgreSQL, Redis, or Celery)

FROM python:3.12-slim

WORKDIR /app

# System deps for Python packages
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Run with local backend (SQLite + DiskCache)
ENV DATABASE_URL=sqlite+aiosqlite:///data/research_layer.db
ENV CACHE_DIR=/data/cache

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
