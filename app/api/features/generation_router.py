"""
Feature generation endpoints.

Separate from the plain CRUD router (`router.py`) since these trigger
actual computation through the FeaturePipeline rather than just persisting
metadata. Mounted under the same `/features` prefix.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud_base import CRUDRepository
from app.db.session import get_db
from app.domain.feature.generation_schemas import (
    FeatureDatasetRead,
    FeatureGenerateRequest,
    FeatureGenerateResponse,
)
from app.domain.feature.orm import Feature
from app.domain.feature.schemas import FeatureCreate, FeatureUpdate
from app.engines.feature_engine.market_data_client import get_market_data_client
from app.engines.feature_engine.pipeline import FeaturePipeline

router = APIRouter(prefix="/features", tags=["features"])
_feature_repo = CRUDRepository[Feature, FeatureCreate, FeatureUpdate](Feature)


@router.post("/{feature_id}/generate", response_model=FeatureGenerateResponse)
async def generate_feature(
    feature_id: uuid.UUID,
    payload: FeatureGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compute (or reuse a cached/stored version of) a feature for a symbol/date range."""
    feature = await _feature_repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")

    market_data_client = get_market_data_client()
    ohlcv = await market_data_client.get_ohlcv(
        payload.symbol, payload.timeframe, payload.start_date, payload.end_date
    )
    if ohlcv.empty:
        raise HTTPException(
            status_code=422,
            detail="Market Data Layer returned no OHLCV data for the requested range",
        )

    pipeline = FeaturePipeline()
    result = await pipeline.run_one(
        db=db,
        feature=feature,
        market_data=ohlcv,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )

    preview = result.dataframe.head(10).reset_index().to_dict(orient="records")
    return FeatureGenerateResponse(
        dataset=FeatureDatasetRead.model_validate(result.dataset), preview=preview
    )


@router.post("/{feature_id}/regenerate", response_model=FeatureGenerateResponse)
async def regenerate_feature(
    feature_id: uuid.UUID,
    payload: FeatureGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Force recomputation against the current Market Data Layer state.

    Produces a new dataset version automatically if source data has changed
    since the last generation (content-hash versioning); otherwise resolves
    to the existing version.
    """
    feature = await _feature_repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")

    market_data_client = get_market_data_client()
    ohlcv = await market_data_client.get_ohlcv(
        payload.symbol, payload.timeframe, payload.start_date, payload.end_date
    )
    if ohlcv.empty:
        raise HTTPException(
            status_code=422,
            detail="Market Data Layer returned no OHLCV data for the requested range",
        )

    pipeline = FeaturePipeline()
    result = await pipeline.regenerate(
        db=db,
        feature=feature,
        market_data=ohlcv,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )

    preview = result.dataframe.head(10).reset_index().to_dict(orient="records")
    return FeatureGenerateResponse(
        dataset=FeatureDatasetRead.model_validate(result.dataset), preview=preview
    )


@router.get("/{feature_id}/versions", response_model=list[FeatureDatasetRead])
async def list_feature_versions(
    feature_id: uuid.UUID,
    symbol: str,
    timeframe: str = "1d",
    db: AsyncSession = Depends(get_db),
):
    """All historical generated versions of a feature for a symbol/timeframe."""
    feature = await _feature_repo.get(db, feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")

    pipeline = FeaturePipeline()
    versions = await pipeline.store.list_versions(db, feature_id, symbol, timeframe)
    return versions


@router.get("/plugins/available")
async def list_available_feature_plugins():
    """Plugin keys the Feature Builder UI can offer (technical/statistical/etc.)."""
    from app.plugins.features import feature_registry

    return {"plugins": feature_registry.list_keys()}
