"""
Object storage client.

Thin wrapper around boto3's S3 client pointed at MinIO (or any S3-compatible
endpoint). Used by the Feature Store for parquet datasets and by other
layers (model artifacts, backtest trade/equity blobs) for large binary
payloads that shouldn't live in Postgres rows.
"""
import io
from functools import lru_cache
from typing import Optional

import boto3
from botocore.client import Config

from app.core.config import get_settings


class ObjectStorageClient:
    def __init__(self):
        settings = get_settings()
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )

    def ensure_bucket(self, bucket: str) -> None:
        existing = {b["Name"] for b in self._client.list_buckets().get("Buckets", [])}
        if bucket not in existing:
            self._client.create_bucket(Bucket=bucket)

    def put_bytes(self, bucket: str, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        self.ensure_bucket(bucket)
        extra = {"ContentType": content_type} if content_type else {}
        self._client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return f"s3://{bucket}/{key}"

    def get_bytes(self, bucket: str, key: str) -> bytes:
        obj = self._client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def delete(self, bucket: str, key: str) -> None:
        self._client.delete_object(Bucket=bucket, Key=key)

    def list_keys(self, bucket: str, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        """s3://bucket/key/path -> (bucket, key/path)"""
        assert uri.startswith("s3://"), f"Not an s3 URI: {uri}"
        rest = uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        return bucket, key


@lru_cache
def get_storage_client() -> ObjectStorageClient:
    return ObjectStorageClient()
