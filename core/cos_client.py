"""
IBM Cloud Object Storage client.
Falls back gracefully to local filesystem if COS credentials are absent.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from core.config import (
    COS_API_KEY,
    COS_BUCKET,
    COS_ENDPOINT,
    COS_INSTANCE_CRN,
    GRANTS_DIR,
)

logger = logging.getLogger(__name__)

_cos_client = None
_cos_available: bool | None = None


def _get_cos():
    """Lazy initialise the IBM COS client. Returns None if not configured."""
    global _cos_client, _cos_available
    if _cos_available is not None:
        return _cos_client if _cos_available else None

    if not COS_API_KEY or not COS_INSTANCE_CRN:
        logger.info("COS credentials not set – using local filesystem fallback.")
        _cos_available = False
        return None

    try:
        import ibm_boto3
        from ibm_botocore.client import Config

        _cos_client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=COS_API_KEY,
            ibm_service_instance_id=COS_INSTANCE_CRN,
            config=Config(signature_version="oauth"),
            endpoint_url=COS_ENDPOINT,
        )
        # Quick connectivity check
        _cos_client.head_bucket(Bucket=COS_BUCKET)
        _cos_available = True
        logger.info("IBM COS connected. Bucket: %s", COS_BUCKET)
    except Exception as exc:
        logger.warning("IBM COS unavailable (%s) – using local fallback.", exc)
        _cos_available = False
        _cos_client = None

    return _cos_client if _cos_available else None


def upload_file(file_bytes: bytes, object_key: str) -> str:
    """
    Upload bytes to COS or save to local grants directory.

    Returns
    -------
    str
        COS object key (if COS) or absolute local path.
    """
    cos = _get_cos()
    if cos is not None:
        try:
            cos.put_object(Bucket=COS_BUCKET, Key=object_key, Body=file_bytes)
            logger.info("Uploaded %s to COS bucket %s", object_key, COS_BUCKET)
            return object_key
        except Exception as exc:
            logger.error("COS upload failed: %s – saving locally.", exc)

    # Local fallback
    local_path = GRANTS_DIR / Path(object_key).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(file_bytes)
    logger.info("Saved %s locally at %s", object_key, local_path)
    return str(local_path)


def download_file(object_key: str) -> bytes:
    """
    Download bytes from COS or read from local grants directory.
    """
    cos = _get_cos()
    if cos is not None:
        try:
            response = cos.get_object(Bucket=COS_BUCKET, Key=object_key)
            return response["Body"].read()
        except Exception as exc:
            logger.warning("COS download failed for %s: %s", object_key, exc)

    # Local fallback
    local_path = GRANTS_DIR / Path(object_key).name
    if local_path.exists():
        return local_path.read_bytes()
    raise FileNotFoundError(f"File not found locally or in COS: {object_key}")


def list_objects(prefix: str = "") -> list[str]:
    """List object keys in COS bucket or filenames in local grants directory."""
    cos = _get_cos()
    if cos is not None:
        try:
            response = cos.list_objects_v2(Bucket=COS_BUCKET, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as exc:
            logger.warning("COS list failed: %s – listing locally.", exc)

    return [str(p) for p in GRANTS_DIR.glob("**/*") if p.is_file()]


def cos_available() -> bool:
    """Return True if COS is configured and reachable."""
    return _get_cos() is not None
