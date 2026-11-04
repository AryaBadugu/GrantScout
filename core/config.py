"""
GrantScout configuration.
All secrets are read from environment variables (loaded from .env by python-dotenv).
Model IDs are configurable so they can be swapped when IBM Lite tier changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from core/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ─── IBM watsonx.ai ───────────────────────────────────────────────────────────
WATSONX_API_KEY: str = os.environ.get("WATSONX_API_KEY", "")
WATSONX_URL: str = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID: str = os.environ.get("WATSONX_PROJECT_ID", "")

# Granite model IDs – override via .env if your Lite account has different ones
GRANITE_INSTRUCT_MODEL: str = os.environ.get(
    "GRANITE_INSTRUCT_MODEL", "ibm/granite-4-h-small"
)
GRANITE_EMBEDDING_MODEL: str = os.environ.get(
    "GRANITE_EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"
)

# ─── IBM Cloud Object Storage ─────────────────────────────────────────────────
COS_API_KEY: str = os.environ.get("COS_API_KEY", "")
COS_INSTANCE_CRN: str = os.environ.get("COS_INSTANCE_CRN", "")
COS_ENDPOINT: str = os.environ.get("COS_ENDPOINT", "https://s3.us-south.cloud-object-storage.appdomain.cloud")
COS_BUCKET: str = os.environ.get("COS_BUCKET", "grantscout-docs")

# ─── Local paths ──────────────────────────────────────────────────────────────
DATA_DIR: Path = _ROOT / "data"
GRANTS_DIR: Path = DATA_DIR / "grants"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
CHROMA_DIR: Path = _ROOT / ".chroma_db"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "100"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", "8"))

# ─── LLM generation ───────────────────────────────────────────────────────────
LLM_MAX_NEW_TOKENS: int = int(os.environ.get("LLM_MAX_NEW_TOKENS", "1500"))
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

# ─── Extraction retries ───────────────────────────────────────────────────────
EXTRACTION_MAX_RETRIES: int = int(os.environ.get("EXTRACTION_MAX_RETRIES", "3"))

CHROMA_COLLECTION: str = "grants"


def validate_watsonx() -> None:
    """Raise ValueError if mandatory watsonx credentials are missing."""
    missing = [
        name
        for name, val in [
            ("WATSONX_API_KEY", WATSONX_API_KEY),
            ("WATSONX_URL", WATSONX_URL),
            ("WATSONX_PROJECT_ID", WATSONX_PROJECT_ID),
        ]
        if not val
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please copy .env.example to .env and fill in your IBM Cloud credentials."
        )
