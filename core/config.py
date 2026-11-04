"""
GrantScout configuration.
Secrets are read from (in priority order):
  1. Environment variables (set directly or loaded from .env)
  2. Streamlit secrets (st.secrets) — used on Streamlit Community Cloud

IMPORTANT: The module-level constants (WATSONX_API_KEY etc.) are computed once
at import time.  On Streamlit Cloud the import happens inside @st.cache_resource
which runs after st.secrets is available, so this is safe.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from core/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


def _get(key: str, default: str = "") -> str:
    """
    Read a config value from:
      1. os.environ  (populated from .env via load_dotenv above, or real env vars)
      2. st.secrets  (Streamlit Community Cloud secret manager)
      3. default
    """
    val = os.environ.get(key, "")
    if val:
        return val
    # Try st.secrets — available after Streamlit runtime starts
    try:
        import streamlit as st  # noqa: PLC0415
        secret = st.secrets.get(key)  # type: ignore[attr-defined]
        if secret:
            return str(secret)
    except Exception:
        pass
    return default


def get_watsonx_api_key() -> str:
    return _get("WATSONX_API_KEY")

def get_watsonx_url() -> str:
    return _get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

def get_watsonx_project_id() -> str:
    return _get("WATSONX_PROJECT_ID")

def get_granite_instruct_model() -> str:
    return _get("GRANITE_INSTRUCT_MODEL", "ibm/granite-4-h-small")

def get_granite_embedding_model() -> str:
    return _get("GRANITE_EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual")


# ─── Module-level constants (kept for backwards compat with other modules) ────
# These are evaluated at import time.  When imported inside @st.cache_resource
# (as _core() does) st.secrets is already available on Streamlit Cloud.
WATSONX_API_KEY: str     = _get("WATSONX_API_KEY")
WATSONX_URL: str         = _get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID: str  = _get("WATSONX_PROJECT_ID")

GRANITE_INSTRUCT_MODEL: str  = _get("GRANITE_INSTRUCT_MODEL",  "ibm/granite-4-h-small")
GRANITE_EMBEDDING_MODEL: str = _get("GRANITE_EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual")

# ─── IBM Cloud Object Storage ─────────────────────────────────────────────────
COS_API_KEY: str      = _get("COS_API_KEY")
COS_INSTANCE_CRN: str = _get("COS_INSTANCE_CRN")
COS_ENDPOINT: str     = _get("COS_ENDPOINT", "https://s3.us-south.cloud-object-storage.appdomain.cloud")
COS_BUCKET: str       = _get("COS_BUCKET", "grantscout-docs")

# ─── Local paths ──────────────────────────────────────────────────────────────
DATA_DIR:      Path = _ROOT / "data"
GRANTS_DIR:    Path = DATA_DIR / "grants"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
CHROMA_DIR:    Path = _ROOT / ".chroma_db"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = int(_get("CHUNK_SIZE",    "800"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "100"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(_get("RETRIEVAL_TOP_K", "8"))

# ─── LLM generation ───────────────────────────────────────────────────────────
LLM_MAX_NEW_TOKENS: int   = int(_get("LLM_MAX_NEW_TOKENS", "900"))
LLM_TEMPERATURE: float    = float(_get("LLM_TEMPERATURE",  "0.1"))

# ─── Extraction retries ───────────────────────────────────────────────────────
EXTRACTION_MAX_RETRIES: int = int(_get("EXTRACTION_MAX_RETRIES", "3"))

CHROMA_COLLECTION: str = "grants"


def validate_watsonx() -> None:
    """Raise ValueError if mandatory watsonx credentials are missing.

    Checks both the module-level constants (so tests can patch them) and
    the live getter functions (so Streamlit Cloud st.secrets values work).
    """
    key     = WATSONX_API_KEY     or get_watsonx_api_key()
    url     = WATSONX_URL         or get_watsonx_url()
    proj_id = WATSONX_PROJECT_ID  or get_watsonx_project_id()
    missing = [
        name for name, val in [
            ("WATSONX_API_KEY",    key),
            ("WATSONX_URL",        url),
            ("WATSONX_PROJECT_ID", proj_id),
        ] if not val
    ]
    if missing:
        raise ValueError(
            f"Missing required credentials: {', '.join(missing)}. "
            "Set them in .env (local) or Streamlit Cloud → App settings → Secrets."
        )
