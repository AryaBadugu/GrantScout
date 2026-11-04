"""
IBM watsonx.ai client wrapper.
Provides:
  - text generation via Granite instruct model
  - embeddings via Granite embedding model
Both use ibm-watsonx-ai SDK; model IDs come from config so they are
swappable without touching this file.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from core.config import (
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    validate_watsonx,
    get_watsonx_api_key,
    get_watsonx_url,
    get_watsonx_project_id,
    get_granite_instruct_model,
    get_granite_embedding_model,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _credentials() -> Credentials:
    """Return cached IBM Cloud credentials (read fresh so st.secrets is available)."""
    validate_watsonx()
    return Credentials(url=get_watsonx_url(), api_key=get_watsonx_api_key())


@lru_cache(maxsize=1)
def get_llm() -> ModelInference:
    """Return a cached ModelInference instance for Granite instruct."""
    creds = _credentials()
    model_id = get_granite_instruct_model()
    logger.info("Initialising Granite instruct model: %s", model_id)
    return ModelInference(
        model_id=model_id,
        credentials=creds,
        project_id=get_watsonx_project_id(),
        params={
            GenParams.MAX_NEW_TOKENS: LLM_MAX_NEW_TOKENS,
            GenParams.TEMPERATURE: LLM_TEMPERATURE,
        },
    )


@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    """Return a cached Embeddings instance for Granite embedding."""
    creds = _credentials()
    model_id = get_granite_embedding_model()
    logger.info("Initialising Granite embedding model: %s", model_id)
    return Embeddings(
        model_id=model_id,
        credentials=creds,
        project_id=get_watsonx_project_id(),
    )


def generate_text(prompt: str, max_new_tokens: int | None = None) -> str:
    """
    Generate text using the Granite instruct model via the chat API.

    Parameters
    ----------
    prompt: str
        The full prompt string (system + user content already formatted).
    max_new_tokens: int | None
        Override token limit for this call only.

    Returns
    -------
    str
        Generated text stripped of leading/trailing whitespace.
    """
    llm = get_llm()
    extra: dict[str, Any] = {}
    if max_new_tokens is not None:
        extra[GenParams.MAX_NEW_TOKENS] = max_new_tokens
    try:
        # Use chat API (preferred over deprecated generate_text endpoint)
        messages = [{"role": "user", "content": prompt}]
        response = llm.chat(messages=messages, params=extra if extra else None)
        # Extract text from chat response structure
        result = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return result.strip()
    except Exception as exc:
        logger.error("watsonx generate_text failed: %s", exc)
        raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the Granite embedding model.

    Parameters
    ----------
    texts: list[str]
        Texts to embed. Batch size limited by API; pass reasonably sized lists.

    Returns
    -------
    list[list[float]]
        List of embedding vectors in the same order as `texts`.
    """
    embedder = get_embedder()
    try:
        response = embedder.embed_documents(texts=texts)
        # ibm-watsonx-ai returns list[list[float]] directly
        return response
    except Exception as exc:
        logger.error("watsonx embed_texts failed: %s", exc)
        raise


def smoke_test() -> dict[str, Any]:
    """
    Quick smoke test: verify credentials, generate one token, produce one embedding.
    Returns a dict with status fields; does NOT raise so callers can report gracefully.
    """
    result: dict[str, Any] = {
        "credentials_valid": False,
        "llm_ok": False,
        "embedder_ok": False,
        "llm_model": GRANITE_INSTRUCT_MODEL,
        "embedding_model": GRANITE_EMBEDDING_MODEL,
        "errors": [],
    }
    try:
        validate_watsonx()
        result["credentials_valid"] = True
    except ValueError as exc:
        result["errors"].append(str(exc))
        return result

    try:
        text = generate_text("Reply with exactly the word: OK", max_new_tokens=20)
        result["llm_ok"] = True
        result["llm_response_sample"] = text[:80]
    except Exception as exc:
        result["errors"].append(f"LLM error: {exc}")

    try:
        vecs = embed_texts(["GrantScout smoke test"])
        result["embedder_ok"] = True
        result["embedding_dim"] = len(vecs[0]) if vecs else 0
    except Exception as exc:
        result["errors"].append(f"Embedder error: {exc}")

    return result
