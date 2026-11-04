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
    WATSONX_API_KEY,
    WATSONX_URL,
    WATSONX_PROJECT_ID,
    GRANITE_INSTRUCT_MODEL,
    GRANITE_EMBEDDING_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    validate_watsonx,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _credentials() -> Credentials:
    """Return cached IBM Cloud credentials."""
    validate_watsonx()
    return Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)


@lru_cache(maxsize=1)
def get_llm() -> ModelInference:
    """Return a cached ModelInference instance for Granite instruct."""
    creds = _credentials()
    logger.info("Initialising Granite instruct model: %s", GRANITE_INSTRUCT_MODEL)
    return ModelInference(
        model_id=GRANITE_INSTRUCT_MODEL,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID,
        params={
            GenParams.MAX_NEW_TOKENS: LLM_MAX_NEW_TOKENS,
            GenParams.TEMPERATURE: LLM_TEMPERATURE,
        },
    )


@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    """Return a cached Embeddings instance for Granite embedding."""
    creds = _credentials()
    logger.info("Initialising Granite embedding model: %s", GRANITE_EMBEDDING_MODEL)
    return Embeddings(
        model_id=GRANITE_EMBEDDING_MODEL,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID,
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
