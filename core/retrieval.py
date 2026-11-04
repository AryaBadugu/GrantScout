"""
Retrieval module.
Semantic similarity search over ChromaDB using Granite embeddings.
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from core.config import CHROMA_DIR, CHROMA_COLLECTION, RETRIEVAL_TOP_K
from core.schemas import DocumentChunk
from core.watsonx_client import embed_texts

logger = logging.getLogger(__name__)


def _get_collection() -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_similar_chunks(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    source_file_filter: str | None = None,
) -> list[DocumentChunk]:
    """
    Embed the query and retrieve the top-k most similar chunks from ChromaDB.

    Parameters
    ----------
    query: str
        Natural language query.
    top_k: int
        Number of results to return.
    source_file_filter: str | None
        If set, restrict results to chunks from this file.

    Returns
    -------
    list[DocumentChunk]
        Matching chunks ordered by cosine similarity (highest first).
    """
    collection = _get_collection()
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty – no documents indexed yet.")
        return []

    query_vec = embed_texts([query])[0]

    where: dict[str, Any] | None = None
    if source_file_filter:
        where = {"source_file": source_file_filter}

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    chunks: list[DocumentChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunk = DocumentChunk(
            chunk_id=meta.get("chunk_id", ""),
            source_file=meta.get("source_file", ""),
            page=meta.get("page"),
            text=doc,
            metadata={**meta, "distance": dist, "similarity": round(1 - dist, 4)},
        )
        chunks.append(chunk)

    return chunks


def retrieve_for_grant_query(
    startup_description: str,
    sector: str,
    stage: str,
    state: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[DocumentChunk]:
    """
    Build a rich retrieval query from a startup profile and retrieve chunks.
    """
    query = (
        f"Grant funding scheme for {stage} startup in {sector} sector "
        f"located in {state}. {startup_description}"
    )
    return retrieve_similar_chunks(query, top_k=top_k)


def get_context_for_source(source_file: str, max_chars: int = 6000) -> str:
    """
    Return concatenated text from all chunks belonging to source_file.
    Used by drafting and extraction to get full document context.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return ""
    try:
        results = collection.get(
            where={"source_file": source_file},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        # Sort by page number
        paired = sorted(zip(metas, docs), key=lambda x: x[0].get("page") or 0)
        combined = " ".join(d for _, d in paired)
        return combined[:max_chars]
    except Exception as exc:
        logger.error("get_context_for_source failed: %s", exc)
        return ""
