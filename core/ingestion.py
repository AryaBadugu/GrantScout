"""
Document ingestion pipeline.
PDF/DOCX/HTML -> Docling -> clean text chunks -> embeddings -> ChromaDB.
Originals are stored in IBM COS (or local fallback).
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import chromadb
from chromadb.config import Settings

from core.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
)
from core.cos_client import upload_file
from core.schemas import DocumentChunk
from core.watsonx_client import embed_texts

logger = logging.getLogger(__name__)

# ─── ChromaDB setup ───────────────────────────────────────────────────────────

def _get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Docling parsing ──────────────────────────────────────────────────────────

def _parse_with_docling(file_path: Path) -> list[dict[str, Any]]:
    """
    Parse a document with Docling and return a list of
    {'text': str, 'page': int | None} dicts.
    Falls back to plain text reading for .txt files.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        pages = []
        text = file_path.read_text(encoding="utf-8", errors="replace")
        pages.append({"text": text, "page": 1})
        return pages

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document

        # Collect text grouped by page
        pages: list[dict[str, Any]] = []
        for element, _level in doc.iterate_items():
            text = getattr(element, "text", None)
            if not text or not text.strip():
                continue
            prov = getattr(element, "prov", None)
            page_no: int | None = None
            if prov:
                # prov may be a list of ProvenanceItem
                if hasattr(prov, "__iter__"):
                    for p in prov:
                        if hasattr(p, "page_no"):
                            page_no = p.page_no
                            break
            pages.append({"text": text.strip(), "page": page_no})
        return pages
    except Exception as exc:
        logger.warning("Docling failed for %s: %s – trying fallback.", file_path.name, exc)
        return _fallback_parse(file_path)


def _fallback_parse(file_path: Path) -> list[dict[str, Any]]:
    """Best-effort text extraction without Docling."""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pdfplumber

            pages = []
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"text": text.strip(), "page": i})
            return pages
        elif suffix in (".docx",):
            import docx

            doc = docx.Document(str(file_path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [{"text": text, "page": None}]
        elif suffix in (".html", ".htm"):
            from bs4 import BeautifulSoup

            html = file_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            return [{"text": text.strip(), "page": None}]
    except Exception as exc:
        logger.error("Fallback parse failed for %s: %s", file_path.name, exc)
    # Last resort
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [{"text": text, "page": None}]
    except Exception:
        return []


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─── Main ingestion entry point ───────────────────────────────────────────────

def ingest_document(
    file_path: Path,
    store_in_cos: bool = True,
    trace: list[dict] | None = None,
) -> list[DocumentChunk]:
    """
    Full ingestion pipeline for a single document.

    Steps:
      1. Upload original to COS / local
      2. Parse with Docling
      3. Chunk text
      4. Generate embeddings (Granite)
      5. Store in ChromaDB

    Parameters
    ----------
    file_path: Path
        Local path to the document file.
    store_in_cos: bool
        Whether to attempt COS upload.
    trace: list[dict] | None
        If provided, append step dicts for the agent trace panel.

    Returns
    -------
    list[DocumentChunk]
        The chunks that were indexed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    _trace = trace if trace is not None else []

    # Step 1 – store original
    t0 = time.time()
    if store_in_cos:
        file_bytes = file_path.read_bytes()
        object_key = f"grants/{file_path.name}"
        stored_at = upload_file(file_bytes, object_key)
    else:
        stored_at = str(file_path)
    _trace.append({
        "step": "store_original",
        "file": file_path.name,
        "stored_at": stored_at,
        "ms": round((time.time() - t0) * 1000),
    })
    logger.info("Stored %s at %s", file_path.name, stored_at)

    # Step 2 – parse
    t0 = time.time()
    raw_pages = _parse_with_docling(file_path)
    _trace.append({
        "step": "parse",
        "file": file_path.name,
        "pages_extracted": len(raw_pages),
        "ms": round((time.time() - t0) * 1000),
    })
    logger.info("Parsed %d page-blocks from %s", len(raw_pages), file_path.name)

    # Step 3 – chunk
    t0 = time.time()
    chunks: list[DocumentChunk] = []
    for page_block in raw_pages:
        text = page_block["text"]
        page_no = page_block.get("page")
        for idx, chunk_text in enumerate(_chunk_text(text)):
            chunk_id = hashlib.sha256(
                f"{file_path.name}:p{page_no}:c{idx}:{chunk_text[:40]}".encode()
            ).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_file=file_path.name,
                    page=page_no,
                    text=chunk_text,
                    metadata={"page": page_no, "source_file": file_path.name},
                )
            )
    _trace.append({
        "step": "chunk",
        "num_chunks": len(chunks),
        "ms": round((time.time() - t0) * 1000),
    })
    logger.info("Created %d chunks from %s", len(chunks), file_path.name)

    if not chunks:
        logger.warning("No text extracted from %s – skipping indexing.", file_path.name)
        return []

    # Step 4 – embed (batch to avoid large payload)
    t0 = time.time()
    BATCH = 32
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), BATCH):
        batch_texts = [c.text for c in chunks[i : i + BATCH]]
        all_embeddings.extend(embed_texts(batch_texts))
    _trace.append({
        "step": "embed",
        "num_vectors": len(all_embeddings),
        "ms": round((time.time() - t0) * 1000),
    })

    # Step 5 – upsert to ChromaDB
    t0 = time.time()
    collection = _get_chroma_collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=all_embeddings,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    _trace.append({
        "step": "index",
        "collection": CHROMA_COLLECTION,
        "upserted": len(chunks),
        "ms": round((time.time() - t0) * 1000),
    })
    logger.info("Indexed %d chunks to ChromaDB from %s", len(chunks), file_path.name)

    return chunks


def get_collection_stats() -> dict[str, Any]:
    """Return basic stats about the ChromaDB collection."""
    try:
        col = _get_chroma_collection()
        return {"count": col.count(), "collection": CHROMA_COLLECTION}
    except Exception as exc:
        return {"count": 0, "error": str(exc)}


def clear_collection() -> None:
    """Delete and recreate the ChromaDB collection (used by clear-data button)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection cleared.")
