"""
Chat Q&A agent.
Provides grounded answers with citations (document name + page).
If retrieved context does not support the answer, says so explicitly.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.retrieval import retrieve_similar_chunks
from core.schemas import ChatResponse, Citation
from core.watsonx_client import generate_text

logger = logging.getLogger(__name__)

_CHAT_SYSTEM = """\
You are GrantScout, an assistant that answers questions about Indian government grants.
Answer ONLY using the context below. Quote exact values (amounts, dates, stage names) \
directly from the text — do NOT paraphrase or say "the context does not specify" if the \
value is present verbatim. If the answer truly cannot be found in the context, say: \
"not found in the provided documents."

Context:
{context}

Question: {question}

Answer concisely (1-3 sentences). Cite the source file for each fact, e.g. \
[source: filename.txt]:"""


def _build_context_str(chunks: list[Any]) -> tuple[str, list[Citation]]:
    """Build a context string with full chunk text and extract citations."""
    parts: list[str] = []
    citations: list[Citation] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.page
        source = chunk.source_file or "unknown"
        # Use the full chunk — these are single-file chunks ~1500 chars each
        snippet = chunk.text
        label = f"[{i}] {source}" + (f" p.{page}" if page else "")
        parts.append(f"{label}:\n{snippet}")
        citations.append(Citation(source_file=source, page=page, snippet=snippet[:300]))
    return "\n\n".join(parts), citations


def answer_question(
    question: str,
    top_k: int = 2,
    trace: list[dict] | None = None,
) -> ChatResponse:
    """
    Answer a user question about grants using RAG.

    Retrieves the top-2 most similar chunks so Granite can focus on the
    most relevant document rather than being distracted by 6 at once.

    Parameters
    ----------
    question: str
    top_k: int
        Number of chunks to retrieve for context. Default 2 keeps the
        prompt tight for the Granite Lite token budget.
    trace: list[dict] | None

    Returns
    -------
    ChatResponse with answer, citations, and grounded flag.
    """
    _trace = trace if trace is not None else []
    t0 = time.time()

    # Retrieve top-2 most relevant chunks
    chunks = retrieve_similar_chunks(question, top_k=top_k)
    _trace.append({
        "step": "retrieve",
        "query": question[:100],
        "chunks_retrieved": len(chunks),
        "ms": round((time.time() - t0) * 1000),
    })

    if not chunks:
        return ChatResponse(
            answer="not found in the provided documents. Please upload grant documents first.",
            citations=[],
            grounded=False,
        )

    context_str, citations = _build_context_str(chunks)

    # Cap at 4000 chars to stay within Granite Lite token budget
    prompt = _CHAT_SYSTEM.format(context=context_str[:4000], question=question)

    t1 = time.time()
    try:
        answer = generate_text(prompt, max_new_tokens=400)
    except Exception as exc:
        logger.error("Chat generation failed: %s", exc)
        answer = f"[Error generating answer: {exc}]"

    grounded = "not found in the provided documents" not in answer.lower()

    _trace.append({
        "step": "generate_answer",
        "answer_chars": len(answer),
        "grounded": grounded,
        "ms": round((time.time() - t1) * 1000),
    })

    # Only include citations if answer is grounded
    used_citations = citations if grounded else []

    return ChatResponse(
        answer=answer.strip(),
        citations=used_citations,
        grounded=grounded,
    )
