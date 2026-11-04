"""
Structured grant extraction using IBM Granite.
Each grant document is converted to a validated GrantInfo JSON object.
Uses JSON-only prompts with retry on invalid output.
Missing fields are always null – never guessed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.config import EXTRACTION_MAX_RETRIES
from core.schemas import GrantInfo
from core.watsonx_client import generate_text

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """\
You are a precise grant data extractor. Given text from an Indian government grant or funding scheme document, extract information and return ONLY a valid JSON object. Do not add any text before or after the JSON.

Required JSON schema:
{
  "name": "string – official name of the scheme",
  "issuing_body": "string or null",
  "funding_amount": "string as written in the text, e.g. 'up to ₹50 lakh' or null",
  "funding_amount_inr": number or null,
  "eligible_stages": ["ideation"|"prototype"|"early_stage"|"growth"|"scaling"|"any"],
  "sectors": ["list of sectors as strings"],
  "location_restrictions": "string or null – state/UT restrictions",
  "eligibility_criteria": ["list of key eligibility rules as strings"],
  "deadline": "string as written or null",
  "deadline_date": "YYYY-MM-DD or null",
  "required_documents": ["list of required documents"],
  "application_link": "URL string or null",
  "source_file": "string or null",
  "source_page": number or null
}

Rules:
- Output ONLY the JSON object. No markdown fences, no explanation.
- If a field is not mentioned in the text, set it to null or empty list [].
- Never invent or guess values. Only extract what is explicitly stated.
- For eligible_stages use ONLY the allowed enum values above.
- funding_amount_inr should be the numeric value in INR (e.g. 5000000 for 50 lakh). null if not parseable.
"""


def _build_extraction_prompt(text: str, source_file: str = "", source_page: int | None = None) -> str:
    meta = f"Source file: {source_file}" if source_file else ""
    if source_page:
        meta += f"  Page: {source_page}"
    return f"""{_EXTRACTION_SYSTEM}

{meta}

--- DOCUMENT TEXT START ---
{text[:4000]}
--- DOCUMENT TEXT END ---

JSON:"""


def _parse_json_response(response: str) -> dict[str, Any]:
    """Extract the first JSON object from the model response."""
    response = response.strip()
    # Remove markdown code fences if present
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    # Find the first { ... } block
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in response: {response[:200]}")


def extract_grant_info(
    text: str,
    source_file: str = "",
    source_page: int | None = None,
    trace: list[dict] | None = None,
) -> GrantInfo | None:
    """
    Extract structured grant information from raw text using Granite.

    Parameters
    ----------
    text: str
        Raw text from the grant document (will be truncated to ~4000 chars).
    source_file: str
        Filename for provenance.
    source_page: int | None
        Page number for provenance.
    trace: list[dict] | None
        Agent trace list to append steps to.

    Returns
    -------
    GrantInfo | None
        Validated grant metadata, or None if all retries fail.
    """
    _trace = trace if trace is not None else []
    prompt = _build_extraction_prompt(text, source_file, source_page)

    last_error: Exception | None = None
    for attempt in range(1, EXTRACTION_MAX_RETRIES + 1):
        t0 = time.time()
        try:
            raw = generate_text(prompt, max_new_tokens=1200)
            parsed = _parse_json_response(raw)
            # Inject provenance
            if not parsed.get("source_file"):
                parsed["source_file"] = source_file
            if parsed.get("source_page") is None and source_page is not None:
                parsed["source_page"] = source_page

            grant = GrantInfo.model_validate(parsed)
            ms = round((time.time() - t0) * 1000)
            _trace.append({
                "step": "extract",
                "attempt": attempt,
                "source_file": source_file,
                "grant_name": grant.name,
                "ms": ms,
            })
            logger.info("Extracted grant '%s' from %s (attempt %d)", grant.name, source_file, attempt)
            return grant

        except Exception as exc:
            last_error = exc
            ms = round((time.time() - t0) * 1000)
            logger.warning(
                "Extraction attempt %d/%d failed for %s: %s",
                attempt, EXTRACTION_MAX_RETRIES, source_file, exc,
            )
            _trace.append({
                "step": "extract_retry",
                "attempt": attempt,
                "error": str(exc)[:120],
                "ms": ms,
            })

    logger.error("All extraction attempts failed for %s: %s", source_file, last_error)
    return None


def extract_from_chunks(
    chunks: list[Any],  # list[DocumentChunk]
    trace: list[dict] | None = None,
) -> list[GrantInfo]:
    """
    Run extraction on a list of DocumentChunks.
    Groups chunks by source file and concatenates the first ~4000 chars
    to give Granite as much context as possible per grant.

    Returns a deduplicated list of GrantInfo objects (one per source file).
    """
    # Group by source file
    by_file: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_file.setdefault(chunk.source_file, []).append(chunk)

    results: list[GrantInfo] = []
    for source_file, file_chunks in by_file.items():
        # Combine chunk texts up to ~5000 chars for context richness
        combined_text = " ".join(c.text for c in file_chunks)[:5000]
        first_page = file_chunks[0].page if file_chunks else None
        grant = extract_grant_info(
            text=combined_text,
            source_file=source_file,
            source_page=first_page,
            trace=trace,
        )
        if grant:
            results.append(grant)
    return results
