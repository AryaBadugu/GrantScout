"""
Pitch-deck to profile: parse a startup pitch deck or one-pager (PDF/PPTX/DOCX)
with Docling, then let Granite pre-fill the startup profile form.
User must review and confirm before saving.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PREFILL_PROMPT = """\
You are a startup analyst. Read the pitch deck text below and extract the startup profile.
Return ONLY a valid JSON object with these fields (null if not found):
{{
  "name": "startup name",
  "stage": "ideation|prototype|early_stage|growth|scaling",
  "sector": "primary sector",
  "state": "Indian state of registration or HQ",
  "team_size": integer or null,
  "annual_revenue_inr": number or null,
  "dpiit_recognized": true/false or null,
  "funding_needed_inr": number or null,
  "description": "one-sentence description"
}}

Pitch deck text:
---
{text}
---
JSON:"""


def prefill_from_pitchdeck(
    file_bytes: bytes,
    filename: str,
    trace: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Parse a pitch deck and return a pre-filled profile dict.
    Returns dict with keys: profile (dict), confidence (str), raw_text (str).
    User MUST review and confirm before saving.
    """
    from core.watsonx_client import generate_text

    _trace = trace if trace is not None else []

    # Write to temp file for Docling
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        # Parse with Docling / fallback
        from core.ingestion import _parse_with_docling
        pages = _parse_with_docling(tmp_path)
        raw_text = " ".join(p["text"] for p in pages)[:4000]
    except Exception as exc:
        logger.warning("Pitch deck parse failed: %s", exc)
        raw_text = ""
    finally:
        tmp_path.unlink(missing_ok=True)

    if not raw_text.strip():
        return {"profile": {}, "confidence": "low", "raw_text": ""}

    _trace.append({"step": "pitchdeck_parse", "chars": len(raw_text)})

    prompt = _PREFILL_PROMPT.format(text=raw_text[:3500])
    try:
        response = generate_text(prompt, max_new_tokens=400)
        response = re.sub(r"^```(?:json)?\s*", "", response.strip())
        response = re.sub(r"\s*```$", "", response)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        profile_data = json.loads(match.group()) if match else {}
    except Exception as exc:
        logger.error("Pitch deck profile extraction failed: %s", exc)
        profile_data = {}

    # Estimate confidence by filled fields
    filled = sum(1 for v in profile_data.values() if v is not None and v != "")
    confidence = "high" if filled >= 6 else "medium" if filled >= 3 else "low"

    _trace.append({"step": "pitchdeck_extract", "fields_filled": filled, "confidence": confidence})

    return {
        "profile": profile_data,
        "confidence": confidence,
        "raw_text": raw_text[:500],
    }
