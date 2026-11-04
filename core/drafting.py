"""
Proposal drafting agent.
Generates a tailored grant application draft grounded ONLY in the
startup profile and the retrieved grant document text.
Drafts are labelled DRAFT and require explicit user approval before export.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.retrieval import get_context_for_source, retrieve_similar_chunks
from core.schemas import GrantInfo, MatchResult, StartupProfile
from core.watsonx_client import generate_text

logger = logging.getLogger(__name__)

_DRAFT_PROMPT = """\
You are an expert grant proposal writer helping an Indian startup prepare a government grant application.

IMPORTANT:
- Ground EVERY claim in the startup profile and the grant document provided.
- Do NOT invent facts, statistics, or eligibility claims.
- Output ONLY the draft proposal sections as requested. No preamble.

Startup Profile:
- Name: {name}
- Stage: {stage}
- Sector: {sector}
- State: {state}
- Team size: {team_size}
- DPIIT recognised: {dpiit}
- Funding needed: {funding_needed}
- Description: {description}

Grant Information (from official document):
{grant_context}

Write a tailored grant application draft with these numbered sections:
1. Executive Summary (3-4 sentences)
2. Problem Statement (what problem does the startup solve, citing the sector relevance to the grant)
3. Proposed Solution & Innovation (what makes this startup's approach unique)
4. Expected Impact (social, economic, sector-level; cite the grant's stated objectives)
5. Budget Justification (how the requested funds will be used; align with grant guidelines)
6. Eligibility Mapping (bullet list: each grant criterion -> how the startup meets it)
7. Team & Capabilities (brief team overview supporting execution ability)

⚠️ DRAFT - REVIEW BEFORE SUBMISSION ⚠️
All claims must be verified against the official grant portal before submission.
Deadlines and amounts carry a "verify on the official portal" notice.
"""


def _build_grant_context(grant: GrantInfo, source_context: str) -> str:
    """Format grant information for the prompt."""
    lines = [
        f"Grant Name: {grant.name}",
        f"Issuing Body: {grant.issuing_body or 'N/A'}",
        f"Funding: {grant.funding_amount or 'N/A'} (verify on official portal)",
        f"Deadline: {grant.deadline or 'N/A'} (verify on official portal)",
        f"Eligible Stages: {', '.join(str(s) for s in grant.eligible_stages) or 'Any'}",
        f"Sectors: {', '.join(grant.sectors) or 'Any'}",
        f"Location: {grant.location_restrictions or 'Pan-India'}",
        "Eligibility Criteria:",
    ]
    for c in grant.eligibility_criteria[:8]:
        lines.append(f"  - {c}")
    lines.append("Required Documents:")
    for d in grant.required_documents[:8]:
        lines.append(f"  - {d}")
    if grant.application_link:
        lines.append(f"Application Link: {grant.application_link}")
    if source_context:
        lines.append("\n--- Relevant Grant Document Excerpt ---")
        lines.append(source_context[:2000])
    return "\n".join(lines)


def draft_proposal(
    grant: GrantInfo,
    profile: StartupProfile,
    trace: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Generate a grant application draft for a startup.

    Returns
    -------
    dict with keys:
      - draft: str – the full proposal text (labelled DRAFT)
      - checklist: list[str] – required documents checklist
      - sources: list[str] – citations
      - warning: str – responsible AI notice
    """
    _trace = trace if trace is not None else []
    t0 = time.time()

    # Retrieve relevant context from the grant source document
    source_context = ""
    if grant.source_file:
        source_context = get_context_for_source(grant.source_file, max_chars=3000)
    if not source_context:
        # Fallback: semantic search
        query = f"{grant.name} eligibility criteria application requirements"
        chunks = retrieve_similar_chunks(query, top_k=5)
        source_context = " ".join(c.text for c in chunks)[:3000]

    _trace.append({
        "step": "retrieve_context",
        "grant": grant.name,
        "context_chars": len(source_context),
        "ms": round((time.time() - t0) * 1000),
    })

    grant_context = _build_grant_context(grant, source_context)

    prompt = _DRAFT_PROMPT.format(
        name=profile.name,
        stage=profile.stage,
        sector=profile.sector,
        state=profile.state,
        team_size=profile.team_size,
        dpiit="Yes" if profile.dpiit_recognized else "No",
        funding_needed=f"₹{profile.funding_needed_inr:,.0f}" if profile.funding_needed_inr else "Not specified",
        description=profile.description[:500],
        grant_context=grant_context,
    )

    t1 = time.time()
    try:
        draft_text = generate_text(prompt, max_new_tokens=1800)
    except Exception as exc:
        logger.error("Draft generation failed: %s", exc)
        draft_text = f"[Draft generation failed: {exc}]"

    _trace.append({
        "step": "draft",
        "grant": grant.name,
        "draft_chars": len(draft_text),
        "ms": round((time.time() - t1) * 1000),
    })

    checklist = grant.required_documents or ["No required documents listed in source"]

    sources = []
    if grant.source_file:
        page_str = f" (p.{grant.source_page})" if grant.source_page else ""
        sources.append(f"{grant.source_file}{page_str}")

    warning = (
        "⚠️ DRAFT – REVIEW BEFORE SUBMISSION. "
        "All grant details (deadlines, amounts, eligibility) must be independently "
        "verified on the official portal before submission. "
        "This draft was AI-generated and may contain errors."
    )

    return {
        "draft": f"⚠️ DRAFT - REVIEW BEFORE SUBMISSION ⚠️\n\n{draft_text}",
        "checklist": checklist,
        "sources": sources,
        "warning": warning,
        "approved": False,  # requires explicit user approval
    }
