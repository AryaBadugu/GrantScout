"""
Eligibility Gap Analyzer.
For near-miss grants (score 40-70%), produce a concrete "how to become eligible"
action list grounded in the grant text, with citations.
"""
from __future__ import annotations

import logging
from typing import Any

from core.retrieval import get_context_for_source
from core.schemas import GrantInfo, MatchResult, StartupProfile
from core.watsonx_client import generate_text

logger = logging.getLogger(__name__)

_GAP_PROMPT = """\
You are a startup eligibility advisor. A startup is close to qualifying for a grant but has gaps.
List SPECIFIC, ACTIONABLE steps the startup can take to become eligible.
Ground every suggestion in the grant eligibility criteria provided. Cite the criterion.
Do NOT invent requirements not stated in the grant text.

Startup Profile:
- Name: {name}
- Stage: {stage}
- Sector: {sector}
- State: {state}
- DPIIT Recognised: {dpiit}
- Team size: {team_size}

Grant: {grant_name}
Issuing Body: {issuing_body}

Known Gaps:
{gaps}

Grant Eligibility Criteria (from document):
{criteria}

Respond with a numbered list of 3-5 specific actions. For each, cite the criterion it addresses.
Format:
1. [Action] — addresses criterion: "[criterion text]" [source: {source_file}]
"""


def analyze_gaps(
    match: MatchResult,
    profile: StartupProfile,
    trace: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Generate a concrete gap-closing action list for a near-miss grant.

    Parameters
    ----------
    match : MatchResult
        A match with score between 0.35 and 0.72 (near-miss range).
    profile : StartupProfile
    trace : list[dict] | None

    Returns
    -------
    dict with keys: actions (list[str]), grant_name (str), sources (list[str])
    """
    _trace = trace if trace is not None else []
    grant = match.grant

    # Pull grant context
    source_context = ""
    if grant.source_file:
        source_context = get_context_for_source(grant.source_file, max_chars=2000)

    criteria_text = "\n".join(
        f"  - {c}" for c in grant.eligibility_criteria[:10]
    ) or "  (No criteria extracted from document)"

    gaps_text = "\n".join(
        f"  - {g}" for g in match.what_is_missing[:5]
    ) or "  (No specific gaps identified)"

    prompt = _GAP_PROMPT.format(
        name=profile.name,
        stage=profile.stage,
        sector=profile.sector,
        state=profile.state,
        dpiit="Yes" if profile.dpiit_recognized else "No",
        team_size=profile.team_size,
        grant_name=grant.name,
        issuing_body=grant.issuing_body or "N/A",
        gaps=gaps_text,
        criteria=criteria_text + ("\n\n" + source_context[:1000] if source_context else ""),
        source_file=grant.source_file or "grant document",
    )

    try:
        raw = generate_text(prompt, max_new_tokens=600)
        # Split into numbered action lines
        actions = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                actions.append(line.lstrip("0123456789.-) ").strip())
        if not actions:
            actions = [raw.strip()]
    except Exception as exc:
        logger.error("Gap analyzer failed: %s", exc)
        actions = [f"[Gap analysis unavailable: {exc}]"]

    sources = []
    if grant.source_file:
        sources.append(grant.source_file + (f" p.{grant.source_page}" if grant.source_page else ""))

    _trace.append({
        "step": "gap_analysis",
        "grant": grant.name,
        "actions_generated": len(actions),
    })

    return {
        "grant_name": grant.name,
        "actions": actions,
        "sources": sources,
    }
