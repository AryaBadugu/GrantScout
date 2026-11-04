"""
Matching agent.
Hybrid scoring = hard eligibility filters (rule-based) +
                 semantic similarity (RAG) +
                 Granite LLM reasoning.
Outputs ranked MatchResult list with explanations and deadline countdowns.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

from core.schemas import GrantInfo, MatchResult, StartupProfile
from core.retrieval import retrieve_for_grant_query, retrieve_similar_chunks
from core.watsonx_client import generate_text

logger = logging.getLogger(__name__)

# Weights for hybrid score
W_RULE = 0.35
W_SEMANTIC = 0.35
W_REASONING = 0.30


# ─── Rule-based eligibility filter ───────────────────────────────────────────

def _rule_score(grant: GrantInfo, profile: StartupProfile) -> tuple[float, list[str], list[str]]:
    """
    Returns (score 0-1, why_matches list, what_is_missing list).
    Hard filters that fail -> score 0. Soft matches accumulate.
    """
    why: list[str] = []
    missing: list[str] = []
    total_checks = 0
    passed = 0

    # Deadline check – skip expired grants
    if grant.deadline_date:
        today = date.today()
        if grant.deadline_date < today:
            missing.append(f"Deadline passed ({grant.deadline_date})")
            return 0.0, why, missing

    # Stage match
    if grant.eligible_stages:
        total_checks += 1
        profile_stage = profile.stage if isinstance(profile.stage, str) else profile.stage.value
        allowed_stages = [
            s if isinstance(s, str) else s.value for s in grant.eligible_stages
        ]
        if "any" in allowed_stages or profile_stage in allowed_stages:
            passed += 1
            why.append(f"Stage '{profile_stage}' matches eligible stages")
        else:
            missing.append(f"Stage '{profile_stage}' not in eligible stages {allowed_stages}")

    # Location check
    if grant.location_restrictions:
        total_checks += 1
        loc = grant.location_restrictions.lower()
        state = profile.state.lower()
        if state in loc or "all" in loc or "pan india" in loc or "any" in loc:
            passed += 1
            why.append(f"Location '{profile.state}' is covered")
        else:
            missing.append(f"Grant restricted to: {grant.location_restrictions}")

    # DPIIT check
    dpiit_keywords = ["dpiit", "startup india", "dipp recognised", "recognised startup"]
    if grant.eligibility_criteria:
        total_checks += 1
        criteria_text = " ".join(grant.eligibility_criteria).lower()
        if any(kw in criteria_text for kw in dpiit_keywords):
            if profile.dpiit_recognized:
                passed += 1
                why.append("Startup has DPIIT recognition as required")
            else:
                missing.append("Grant requires DPIIT / Startup India recognition")
        else:
            passed += 1  # no DPIIT requirement

    # Sector match (soft – both strings)
    if grant.sectors:
        total_checks += 1
        profile_sector = profile.sector.lower()
        grant_sectors = [s.lower() for s in grant.sectors]
        if "any" in grant_sectors or any(
            profile_sector in gs or gs in profile_sector for gs in grant_sectors
        ):
            passed += 1
            why.append(f"Sector '{profile.sector}' aligns with grant sectors")
        else:
            missing.append(f"Sector '{profile.sector}' not in {grant.sectors}")

    # Funding amount alignment (soft)
    if grant.funding_amount_inr and profile.funding_needed_inr:
        total_checks += 1
        if profile.funding_needed_inr <= grant.funding_amount_inr:
            passed += 1
            why.append(
                f"Requested ₹{profile.funding_needed_inr:,.0f} ≤ "
                f"grant ceiling ₹{grant.funding_amount_inr:,.0f}"
            )
        else:
            missing.append(
                f"Requested ₹{profile.funding_needed_inr:,.0f} exceeds "
                f"grant ceiling ₹{grant.funding_amount_inr:,.0f}"
            )

    if total_checks == 0:
        return 0.5, why, missing  # not enough info to filter

    return round(passed / total_checks, 3), why, missing


# ─── Semantic score ───────────────────────────────────────────────────────────

def _semantic_score(grant: GrantInfo, profile: StartupProfile, chunks_map: dict[str, float]) -> float:
    """
    Look up the best similarity score for the grant's source file
    from the pre-computed chunks_map {source_file: max_similarity}.
    """
    if grant.source_file and grant.source_file in chunks_map:
        return chunks_map[grant.source_file]
    return 0.0


# ─── Granite reasoning ────────────────────────────────────────────────────────

_REASONING_PROMPT = """\
You are a grant eligibility analyst. Given the startup profile and grant details below, assess how well the startup fits this grant.

Startup Profile:
- Name: {name}
- Stage: {stage}
- Sector: {sector}
- State: {state}
- Team size: {team_size}
- DPIIT recognised: {dpiit}
- Funding needed: {funding_needed}
- Description: {description}

Grant:
- Name: {grant_name}
- Issuing body: {issuing_body}
- Funding: {funding_amount}
- Eligible stages: {eligible_stages}
- Sectors: {sectors}
- Location: {location}
- Key eligibility: {eligibility}

Rate the fit as a number from 0.0 to 1.0 and list up to 3 reasons why this grant matches and up to 3 gaps.

Respond ONLY in this JSON format:
{{"score": <float 0.0-1.0>, "matches": ["reason1", "reason2"], "gaps": ["gap1"]}}
"""


def _reasoning_score(
    grant: GrantInfo, profile: StartupProfile
) -> tuple[float, list[str], list[str]]:
    """Call Granite to get a reasoning-based match score."""
    prompt = _REASONING_PROMPT.format(
        name=profile.name,
        stage=profile.stage,
        sector=profile.sector,
        state=profile.state,
        team_size=profile.team_size,
        dpiit="Yes" if profile.dpiit_recognized else "No",
        funding_needed=f"₹{profile.funding_needed_inr:,.0f}" if profile.funding_needed_inr else "Not specified",
        description=profile.description[:400],
        grant_name=grant.name,
        issuing_body=grant.issuing_body or "N/A",
        funding_amount=grant.funding_amount or "N/A",
        eligible_stages=", ".join(str(s) for s in grant.eligible_stages) or "Any",
        sectors=", ".join(grant.sectors) or "Any",
        location=grant.location_restrictions or "Pan-India",
        eligibility="; ".join(grant.eligibility_criteria[:3]) or "N/A",
    )
    try:
        import json, re
        raw = generate_text(prompt, max_new_tokens=300)
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            return score, data.get("matches", []), data.get("gaps", [])
    except Exception as exc:
        logger.warning("Granite reasoning score failed: %s", exc)
    return 0.5, [], []


# ─── Days until deadline ──────────────────────────────────────────────────────

def _days_until_deadline(grant: GrantInfo) -> int | None:
    if not grant.deadline_date:
        return None
    delta = (grant.deadline_date - date.today()).days
    return delta


# ─── Main matching pipeline ───────────────────────────────────────────────────

def match_grants(
    profile: StartupProfile,
    grants: list[GrantInfo],
    top_k: int = 10,
    trace: list[dict] | None = None,
) -> list[MatchResult]:
    """
    Score and rank a list of GrantInfo objects against a startup profile.

    Parameters
    ----------
    profile: StartupProfile
    grants: list[GrantInfo]
        All known grants (from extraction or synthetic data).
    top_k: int
        Return only the top-k results.
    trace: list[dict] | None

    Returns
    -------
    list[MatchResult]
        Ranked list, highest match_score first.
    """
    _trace = trace if trace is not None else []
    t0 = time.time()

    # Retrieve semantic similarity for all grants at once
    _trace.append({"step": "retrieve", "description": "Semantic retrieval for startup profile"})
    query_chunks = retrieve_for_grant_query(
        startup_description=profile.description,
        sector=profile.sector,
        stage=str(profile.stage),
        state=profile.state,
        top_k=50,
    )

    # Build map: source_file -> best similarity score
    chunks_map: dict[str, float] = {}
    for chunk in query_chunks:
        sim = chunk.metadata.get("similarity", 0.0)
        if chunk.source_file not in chunks_map or sim > chunks_map[chunk.source_file]:
            chunks_map[chunk.source_file] = sim

    _trace.append({
        "step": "filter",
        "description": "Applying hard eligibility filters",
        "grants_evaluated": len(grants),
    })

    results: list[MatchResult] = []
    for grant in grants:
        # Rule-based
        rule_s, why, missing = _rule_score(grant, profile)
        if rule_s == 0.0 and missing and "Deadline passed" in str(missing):
            continue  # Hard exclude expired grants

        # Semantic
        sem_s = _semantic_score(grant, profile, chunks_map)

        # Granite reasoning (only if rule_score > 0)
        if rule_s > 0:
            reas_s, llm_why, llm_missing = _reasoning_score(grant, profile)
        else:
            reas_s, llm_why, llm_missing = 0.0, [], []

        # Combine
        combined = (W_RULE * rule_s) + (W_SEMANTIC * sem_s) + (W_REASONING * reas_s)
        why_all = why + llm_why
        missing_all = list(set(missing + llm_missing))

        sources = []
        if grant.source_file:
            page_str = f" p.{grant.source_page}" if grant.source_page else ""
            sources.append(f"{grant.source_file}{page_str}")

        results.append(
            MatchResult(
                grant=grant,
                match_score=round(combined, 3),
                semantic_score=round(sem_s, 3),
                rule_score=round(rule_s, 3),
                reasoning_score=round(reas_s, 3),
                why_matches=why_all[:5],
                what_is_missing=missing_all[:5],
                days_until_deadline=_days_until_deadline(grant),
                sources=sources,
            )
        )

    results.sort(key=lambda r: r.match_score, reverse=True)
    top = results[:top_k]

    _trace.append({
        "step": "score",
        "description": "Hybrid scoring complete",
        "total_ranked": len(results),
        "returned": len(top),
        "ms": round((time.time() - t0) * 1000),
    })
    return top
