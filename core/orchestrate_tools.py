"""
Orchestrate tool wrappers.
Exposes the GrantScout pipeline steps as clean tool functions with JSON schemas,
compatible with watsonx Orchestrate and Langflow.
Does NOT add a hard dependency on Orchestrate.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Tool: extract_grant ──────────────────────────────────────────────────────

TOOL_EXTRACT_GRANT = {
    "name": "extract_grant",
    "description": "Extract structured grant metadata from raw document text.",
    "parameters": {
        "type": "object",
        "properties": {
            "text":        {"type": "string", "description": "Raw text from the grant document"},
            "source_file": {"type": "string", "description": "Filename for provenance"},
            "source_page": {"type": "integer", "description": "Page number"},
        },
        "required": ["text"],
    },
}


def extract_grant(text: str, source_file: str = "", source_page: int | None = None) -> dict[str, Any]:
    """Extract structured grant info. Returns GrantInfo as dict or error."""
    from core.extraction import extract_grant_info
    result = extract_grant_info(text=text, source_file=source_file, source_page=source_page)
    if result:
        return result.model_dump()
    return {"error": "Extraction failed after retries"}


# ─── Tool: match_grants ───────────────────────────────────────────────────────

TOOL_MATCH_GRANTS = {
    "name": "match_grants",
    "description": "Match a startup profile against the indexed grant database and return ranked results.",
    "parameters": {
        "type": "object",
        "properties": {
            "startup_name":         {"type": "string"},
            "stage":                {"type": "string", "enum": ["ideation","prototype","early_stage","growth","scaling","any"]},
            "sector":               {"type": "string"},
            "state":                {"type": "string"},
            "team_size":            {"type": "integer"},
            "dpiit_recognized":     {"type": "boolean"},
            "funding_needed_inr":   {"type": "number"},
            "description":          {"type": "string"},
            "top_k":                {"type": "integer", "default": 5},
        },
        "required": ["startup_name", "stage", "sector", "state", "team_size"],
    },
}


def match_grants_tool(
    startup_name: str, stage: str, sector: str, state: str,
    team_size: int, dpiit_recognized: bool = False,
    funding_needed_inr: float | None = None,
    description: str = "", top_k: int = 5,
    grants: list | None = None,
) -> list[dict[str, Any]]:
    """
    Match grants. If grants list not provided, returns empty (caller must ingest first).
    Returns list of MatchResult dicts.
    """
    from core.schemas import StartupProfile
    from core.matching import match_grants

    profile = StartupProfile(
        name=startup_name, stage=stage, sector=sector, state=state,
        team_size=team_size, dpiit_recognized=dpiit_recognized,
        funding_needed_inr=funding_needed_inr, description=description,
    )
    results = match_grants(profile=profile, grants=grants or [], top_k=top_k)
    return [
        {
            "grant_name": r.grant.name,
            "match_score": r.match_score,
            "why_matches": r.why_matches,
            "what_is_missing": r.what_is_missing,
            "days_until_deadline": r.days_until_deadline,
            "sources": r.sources,
        }
        for r in results
    ]


# ─── Tool: draft_application ─────────────────────────────────────────────────

TOOL_DRAFT_APPLICATION = {
    "name": "draft_application",
    "description": "Generate a tailored grant application draft. Returns DRAFT text requiring human review.",
    "parameters": {
        "type": "object",
        "properties": {
            "grant_name":      {"type": "string", "description": "Name of the grant to draft for"},
            "startup_profile": {"type": "object", "description": "StartupProfile fields"},
        },
        "required": ["grant_name", "startup_profile"],
    },
}


def draft_application_tool(grant_name: str, startup_profile: dict, grants: list | None = None) -> dict[str, Any]:
    """Draft a grant application. grant_name must match an indexed grant."""
    from core.schemas import StartupProfile
    from core.drafting import draft_proposal

    grant_list = grants or []
    grant = next((g for g in grant_list if g.name.lower() == grant_name.lower()), None)
    if not grant:
        return {"error": f"Grant '{grant_name}' not found in provided grants list"}

    try:
        profile = StartupProfile(**startup_profile)
    except Exception as exc:
        return {"error": f"Invalid startup profile: {exc}"}

    return draft_proposal(grant=grant, profile=profile)


# ─── Tool: ask_question ───────────────────────────────────────────────────────

TOOL_ASK_QUESTION = {
    "name": "ask_question",
    "description": "Answer a question about grants using RAG over indexed documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Natural language question about grants"},
            "top_k":    {"type": "integer", "default": 6},
        },
        "required": ["question"],
    },
}


def ask_question_tool(question: str, top_k: int = 6) -> dict[str, Any]:
    """Ask a question about grants. Returns answer with citations."""
    from core.chat import answer_question
    response = answer_question(question=question, top_k=top_k)
    return {
        "answer": response.answer,
        "grounded": response.grounded,
        "citations": [
            {"source": c.source_file, "page": c.page, "snippet": c.snippet[:150]}
            for c in response.citations
        ],
    }


# ─── All tools manifest ───────────────────────────────────────────────────────

ALL_TOOLS = [
    TOOL_EXTRACT_GRANT,
    TOOL_MATCH_GRANTS,
    TOOL_DRAFT_APPLICATION,
    TOOL_ASK_QUESTION,
]

ALL_FUNCTIONS = {
    "extract_grant":       extract_grant,
    "match_grants":        match_grants_tool,
    "draft_application":   draft_application_tool,
    "ask_question":        ask_question_tool,
}


def dispatch(tool_name: str, arguments: dict[str, Any], **kwargs) -> Any:
    """Route a tool call by name. kwargs passed through to the function."""
    fn = ALL_FUNCTIONS.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool: {tool_name}"}
    merged = {**arguments, **kwargs}
    return fn(**merged)


if __name__ == "__main__":
    print(json.dumps(ALL_TOOLS, indent=2))
