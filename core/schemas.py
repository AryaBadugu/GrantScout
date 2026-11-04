"""
Pydantic schemas for GrantScout.
All domain models live here for easy validation and serialisation.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class StartupStage(str, Enum):
    IDEATION = "ideation"
    PROTOTYPE = "prototype"
    EARLY_STAGE = "early_stage"
    GROWTH = "growth"
    SCALING = "scaling"
    ANY = "any"


class Sector(str, Enum):
    AGRITECH = "agritech"
    CLEANTECH = "cleantech"
    EDTECH = "edtech"
    FINTECH = "fintech"
    HEALTHTECH = "healthtech"
    DEEPTECH = "deeptech"
    MANUFACTURING = "manufacturing"
    SOCIAL_IMPACT = "social_impact"
    SPACETECH = "spacetech"
    DEFENSE = "defense"
    ANY = "any"
    OTHER = "other"


# ─── Grant Schema ─────────────────────────────────────────────────────────────

class GrantInfo(BaseModel):
    """Validated grant metadata extracted from a document."""

    name: str = Field(description="Official name of the grant/scheme")
    issuing_body: Optional[str] = Field(None, description="Ministry, department or agency")
    funding_amount: Optional[str] = Field(None, description="Amount as stated (e.g. 'up to ₹50 lakh')")
    funding_amount_inr: Optional[float] = Field(None, description="Numeric value in INR if parseable")
    eligible_stages: list[StartupStage] = Field(
        default_factory=list,
        description="Startup stages eligible to apply",
    )
    sectors: list[str] = Field(default_factory=list, description="Target sectors")
    location_restrictions: Optional[str] = Field(
        None, description="State/UT restrictions if any"
    )
    eligibility_criteria: list[str] = Field(
        default_factory=list, description="Key eligibility rules"
    )
    deadline: Optional[str] = Field(None, description="Application deadline as text")
    deadline_date: Optional[date] = Field(None, description="Parsed deadline date if available")
    required_documents: list[str] = Field(
        default_factory=list, description="Documents required for application"
    )
    application_link: Optional[str] = Field(None, description="Official application URL")
    source_file: Optional[str] = Field(None, description="Source filename")
    source_page: Optional[int] = Field(None, description="Page number in source document")
    is_synthetic: bool = Field(False, description="True if this is a synthetic test record")

    model_config = {"use_enum_values": True}

    @field_validator("eligible_stages", mode="before")
    @classmethod
    def coerce_stages(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v else []
        return v or []

    @field_validator("sectors", mode="before")
    @classmethod
    def coerce_sectors(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []


# ─── Startup Profile ──────────────────────────────────────────────────────────

class StartupProfile(BaseModel):
    """Startup profile submitted by the user."""

    name: str = Field(description="Startup name")
    stage: StartupStage = Field(description="Current stage")
    sector: str = Field(description="Primary sector/domain")
    state: str = Field(description="State/UT of registration")
    team_size: int = Field(ge=1, description="Number of full-time team members")
    annual_revenue_inr: Optional[float] = Field(
        None, ge=0, description="Annual revenue in INR (0 for pre-revenue)"
    )
    dpiit_recognized: bool = Field(False, description="DPIIT / Startup India recognition")
    funding_needed_inr: Optional[float] = Field(
        None, ge=0, description="Funding required in INR"
    )
    description: str = Field("", description="One-paragraph startup description")

    model_config = {"use_enum_values": True}


# ─── Match Result ─────────────────────────────────────────────────────────────

class MatchResult(BaseModel):
    """Ranked grant match for a startup profile."""

    grant: GrantInfo
    match_score: float = Field(ge=0.0, le=1.0, description="Overall match score 0-1")
    semantic_score: float = Field(ge=0.0, le=1.0)
    rule_score: float = Field(ge=0.0, le=1.0)
    reasoning_score: float = Field(ge=0.0, le=1.0)
    why_matches: list[str] = Field(default_factory=list)
    what_is_missing: list[str] = Field(default_factory=list)
    days_until_deadline: Optional[int] = Field(None)
    sources: list[str] = Field(default_factory=list, description="Source doc + page citations")


# ─── Chunk schema ─────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A text chunk with metadata, ready for embedding."""

    chunk_id: str
    source_file: str
    page: Optional[int] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Chat schemas ─────────────────────────────────────────────────────────────

class Citation(BaseModel):
    source_file: str
    page: Optional[int] = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = Field(True, description="False if context did not support the answer")


# ─── Agent trace ──────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    step_name: str
    description: str
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: Optional[float] = None
