"""
Pytest tests for GrantScout.
- Schema validation
- Matching logic (rule-based)
- Extraction JSON parsing
All IBM watsonx calls are mocked.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Stub ibm_watsonx_ai SDK before any core imports ─────────────────────────
# This lets tests run without the real IBM SDK installed.

def _make_ibm_stub():
    """Build a minimal sys.modules stub for ibm_watsonx_ai."""
    root = types.ModuleType("ibm_watsonx_ai")
    root.Credentials = MagicMock()
    root.APIClient = MagicMock()

    creds_mod = types.ModuleType("ibm_watsonx_ai.foundation_models")
    creds_mod.ModelInference = MagicMock()
    root.foundation_models = creds_mod

    emb_mod = types.ModuleType("ibm_watsonx_ai.foundation_models.embeddings")
    emb_mod.Embeddings = MagicMock()
    creds_mod.embeddings = emb_mod

    meta_mod = types.ModuleType("ibm_watsonx_ai.metanames")
    gp = MagicMock()
    gp.MAX_NEW_TOKENS = "max_new_tokens"
    gp.TEMPERATURE = "temperature"
    gp.STOP_SEQUENCES = "stop_sequences"
    meta_mod.GenTextParamsMetaNames = gp
    root.metanames = meta_mod

    sys.modules.setdefault("ibm_watsonx_ai", root)
    sys.modules.setdefault("ibm_watsonx_ai.foundation_models", creds_mod)
    sys.modules.setdefault("ibm_watsonx_ai.foundation_models.embeddings", emb_mod)
    sys.modules.setdefault("ibm_watsonx_ai.metanames", meta_mod)

    # IBM boto3 stub
    boto3_mod = types.ModuleType("ibm_boto3")
    boto3_mod.client = MagicMock()
    sys.modules.setdefault("ibm_boto3", boto3_mod)

    bc_mod = types.ModuleType("ibm_botocore")
    bc_cfg = types.ModuleType("ibm_botocore.client")
    bc_cfg.Config = MagicMock()
    bc_mod.client = bc_cfg
    sys.modules.setdefault("ibm_botocore", bc_mod)
    sys.modules.setdefault("ibm_botocore.client", bc_cfg)


_make_ibm_stub()


# ─── Schema tests ─────────────────────────────────────────────────────────────

class TestGrantInfoSchema:
    def test_valid_grant(self):
        from core.schemas import GrantInfo, StartupStage
        g = GrantInfo(
            name="Test Grant",
            issuing_body="DPIIT",
            funding_amount="up to ₹50 lakh",
            funding_amount_inr=5_000_000,
            eligible_stages=["ideation", "prototype"],
            sectors=["agritech"],
            deadline="Rolling basis",
            source_file="test.pdf",
        )
        assert g.name == "Test Grant"
        assert g.funding_amount_inr == 5_000_000
        assert "ideation" in g.eligible_stages

    def test_missing_optional_fields_are_null(self):
        from core.schemas import GrantInfo
        g = GrantInfo(name="Minimal Grant")
        assert g.issuing_body is None
        assert g.funding_amount is None
        assert g.deadline is None
        assert g.eligible_stages == []

    def test_sectors_coerced_from_string(self):
        from core.schemas import GrantInfo
        g = GrantInfo(name="X", sectors="agritech, cleantech, edtech")
        assert g.sectors == ["agritech", "cleantech", "edtech"]

    def test_stages_coerced_from_string(self):
        from core.schemas import GrantInfo
        g = GrantInfo(name="X", eligible_stages="ideation")
        assert g.eligible_stages == ["ideation"]


class TestStartupProfileSchema:
    def test_valid_profile(self):
        from core.schemas import StartupProfile
        p = StartupProfile(
            name="TestStartup",
            stage="prototype",
            sector="agritech",
            state="Maharashtra",
            team_size=4,
            annual_revenue_inr=0,
            dpiit_recognized=True,
            funding_needed_inr=2_000_000,
            description="Test description",
        )
        assert p.name == "TestStartup"
        assert p.dpiit_recognized is True

    def test_team_size_minimum(self):
        from core.schemas import StartupProfile
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StartupProfile(
                name="X", stage="ideation", sector="any", state="Delhi",
                team_size=0,  # invalid
            )


class TestMatchResultSchema:
    def test_score_bounds(self):
        from core.schemas import GrantInfo, MatchResult
        from pydantic import ValidationError
        grant = GrantInfo(name="G")
        with pytest.raises(ValidationError):
            MatchResult(
                grant=grant,
                match_score=1.5,  # out of bounds
                semantic_score=0.5,
                rule_score=0.5,
                reasoning_score=0.5,
            )


# ─── Matching logic tests (rule-based, no LLM) ───────────────────────────────

class TestRuleScore:
    def _make_grant(self, **kwargs):
        from core.schemas import GrantInfo
        defaults = dict(name="TestGrant", eligible_stages=["prototype"], sectors=["agritech"])
        defaults.update(kwargs)
        return GrantInfo(**defaults)

    def _make_profile(self, **kwargs):
        from core.schemas import StartupProfile
        defaults = dict(
            name="TestStartup", stage="prototype", sector="agritech",
            state="Maharashtra", team_size=3, dpiit_recognized=True,
            funding_needed_inr=2_000_000,
        )
        defaults.update(kwargs)
        return StartupProfile(**defaults)

    def test_matching_stage(self):
        from core.matching import _rule_score
        g = self._make_grant(eligible_stages=["prototype"])
        p = self._make_profile(stage="prototype")
        score, why, missing = _rule_score(g, p)
        assert score > 0
        assert any("Stage" in w for w in why)

    def test_non_matching_stage(self):
        from core.matching import _rule_score
        g = self._make_grant(eligible_stages=["growth"])
        p = self._make_profile(stage="ideation")
        score, why, missing = _rule_score(g, p)
        # Should reduce score
        assert any("Stage" in m for m in missing)

    def test_location_restriction_pass(self):
        from core.matching import _rule_score
        g = self._make_grant(location_restrictions="Maharashtra only")
        p = self._make_profile(state="Maharashtra")
        score, why, missing = _rule_score(g, p)
        assert any("Location" in w for w in why)

    def test_location_restriction_fail(self):
        from core.matching import _rule_score
        g = self._make_grant(location_restrictions="Karnataka only")
        p = self._make_profile(state="Maharashtra")
        score, why, missing = _rule_score(g, p)
        assert any("restricted" in m.lower() for m in missing)

    def test_dpiit_required_not_met(self):
        from core.matching import _rule_score
        g = self._make_grant(
            eligibility_criteria=["Startup must be DPIIT recognized"]
        )
        p = self._make_profile(dpiit_recognized=False)
        score, why, missing = _rule_score(g, p)
        assert any("DPIIT" in m for m in missing)

    def test_dpiit_required_met(self):
        from core.matching import _rule_score
        g = self._make_grant(
            eligibility_criteria=["Startup must be DPIIT recognized"]
        )
        p = self._make_profile(dpiit_recognized=True)
        score, why, missing = _rule_score(g, p)
        assert any("DPIIT" in w for w in why)

    def test_funding_ceiling_match(self):
        from core.matching import _rule_score
        g = self._make_grant(funding_amount_inr=5_000_000)
        p = self._make_profile(funding_needed_inr=2_000_000)
        score, why, missing = _rule_score(g, p)
        assert any("₹" in w for w in why)

    def test_funding_exceeds_ceiling(self):
        from core.matching import _rule_score
        g = self._make_grant(funding_amount_inr=1_000_000)
        p = self._make_profile(funding_needed_inr=5_000_000)
        score, why, missing = _rule_score(g, p)
        assert any("exceeds" in m.lower() for m in missing)

    def test_any_stage_always_matches(self):
        from core.matching import _rule_score
        g = self._make_grant(eligible_stages=["any"])
        p = self._make_profile(stage="scaling")
        score, why, missing = _rule_score(g, p)
        assert any("Stage" in w for w in why)


# ─── Extraction JSON parsing tests (no LLM calls) ────────────────────────────

class TestExtractionParsing:
    def test_parse_clean_json(self):
        from core.extraction import _parse_json_response
        raw = '{"name": "Test Grant", "issuing_body": "DPIIT"}'
        result = _parse_json_response(raw)
        assert result["name"] == "Test Grant"

    def test_parse_json_with_markdown_fence(self):
        from core.extraction import _parse_json_response
        raw = '```json\n{"name": "Test Grant"}\n```'
        result = _parse_json_response(raw)
        assert result["name"] == "Test Grant"

    def test_parse_json_with_preamble(self):
        from core.extraction import _parse_json_response
        raw = 'Here is the result:\n{"name": "Grant X", "issuing_body": null}'
        result = _parse_json_response(raw)
        assert result["name"] == "Grant X"
        assert result["issuing_body"] is None

    def test_parse_invalid_json_raises(self):
        from core.extraction import _parse_json_response
        with pytest.raises((ValueError, Exception)):
            _parse_json_response("This is not JSON at all.")

    def test_extract_grant_info_mocked(self):
        """Test full extraction flow with mocked LLM."""
        from core.extraction import extract_grant_info

        mock_response = json.dumps({
            "name": "Mock Grant",
            "issuing_body": "Mock Ministry",
            "funding_amount": "up to ₹25 lakh",
            "funding_amount_inr": 2500000,
            "eligible_stages": ["prototype", "early_stage"],
            "sectors": ["agritech"],
            "location_restrictions": None,
            "eligibility_criteria": ["Must be DPIIT recognized"],
            "deadline": "2025-12-31",
            "deadline_date": None,
            "required_documents": ["DPIIT certificate"],
            "application_link": "https://example.gov.in",
            "source_file": "test.txt",
            "source_page": 1,
        })

        with patch("core.extraction.generate_text", return_value=mock_response):
            grant = extract_grant_info(
                text="Some grant text here",
                source_file="test.txt",
                source_page=1,
            )

        assert grant is not None
        assert grant.name == "Mock Grant"
        assert grant.funding_amount_inr == 2_500_000
        assert "agritech" in grant.sectors

    def test_extract_retries_on_bad_json(self):
        """Extraction should retry and eventually return None after max retries."""
        from core.extraction import extract_grant_info
        from core.config import EXTRACTION_MAX_RETRIES

        with patch("core.extraction.generate_text", return_value="not valid json"):
            grant = extract_grant_info(text="text", source_file="bad.txt")

        assert grant is None


# ─── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_validate_watsonx_raises_without_credentials(self):
        import os
        from core.config import validate_watsonx
        import core.config as cfg

        # Blank out both the module constant AND the env var so both
        # the constant path and the getter path return empty strings.
        original_key    = cfg.WATSONX_API_KEY
        original_proj   = cfg.WATSONX_PROJECT_ID
        original_env_k  = os.environ.pop("WATSONX_API_KEY", "")
        original_env_p  = os.environ.pop("WATSONX_PROJECT_ID", "")
        cfg.WATSONX_API_KEY   = ""
        cfg.WATSONX_PROJECT_ID = ""
        try:
            with pytest.raises(ValueError, match="WATSONX_API_KEY"):
                validate_watsonx()
        finally:
            cfg.WATSONX_API_KEY    = original_key
            cfg.WATSONX_PROJECT_ID = original_proj
            if original_env_k:
                os.environ["WATSONX_API_KEY"]    = original_env_k
            if original_env_p:
                os.environ["WATSONX_PROJECT_ID"] = original_env_p

    def test_model_ids_configurable(self):
        from core.config import GRANITE_INSTRUCT_MODEL, GRANITE_EMBEDDING_MODEL
        assert isinstance(GRANITE_INSTRUCT_MODEL, str)
        assert len(GRANITE_INSTRUCT_MODEL) > 0
        assert isinstance(GRANITE_EMBEDDING_MODEL, str)
        assert len(GRANITE_EMBEDDING_MODEL) > 0


# ─── Chunking tests ───────────────────────────────────────────────────────────

class TestChunking:
    def test_chunks_not_exceed_size(self):
        from core.ingestion import _chunk_text
        text = " ".join(["word"] * 2000)
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 100

    def test_overlap_creates_more_chunks(self):
        from core.ingestion import _chunk_text
        text = " ".join(["word"] * 500)
        chunks_no_overlap = _chunk_text(text, chunk_size=100, overlap=0)
        chunks_with_overlap = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_empty_text_returns_empty(self):
        from core.ingestion import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []


# ─── Chat tests ───────────────────────────────────────────────────────────────

class TestChat:
    def test_empty_retrieval_returns_not_found(self):
        from core.chat import answer_question

        with patch("core.chat.retrieve_similar_chunks", return_value=[]):
            response = answer_question("What is the SISFS scheme?")

        assert response.grounded is False
        assert "not found" in response.answer.lower()

    def test_grounded_answer(self):
        from core.chat import answer_question
        from core.schemas import DocumentChunk

        mock_chunk = DocumentChunk(
            chunk_id="abc",
            source_file="test.pdf",
            page=3,
            text="SISFS provides up to 50 lakh to DPIIT recognized startups.",
        )

        with patch("core.chat.retrieve_similar_chunks", return_value=[mock_chunk]):
            with patch(
                "core.chat.generate_text",
                return_value="SISFS provides up to 50 lakh. [source: test.pdf p.3]",
            ):
                response = answer_question("What does SISFS provide?")

        assert response.grounded is True
        assert len(response.citations) == 1
        assert response.citations[0].source_file == "test.pdf"
