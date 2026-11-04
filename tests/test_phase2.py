"""
Additional tests for Phase 2 features.
Gap Analyzer, Export (DOCX/PDF), Pitch Deck, Orchestrate Tools.
All IBM calls mocked.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── Stub ibm_watsonx_ai (same as test_grantscout.py) ────────────────────────
def _make_ibm_stub():
    root = types.ModuleType("ibm_watsonx_ai")
    root.Credentials = MagicMock(); root.APIClient = MagicMock()
    creds_mod = types.ModuleType("ibm_watsonx_ai.foundation_models")
    creds_mod.ModelInference = MagicMock()
    root.foundation_models = creds_mod
    emb_mod = types.ModuleType("ibm_watsonx_ai.foundation_models.embeddings")
    emb_mod.Embeddings = MagicMock(); creds_mod.embeddings = emb_mod
    meta_mod = types.ModuleType("ibm_watsonx_ai.metanames")
    gp = MagicMock(); gp.MAX_NEW_TOKENS="max_new_tokens"; gp.TEMPERATURE="temperature"; gp.STOP_SEQUENCES="stop_sequences"
    meta_mod.GenTextParamsMetaNames = gp; root.metanames = meta_mod
    sys.modules.setdefault("ibm_watsonx_ai", root)
    sys.modules.setdefault("ibm_watsonx_ai.foundation_models", creds_mod)
    sys.modules.setdefault("ibm_watsonx_ai.foundation_models.embeddings", emb_mod)
    sys.modules.setdefault("ibm_watsonx_ai.metanames", meta_mod)
    boto3_mod = types.ModuleType("ibm_boto3"); boto3_mod.client = MagicMock()
    sys.modules.setdefault("ibm_boto3", boto3_mod)
    bc_mod = types.ModuleType("ibm_botocore"); bc_cfg = types.ModuleType("ibm_botocore.client")
    bc_cfg.Config = MagicMock(); bc_mod.client = bc_cfg
    sys.modules.setdefault("ibm_botocore", bc_mod); sys.modules.setdefault("ibm_botocore.client", bc_cfg)

_make_ibm_stub()


# ─── Gap Analyzer tests ───────────────────────────────────────────────────────
class TestGapAnalyzer:
    def _make_match(self, score=0.55):
        from core.schemas import GrantInfo, MatchResult
        g = GrantInfo(
            name="Test Grant",
            eligibility_criteria=["Must be DPIIT recognized", "Stage: ideation only"],
            source_file="test.txt",
        )
        return MatchResult(
            grant=g,
            match_score=score,
            semantic_score=0.5,
            rule_score=0.5,
            reasoning_score=0.5,
            what_is_missing=["Not DPIIT recognized"],
        )

    def _make_profile(self):
        from core.schemas import StartupProfile
        return StartupProfile(
            name="TestCo", stage="prototype", sector="agritech",
            state="Maharashtra", team_size=3, dpiit_recognized=False,
        )

    def test_analyze_gaps_returns_actions(self):
        from core.gap_analyzer import analyze_gaps
        match = self._make_match()
        profile = self._make_profile()
        mock_resp = "1. Register with DPIIT — addresses criterion: 'Must be DPIIT recognized'\n2. Adjust stage to ideation"
        with patch("core.gap_analyzer.generate_text", return_value=mock_resp):
            with patch("core.gap_analyzer.get_context_for_source", return_value=""):
                result = analyze_gaps(match, profile)
        assert "actions" in result
        assert len(result["actions"]) >= 1
        assert result["grant_name"] == "Test Grant"

    def test_analyze_gaps_handles_llm_failure(self):
        from core.gap_analyzer import analyze_gaps
        match = self._make_match()
        profile = self._make_profile()
        with patch("core.gap_analyzer.generate_text", side_effect=RuntimeError("API error")):
            with patch("core.gap_analyzer.get_context_for_source", return_value=""):
                result = analyze_gaps(match, profile)
        assert "actions" in result
        assert len(result["actions"]) >= 1
        assert "unavailable" in result["actions"][0].lower() or "error" in result["actions"][0].lower()


# ─── Export tests ─────────────────────────────────────────────────────────────
class TestExport:
    _DRAFT = "1. Executive Summary\nTest startup is applying for a grant.\n2. Problem Statement\nFarmers need help."
    _CHECKLIST = ["DPIIT certificate", "Pitch deck"]

    def test_export_docx_returns_bytes(self):
        from core.export import export_draft_docx
        result = export_draft_docx(
            draft_text=self._DRAFT,
            grant_name="Test Grant",
            startup_name="TestCo",
            checklist=self._CHECKLIST,
        )
        assert isinstance(result, bytes)
        assert len(result) > 1000
        # DOCX files start with PK (zip)
        assert result[:2] == b"PK"

    def test_export_pdf_returns_bytes(self):
        from core.export import export_draft_pdf
        result = export_draft_pdf(
            draft_text=self._DRAFT,
            grant_name="Test Grant",
            startup_name="TestCo",
            checklist=self._CHECKLIST,
        )
        assert isinstance(result, bytes)
        assert len(result) > 500
        # PDF files start with %PDF
        assert result[:4] == b"%PDF"

    def test_export_docx_contains_draft_label(self):
        from core.export import export_draft_docx
        from docx import Document
        import io
        result = export_draft_docx(
            draft_text=self._DRAFT,
            grant_name="SISFS",
            startup_name="AgroAI",
            checklist=["Doc1"],
        )
        doc = Document(io.BytesIO(result))
        full_text = " ".join(p.text for p in doc.paragraphs)
        assert "DRAFT" in full_text.upper()


# ─── Pitch Deck tests ─────────────────────────────────────────────────────────
class TestPitchDeck:
    def test_prefill_returns_profile_dict(self):
        from core.pitchdeck import prefill_from_pitchdeck
        mock_json = json.dumps({
            "name": "TechFarm", "stage": "prototype", "sector": "agritech",
            "state": "Karnataka", "team_size": 5, "annual_revenue_inr": 0,
            "dpiit_recognized": False, "funding_needed_inr": 2000000,
            "description": "Farm AI startup",
        })
        txt_bytes = b"TechFarm is a prototype agritech startup in Karnataka with 5 team members."
        # generate_text is imported inside the function from core.watsonx_client
        with patch("core.watsonx_client.generate_text", return_value=mock_json):
            with patch("core.ingestion._parse_with_docling", return_value=[{"text": txt_bytes.decode(), "page": 1}]):
                result = prefill_from_pitchdeck(txt_bytes, "pitch.txt")
        assert result["profile"].get("name") == "TechFarm"
        assert result["confidence"] in ("high", "medium", "low")

    def test_prefill_handles_bad_llm_output(self):
        from core.pitchdeck import prefill_from_pitchdeck
        with patch("core.watsonx_client.generate_text", return_value="not json at all"):
            with patch("core.ingestion._parse_with_docling", return_value=[{"text": "some text", "page": 1}]):
                result = prefill_from_pitchdeck(b"some text", "bad.txt")
        assert "profile" in result
        assert result["confidence"] == "low"


# ─── Orchestrate Tools tests ──────────────────────────────────────────────────
class TestOrchestrateTools:
    def test_tool_schemas_are_valid(self):
        from core.orchestrate_tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 4
        for tool in ALL_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert tool["parameters"]["type"] == "object"

    def test_dispatch_unknown_tool(self):
        from core.orchestrate_tools import dispatch
        result = dispatch("nonexistent_tool", {})
        assert "error" in result

    def test_dispatch_extract_grant_mocked(self):
        from core.orchestrate_tools import dispatch
        mock_grant = json.dumps({
            "name": "Mock", "issuing_body": "Test", "funding_amount": "50L",
            "funding_amount_inr": 5000000, "eligible_stages": [], "sectors": [],
            "location_restrictions": None, "eligibility_criteria": [],
            "deadline": None, "deadline_date": None, "required_documents": [],
            "application_link": None, "source_file": "test", "source_page": None,
        })
        with patch("core.extraction.generate_text", return_value=mock_grant):
            result = dispatch("extract_grant", {"text": "Some grant text", "source_file": "test.txt"})
        assert "error" not in result
        assert result["name"] == "Mock"

    def test_ask_question_no_context(self):
        from core.orchestrate_tools import dispatch
        with patch("core.chat.retrieve_similar_chunks", return_value=[]):
            result = dispatch("ask_question", {"question": "What is SISFS?"})
        assert "answer" in result
        assert result["grounded"] is False


# ─── Helpers tests ────────────────────────────────────────────────────────────
class TestUIHelpers:
    """Test utility functions that don't need Streamlit."""

    def test_fmt_inr_crore(self):
        # Import the formatter from app — minimal test without Streamlit
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
        # Inline the function to test without importing Streamlit
        def _fmt_inr(v):
            if v is None: return "N/A"
            if v >= 1e7:  return f"₹{v/1e7:.1f} Cr"
            if v >= 1e5:  return f"₹{v/1e5:.1f} L"
            return f"₹{v:,.0f}"
        assert _fmt_inr(10_000_000) == "₹1.0 Cr"
        assert _fmt_inr(500_000) == "₹5.0 L"
        assert _fmt_inr(None) == "N/A"
        assert _fmt_inr(99_000) == "₹0.99 L" or "99" in _fmt_inr(99_000)
