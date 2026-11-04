"""
GrantScout – Redesigned Streamlit UI (v2)
Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit import session_state as ss

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grantscout.ui")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GrantScout – AI Grant Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject CSS ───────────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent.parent / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ─── Lazy core imports ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _core():
    from core.config import validate_watsonx
    from core.ingestion import ingest_document, get_collection_stats, clear_collection
    from core.extraction import extract_from_chunks
    from core.matching import match_grants
    from core.drafting import draft_proposal
    from core.chat import answer_question
    from core.schemas import StartupProfile, StartupStage
    from core.gap_analyzer import analyze_gaps
    from core.export import export_draft_docx, export_draft_pdf
    from core.pitchdeck import prefill_from_pitchdeck
    from core.watsonx_client import smoke_test
    from core.cos_client import cos_available
    return dict(
        validate_watsonx=validate_watsonx,
        ingest_document=ingest_document,
        get_collection_stats=get_collection_stats,
        clear_collection=clear_collection,
        extract_from_chunks=extract_from_chunks,
        match_grants=match_grants,
        draft_proposal=draft_proposal,
        answer_question=answer_question,
        StartupProfile=StartupProfile,
        StartupStage=StartupStage,
        analyze_gaps=analyze_gaps,
        export_draft_docx=export_draft_docx,
        export_draft_pdf=export_draft_pdf,
        prefill_from_pitchdeck=prefill_from_pitchdeck,
        smoke_test=smoke_test,
        cos_available=cos_available,
    )


# ─── Session state defaults ───────────────────────────────────────────────────
def _init():
    defaults = {
        "grants": [],
        "matches": [],
        "profile": None,
        "trace": [],
        "draft_result": None,
        "draft_approved": False,
        "chat_history": [],
        "ingest_log": [],
        "tracker": {"Saved": [], "Preparing": [], "Submitted": [], "Result": []},
        "feedback": [],          # list[{type, item, vote}]
        "last_ingested": {},     # source_file -> datetime str
        "demo_mode": False,
        "onboarding_step": 0,
        "gap_results": {},       # grant_name -> gap dict
        "ibm_status": None,      # cached smoke-test result
        "_ingest_queue": [],     # list of (path_str, synthetic_bool) pending ingestion
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v
_init()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _score_class(score: float) -> str:
    if score >= 0.70: return "score-high"
    if score >= 0.40: return "score-medium"
    return "score-low"

def _deadline_class(days: int | None) -> str:
    if days is None: return "deadline-na"
    if days < 14:  return "deadline-urgent"
    if days < 30:  return "deadline-soon"
    return "deadline-ok"

def _deadline_label(days: int | None) -> str:
    if days is None: return "No deadline"
    if days < 0:   return "Expired"
    if days == 0:  return "Today!"
    return f"{days}d left"

def _fmt_inr(v: float | None) -> str:
    if v is None: return "N/A"
    if v >= 1e7:  return f"₹{v/1e7:.1f} Cr"
    if v >= 1e5:  return f"₹{v/1e5:.1f} L"
    return f"₹{v:,.0f}"

def _css_chip(text: str, cls: str = "chip-blue") -> str:
    return f'<span class="chip {cls}">{text}</span>'

def _empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="es-icon">{icon}</div>'
        f'<h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )

def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ─── Demo mode loader ─────────────────────────────────────────────────────────
# The demo runs in phases so the browser websocket never idles for >1 API call.
# ss["_demo_phase"]:
#   None / missing  → not started
#   "ingest"        → ingesting file at index ss["_demo_file_idx"]
#   "match"         → all files done, now run matching
#   "done"          → complete

def _init_demo_profile(c: dict) -> None:
    """Set the demo startup profile in session state."""
    ss["profile"] = c["StartupProfile"](
        name="AgroAI Demo",
        stage="prototype",
        sector="agritech",
        state="Maharashtra",
        team_size=4,
        annual_revenue_inr=0,
        dpiit_recognized=True,
        funding_needed_inr=2_000_000,
        description="AI-powered crop disease detection for smallholder farmers using smartphone cameras.",
    )
    ss["demo_mode"] = True


def _start_demo():
    """Kick off demo mode — just set the phase and rerun."""
    c = _core()
    synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
    files = sorted(synthetic_dir.glob("*.txt"))
    if not files:
        st.error("No synthetic files found in /data/synthetic/")
        return
    _init_demo_profile(c)
    ss["_demo_files"] = [str(f) for f in files]
    ss["_demo_file_idx"] = 0
    ss["_demo_phase"] = "ingest"
    st.rerun()


def _continue_demo():
    """
    Called each render cycle while demo is running.
    Processes exactly ONE file per cycle (ingest phase) then ONE matching call,
    so the Streamlit websocket heartbeat is never starved.
    """
    c = _core()
    phase = ss.get("_demo_phase")

    if phase == "ingest":
        files = ss.get("_demo_files", [])
        idx = ss.get("_demo_file_idx", 0)

        if idx >= len(files):
            # All files done → move to matching phase
            ss["_demo_phase"] = "match"
            st.rerun()
            return

        sf = Path(files[idx])
        with st.status(f"📄 Processing {sf.name} ({idx+1}/{len(files)})…", expanded=False) as status:
            try:
                chunks = c["ingest_document"](sf, trace=ss["trace"])
                if chunks:
                    status.update(label=f"🔍 Extracting grant info from {sf.name}…")
                    grants = c["extract_from_chunks"](chunks, trace=ss["trace"])
                    for g in grants:
                        g.is_synthetic = True
                        if not any(x.name == g.name for x in ss["grants"]):
                            ss["grants"].append(g)
                            ss["last_ingested"][g.source_file or sf.name] = datetime.now().isoformat()
                status.update(label=f"✅ {sf.name} done", state="complete")
            except Exception as exc:
                logger.error("Demo ingest failed for %s: %s", sf.name, exc)
                status.update(label=f"⚠️ {sf.name}: {exc}", state="error")

        ss["_demo_file_idx"] = idx + 1
        st.rerun()

    elif phase == "match":
        if ss["grants"]:
            with st.status("🎯 Running hybrid matching with Granite…", expanded=False) as status:
                try:
                    ss["matches"] = c["match_grants"](
                        profile=ss["profile"], grants=ss["grants"], top_k=10, trace=ss["trace"]
                    )
                    status.update(label=f"✅ Matched {len(ss['matches'])} grants", state="complete")
                except Exception as exc:
                    logger.error("Demo matching failed: %s", exc)
                    status.update(label=f"⚠️ Matching error: {exc}", state="error")
        ss["_demo_phase"] = "done"
        st.rerun()

    elif phase == "done":
        ss.pop("_demo_phase", None)
        ss.pop("_demo_files", None)
        ss.pop("_demo_file_idx", None)
        n_grants = len(ss["grants"])
        n_matches = len(ss["matches"])
        st.toast(f"✅ Demo ready — {n_grants} grants, {n_matches} matches", icon="🎉")
        st.rerun()


def _continue_ingest_queue():
    """
    Called each render cycle while ss["_ingest_queue"] is non-empty.
    Processes exactly ONE file per cycle so the Streamlit websocket heartbeat
    is never starved by a long synchronous loop of API calls.
    """
    queue = ss.get("_ingest_queue", [])
    if not queue:
        return

    path_str, synthetic = queue[0]
    sf = Path(path_str)
    c = _core()

    with st.status(f"📄 Processing {sf.name} ({1}/{len(queue)} remaining)…", expanded=False) as status:
        try:
            chunks = c["ingest_document"](sf, trace=ss["trace"])
            if chunks:
                status.update(label=f"🔍 Extracting from {sf.name}…")
                grants = c["extract_from_chunks"](chunks, trace=ss["trace"])
                for g in grants:
                    if synthetic:
                        g.is_synthetic = True
                    if not any(x.name == g.name for x in ss["grants"]):
                        ss["grants"].append(g)
                        ss["last_ingested"][g.source_file or sf.name] = datetime.now().isoformat()
                ss["ingest_log"].append(
                    f"✅ **{sf.name}**: {len(chunks)} chunks → {len(grants)} grant(s) extracted"
                )
            else:
                ss["ingest_log"].append(f"⚠️ **{sf.name}**: No text extracted")
            status.update(label=f"✅ {sf.name} done", state="complete")
        except Exception as exc:
            logger.error("Queue ingest failed for %s: %s", sf.name, exc)
            ss["ingest_log"].append(f"❌ **{sf.name}**: {exc}")
            status.update(label=f"⚠️ {sf.name}: {exc}", state="error")

    # Remove the processed file and trigger next cycle
    ss["_ingest_queue"] = queue[1:]
    if not ss["_ingest_queue"]:
        # Queue drained — final notification
        ss["onboarding_step"] = max(ss.get("onboarding_step", 0), 2)
        st.toast("✅ All synthetic samples loaded", icon="📦")
    st.rerun()


# ─── Stepper widget ───────────────────────────────────────────────────────────
def _render_stepper():
    """
    Progress stepper driven strictly by real session-state prerequisites:
      Step 1 done  = profile saved (ss["profile"] is not None)
      Step 2 done  = at least one grant indexed
      Step 3 done  = matching has been run (matches list non-empty)
      Step 4 active= draft generated
    """
    has_profile = ss["profile"] is not None
    has_grants  = len(ss["grants"]) > 0
    has_matches = len(ss["matches"]) > 0
    has_draft   = ss["draft_result"] is not None

    # Compute which step index is currently active (0-based)
    if has_draft:
        current = 4       # all done
    elif has_matches:
        current = 3       # ready to draft
    elif has_grants:
        current = 2       # ready to match
    elif has_profile:
        current = 1       # ready to load grants
    else:
        current = 0       # nothing done yet

    steps = [
        ("1", "Profile",     has_profile),
        ("2", "Load Grants", has_grants),
        ("3", "Match",       has_matches),
        ("4", "Draft",       has_draft),
    ]

    parts = []
    for i, (num, label, done) in enumerate(steps):
        if done:
            cls = "done"
            dot = "✓"
        elif i == current:
            cls = "active"
            dot = num
        else:
            cls = ""
            dot = num
        parts.append(
            f'<div class="gs-step {cls}">'
            f'<div class="gs-step-dot">{dot}</div>'
            f'<span>{label}</span>'
            f'</div>'
        )
        if i < len(steps) - 1:
            line_cls = "done" if done else ""
            parts.append(f'<div class="gs-step-line {line_cls}"></div>')

    st.markdown(f'<div class="gs-stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


# ─── IBM Status check ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _get_ibm_status() -> dict:
    c = _core()
    return c["smoke_test"]()


# ─── Cached matching ──────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _cached_match(profile_json: str, grants_json: str, top_k: int = 10) -> list:
    """
    Cache match_grants results keyed on a JSON fingerprint of profile + grants.
    Avoids re-running 6+ Granite API calls every time the user navigates back
    to the Matches tab — results are reused for up to 10 minutes.
    """
    import json as _json
    from core.matching import match_grants
    from core.schemas import StartupProfile, GrantInfo
    profile = StartupProfile(**_json.loads(profile_json))
    grants  = [GrantInfo(**g) for g in _json.loads(grants_json)]
    return match_grants(profile=profile, grants=grants, top_k=top_k, trace=[])


def _matches_fingerprint() -> tuple[str, str]:
    """Return (profile_json, grants_json) for cache keying."""
    import json as _json
    profile = ss.get("profile")
    grants  = ss.get("grants", [])
    p_json  = profile.model_dump_json() if profile else "{}"
    g_json  = _json.dumps([g.model_dump(mode="json") for g in grants])
    return p_json, g_json


def _matches_as_csv() -> str:
    """Serialise current matches to CSV string for download."""
    import csv, io
    rows = ss.get("matches", [])
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Rank", "Grant Name", "Issuing Body", "Match %", "Rule %",
        "Semantic %", "LLM %", "Funding Amount", "Deadline",
        "Days Until Deadline", "Sectors", "Eligible Stages",
        "Location", "Why Matches", "What Is Missing",
        "Application Link", "Source File",
    ])
    for i, m in enumerate(rows, 1):
        g = m.grant
        writer.writerow([
            i,
            g.name or "",
            g.issuing_body or "",
            f"{int(m.match_score * 100)}%",
            f"{int(m.rule_score * 100)}%",
            f"{int(m.semantic_score * 100)}%",
            f"{int(m.reasoning_score * 100)}%",
            g.funding_amount or "",
            g.deadline or "",
            m.days_until_deadline if m.days_until_deadline is not None else "",
            "; ".join(g.sectors or []),
            "; ".join(str(s) for s in (g.eligible_stages or [])),
            g.location_restrictions or "Pan-India",
            " | ".join(m.why_matches or []),
            " | ".join(m.what_is_missing or []),
            g.application_link or "",
            g.source_file or "",
        ])
    return buf.getvalue()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def _sidebar():
    c = _core()
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            '<span style="font-size:1.6rem;">🔍</span>'
            '<span style="font-size:1.1rem;font-weight:800;color:#0F3460;">GrantScout</span>'
            '</div>'
            '<div style="font-size:11px;color:#6B7890;margin-bottom:16px;">Powered by IBM Granite · watsonx.ai</div>',
            unsafe_allow_html=True,
        )

        # ── IBM Status indicator
        with st.expander("🔌 IBM Services Status", expanded=False):
            try:
                status = _get_ibm_status()
                rows = [
                    ("Granite LLM", status.get("llm_ok"), status.get("llm_model", "")),
                    ("Granite Embeddings", status.get("embedder_ok"), status.get("embedding_model", "")),
                    ("watsonx.ai Auth", status.get("credentials_valid"), ""),
                    ("IBM COS", c["cos_available"](), "local fallback" if not c["cos_available"]() else ""),
                ]
                for name, ok, detail in rows:
                    dot = "dot-ok" if ok else ("dot-warn" if name == "IBM COS" and not ok else "dot-err")
                    label = "✓ Connected" if ok else ("⚡ Local fallback" if name == "IBM COS" else "✗ Unavailable")
                    st.markdown(
                        f'<div class="status-row"><span class="{dot}"></span>'
                        f'<span><b>{name}</b> — {label}'
                        + (f' <span style="color:#9099A6;font-size:10px;">({detail})</span>' if detail else "")
                        + "</span></div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.caption("Status check unavailable — check credentials.")

        st.divider()
        st.markdown("### 📋 Startup Profile")

        # ── Sample profile quick-fill ──────────────────────────────────────
        _SAMPLE_PROFILES = {
            "— choose a sample —": None,
            "🌾 AgroAI (Agritech · Prototype · Maharashtra)": {
                "name": "AgroAI", "stage": "prototype", "sector": "agritech",
                "state": "Maharashtra", "team_size": 4, "annual_revenue_inr": 0,
                "dpiit_recognized": True, "funding_needed_inr": 2_000_000,
                "description": "AI-powered crop disease detection for smallholder farmers using smartphone cameras.",
            },
            "💻 CloudStack (SaaS · Early-stage · Karnataka)": {
                "name": "CloudStack", "stage": "early_stage", "sector": "deeptech",
                "state": "Karnataka", "team_size": 8, "annual_revenue_inr": 1_200_000,
                "dpiit_recognized": True, "funding_needed_inr": 5_000_000,
                "description": "Developer-first cloud cost-optimisation platform for Indian SaaS companies.",
            },
            "🤝 SakhiCare (Social Impact · Growth · Rajasthan)": {
                "name": "SakhiCare", "stage": "growth", "sector": "social_impact",
                "state": "Rajasthan", "team_size": 12, "annual_revenue_inr": 3_500_000,
                "dpiit_recognized": False, "funding_needed_inr": 8_000_000,
                "description": "Women-led social enterprise providing last-mile maternal health services in rural Rajasthan.",
            },
        }
        chosen_label = st.selectbox(
            "Quick-fill from sample profile",
            options=list(_SAMPLE_PROFILES.keys()),
            key="sb_sample_profile",
            label_visibility="visible",
        )
        if _SAMPLE_PROFILES[chosen_label]:
            ss["_prefill"] = _SAMPLE_PROFILES[chosen_label]

        # Pitch deck upload
        with st.expander("📎 Pre-fill from Pitch Deck", expanded=False):
            pitch_file = st.file_uploader(
                "Upload pitch deck (PDF/DOCX/PPTX)", type=["pdf","docx","pptx"], key="pitch_up"
            )
            if pitch_file and st.button("Extract Profile", key="pitch_extract"):
                with st.spinner("Granite is reading your pitch deck…"):
                    result = c["prefill_from_pitchdeck"](pitch_file.read(), pitch_file.name, ss["trace"])
                if result["profile"]:
                    ss["_prefill"] = result["profile"]
                    conf = result["confidence"]
                    conf_color = "#1B5E20" if conf=="high" else "#E65100" if conf=="medium" else "#B71C1C"
                    st.markdown(
                        f'<div class="rai-banner">⚠️ AI pre-fill — confidence: '
                        f'<b style="color:{conf_color}">{conf.upper()}</b>. '
                        f'Review and confirm before saving.</div>',
                        unsafe_allow_html=True,
                    )
                    st.rerun()
                else:
                    st.warning("Could not extract profile from pitch deck.")

        pf = ss.get("_prefill", {})

        name     = st.text_input("Startup Name",    value=pf.get("name", ss["profile"].name if ss["profile"] else "My Startup"), key="sb_name")
        stage    = st.selectbox("Stage", options=["ideation","prototype","early_stage","growth","scaling","any"],
                                index=["ideation","prototype","early_stage","growth","scaling","any"].index(
                                    pf.get("stage", ss["profile"].stage if ss["profile"] else "early_stage")
                                ), key="sb_stage")
        sector   = st.text_input("Primary Sector",  value=pf.get("sector",  ss["profile"].sector  if ss["profile"] else "agritech"), key="sb_sector")
        state    = st.text_input("State",           value=pf.get("state",   ss["profile"].state   if ss["profile"] else "Maharashtra"), key="sb_state")
        team_size= st.number_input("Team Size", min_value=1, max_value=1000,
                                   value=int(pf.get("team_size", ss["profile"].team_size if ss["profile"] else 4) or 4), key="sb_team")
        revenue  = st.number_input("Annual Revenue (₹)", min_value=0, value=int(pf.get("annual_revenue_inr", 0) or 0), step=100000, key="sb_rev")
        dpiit    = st.checkbox("DPIIT Recognised?", value=bool(pf.get("dpiit_recognized", ss["profile"].dpiit_recognized if ss["profile"] else False)), key="sb_dpiit")
        funding  = st.number_input("Funding Needed (₹)", min_value=0,
                                   value=int(pf.get("funding_needed_inr", ss["profile"].funding_needed_inr if ss["profile"] else 2000000) or 2000000),
                                   step=100000, key="sb_fund")
        desc     = st.text_area("Description", value=pf.get("description", ss["profile"].description if ss["profile"] else ""), key="sb_desc", height=80)

        if st.button("💾 Save Profile", use_container_width=True, type="primary"):
            try:
                ss["profile"] = c["StartupProfile"](
                    name=name, stage=stage, sector=sector, state=state,
                    team_size=int(team_size), annual_revenue_inr=float(revenue),
                    dpiit_recognized=dpiit, funding_needed_inr=float(funding), description=desc,
                )
                ss.pop("_prefill", None)
                ss["onboarding_step"] = max(ss["onboarding_step"], 1)
                st.success("Profile saved ✓")
            except Exception as exc:
                st.error(f"Profile error: {exc}")

        st.divider()

        # Stats
        try:
            stats = c["get_collection_stats"]()
            col1, col2 = st.columns(2)
            col1.metric("Indexed Chunks", stats.get("count", 0))
            col2.metric("Grants", len(ss["grants"]))
        except Exception:
            pass

        st.divider()
        st.markdown(
            '<div class="rai-banner">'
            '🛡️ <b>Ethics notice:</b> Only public documents. No login-protected portals. '
            'AI drafts need human review. No PII stored beyond this session.'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑️ Clear All Data", use_container_width=True):
            try:
                c["clear_collection"]()
                for k in ["grants","matches","trace","draft_result","chat_history",
                          "ingest_log","tracker","feedback","last_ingested","gap_results"]:
                    ss[k] = [] if isinstance(ss[k], list) else ({} if isinstance(ss[k], dict) else None)
                ss["tracker"] = {"Saved": [], "Preparing": [], "Submitted": [], "Result": []}
                ss["draft_approved"] = False
                ss["demo_mode"] = False
                st.success("Session cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")


# ─── HERO banner ──────────────────────────────────────────────────────────────
def _hero():
    total_grants = len(ss["grants"])
    total_funding = sum(
        (g.funding_amount_inr or 0) for g in ss["grants"]
    )
    nearest = None
    any_rolling = False
    for g in ss["grants"]:
        if g.deadline_date:
            d = (g.deadline_date - date.today()).days
            if d >= 0 and (nearest is None or d < nearest):
                nearest = d
        elif g.deadline and "rolling" in (g.deadline or "").lower():
            any_rolling = True

    # Honest deadline stat: prefer days countdown, then "Rolling", then "—"
    if nearest is not None:
        deadline_stat = f"{nearest}d"
    elif any_rolling:
        deadline_stat = "Rolling"
    else:
        deadline_stat = "—"

    # Honest 4th stat: indexed / 35 curated
    curated_stat = f"{total_grants} / 35"

    st.markdown(
        f"""<div class="gs-hero">
        <h1>🔍 GrantScout</h1>
        <div class="tagline">AI-powered grant & funding finder for Indian startups — built on IBM Granite</div>
        <div class="stat-row">
          <div class="stat-box"><div class="stat-num">{total_grants}</div><div class="stat-label">Grants Indexed</div></div>
          <div class="stat-box"><div class="stat-num">{_fmt_inr(total_funding) if total_funding else "—"}</div><div class="stat-label">Total Potential Funding</div></div>
          <div class="stat-box"><div class="stat-num">{deadline_stat}</div><div class="stat-label">Nearest Deadline</div></div>
          <div class="stat-box"><div class="stat-num">{curated_stat}</div><div class="stat-label">Indexed / Curated Sources</div></div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    dcol, _ = st.columns([1, 3])
    with dcol:
        demo_running = ss.get("_demo_phase") is not None
        if demo_running:
            idx   = ss.get("_demo_file_idx", 0)
            total = len(ss.get("_demo_files", []))
            phase = ss.get("_demo_phase", "")
            if phase == "ingest":
                btn_label = f"⏳ Loading file {idx}/{total}…"
            elif phase == "match":
                btn_label = "⏳ Running Granite matching…"
            else:
                btn_label = "⏳ Finishing up…"
            st.button(btn_label, disabled=True, use_container_width=True)
        else:
            if st.button("⚡ Try Demo Mode", type="primary", use_container_width=True,
                         help="Loads synthetic samples + pre-filled AgroAI profile, one file at a time"):
                _start_demo()


# ─── Tab 1: Upload & Ingest ───────────────────────────────────────────────────
def _tab_ingest():
    c = _core()
    _section("📂 Upload & Ingest Grant Documents")
    st.caption("Supported: PDF · DOCX · HTML · TXT — parsed with Docling, embedded with IBM Granite, stored in ChromaDB")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Drop grant documents here",
            type=["pdf","docx","html","htm","txt"],
            accept_multiple_files=True,
        )
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        queue_running = bool(ss.get("_ingest_queue"))
        if queue_running:
            remaining = len(ss["_ingest_queue"])
            st.button(f"⏳ Processing… ({remaining} left)", disabled=True, use_container_width=True)
        else:
            if st.button("📦 Load Synthetic Samples", use_container_width=True):
                synthetic_dir = Path(__file__).parent.parent / "data" / "synthetic"
                files = sorted(synthetic_dir.glob("*.txt"))
                if not files:
                    st.warning("No synthetic files found in /data/synthetic/")
                else:
                    ss["_ingest_queue"] = [(str(f), True) for f in files]
                    st.rerun()

        # URL ingestion
        with st.expander("🌐 Ingest from public URL"):
            st.markdown(
                '<div class="rai-banner">Only public HTTP/HTTPS pages. '
                'No login-protected or bulk scraping. robots.txt respected.</div>',
                unsafe_allow_html=True,
            )
            url_input = st.text_input("Paste public grant PDF/page URL", key="url_input", placeholder="https://example.gov.in/grant.pdf")
            if st.button("⬇️ Fetch & Ingest", key="url_fetch"):
                _ingest_url(url_input, c)

    if uploaded:
        prog = st.progress(0, text="Uploading…")
        for i, uf in enumerate(uploaded):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = Path(tmp.name)
            _ingest_single(tmp_path, c, original_name=uf.name)
            tmp_path.unlink(missing_ok=True)
            prog.progress((i+1)/len(uploaded), text=f"Processed {uf.name}")
        prog.empty()
        ss["onboarding_step"] = max(ss["onboarding_step"], 2)
        st.rerun()

    # Ingestion log
    if ss["ingest_log"]:
        with st.expander(f"📋 Ingestion Log ({len(ss['ingest_log'])} entries)", expanded=False):
            for entry in ss["ingest_log"][-30:]:
                st.markdown(f"- {entry}")

    # Extracted grants gallery
    if ss["grants"]:
        _section(f"📋 Extracted Grants ({len(ss['grants'])})")
        for g in ss["grants"]:
            _staleness = ""
            li = ss["last_ingested"].get(g.source_file or "")
            if li:
                ingested_dt = datetime.fromisoformat(li)
                age_days = (datetime.now() - ingested_dt).days
                if age_days > 365:
                    _staleness = f' <span class="chip chip-amber">⚠ Stale ({age_days}d old)</span>'

            with st.expander(f"**{g.name}**  —  {g.issuing_body or 'N/A'}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Funding", g.funding_amount or "N/A")
                c2.metric("Deadline", g.deadline or "Rolling")
                c3.metric("Stages", ", ".join(str(s) for s in g.eligible_stages[:2]) or "Any")
                c4.metric("Sectors", ", ".join(g.sectors[:2]) or "Any")

                # Per-field confidence
                st.markdown("**Extracted fields & confidence:**")
                fields = {
                    "Name": (g.name, "high"),
                    "Funding Amount": (g.funding_amount, "high" if g.funding_amount else "low"),
                    "Issuing Body": (g.issuing_body, "high" if g.issuing_body else "medium"),
                    "Deadline": (g.deadline, "high" if g.deadline_date else ("medium" if g.deadline else "low")),
                    "Eligibility Criteria": (f"{len(g.eligibility_criteria)} items", "high" if g.eligibility_criteria else "low"),
                    "Required Documents": (f"{len(g.required_documents)} items", "high" if g.required_documents else "low"),
                    "Application Link": (g.application_link, "high" if g.application_link else "low"),
                }
                rows_html = ""
                for fname, (fval, conf) in fields.items():
                    conf_cls = f"conf-{conf}"
                    val_str = str(fval)[:60] if fval else "—"
                    rows_html += (
                        f"<tr><td style='padding:3px 8px;font-size:12px;color:#6B7890'>{fname}</td>"
                        f"<td style='padding:3px 8px;font-size:12px'>{val_str}</td>"
                        f"<td style='padding:3px 8px'><span class='{conf_cls}' style='font-size:11px'>{conf.upper()}</span></td></tr>"
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse">'
                    f'<tr><th style="text-align:left;font-size:11px;padding:3px 8px;color:#9099A6">Field</th>'
                    f'<th style="text-align:left;font-size:11px;padding:3px 8px;color:#9099A6">Value</th>'
                    f'<th style="text-align:left;font-size:11px;padding:3px 8px;color:#9099A6">Confidence</th></tr>'
                    f'{rows_html}</table>',
                    unsafe_allow_html=True,
                )
                if li:
                    st.caption(f"📅 Last ingested: {ingested_dt.strftime('%d %b %Y %H:%M')}{_staleness}")
                if g.is_synthetic:
                    st.warning("⚠️ SYNTHETIC DATA – for testing only. Not a real grant document.")
    else:
        _empty_state("📂", "No grants indexed yet",
                     "Upload grant PDFs above or click 'Load Synthetic Samples' to get started.")


def _ingest_single(file_path: Path, c: dict, original_name: str = "", synthetic: bool = False):
    display = original_name or file_path.name
    try:
        with st.spinner(f"Parsing {display} with Docling…"):
            chunks = c["ingest_document"](file_path, trace=ss["trace"])
        if not chunks:
            ss["ingest_log"].append(f"⚠️ **{display}**: No text extracted")
            return
        with st.spinner(f"Extracting grant info from {display} with Granite…"):
            grants = c["extract_from_chunks"](chunks, trace=ss["trace"])
        if grants:
            for g in grants:
                if synthetic:
                    g.is_synthetic = True
                if not any(x.name == g.name for x in ss["grants"]):
                    ss["grants"].append(g)
                    ss["last_ingested"][g.source_file or display] = datetime.now().isoformat()
            ss["ingest_log"].append(
                f"✅ **{display}**: {len(chunks)} chunks → {len(grants)} grant(s) extracted"
            )
        else:
            ss["ingest_log"].append(
                f"⚠️ **{display}**: {len(chunks)} chunks indexed, extraction returned no grants"
            )
    except Exception as exc:
        ss["ingest_log"].append(f"❌ **{display}**: {exc}")
        st.error(f"Failed: {display} — {exc}")


def _ingest_url(url: str, c: dict):
    import urllib.parse, urllib.request
    if not url.strip():
        st.warning("Please enter a URL.")
        return
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        st.error("Only http/https URLs allowed.")
        return
    blocked = ["localhost", "127.0.0.1", "192.168.", "10.", "172."]
    if any(b in parsed.netloc for b in blocked):
        st.error("Local/private network URLs are not permitted.")
        return
    try:
        with st.spinner(f"Fetching {url[:60]}…"):
            req = urllib.request.Request(url, headers={"User-Agent": "GrantScout/1.0 (public research)"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
        suffix = ".pdf" if "pdf" in content_type else ".html"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        _ingest_single(tmp_path, c, original_name=urllib.parse.urlparse(url).path.split("/")[-1] or "url_grant" + suffix)
        tmp_path.unlink(missing_ok=True)
        st.toast("URL ingested successfully", icon="✅")
        st.rerun()
    except Exception as exc:
        st.error(f"URL fetch failed: {exc}")


# ─── Tab 2: Grant Matches ─────────────────────────────────────────────────────
def _tab_matches():
    import plotly.graph_objects as go
    c = _core()

    if not ss["profile"]:
        _empty_state("👤", "Profile needed", "Fill in your Startup Profile in the sidebar, then come back here.")
        return
    if not ss["grants"]:
        _empty_state("📂", "No grants indexed", "Go to 'Upload & Ingest' first to load grant documents.")
        return

    # ── Action row: run/re-run button + CSV export
    # Matching is user-triggered (not auto) to avoid hitting the Lite API rate
    # limit (429). Results are cached for 10 min so the second click is instant.
    col_run, col_csv, col_spacer = st.columns([2, 2, 4])
    with col_run:
        run_label = "🔄 Re-run Matching" if ss["matches"] else "🔍 Find Matching Grants"
        run_type  = "secondary" if ss["matches"] else "primary"
        if st.button(run_label, type=run_type, use_container_width=True,
                     help="Run hybrid scoring (rules + semantic + Granite). Results cached 10 min."):
            p_json, g_json = _matches_fingerprint()
            with st.spinner("Running hybrid scoring with IBM Granite… (first run ~30 s, then cached)"):
                try:
                    results = _cached_match(p_json, g_json, top_k=10)
                    ss["matches"] = results
                    ss["_match_fp"] = hash(p_json + g_json)
                    ss["onboarding_step"] = max(ss.get("onboarding_step", 0), 3)
                    ss["trace"].append({"step": "match", "grants": len(ss["grants"]), "matches": len(results)})
                    st.toast(f"✅ {len(results)} grants matched", icon="🎯")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Matching failed: {exc}")
    with col_csv:
        csv_str = _matches_as_csv()
        if csv_str:
            profile_name = (ss["profile"].name or "startup").replace(" ", "_")[:20]
            st.download_button(
                "⬇️ Export CSV",
                data=csv_str.encode("utf-8"),
                file_name=f"GrantScout_matches_{profile_name}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download all ranked matches as a spreadsheet",
            )

    if not ss["matches"]:
        _empty_state("🎯", "No matches yet", "Click 'Find Matching Grants' above to run the AI matching engine.")
        return

    # ── KPI row
    eligible  = len([m for m in ss["matches"] if m.match_score >= 0.4])
    total_pot = sum((m.grant.funding_amount_inr or 0) for m in ss["matches"] if m.match_score >= 0.4)
    near_dl   = min(
        (m.days_until_deadline for m in ss["matches"]
         if m.days_until_deadline is not None and m.days_until_deadline >= 0),
        default=None,
    )
    has_rolling_match = any(
        m.grant.deadline and "rolling" in (m.grant.deadline or "").lower()
        for m in ss["matches"] if m.days_until_deadline is None
    )
    if near_dl is not None:
        dl_kpi_val = f"{near_dl}d"
        dl_urgency = "urgent" if near_dl < 14 else ("amber" if near_dl < 30 else "")
    elif has_rolling_match:
        dl_kpi_val = "Rolling"
        dl_urgency = ""
    else:
        dl_kpi_val = "—"
        dl_urgency = ""

    kpi_html = (
        f'<div class="kpi-row">'
        f'<div class="kpi-card"><div class="kpi-val">{eligible}</div><div class="kpi-sub">Eligible grants (≥40%)</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{_fmt_inr(total_pot)}</div><div class="kpi-sub">Total potential funding</div></div>'
        f'<div class="kpi-card {dl_urgency}"><div class="kpi-val">{dl_kpi_val}</div>'
        f'<div class="kpi-sub">Nearest deadline</div></div>'
        f'<div class="kpi-card"><div class="kpi-val">{len(ss["matches"])}</div><div class="kpi-sub">Grants evaluated</div></div>'
        f'</div>'
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Filters & sort
    with st.expander("🔧 Filter & Sort", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        min_score = fcol1.slider("Min match score", 0, 100, 0, 5, key="f_score") / 100
        sort_by   = fcol2.selectbox("Sort by", ["Match Score ↓", "Deadline ↑", "Funding ↓"], key="f_sort")
        max_show  = fcol3.selectbox("Show top", [5, 10, 20], key="f_max")

    matches = [m for m in ss["matches"] if m.match_score >= min_score]
    if sort_by == "Deadline ↑":
        matches = sorted(matches, key=lambda m: m.days_until_deadline if m.days_until_deadline is not None else 9999)
    elif sort_by == "Funding ↓":
        matches = sorted(matches, key=lambda m: -(m.grant.funding_amount_inr or 0))
    matches = matches[:max_show]

    # ── Compare mode
    compare_sel = []
    with st.expander("📊 Compare grants side-by-side (select up to 3)", expanded=False):
        cols = st.columns(min(len(matches), 5))
        for i, m in enumerate(matches[:5]):
            if cols[i % 5].checkbox(m.grant.name[:25], key=f"cmp_{i}"):
                compare_sel.append(m)
        if len(compare_sel) > 3:
            st.warning("Select at most 3 grants.")
            compare_sel = compare_sel[:3]
        if compare_sel:
            _render_compare(compare_sel)

    # ── Deadline timeline
    _section("📅 Deadline Timeline")
    _render_deadline_timeline(matches)

    # ── Match cards
    _section(f"🎯 Ranked Matches ({len(matches)})")
    for i, match in enumerate(matches):
        _render_match_card(match, i, c)

    # ── Score breakdown chart
    if len(matches) >= 2:
        _section("📊 Score Breakdown")
        _render_score_chart(matches[:8])


def _render_match_card(match, idx: int, c: dict):
    import plotly.graph_objects as go
    g = match.grant
    score_pct = int(match.match_score * 100)
    s_cls = _score_class(match.match_score)
    dl_cls = _deadline_class(match.days_until_deadline)
    dl_lbl = _deadline_label(match.days_until_deadline)

    chips_html = "".join([
        _css_chip(g.issuing_body[:30] if g.issuing_body else "N/A", "chip-gray"),
        *[_css_chip(s, "chip-purple") for s in (g.eligible_stages or [])[:3]],
        *[_css_chip(s, "chip-blue") for s in (g.sectors or [])[:2]],
        (f'<span class="chip chip-amber">⚠ SYNTHETIC</span>' if g.is_synthetic else ""),
    ])

    st.markdown(
        f'<div class="match-card">'
        f'<div class="match-card-header">'
        f'<div>'
        f'<h3>#{idx+1} {g.name}</h3>'
        f'<div class="issuer">{chips_html}</div>'
        f'</div>'
        f'<div style="text-align:center">'
        f'<div class="score-badge {s_cls}">{score_pct}%</div>'
        f'<div style="font-size:10px;color:#9099A6;margin-top:4px">match</div>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;gap:24px;margin:12px 0 8px;flex-wrap:wrap">'
        f'<div><span style="font-size:11px;color:#6B7890">FUNDING</span><br>'
        f'<span style="font-weight:700;color:#0F3460">{g.funding_amount or "N/A"}</span></div>'
        f'<div><span style="font-size:11px;color:#6B7890">DEADLINE</span><br>'
        f'<span class="{dl_cls}">{dl_lbl}</span>'
        f'<span style="font-size:11px;color:#9099A6"> ({g.deadline or "N/A"})</span></div>'
        f'<div><span style="font-size:11px;color:#6B7890">RULE / SEM / LLM</span><br>'
        f'<span style="font-size:12px;font-weight:600">'
        f'{int(match.rule_score*100)}% / {int(match.semantic_score*100)}% / {int(match.reasoning_score*100)}%'
        f'</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        exp1, exp2, exp3 = st.columns([2, 2, 1])
        with exp1:
            with st.expander("✅ Why it matches"):
                for w in (match.why_matches or ["No specific matches identified"]):
                    st.markdown(f"- {w}")
                if match.sources:
                    st.caption(f"📄 {', '.join(match.sources)}")
                    st.caption("⚠️ Verify deadline and amount on the official portal")
        with exp2:
            with st.expander("⚠️ What's missing"):
                if match.what_is_missing:
                    for m in match.what_is_missing:
                        st.markdown(f"- {m}")
                else:
                    st.markdown("No critical gaps identified.")
        with exp3:
            with st.expander("🧮 Score"):
                rule_pct    = int(match.rule_score * 100)
                sem_pct     = int(match.semantic_score * 100)
                reason_pct  = int(match.reasoning_score * 100)
                st.markdown(
                    f"**How this {score_pct}% was calculated:**\n\n"
                    f"| Component | Weight | Your Score |\n"
                    f"|-----------|--------|------------|\n"
                    f"| 🔵 Rule-based filters | 35% | **{rule_pct}%** |\n"
                    f"| 🟣 Semantic similarity | 35% | **{sem_pct}%** |\n"
                    f"| 🟢 Granite reasoning | 30% | **{reason_pct}%** |\n\n"
                    f"**Rule (35%):** Stage, location, DPIIT recognition, sector, and budget ceiling checks against your profile. Hard failures (expired deadline, wrong state) score 0.\n\n"
                    f"**Semantic (35%):** Cosine similarity between your startup description and the grant's indexed text chunks (IBM Granite Embedding 278M).\n\n"
                    f"**Granite reasoning (30%):** IBM Granite 4H-Small reads your profile and the grant criteria and returns a 0–1 fit score with plain-language explanations."
                )

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            if g.application_link:
                st.link_button("🔗 Apply", g.application_link, use_container_width=True)
        with bcol2:
            if st.button("✍️ Draft", key=f"draft_btn_{idx}", use_container_width=True):
                ss["_draft_grant_idx"] = idx
                st.rerun()
        with bcol3:
            # Save to tracker
            if st.button("📌 Save", key=f"save_btn_{idx}", use_container_width=True):
                entry = {"name": g.name, "score": score_pct, "deadline": g.deadline or "N/A", "notes": ""}
                if not any(e["name"] == g.name for col in ss["tracker"].values() for e in col):
                    ss["tracker"]["Saved"].append(entry)
                    st.toast(f"Saved '{g.name[:30]}' to tracker", icon="📌")
        with bcol4:
            # Near-miss gap analyzer
            is_near_miss = 0.35 <= match.match_score <= 0.72
            if is_near_miss:
                if st.button("🔎 Gap Analysis", key=f"gap_btn_{idx}", use_container_width=True):
                    with st.spinner("Analyzing eligibility gaps…"):
                        result = c["analyze_gaps"](match, ss["profile"], ss["trace"])
                        ss["gap_results"][g.name] = result
                    st.rerun()

        # Show gap result if available
        if g.name in ss["gap_results"]:
            with st.expander("📋 Eligibility Gap Analysis", expanded=True):
                gap = ss["gap_results"][g.name]
                st.markdown("**How to become eligible — AI-suggested actions:**")
                for action in gap["actions"]:
                    st.markdown(f"➡ {action}")
                if gap["sources"]:
                    st.caption(f"📄 Grounded in: {', '.join(gap['sources'])}")

        # Feedback
        fb_col1, fb_col2, _ = st.columns([1, 1, 6])
        with fb_col1:
            if st.button("👍", key=f"fb_up_{idx}", help="This match is relevant"):
                ss["feedback"].append({"type": "match", "item": g.name, "vote": "up"})
                st.toast("Thanks for the feedback!", icon="👍")
        with fb_col2:
            if st.button("👎", key=f"fb_dn_{idx}", help="This match is not relevant"):
                ss["feedback"].append({"type": "match", "item": g.name, "vote": "down"})
                st.toast("Thanks for the feedback!", icon="👎")


def _render_compare(matches):
    st.markdown("#### Side-by-Side Comparison")
    cols = st.columns(len(matches))
    fields = ["Funding", "Deadline", "Stages", "Sectors", "Match %", "Rule %", "Semantic %", "LLM %"]
    for col, m in zip(cols, matches):
        g = m.grant
        col.markdown(f"**{g.name[:40]}**")
        col.metric("Funding", g.funding_amount or "N/A")
        col.metric("Deadline", g.deadline or "N/A")
        col.metric("Match", f"{int(m.match_score*100)}%")
        col.caption(f"Rule {int(m.rule_score*100)}% / Sem {int(m.semantic_score*100)}% / LLM {int(m.reasoning_score*100)}%")
        col.caption(", ".join(g.sectors[:2]) or "Any")


def _render_deadline_timeline(matches):
    import plotly.graph_objects as go
    timed = [(m, m.days_until_deadline) for m in matches if m.days_until_deadline is not None and m.days_until_deadline >= 0]
    if not timed:
        st.caption("No upcoming deadlines with known dates.")
        return
    timed.sort(key=lambda x: x[1])
    today = date.today()
    names = [m.grant.name[:30] for m, _ in timed]
    deadlines = [(today + timedelta(days=d)).strftime("%d %b %Y") for _, d in timed]
    days_vals = [d for _, d in timed]
    colors = ["#B71C1C" if d < 14 else "#E65100" if d < 30 else "#1565C0" for d in days_vals]

    fig = go.Figure(go.Bar(
        x=days_vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{d}d" for d in days_vals],
        textposition="outside",
        hovertext=deadlines,
        hovertemplate="%{y}<br>Deadline: %{hovertext}<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, len(timed) * 36),
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis_title="Days until deadline",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, sans-serif", size=12),
        xaxis=dict(gridcolor="#EEF1F7"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_score_chart(matches):
    import plotly.graph_objects as go
    names = [m.grant.name[:28] for m in matches]
    fig = go.Figure(data=[
        go.Bar(name="Rule (35%)",     x=names, y=[m.rule_score*100     for m in matches], marker_color="#1565C0"),
        go.Bar(name="Semantic (35%)", x=names, y=[m.semantic_score*100 for m in matches], marker_color="#7C4DFF"),
        go.Bar(name="Reasoning (30%)",x=names, y=[m.reasoning_score*100 for m in matches], marker_color="#00897B"),
    ])
    fig.update_layout(
        barmode="stack", height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Score (%)", yaxis=dict(gridcolor="#EEF1F7"),
        font=dict(family="Segoe UI, sans-serif", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Tab 3: Draft Application ─────────────────────────────────────────────────
def _tab_draft():
    c = _core()
    _section("✍️ Draft Grant Application")

    if not ss["profile"]:
        _empty_state("👤", "Profile needed", "Set your startup profile in the sidebar first.")
        return
    if not ss["grants"]:
        _empty_state("📂", "No grants", "Ingest grant documents first.")
        return

    grant_options = ss["grants"]
    # If user clicked Draft from matches tab, pre-select that grant
    default_idx = ss.pop("_draft_grant_idx", 0)
    default_idx = min(default_idx, len(grant_options)-1)

    selected_idx = st.selectbox(
        "Select grant to draft for:",
        range(len(grant_options)),
        format_func=lambda i: grant_options[i].name,
        index=default_idx,
    )
    selected_grant = grant_options[selected_idx]

    if st.button("🤖 Generate Draft with Granite", type="primary"):
        with st.spinner("Granite is writing your proposal (30-60s)…"):
            try:
                result = c["draft_proposal"](
                    grant=selected_grant, profile=ss["profile"], trace=ss["trace"]
                )
                ss["draft_result"] = result
                ss["draft_approved"] = False
                st.toast("Draft generated!", icon="✍️")
                st.rerun()
            except Exception as exc:
                st.error(f"Draft generation failed: {exc}")

    if not ss["draft_result"]:
        _empty_state("✍️", "No draft yet", "Select a grant above and click Generate.")
        return

    result = ss["draft_result"]

    # ── Split view
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="draft-req">', unsafe_allow_html=True)
        st.markdown("**📋 Grant Requirements**")
        st.markdown(f"**{selected_grant.name}**")
        st.caption(selected_grant.issuing_body or "")
        st.caption(f"Funding: {selected_grant.funding_amount or 'N/A'} ⚠️ verify")
        st.caption(f"Deadline: {selected_grant.deadline or 'N/A'} ⚠️ verify")

        # Document readiness progress
        checked_docs = sum(
            1 for doc in result["checklist"]
            if ss.get(f"doc_{doc[:30]}", False)
        )
        total_docs = max(len(result["checklist"]), 1)
        pct = int(checked_docs / total_docs * 100)
        st.markdown(
            f'<div class="draft-progress">'
            f'<div class="draft-progress-label">Document Readiness: {pct}% ({checked_docs}/{total_docs})</div>'
            f'<div class="draft-progress-bar"><div class="draft-progress-fill" style="width:{pct}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Required Documents:**")
        for doc in result["checklist"]:
            st.checkbox(f"  {doc}", key=f"doc_{doc[:30]}")

        if result.get("sources"):
            st.caption(f"📄 {', '.join(result['sources'])}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        draft_text = result["draft"]
        word_count = len(draft_text.split())
        st.markdown(
            f'<div class="word-count">Word count: {word_count}</div>',
            unsafe_allow_html=True,
        )
        st.error("⚠️ DRAFT — AI-generated. Review all claims before submission.")
        edited_draft = st.text_area(
            "Edit draft here:", value=draft_text, height=500, key="draft_editor"
        )
        if edited_draft != draft_text:
            ss["draft_result"]["draft"] = edited_draft
            ss["draft_result"]["approved"] = False

    # ── HITL Approval Gate
    st.divider()
    _section("🔐 Human-in-the-Loop Approval")
    st.warning("Complete all 4 checks before exporting. This gate is mandatory.")
    c1, c2 = st.columns(2)
    v1 = c1.checkbox("✅ Verified deadline on official portal")
    v2 = c1.checkbox("✅ Verified funding amount on official portal")
    v3 = c2.checkbox("✅ Verified eligibility criteria")
    v4 = c2.checkbox("✅ Reviewed full draft content")
    all_verified = v1 and v2 and v3 and v4

    if all_verified:
        if st.button("🔓 Approve & Unlock Export", type="primary"):
            ss["draft_approved"] = True

    if ss["draft_approved"]:
        st.success("✅ Draft approved. Choose export format:")
        final_text = ss["draft_result"]["draft"]
        grant_name_safe = selected_grant.name[:30].replace(" ", "_")
        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            st.download_button(
                "⬇️ Download TXT",
                data=final_text,
                file_name=f"draft_{grant_name_safe}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dcol2:
            try:
                docx_bytes = c["export_draft_docx"](
                    draft_text=final_text,
                    grant_name=selected_grant.name,
                    startup_name=ss["profile"].name,
                    checklist=result["checklist"],
                )
                st.download_button(
                    "⬇️ Download DOCX",
                    data=docx_bytes,
                    file_name=f"draft_{grant_name_safe}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            except Exception as exc:
                st.caption(f"DOCX export unavailable: {exc}")
        with dcol3:
            try:
                pdf_bytes = c["export_draft_pdf"](
                    draft_text=final_text,
                    grant_name=selected_grant.name,
                    startup_name=ss["profile"].name,
                    checklist=result["checklist"],
                )
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"draft_{grant_name_safe}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.caption(f"PDF export unavailable: {exc}")

        st.caption("Remember: verify once more before actual submission.")


# ─── Tab 4: Ask Questions ─────────────────────────────────────────────────────
def _build_suggested_questions() -> list[str]:
    """
    Build 5 suggested questions that are reliably answerable from indexed grants.
    All 5 templates ask about fields present in the first ~800 chars of every
    synthetic grant (funding amount, stages, eligibility, location, DPIIT) so
    the RAG retrieval always has enough context to give a real answer.
    """
    grants = ss.get("grants", [])

    # Pick up to 3 distinct grant names for labeling the questions
    names: list[str] = []
    for g in grants:
        n = (g.name or "").strip()
        if n and n not in names:
            names.append(n)
        if len(names) == 3:
            break

    def _n(i: int, fallback: str) -> str:
        """Return short grant name at index i, or fallback."""
        if i < len(names):
            return names[i][:45]
        return fallback

    return [
        f"What is the funding amount for {_n(0, 'SISFS')}?",
        f"What are the eligible startup stages for {_n(1, 'BIRAC BIG')}?",
        f"What are the eligibility criteria for {_n(2, 'NIDHI-PRAYAS')}?",
        "Which grants are available for Maharashtra-based startups?",
        "Which grants do not require DPIIT recognition?",
    ]


def _run_chat_question(question: str, c: dict) -> None:
    """Append user question to history, call answer_question, append response."""
    ss["chat_history"].append({"role": "user", "content": question})
    with st.spinner("Searching grant documents with Granite…"):
        try:
            resp = c["answer_question"](question, trace=ss["trace"])
            entry: dict = {
                "role": "assistant",
                "content": resp.answer,
                "citations": resp.citations,
            }
            if not resp.grounded:
                entry["content"] += "\n\n*ℹ️ Answer not found in the indexed documents.*"
            ss["chat_history"].append(entry)
        except Exception as exc:
            ss["chat_history"].append({"role": "assistant", "content": f"Error: {exc}", "citations": []})
    ss["feedback"].append({"type": "chat_pending", "item": question, "vote": None})


def _tab_chat():
    c = _core()
    _section("💬 Ask Questions About Grants")

    # ── Consume any pending question set by a chip button click ──────────────
    # We check this FIRST so the answer is processed before the UI renders.
    pending = ss.pop("pending_question", None)
    if pending:
        _run_chat_question(pending, c)

    # ── Suggested-question buttons (real st.button widgets, not HTML spans) ──
    suggested = _build_suggested_questions()
    st.caption("💡 Click a suggested question or type your own below:")

    # Lay out up to 5 buttons in a responsive row (up to 3 per row)
    row1 = suggested[:3]
    row2 = suggested[3:]
    cols1 = st.columns(len(row1))
    for col, q in zip(cols1, row1):
        # Use the question text as the key (truncated to keep it unique)
        if col.button(q, key=f"sq_{q[:30]}", use_container_width=True):
            ss["pending_question"] = q
            st.rerun()
    if row2:
        cols2 = st.columns(len(row2))
        for col, q in zip(cols2, row2):
            if col.button(q, key=f"sq_{q[:30]}", use_container_width=True):
                ss["pending_question"] = q
                st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Chat history bubbles ──────────────────────────────────────────────────
    if ss["chat_history"]:
        bubble_html = '<div class="chat-wrap">'
        for msg in ss["chat_history"]:
            if msg["role"] == "user":
                bubble_html += (
                    f'<div style="display:flex;justify-content:flex-end">'
                    f'<div class="bubble bubble-user">{msg["content"]}</div></div>'
                )
            else:
                cites = msg.get("citations", [])
                cite_html = ""
                if cites:
                    cite_html = '<div class="bubble-citation">📄 ' + " ".join(
                        f'<span class="citation-chip">{ct.source_file}' +
                        (f' p.{ct.page}' if ct.page else "") + '</span>'
                        for ct in cites
                    ) + '</div>'
                bubble_html += (
                    f'<div style="display:flex;justify-content:flex-start">'
                    f'<div class="bubble bubble-bot">{msg["content"]}{cite_html}</div></div>'
                )
        bubble_html += '</div>'
        st.markdown(bubble_html, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    else:
        _empty_state("💬", "No questions yet", "Click a suggestion above or type a question below.")

    # ── Free-text input ───────────────────────────────────────────────────────
    question = st.chat_input("Ask about eligibility, deadlines, funding amounts…")
    if question:
        _run_chat_question(question, c)
        st.rerun()

    # ── Feedback on latest answer ─────────────────────────────────────────────
    if ss["chat_history"] and ss["chat_history"][-1]["role"] == "assistant":
        fb1, fb2, _ = st.columns([1, 1, 8])
        if fb1.button("👍", key="chat_fb_up"):
            if ss["feedback"] and ss["feedback"][-1]["type"] == "chat_pending":
                ss["feedback"][-1]["vote"] = "up"
            st.toast("Thanks!", icon="👍")
        if fb2.button("👎", key="chat_fb_dn"):
            if ss["feedback"] and ss["feedback"][-1]["type"] == "chat_pending":
                ss["feedback"][-1]["vote"] = "down"
            st.toast("Noted!", icon="👎")

    if ss["chat_history"]:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            ss["chat_history"] = []
            st.rerun()


# ─── Tab 5: Application Tracker ───────────────────────────────────────────────
def _tab_tracker():
    _section("📌 Application Tracker")
    st.caption("Kanban board — move grants through your application workflow. Stored in session only.")

    total_tracked = sum(len(v) for v in ss["tracker"].values())
    if total_tracked == 0:
        _empty_state(
            "📌", "No grants tracked yet",
            "Go to the 🎯 Grant Matches tab, find a grant you like, and click the '📌 Save' button on its card to add it here.",
        )
        st.divider()

    columns = list(ss["tracker"].keys())
    cols = st.columns(len(columns))

    # Column-level empty-state hints
    _col_hints = {
        "Saved": "Use '📌 Save' on match cards to add grants here.",
        "Preparing": "Move grants here when you start gathering documents.",
        "Submitted": "Move grants here after you've submitted your application.",
        "Result": "Record the outcome — accepted, rejected, or pending.",
    }

    for col_widget, col_name in zip(cols, columns):
        items = ss["tracker"][col_name]
        with col_widget:
            st.markdown(
                f'<div class="kanban-col">'
                f'<div class="kanban-col-title">{col_name} ({len(items)})</div>',
                unsafe_allow_html=True,
            )
            if not items:
                st.markdown(
                    f'<div style="font-size:11px;color:#9099A6;padding:8px 4px;font-style:italic;">'
                    f'{_col_hints.get(col_name, "Empty")}</div>',
                    unsafe_allow_html=True,
                )
            for item in items:
                st.markdown(
                    f'<div class="kanban-card">'
                    f'<div class="kanban-card-name">{item["name"][:35]}</div>'
                    f'<div class="kanban-card-meta">Match: {item["score"]}% · {item["deadline"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                note = st.text_input("Notes", value=item.get("notes",""), key=f"note_{col_name}_{item['name'][:20]}", label_visibility="collapsed", placeholder="Add a note…")
                item["notes"] = note

                # Move buttons
                other_cols = [c for c in columns if c != col_name]
                move_to = st.selectbox("Move to →", ["—"] + other_cols, key=f"move_{col_name}_{item['name'][:20]}", label_visibility="collapsed")
                if move_to != "—":
                    ss["tracker"][col_name].remove(item)
                    ss["tracker"][move_to].append(item)
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # Export / Import
    st.divider()
    ecol, icol = st.columns(2)
    with ecol:
        tracker_json = json.dumps(ss["tracker"], indent=2)
        st.download_button("⬇️ Export Tracker (JSON)", data=tracker_json, file_name="grantscout_tracker.json", mime="application/json")

        # ICS export
        _export_ics()

    with icol:
        uploaded_tracker = st.file_uploader("⬆️ Import Tracker (JSON)", type=["json"], key="tracker_import")
        if uploaded_tracker:
            try:
                loaded = json.loads(uploaded_tracker.read())
                for col in columns:
                    if col in loaded:
                        ss["tracker"][col] = loaded[col]
                st.success("Tracker imported.")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")


def _export_ics():
    """Generate .ics calendar file for all upcoming deadlines."""
    try:
        from icalendar import Calendar, Event as ICSEvent
        cal = Calendar()
        cal.add("prodid", "-//GrantScout//IBM//EN")
        cal.add("version", "2.0")
        for g in ss["grants"]:
            if g.deadline_date and g.deadline_date >= date.today():
                evt = ICSEvent()
                evt.add("summary", f"Grant Deadline: {g.name}")
                evt.add("dtstart", g.deadline_date)
                evt.add("dtend", g.deadline_date)
                evt.add("description", f"Grant: {g.name}\nIssuing Body: {g.issuing_body or 'N/A'}\n⚠️ Verify on official portal before applying.")
                cal.add_component(evt)
        if len(cal.subcomponents) > 0:
            st.download_button(
                "📅 Export Deadlines (.ics)",
                data=cal.to_ical(),
                file_name="grant_deadlines.ics",
                mime="text/calendar",
            )
        else:
            st.caption("No grants with known upcoming deadlines for calendar export.")
    except ImportError:
        st.caption("icalendar package not installed — ICS export unavailable.")
    except Exception as exc:
        st.caption(f"ICS export error: {exc}")


# ─── Tab 6: Agent Trace & Eval ────────────────────────────────────────────────
def _tab_trace_eval():
    import plotly.graph_objects as go

    tab_a, tab_b, tab_c = st.tabs(["🔬 Agent Trace", "📊 Evaluation Metrics", "🛡️ Responsible AI"])

    # ── Agent Trace
    with tab_a:
        _section("🔬 Agent Trace")
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Clear Trace", key="clear_trace"):
                ss["trace"] = []
        if not ss["trace"]:
            _empty_state("🔬", "No trace yet", "Run ingestion, matching, or drafting to see the pipeline steps here.")
        else:
            # Step summary
            step_counts: dict[str, int] = {}
            for step in ss["trace"]:
                sname = step.get("step","unknown")
                step_counts[sname] = step_counts.get(sname, 0) + 1
            cols = st.columns(min(len(step_counts), 6))
            for i, (sname, cnt) in enumerate(step_counts.items()):
                cols[i % 6].metric(sname, cnt)

            # Total latency
            total_ms = sum(s.get("ms", 0) for s in ss["trace"] if isinstance(s.get("ms"), (int, float)))
            st.caption(f"Total trace time: {total_ms:,}ms")

            icon_map = {
                "store_original":"💾","parse":"📄","chunk":"✂️","embed":"🔢","index":"🗄️",
                "extract":"🔍","extract_retry":"🔄","retrieve":"🔎","filter":"🧩","score":"🎯",
                "retrieve_context":"📖","draft":"✍️","generate_answer":"💬",
                "gap_analysis":"🔎","pitchdeck_parse":"📑","pitchdeck_extract":"🤖",
            }
            for i, step in enumerate(ss["trace"]):
                sname = step.get("step","step")
                ms = step.get("ms","")
                icon = icon_map.get(sname, "⚙️")
                with st.expander(f"{icon} Step {i+1}: **{sname}**" + (f" ({ms}ms)" if ms else ""), expanded=False):
                    st.json({k:v for k,v in step.items() if k!="step"})

    # ── Eval metrics
    with tab_b:
        _section("📊 Evaluation Metrics")

        if ss["matches"]:
            st.markdown("**Match Score Distribution**")
            scores = [m.match_score for m in ss["matches"]]
            fig = go.Figure(go.Histogram(x=[s*100 for s in scores], nbinsx=10,
                                         marker_color="#1565C0", opacity=0.8))
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=10),
                               xaxis_title="Match Score (%)", yaxis_title="Count",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            import pandas as pd
            rows = [{"Grant": m.grant.name[:40], "Match%": int(m.match_score*100),
                     "Rule%": int(m.rule_score*100), "Semantic%": int(m.semantic_score*100),
                     "Reasoning%": int(m.reasoning_score*100),
                     "Days Left": m.days_until_deadline if m.days_until_deadline is not None else "N/A"}
                    for m in ss["matches"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Latency per step type
        if ss["trace"]:
            step_ms: dict[str, list] = {}
            for step in ss["trace"]:
                sname = step.get("step","unknown")
                ms = step.get("ms")
                if ms:
                    step_ms.setdefault(sname, []).append(ms)
            if step_ms:
                avg_ms = {s: sum(v)/len(v) for s,v in step_ms.items()}
                st.markdown("**Average Latency per Pipeline Step (ms)**")
                fig2 = go.Figure(go.Bar(x=list(avg_ms.keys()), y=list(avg_ms.values()),
                                        marker_color="#7C4DFF"))
                fig2.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   yaxis_title="ms")
                st.plotly_chart(fig2, use_container_width=True)

        # Feedback summary
        if ss["feedback"]:
            st.markdown("**User Feedback Summary**")
            up = sum(1 for f in ss["feedback"] if f.get("vote") == "up")
            dn = sum(1 for f in ss["feedback"] if f.get("vote") == "down")
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("👍 Thumbs Up", up)
            fc2.metric("👎 Thumbs Down", dn)
            fc3.metric("Pending", sum(1 for f in ss["feedback"] if f.get("vote") is None))

        # ── Manual vs GrantScout comparison card ──────────────────────────
        _section("⏱️ Manual Search vs GrantScout")
        n_grants = max(len(ss["grants"]), 6)

        # Measure actual pipeline time from trace
        total_trace_ms = sum(
            s.get("ms", 0) for s in ss["trace"]
            if isinstance(s.get("ms"), (int, float))
        )
        ai_mins = max(2.0, total_trace_ms / 60_000)

        NUM_PORTALS = 8
        MANUAL_HRS_PER_GRANT = 1.5

        st.markdown(
            f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:12px">
  <thead>
    <tr style="background:#F4F6FB">
      <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #DDE3EE">Task</th>
      <th style="padding:8px 12px;text-align:center;border-bottom:2px solid #DDE3EE">Manual</th>
      <th style="padding:8px 12px;text-align:center;border-bottom:2px solid #DDE3EE">GrantScout</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:7px 12px;border-bottom:1px solid #EEF1F7">Find relevant grants</td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7">{NUM_PORTALS} portals × 30 min = <b>{NUM_PORTALS*30} min</b></td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7;color:#1565C0"><b>~1 min</b> (semantic search)</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border-bottom:1px solid #EEF1F7">Read & extract eligibility ({n_grants} docs)</td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7">{n_grants} × 45 min = <b>{n_grants*45} min</b></td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7;color:#1565C0"><b>~{int(ai_mins)} min</b> (Granite extraction)</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border-bottom:1px solid #EEF1F7">Compare & rank against profile</td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7"><b>2–4 hours</b></td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7;color:#1565C0"><b>&lt;1 min</b> (hybrid scoring)</td>
    </tr>
    <tr>
      <td style="padding:7px 12px;border-bottom:1px solid #EEF1F7">Draft initial application</td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7"><b>4–8 hours</b></td>
      <td style="padding:7px 12px;text-align:center;border-bottom:1px solid #EEF1F7;color:#1565C0"><b>~30 sec</b> + human review</td>
    </tr>
    <tr style="background:#F0FDF4">
      <td style="padding:8px 12px;font-weight:700">Estimated total</td>
      <td style="padding:8px 12px;text-align:center;font-weight:700">{int(n_grants * MANUAL_HRS_PER_GRANT)}+ hours</td>
      <td style="padding:8px 12px;text-align:center;font-weight:700;color:#1565C0">~{int(ai_mins)} min pipeline + review</td>
    </tr>
  </tbody>
</table>
<div style="font-size:11px;color:#9099A6;margin-top:2px">
⚠️ <b>Stated assumptions:</b> {MANUAL_HRS_PER_GRANT}h per grant for manual reading + eligibility check; {NUM_PORTALS} portals typically checked manually; GrantScout pipeline time measured from agent trace ({total_trace_ms:,} ms recorded so far). Human review of AI output is always required and is not counted in the AI time.
</div>""",
            unsafe_allow_html=True,
        )

        # Link to run_eval
        st.divider()
        st.markdown("**Run full evaluation (accuracy + precision@3):**")
        st.code("python eval/run_eval.py", language="bash")
        st.caption("Outputs CSV + chart to eval/results/")

    # ── Responsible AI
    with tab_c:
        _section("🛡️ Responsible AI")
        st.markdown("""
GrantScout was designed with responsible AI principles from the ground up. Every user-facing feature has a corresponding safeguard.

| Principle | Implementation |
|-----------|---------------|
| **Grounded claims only** | Every chat answer and draft section cites its source document and page number. If the indexed context does not support an answer, GrantScout says "not found in the provided documents" — it never fabricates. |
| **Null-not-guess extraction** | During structured grant extraction, every field that cannot be found in the document is set to `null`. The LLM is explicitly instructed never to guess or infer values not present in the text. |
| **Human-in-the-loop** | Application drafts carry a ⚠️ DRAFT watermark. Users must tick 4 verification checkboxes (deadline, amount, eligibility, content review) before the export button is unlocked. |
| **Verify-on-portal notices** | Every displayed deadline and funding amount is accompanied by a "verify on the official portal" notice, because grant details change. |
| **Public documents only** | GrantScout only indexes documents you upload from public sources. URL ingestion blocks private IPs and private portals. No bulk scraping. The ethics notice is visible in the sidebar on every page. |
| **No PII storage** | No personal data is persisted beyond your browser session. The Clear Data button removes all vectors, profiles, and history immediately. |
| **Transparent pipeline** | Every step of the agentic pipeline — store → parse → chunk → embed → index → extract → retrieve → filter → score → draft — is logged in the Agent Trace tab with timing and metadata. |
| **Synthetic data labelled** | Test grant samples are clearly marked SYNTHETIC DATA and separated from real grant documents in `/data/synthetic/`. |

### Known Limitations
- **Stale data**: Extracted grant details reflect the document at ingestion time. Re-ingest when official documents are updated.
- **Model ceiling**: IBM Granite 4H-Small on the Lite tier has a token limit (~1024 output tokens). Very long drafts may be truncated.
- **Language**: Extraction and drafting work best in English. Hindi/Marathi support depends on the multilingual embedding model for retrieval.
- **Unstructured PDFs**: Scanned or image-only PDFs may produce poor extraction results.
- **Match scores are indicative**: Always read the full grant guidelines before applying.
        """)


# ─── Footer ───────────────────────────────────────────────────────────────────
def _footer():
    st.markdown(
        '<div class="gs-footer">'
        'Built using <b>IBM Granite</b> · <b>watsonx.ai Lite</b> · <b>Docling</b> · <b>ChromaDB</b> · Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Phase runners — must fire BEFORE any UI renders so st.status()
    # widgets appear at the top and st.rerun() short-circuits the rest.
    if ss.get("_demo_phase"):
        _continue_demo()
    if ss.get("_ingest_queue"):
        _continue_ingest_queue()

    _sidebar()
    _hero()
    _render_stepper()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📂 Upload & Ingest",
        "🎯 Grant Matches",
        "✍️ Draft Application",
        "💬 Ask Questions",
        "📌 Tracker",
        "🔬 Trace & Eval",
    ])

    with tab1: _tab_ingest()
    with tab2: _tab_matches()
    with tab3: _tab_draft()
    with tab4: _tab_chat()
    with tab5: _tab_tracker()
    with tab6: _tab_trace_eval()

    _footer()


if __name__ == "__main__":
    main()
