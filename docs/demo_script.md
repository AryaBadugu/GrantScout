# GrantScout — Demo Script (2:45 walkthrough)

> Use this script for a 2–3 minute video demo or live presentation.

---

## Scene 1 — Hook (0:00–0:20)

**Say:** "Finding the right government grant for your startup takes weeks of research across dozens of portals. GrantScout does it in seconds — using IBM Granite AI on watsonx.ai."

**Show:** The live app at [https://grant-scout.streamlit.app](https://grant-scout.streamlit.app) — hero banner with live stats.

---

## Scene 2 — Fill Startup Profile (0:20–0:40)

**Say:** "First, tell GrantScout about your startup."

**Do:**
- In the sidebar, fill in: Stage = Prototype, Sector = Agritech, State = Maharashtra, DPIIT = Yes, Funding needed = ₹20 lakh
- Click **Save Profile** — the onboarding stepper advances to Step 2

---

## Scene 3 — Load & Ingest Grants (0:40–1:05)

**Say:** "Now we load grant documents. I'll use the built-in synthetic samples — in production you'd upload real PDFs from government portals."

**Do:**
- Go to **📂 Upload & Ingest** tab
- Click **⚡ Try Demo Mode** — watch the `st.status` bars process each file one at a time
- Point out the **Agent Trace** updating in real time (Docling → embed → ChromaDB)
- Show the Ingestion Log: "6 grants indexed, embeddings stored in ChromaDB"

---

## Scene 4 — AI Matching (1:05–1:35)

**Say:** "GrantScout uses a three-layer hybrid scoring engine — rule-based filters, semantic similarity via IBM Granite embeddings, and Granite LLM reasoning."

**Do:**
- Go to **🎯 Grant Matches** tab
- Click **🔍 Find Matching Grants**
- Show the KPI row: eligible count, total potential funding, nearest deadline
- Click the top match card — expand **✅ Why it matches** and **🧮 Score** breakdown
- Click **⬇️ Export CSV** — show the downloaded spreadsheet

---

## Scene 5 — Draft Application (1:35–2:05)

**Say:** "For your best match, GrantScout drafts a tailored application — grounded only in the grant document and your profile. No hallucinations."

**Do:**
- Click **✍️ Draft** on the top match card
- Go to **✍️ Draft Application** tab
- Show the generated draft: problem statement, solution, impact, budget justification
- Point out the **⚠️ DRAFT — review before submission** watermark
- Check all 4 Human-in-the-Loop checkboxes → **Export DOCX** download

---

## Scene 6 — Chat Q&A (2:05–2:25)

**Say:** "You can also ask questions directly — every answer is grounded with citations."

**Do:**
- Go to **💬 Ask Questions** tab
- Click the suggested question: *"What is the funding amount for Startup India Seed Fund Scheme?"*
- Show the answer with `[source: SISFS_synthetic.txt]` citation chip

---

## Scene 7 — Responsible AI close (2:25–2:45)

**Say:** "GrantScout is built with responsible AI at every layer — citations, human approval, verify notices, no PII storage, and a full agent trace for transparency."

**Do:**
- Show the **Responsible AI** note in the sidebar
- Flash the **🔬 Trace & Eval** tab showing every pipeline step

**Close:** "GrantScout — helping India's early-stage founders access the funding they deserve, powered by IBM Granite on watsonx.ai."

---

## Key talking points for judges

| Criterion | Evidence |
|---|---|
| IBM technology | IBM Granite 4H-Small (LLM) + Granite Embedding 278M — both on watsonx.ai Lite |
| Agentic RAG | 5-step pipeline: retrieve → extract → filter → score → draft, visible in Trace tab |
| Responsible AI | Citations on every claim, HITL approval gate, verify notices, no PII |
| Real problem | 35 real Indian government schemes listed; 6 synthetic samples for demo |
| Production-ready | 41/41 tests, Docker, Streamlit Cloud deploy, CSV export |
