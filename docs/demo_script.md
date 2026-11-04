# GrantScout Demo Script (v2) — 2-3 Minute Video

**Format:** Screen recording with voiceover  
**Start at:** [http://localhost:8501](http://localhost:8501) — app already open, blank state

---

## 0:00 – 0:15 | Problem (voiceover, static screen)

> "India has 35+ government grant schemes for startups — SISFS, BIRAC, SIDBI, state schemes.
> But they're scattered across portals in different formats, with confusing eligibility rules.
> Early founders spend 20+ hours just finding and evaluating grants.
> GrantScout uses IBM Granite on watsonx.ai to do this in under 60 seconds."

---

## 0:15 – 0:55 | Demo Mode — Full flow in 40 seconds

1. Point to the **hero banner** — note the IBM Granite branding and stat boxes
2. Click **"⚡ Try Demo Mode"** — spinner shows "Parsing with Docling… Asking Granite…"
3. While loading (~30-40s): point to the **IBM Services Status** expander in the sidebar
   > "Green dots — Granite LLM, Granite Embeddings, watsonx.ai — all connected live."
4. Demo completes → **Stepper** shows steps 1-3 all checked off
5. Navigate to **🎯 Grant Matches** tab
6. Point to the **KPI row**: "4 eligible grants, ₹1.7 Cr potential funding, 45 days nearest deadline"
7. Scroll through first match card:
   > "85% match — circular score badge in green. SISFS. Funding ₹20L-50L. Stage, sector, DPIIT — all matching. Source cited."
8. Show the **Score Breakdown** stacked bar chart
   > "Rule-based 35%, Semantic RAG 35%, Granite reasoning 30% — fully transparent hybrid scoring."
9. Show **Deadline Timeline** horizontal bar chart with urgency colours

---

## 0:55 – 1:35 | Gap Analyzer + Draft (40 seconds)

1. Find a **near-miss grant** (40-70% match) → click **"🔎 Gap Analysis"**
   > "For near-misses, Granite generates specific actions — 'Register with DPIIT', 'Update entity type to Pvt Ltd' — each citing the grant criterion."
2. Navigate to **✍️ Draft Application** tab
3. Grant is pre-selected → click **"🤖 Generate Draft with Granite"** (~20s)
4. Show the **split view** — grant requirements on left, editable draft on right
   > "7-section proposal: executive summary, problem, solution, impact, budget, eligibility mapping, team."
5. Show **Required Documents** progress bar
6. Point to the **HITL approval gate**:
   > "This is non-negotiable responsible AI. The user must verify deadline, amount, eligibility, and content — four checkboxes — before the export buttons appear."
7. Tick all 4 boxes → **Download DOCX / PDF** buttons appear
   > "One-click export with DRAFT watermark and 'verify on official portal' footer baked in."

---

## 1:35 – 1:55 | Chat with Citations (20 seconds)

1. Navigate to **💬 Ask Questions** tab
2. Click a **suggested question chip**: *"What documents do I need for SISFS?"*
3. Answer appears as a **chat bubble** with blue citation chips
   > "Every answer is grounded in the indexed documents. Citation chip shows filename and page."
4. Type: *"What is the deadline for BIRAC BIG?"*
5. Show citation; note "verify on official portal" notice in the answer

---

## 1:55 – 2:25 | Tracker + Architecture + Eval (30 seconds)

1. Go back to Grant Matches → click **"📌 Save"** on a grant
2. Navigate to **📌 Tracker** tab
   > "Kanban board — Saved, Preparing, Submitted, Result. Export to JSON or .ics calendar."
3. Navigate to **🔬 Trace & Eval** tab → **Agent Trace** sub-tab
   > "Every pipeline step with timings — store → parse → chunk → embed → extract → retrieve → score → draft."
4. Switch to **Responsible AI** sub-tab — briefly show the principles table
   > "Built-in: citations, HITL, null-not-guess, public docs only, no PII."
5. Switch to **Evaluation Metrics** → show score distribution chart + latency bars
   > "Full evaluation with ground-truth CSV and precision@3: `python eval/run_eval.py`"

---

## 2:25 – 2:45 | Closing (voiceover)

> "GrantScout runs entirely on IBM Cloud Lite — zero cost.
> IBM Granite 4H-Small for extraction, reasoning, and drafting.
> Granite Embedding 278M for semantic search.
> ChromaDB for the local vector store.
> Every claim cited. Every draft needs human approval.
> GrantScout — making government grants accessible to every Indian founder."

---

## Pre-Demo Checklist

- [ ] `.env` with valid IBM credentials
- [ ] App running: `streamlit run app/streamlit_app.py`
- [ ] Browser at `http://localhost:8501`
- [ ] Clear browser cache / Streamlit session before recording
- [ ] Test "Demo Mode" once before recording (~40s first run, faster if ChromaDB cached)
