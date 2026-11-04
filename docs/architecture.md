# GrantScout Architecture

## System Overview

```mermaid
flowchart TD
    subgraph User["👤 User (Founder / Student)"]
        U[Streamlit Browser UI]
    end

    subgraph UI["🖥️ Streamlit App (app/streamlit_app.py)"]
        T1[Upload & Ingest Tab]
        T2[Grant Matches Tab]
        T3[Draft Application Tab]
        T4[Ask Questions Tab]
        T5[Agent Trace & Eval Tab]
    end

    subgraph Agent["🤖 Agentic Pipeline (core/)"]
        ING[ingestion.py\nDocling → Chunks → Embed]
        EXT[extraction.py\nGranite JSON extraction]
        RET[retrieval.py\nChromaDB semantic search]
        MAT[matching.py\nHybrid scoring]
        DRA[drafting.py\nProposal generation]
        CHA[chat.py\nGrounded Q&A]
        SCH[schemas.py\nPydantic models]
        CFG[config.py\n.env secrets]
    end

    subgraph IBM["☁️ IBM Cloud (Free/Lite Tier)"]
        WX[watsonx.ai\nibm/granite-4-h-small\nibm/granite-embedding-278m-multilingual]
        COS[Cloud Object Storage\nOriginal documents]
    end

    subgraph Local["💾 Local Storage"]
        CD[ChromaDB\nVector index]
        DATA[/data/grants/\n/data/synthetic/]
    end

    subgraph RAI["🛡️ Responsible AI Layer"]
        CITE[Source citations\n(doc + page)]
        HITL[Human-in-the-loop\nDraft approval gate]
        WARN[Verify-on-portal\nwarnings]
        PRIV[No persistent PII\nClear-data button]
        PUBL[Public docs only\nNo restricted portals]
    end

    U --> UI
    T1 --> ING
    T2 --> MAT
    T3 --> DRA
    T4 --> CHA
    T5 --> Agent

    ING -->|chunks| CD
    ING -->|originals| COS
    ING -->|text| WX
    EXT -->|JSON prompt| WX
    RET -->|query embed| WX
    RET -->|vector search| CD
    MAT --> RET
    MAT --> EXT
    MAT -->|reasoning| WX
    DRA --> RET
    DRA -->|generation| WX
    CHA --> RET
    CHA -->|generation| WX

    Agent --- RAI
    RAI --> CITE
    RAI --> HITL
    RAI --> WARN
    RAI --> PRIV
    RAI --> PUBL
```

## Component Descriptions

| Component | File | Responsibility |
|-----------|------|---------------|
| Config | `core/config.py` | Reads `.env`, exposes typed constants |
| watsonx Client | `core/watsonx_client.py` | Granite LLM + Embedding via `ibm-watsonx-ai` |
| COS Client | `core/cos_client.py` | IBM COS upload/download with local fallback |
| Schemas | `core/schemas.py` | Pydantic: GrantInfo, StartupProfile, MatchResult, … |
| Ingestion | `core/ingestion.py` | Docling → chunk → embed → ChromaDB |
| Extraction | `core/extraction.py` | Granite JSON-only prompts → GrantInfo |
| Retrieval | `core/retrieval.py` | ChromaDB cosine similarity search |
| Matching | `core/matching.py` | Rule filter + Semantic RAG + Granite reasoning |
| Drafting | `core/drafting.py` | Grounded proposal generation + HITL gate |
| Chat | `core/chat.py` | RAG Q&A with source citations |
| UI | `app/streamlit_app.py` | 5-tab Streamlit interface |
| Eval | `eval/run_eval.py` | Field accuracy + precision@3 reporting |

## Data Flow

```
Grant PDF/DOCX/HTML
        │
        ▼
   Docling Parser
        │ pages → text blocks
        ▼
  Text Chunker (800-word chunks, 100-word overlap)
        │
        ▼
  Granite Embeddings (ibm-watsonx-ai)
        │ vectors
        ▼
   ChromaDB (local persistent vector store)
        │
        ├──► Granite Extraction → GrantInfo JSON (pydantic)
        │
Startup Profile (user input in sidebar)
        │
        ▼
  Matching Agent
  ├── Rule-based eligibility filters (stage, location, DPIIT, sector)
  ├── Semantic similarity (ChromaDB cosine score)
  └── Granite LLM reasoning (0-1 score + explanations)
        │
        ▼
  Ranked MatchResult list → Streamlit Match Cards
        │
        ▼
  Drafting Agent (user selects grant)
  ├── Retrieve full grant context from ChromaDB
  ├── Granite generation (grounded in profile + grant text)
  └── Human-in-the-loop approval gate (4 checkboxes)
        │
        ▼
  Exported .txt draft (only after explicit user approval)
```

## Responsible AI Design

1. **Grounding**: Every answer and draft cites its source document and page number.
2. **No hallucination policy**: Extraction marks unknown fields `null`; chat says "not found in the provided documents" when context is insufficient.
3. **Human-in-the-loop**: Drafts carry a DRAFT label; users must verify 4 checkboxes before export.
4. **Deadline/amount warnings**: Every displayed deadline/amount includes "verify on official portal" notice.
5. **Data privacy**: No personal data is stored beyond the session; a Clear Data button removes all vectors.
6. **Public documents only**: The app does not scrape login-protected portals; the ethics note is visible in the sidebar.
