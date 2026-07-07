# Academic Assistant — Adaptive RAG Platform

A Retrieval-Augmented Generation (RAG) chatbot for California higher-education admissions advising. It uses **Adaptive RAG** to dynamically route each question to the best retrieval strategy — local documents, live web search, or both — before generating cited streaming answers.

**Live UI:** `http://127.0.0.1:8000/`  
**Admin panel:** `http://127.0.0.1:8000/admin`  
**API docs:** `http://127.0.0.1:8000/docs`

---

## Architecture (Adaptive RAG)

```
User question
    │
    ├─ Greeting? ──► instant reply (no RAG)
    │
    ├─ Safety filter (Azure OpenAI)
    │
    ├─ Query router (Azure OpenAI)
    │     ├─ DIRECT  → no retrieval, conversational reply
    │     ├─ LOCAL   → ChromaDB only; fallback to web if graded irrelevant
    │     ├─ WEB     → .edu web search only
    │     └─ HYBRID  → local + web; web fallback if local fails grading
    │
    ├─ Document grader (Azure OpenAI) — filters irrelevant chunks
    │
    ├─ Vector search + FlashRank re-rank (when LOCAL/HYBRID)
    │
    └─ Stream cited answer (Azure OpenAI, temperature 0.0)
```

| Component | Technology |
|-----------|------------|
| LLM | Azure OpenAI (GPT-4o deployment) |
| Embeddings | `BAAI/bge-small-en-v1.5` (Sentence Transformers) |
| Vector DB | ChromaDB (persistent, `./chroma_data/`) |
| Re-ranker | FlashRank cross-encoder |
| Web search | DuckDuckGo (`site:.edu` filter) |
| Chat memory | SQLite (`conversation_memory.db`) |
| API + UI | FastAPI, vanilla HTML/JS |

---

## Project structure

```
rag_application/
├── app/
│   ├── main.py              # FastAPI routes
│   ├── config.py            # Environment config
│   ├── auth.py              # Admin API key guard
│   ├── llm/
│   │   ├── adaptive.py      # Adaptive RAG orchestrator
│   │   ├── router.py        # Query routing (DIRECT/LOCAL/WEB/HYBRID)
│   │   ├── grader.py        # Document relevance grading
│   │   ├── generator.py     # LLM prompts & streaming
│   │   └── controller.py    # Context assembly per route
│   ├── services/
│   │   ├── extractor.py     # PDF, DOCX, TXT extraction
│   │   ├── chunker.py       # Text splitting
│   │   ├── embedder.py      # Embedding generation
│   │   ├── vector_db.py     # ChromaDB storage & search
│   │   ├── reranker.py      # FlashRank re-ranking
│   │   ├── memory.py        # Conversation history
│   │   ├── ingest.py        # Incremental file indexing
│   │   └── evaluator.py     # Optional Ragas evaluation
│   ├── tools/
│   │   └── web_search.py    # .edu web search
│   └── static/              # Chat + admin frontend
├── uploads/                 # Document upload folder
├── requirements.txt
└── .env.example
```

---

## Features

- **Chat UI** — ChatGPT-style interface with saved conversations and delete support
- **Streaming answers** — Token-by-token response via `StreamingResponse`
- **Adaptive RAG routing** — chooses direct, local, web, or hybrid retrieval per question
- **Document grader** — drops irrelevant chunks; falls back to web when local docs fail
- **Re-ranking** — Retrieves 10 chunks, re-ranks to top 4 with FlashRank
- **Conversation memory** — Follow-up questions with query condensation
- **Admin-only uploads** — Document ingestion protected by `ADMIN_API_KEY`
- **Auto-ingest on startup** — New or updated files in `uploads/` are indexed automatically
- **Cited answers** — LLM instructed to cite source filenames or URLs inline

---

## Getting started

### Prerequisites

- Python 3.10+
- Azure OpenAI endpoint and deployment

### Installation

```bash
git clone <your-repo-url>
cd rag_application
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Environment variables

```env
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
ADMIN_API_KEY=change-this-to-a-strong-secret
```

### Run

```bash
uvicorn app.main:app --reload
```

Place documents (`.txt`, `.pdf`, `.docx`) in `uploads/`. They are indexed on startup, or via **Admin → Ingest new files**.

---

## API endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /` | Public | Chat UI |
| `GET /admin` | Public | Admin login page |
| `POST /admin/verify` | Public | Validate admin API key |
| `GET /health` | Public | Health check |
| `POST /ask` | Public | Ask a question (streaming) |
| `POST /session/new` | Public | Create chat session |
| `GET /sessions` | Public | List saved chats |
| `GET /session/{id}/history` | Public | Load chat history |
| `DELETE /session/{id}` | Public | Delete a chat |
| `POST /upload` | Admin | Upload & index a document |
| `POST /ingest` | Admin | Index new/updated files in `uploads/` |
| `GET /ingest/status` | Admin | Indexed chunk count & pending files |

Admin endpoints require header: `X-Admin-Key: <ADMIN_API_KEY>`

---

## Optional: RAG evaluation

Install optional dependencies:

```bash
pip install ragas datasets
```

Use `app/services/evaluator.py` (`RAGEvaluator`) to score faithfulness, answer relevancy, and context precision.

---

## Design notes

- **Grounded responses:** The LLM is instructed to answer only from retrieved context; otherwise it returns a fixed refusal message.
- **Greetings** bypass the full RAG pipeline for faster replies.
- **Incremental ingest** tracks file signatures in `ingest_manifest.json` so unchanged files are skipped.
