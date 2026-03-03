# PGP Agentic — RAG + Agentic Pipeline

## Overview

This project implements a **Generative AI–powered enterprise document Q&A system** that lets users upload documents in multiple formats (PDF, TXT, CSV, Excel), index them into a vector database (Chroma), and ask natural-language questions that are answered by a Gemini LLM via a **Retrieval-Augmented Generation (RAG)** + **Agentic AI** workflow.

## Highlights

- **Streamlit UI** for uploading documents and chatting with the agent.
- **Document ingestion pipeline** supporting PDF, TXT, CSV, and Excel (XLS/XLSX).
- Text chunking → embedding → Chroma vector store persistence.
- **LLM**: Gemini via Google Vertex AI (`ChatVertexAI` + `VertexAIEmbeddings`).
- **Agentic flow** (planner → retriever → generator → validator) built with LangChain.
- Optional **FastAPI REST API** for headless / integration use.
- Input validation and output grounding checks to reduce hallucinations.

## Architecture

```
User (Streamlit UI / REST API)
        │
        ▼
┌───────────────────────────────────┐
│          Ingestion Pipeline       │
│  parse_file → chunk → embed       │
│  → store in Chroma                │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│         Agentic RAG Pipeline      │
│  1. Planner  (intent extraction)  │
│  2. Retriever (similarity search) │
│  3. Generator (Gemini LLM)        │
│  4. Validator (grounding check)   │
└───────────────────────────────────┘
        │
        ▼
  Grounded answer + source refs
```

## Files

| Path | Description |
|------|-------------|
| `app/streamlit_app.py` | Streamlit front-end (upload + chat) |
| `app/api.py` | Optional FastAPI REST API |
| `app/ingest.py` | Document ingestion pipeline |
| `app/agent.py` | Agentic RAG flow |
| `app/chroma_store.py` | Chroma vector store wrapper |
| `app/gemini_llm.py` | LLM + Embeddings via Vertex AI |
| `app/utils/parsers.py` | File parsers (PDF, TXT, CSV, Excel) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SahaDiptarup/pgp-agentic-capstone.git
cd pgp-agentic-capstone
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your GCP project / credentials
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | — | Path to GCP service account JSON |
| `GCP_PROJECT` | Yes | — | Your GCP project ID |
| `GCP_LOCATION` | No | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | No | `gemini-1.0-pro` | Vertex AI model name |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Local Chroma storage path |
| `CHROMA_HOST` | No | — | Chroma server host (server mode) |
| `CHROMA_PORT` | No | `8000` | Chroma server port (server mode) |
| `CHUNK_SIZE` | No | `800` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | No | `100` | Chunk overlap (characters) |

## Running

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### REST API (optional)

```bash
python app/api.py
# or
uvicorn app.api:app --reload
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest` | Upload + index documents (multipart/form-data) |
| `POST` | `/query` | Ask a question (JSON body: `{"question": "..."}`) |

Interactive docs available at http://localhost:8080/docs.

## Agentic Flow Details

1. **Planner** – extracts the retrieval intent from the user question.
2. **Retriever** – performs semantic similarity search in Chroma to fetch the top-k relevant chunks.
3. **Generator** – constructs a grounded prompt (system instruction + retrieved context + question) and calls the Gemini LLM.
4. **Validator** – checks whether the answer shares meaningful vocabulary with the retrieved context; appends a caveat if grounding is weak.

## Notes & Limitations

- This is an educational scaffold. Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, LLM temperature, and prompt engineering for production workloads.
- The Gemini LLM is accessed via Vertex AI. Ensure your service account has the **Vertex AI User** role.
- Safety: outputs are not guaranteed to be hallucination-free. The validator step reduces risk but does not eliminate it.
- Large documents may take some time to embed during ingestion.
