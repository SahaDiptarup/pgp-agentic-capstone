"""
Optional minimal REST API for the PGP Agentic RAG application.

Run with:
    python app/api.py
  or:
    uvicorn app.api:app --reload

Endpoints
---------
POST /ingest  – Upload and index one or more documents.
POST /query   – Ask a question; returns the agent's answer + sources.
GET  /health  – Health check.
"""
from __future__ import annotations

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from app.ingest import ingest_files  # noqa: E402
from app.agent import run_agent  # noqa: E402

app = FastAPI(
    title="PGP Agentic RAG API",
    description=(
        "Upload enterprise documents and ask natural-language questions. "
        "Powered by Gemini (Vertex AI) and a Chroma vector store."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/ingest", summary="Upload and index documents")
async def ingest(files: list[UploadFile] = File(...)) -> JSONResponse:
    """
    Accept one or more files (PDF, TXT, CSV, XLS, XLSX), parse them, chunk
    the text, embed the chunks, and store them in Chroma.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    file_pairs = [(f.filename or "upload", await f.read()) for f in files]

    try:
        results = ingest_files(file_pairs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content={"results": results})


@app.post("/query", response_model=QueryResponse, summary="Ask a question")
def query(request: QueryRequest) -> QueryResponse:
    """
    Run the agentic RAG pipeline for the given question and return the answer
    together with the source document names.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = run_agent(request.question, k=request.k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"],
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080, reload=False)
