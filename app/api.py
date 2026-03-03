"""
Optional REST API for the PGP Agentic RAG app.

Run with:
    python app/api.py
  or:
    uvicorn app.api:app --reload

Endpoints:
    POST /ingest  - Upload and index documents.
    POST /query   - Ask a question, get an answer + sources.
    GET  /health  - Health check.
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
    description="Upload documents and ask natural-language questions. Powered by Gemini + Chroma.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/ingest", summary="Upload and index documents")
async def ingest(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Accept one or more files, parse them, chunk, embed, and store in Chroma."""
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
    """Run the agentic RAG pipeline and return the answer with source references."""
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


if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8080, reload=False)
