"""
Agentic RAG pipeline.

Four steps:
  1. Planner   - turns the user question into a retrieval query.
  2. Retriever - fetches relevant chunks from Chroma.
  3. Generator - calls Gemini with the retrieved context.
  4. Validator - checks the answer is grounded in the context.

Main entry point: run_agent()
"""
from __future__ import annotations

import textwrap

from langchain.schema import Document, HumanMessage, SystemMessage

from app.chroma_store import similarity_search
from app.gemini_llm import get_llm

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a helpful knowledge assistant.
    Answer the user's question strictly using the provided context.
    If the answer cannot be found in the context, say:
    "I could not find relevant information in the uploaded documents."
    Do NOT hallucinate or make up information.
    """
)

_CONTEXT_TEMPLATE = textwrap.dedent(
    """\
    === Retrieved Context ===
    {context}
    ========================

    User question: {question}
    """
)


def _plan(question: str) -> str:
    """Planner step: returns the question as-is to use as the retrieval query."""
    return question.strip()


def _retrieve(query: str, k: int = 5) -> list[Document]:
    """Retriever step: fetch the top-k relevant chunks from the vector store."""
    return similarity_search(query, k=k)


def _generate(question: str, docs: list[Document]) -> str:
    """Generator step: call Gemini with the retrieved context."""
    if not docs:
        return (
            "I could not find relevant information in the uploaded documents."
        )

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
    prompt = _CONTEXT_TEMPLATE.format(context=context, question=question)

    llm = get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content


def _validate(answer: str, docs: list[Document]) -> str:
    """
    Validator step: checks that the answer shares words with the retrieved docs.
    Adds a note if grounding looks weak.
    """
    if not docs:
        return answer

    combined_context = " ".join(doc.page_content.lower() for doc in docs)
    answer_words = set(answer.lower().split())
    context_words = set(combined_context.split())

    overlap = answer_words & context_words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "of", "and", "or", "for", "with", "this", "that", "it", "be", "i",
        "not", "found", "could", "did",
    }
    meaningful_overlap = overlap - stop_words

    if len(meaningful_overlap) < 3 and "could not find" not in answer.lower():
        answer += (
            "\n\n*Note: The answer may not be fully grounded in the "
            "retrieved documents. Please verify.*"
        )
    return answer


def run_agent(question: str, k: int = 5) -> dict:
    """
    Run the full agentic RAG pipeline for a user question.

    Returns a dict with:
        - query   : the retrieval query used
        - sources : list of source file names
        - answer  : the final answer string
    """
    query = _plan(question)
    docs = _retrieve(query, k=k)
    raw_answer = _generate(question, docs)
    final_answer = _validate(raw_answer, docs)

    sources = list(
        dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs)
    )

    return {
        "query": query,
        "sources": sources,
        "answer": final_answer,
    }
