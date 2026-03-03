"""
Chroma vector store wrapper.

Provides helper functions to build/load a Chroma collection from text chunks
and to run similarity searches against it.
"""
from __future__ import annotations

import os


import chromadb
from chromadb import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from app.gemini_llm import get_embeddings

_COLLECTION_NAME = "enterprise_docs"


def _get_client() -> chromadb.ClientAPI:
    """Return a Chroma client (server mode if CHROMA_HOST is set, else local)."""
    host = os.environ.get("CHROMA_HOST")
    if host:
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        return chromadb.HttpClient(host=host, port=port)
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_vector_store() -> Chroma:
    """Load (or create) the LangChain Chroma vector store."""
    client = _get_client()
    embeddings = get_embeddings()
    return Chroma(
        client=client,
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
    )


def add_documents(chunks: list[str], metadatas: list[dict] | None = None) -> None:
    """
    Embed *chunks* and add them to the Chroma collection.

    Parameters
    ----------
    chunks :
        List of text strings to embed.
    metadatas :
        Optional list of metadata dicts (one per chunk).
    """
    store = get_vector_store()
    docs = [
        Document(page_content=chunk, metadata=metadatas[i] if metadatas else {})
        for i, chunk in enumerate(chunks)
    ]
    store.add_documents(docs)


def similarity_search(query: str, k: int = 5) -> list[Document]:
    """
    Return the *k* most relevant document chunks for *query*.

    Parameters
    ----------
    query :
        Natural-language query string.
    k :
        Number of results to return.
    """
    store = get_vector_store()
    return store.similarity_search(query, k=k)
