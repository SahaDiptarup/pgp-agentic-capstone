"""
Gemini LLM and Embeddings using Google Generative AI (no GCP needed).
"""
from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# default model, can be changed via env var
_DEFAULT_MODEL = "gemini-1.5-flash"


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)


def get_llm(temperature: float = 0.2, max_output_tokens: int = 1024) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        model=_model_name(),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return an embeddings model instance."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
