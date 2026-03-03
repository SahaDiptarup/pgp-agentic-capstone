"""
Gemini LLM and Embeddings via Google Vertex AI (LangChain wrappers).
"""
from __future__ import annotations

import os

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# ---------------------------------------------------------------------------
# Defaults (overridden by environment variables)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = "gemini-1.0-pro"
_DEFAULT_LOCATION = "us-central1"


def _project() -> str:
    project = os.environ.get("GCP_PROJECT", "")
    if not project:
        raise EnvironmentError(
            "GCP_PROJECT environment variable is required for Vertex AI."
        )
    return project


def _location() -> str:
    return os.environ.get("GCP_LOCATION", _DEFAULT_LOCATION)


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)


def get_llm(temperature: float = 0.2, max_output_tokens: int = 1024) -> ChatVertexAI:
    """
    Return a LangChain ChatVertexAI instance backed by Gemini.

    Parameters
    ----------
    temperature :
        Sampling temperature (0 = deterministic).
    max_output_tokens :
        Maximum number of tokens in the model response.
    """
    return ChatVertexAI(
        model_name=_model_name(),
        project=_project(),
        location=_location(),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def get_embeddings() -> VertexAIEmbeddings:
    """
    Return a LangChain VertexAIEmbeddings instance for the *textembedding-gecko*
    model (default Vertex AI text embedding model).
    """
    return VertexAIEmbeddings(
        model_name="textembedding-gecko@003",
        project=_project(),
        location=_location(),
    )
