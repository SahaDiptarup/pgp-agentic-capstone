"""
Document ingestion pipeline.

Parses uploaded files, splits the text into chunks, embeds the chunks, and
stores them in the Chroma vector database.
"""
from __future__ import annotations

import os


from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.utils.parsers import parse_file
from app.chroma_store import add_documents

# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    length_function=len,
)


def ingest_file(filename: str, content: bytes) -> int:
    """
    Parse *content*, chunk the text, and index it into the vector store.

    Parameters
    ----------
    filename :
        Original file name (used to detect format).
    content :
        Raw file bytes.

    Returns
    -------
    int
        Number of chunks indexed.

    Raises
    ------
    ValueError
        Propagated from the parser when the file type is unsupported.
    """
    raw_text = parse_file(filename, content)
    if not raw_text.strip():
        return 0

    chunks = _splitter.split_text(raw_text)
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    add_documents(chunks, metadatas)
    return len(chunks)


def ingest_files(files: list[tuple[str, bytes]]) -> dict:
    """
    Ingest a list of (filename, bytes) pairs.

    Returns a summary dict mapping each filename to the number of chunks
    indexed, or an error message string on failure.
    """
    results: dict = {}
    for filename, content in files:
        try:
            n = ingest_file(filename, content)
            results[filename] = n
        except Exception as exc:  # noqa: BLE001
            results[filename] = f"ERROR: {exc}"
    return results
