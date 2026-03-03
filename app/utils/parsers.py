"""
File parsers for PDF, TXT, CSV, and Excel documents.
Each function returns the extracted text as a string.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union


def parse_pdf(file: Union[bytes, str, Path]) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    if isinstance(file, (str, Path)):
        reader = PdfReader(str(file))
    else:
        reader = PdfReader(io.BytesIO(file))

    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_txt(file: Union[bytes, str, Path]) -> str:
    """Read text from a plain-text file."""
    if isinstance(file, (str, Path)):
        return Path(file).read_text(encoding="utf-8", errors="replace")
    return file.decode("utf-8", errors="replace")


def parse_csv(file: Union[bytes, str, Path]) -> str:
    """Convert a CSV file to a plain-text string."""
    import pandas as pd

    if isinstance(file, (str, Path)):
        df = pd.read_csv(str(file))
    else:
        df = pd.read_csv(io.BytesIO(file))

    return df.to_string(index=False)


def parse_excel(file: Union[bytes, str, Path]) -> str:
    """Convert all sheets of an Excel workbook to plain text."""
    import pandas as pd

    if isinstance(file, (str, Path)):
        xl = pd.ExcelFile(str(file))
    else:
        xl = pd.ExcelFile(io.BytesIO(file))

    sheets_text: list[str] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        sheets_text.append(f"=== Sheet: {sheet} ===\n{df.to_string(index=False)}")
    return "\n\n".join(sheets_text)


def parse_file(filename: str, content: bytes) -> str:
    """Pick the right parser based on file extension and return extracted text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(content)
    if ext == ".txt":
        return parse_txt(content)
    if ext == ".csv":
        return parse_csv(content)
    if ext in (".xls", ".xlsx"):
        return parse_excel(content)
    raise ValueError(f"Unsupported file type: {ext!r}")
