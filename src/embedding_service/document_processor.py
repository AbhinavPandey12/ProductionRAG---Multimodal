"""
document_processor.py
─────────────────────
Converts PDFs in the data/ directory into page-level chunks:
  • A rendered PNG image of the page (base64 encoded for Gemini vision)
  • A plain-text snippet extracted from the page
  • Rich metadata (document name, page number, file size, ingestion time)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

try:
    import pymupdf as fitz  # Modern PyMuPDF import
except ImportError:
    import fitz

logger = logging.getLogger(__name__)

# Page render resolution (DPI → pixels)
RENDER_DPI = 150  # good balance between quality and token cost
RENDER_MATRIX = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)


def _render_page_to_b64(page: fitz.Page) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    pixmap = page.get_pixmap(matrix=RENDER_MATRIX, colorspace=fitz.csRGB)
    png_bytes = pixmap.tobytes("png")
    return base64.b64encode(png_bytes).decode("utf-8")


def _extract_text_snippet(page: fitz.Page, max_chars: int = 500) -> str:
    """Extract plain text from a page, truncated to max_chars."""
    text = page.get_text("text").strip()
    return text[:max_chars] if len(text) > max_chars else text


def process_pdf(pdf_path: Path) -> Generator[dict, None, None]:
    """
    Yield one chunk dict per page of the given PDF.

    Each dict contains:
        key          : unique vector key  "<stem>_page_<n>"
        image_b64    : base64 PNG of the rendered page
        text_snippet : first 500 chars of page text
        metadata     : filterable metadata dict
    """
    stat = pdf_path.stat()
    doc_name = pdf_path.name
    file_size = stat.st_size
    ingested_at = datetime.now(timezone.utc).isoformat()

    logger.info("Processing PDF: %s", doc_name)

    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            image_b64 = _render_page_to_b64(page)
            text_snippet = _extract_text_snippet(page)

            # Build a safe vector key (no slashes, no spaces)
            stem = pdf_path.stem.replace(" ", "_").replace("/", "_")
            key = f"{stem}_page_{page_num:04d}"

            yield {
                "key": key,
                "image_b64": image_b64,
                "text_snippet": text_snippet,
                "metadata": {
                    "document_name": doc_name,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "file_size": file_size,
                    "ingested_at": ingested_at,
                    "text_snippet": text_snippet,
                },
            }

    logger.info("Finished processing %s — %d pages", doc_name, total_pages)


def scan_data_dir(data_dir: Path) -> list[Path]:
    """Return all PDF files found under data_dir."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    logger.info("Found %d PDF(s) in %s", len(pdfs), data_dir)
    return pdfs


SUPPORTED_EXTENSIONS = {".pdf"}


def render_pdf_page_to_base64(pdf_path: str | Path, page_number: int) -> str | None:
    """
    Render a single page from a PDF file to base64 PNG.
    page_number is 1-indexed.
    """
    path = Path(pdf_path)
    if not path.exists():
        return None
    try:
        with fitz.open(str(path)) as doc:
            idx = page_number - 1
            if 0 <= idx < len(doc):
                return _render_page_to_b64(doc[idx])
    except Exception as exc:
        logger.warning("Failed to render page %d of %s: %s", page_number, path.name, exc)
    return None

