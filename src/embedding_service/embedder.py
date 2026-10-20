"""
embedder.py
───────────
Generates multimodal embeddings using Google Gemini Embedding API.

Uses: gemini-embedding-exp-03-07 (supports image + text inputs)
Output: 3072-dimensional float32 vectors
"""

from __future__ import annotations

import base64
import logging
import os
import time
import warnings
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import google.generativeai as genai
    from google.generativeai import types as genai_types

logger = logging.getLogger(__name__)

# Model to use for embeddings
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
EMBED_TASK_TYPE = "RETRIEVAL_DOCUMENT"


def _configure_client() -> None:
    """Configure the Gemini client with the API key from env."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please add it to your .env file."
        )
    genai.configure(api_key=api_key)


def embed_page(
    image_b64: str,
    text_snippet: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> list[float]:
    """
    Generate a multimodal embedding for a single PDF page.

    Args:
        image_b64:    base64-encoded PNG of the rendered page
        text_snippet: extracted text from the page (context hint)
        retries:      number of retry attempts on API failure
        backoff:      initial backoff time in seconds (doubles each retry)

    Returns:
        List of floats representing the embedding vector.
    """
    _configure_client()

    # Build multimodal content: image + text
    content = [
        {"mime_type": "image/png", "data": image_b64},
        text_snippet or "",
    ]

    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=content,
                task_type=EMBED_TASK_TYPE,
            )
            return result["embedding"]
        except Exception as exc:
            if attempt < retries - 1:
                wait = backoff * (2**attempt)
                logger.warning(
                    "Embedding attempt %d failed (%s). Retrying in %.1fs...",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("All embedding attempts failed for page chunk.")
                raise

    return []  # unreachable, but keeps type checkers happy


def embed_query(query_text: str, image_b64: Optional[str] = None) -> list[float]:
    """
    Generate an embedding for a user query (optionally multimodal).

    Args:
        query_text: the question / search string
        image_b64:  optional base64 image attached by the user

    Returns:
        List of floats representing the query embedding vector.
    """
    _configure_client()

    if image_b64:
        content = [
            {"mime_type": "image/png", "data": image_b64},
            query_text,
        ]
    else:
        content = query_text

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=content,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]


class GeminiEmbedder:
    """Wrapper class providing object-oriented interface for embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def embed_page(self, image_b64: str, text_snippet: str) -> list[float]:
        return embed_page(image_b64, text_snippet)

    def embed_query(self, query_text: str, image_b64: Optional[str] = None) -> list[float]:
        return embed_query(query_text, image_b64)

