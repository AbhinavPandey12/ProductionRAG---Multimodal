"""
groq_rag_llm.py
───────────────
Optional Groq-based text-only generation (fallback / alternative).
Uses llama-3.3-70b-versatile via the Groq API.

Set GROQ_API_KEY in your .env to enable.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a structured text RAG prompt."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        doc = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        snippet = chunk.get("text_snippet", "")
        context_parts.append(
            f"[Source {i}] {doc} — Page {page}\n{snippet}"
        )
    context_str = "\n\n".join(context_parts)

    return f"""You are an expert document analyst. Use the retrieved context to answer the question.

CONTEXT:
{context_str}

QUESTION: {question}

Provide a thorough, citation-backed answer. If context is insufficient, say so."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generate a text-only RAG answer using Groq (llama-3.3-70b).

    Requires GROQ_API_KEY in environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Either set it or use gemini_rag_llm instead."
        )

    try:
        from groq import Groq  # type: ignore
    except ImportError:
        raise ImportError(
            "groq package is not installed. Run: pip install groq"
        )

    client = Groq(api_key=api_key)
    prompt = _build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant specialized in analyzing "
                    "documents and providing accurate, citation-backed answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.2,
    )

    return response.choices[0].message.content or ""
