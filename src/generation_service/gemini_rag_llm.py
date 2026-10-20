"""
gemini_rag_llm.py — Multimodal Generation Module (generation_service)
─────────────────────────────────────────────────────────────────────
Generates citation-backed answers using Google Gemini 2.0 Flash vision,
incorporating both rendered page images and text snippets into the prompt context.
"""

from __future__ import annotations

from typing import List, Optional
from src.embedding_service.gemini_rag_llm import generate_answer

__all__ = ["generate_answer"]
