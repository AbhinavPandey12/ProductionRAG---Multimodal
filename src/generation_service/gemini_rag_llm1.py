"""
gemini_rag_llm1.py — Alternative/Legacy Gemini Generation Shim
─────────────────────────────────────────────────────────────
Maintained for backwards compatibility. Delegates to the unified
Gemini multimodal generation pipeline in src.embedding_service.gemini_rag_llm.
"""

from __future__ import annotations

from typing import Any, List, Optional
from src.embedding_service.gemini_rag_llm import generate_answer

__all__ = ["generate_answer"]
