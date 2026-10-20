"""
generation_service
──────────────────
FastAPI API server, CLI query tools, response schemas, and generation models (Gemini, Groq, Bedrock).
"""

from src.generation_service.app import app
from src.generation_service.bedrock_rag_llm import generate_answer as bedrock_generate_answer
from src.generation_service.gemini_rag_llm import generate_answer as gemini_generate_answer
from src.generation_service.validation import (
    ContextSource,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

__all__ = [
    "app",
    "gemini_generate_answer",
    "bedrock_generate_answer",
    "QueryRequest",
    "QueryResponse",
    "ContextSource",
    "HealthResponse",
]
