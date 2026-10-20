"""
validation.py — Pydantic Schemas for API Request/Response Validation
────────────────────────────────────────────────────────────────────
Defines typed request and response schemas for the Multimodal RAG API endpoints.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for POST /query endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        description="User question to answer using retrieved multimodal context.",
        example="Explain the characteristics of the PN junction diode.",
    )
    model: str = Field(
        default="gemini",
        description="Generation model to use: 'gemini' (Gemini 2.0 Flash) or 'groq' (Groq Llama 3.3).",
        example="gemini",
    )
    filter_document: Optional[str] = Field(
        default=None,
        description="Optional: Restrict retrieval to a specific document name.",
        example="ELECTRONIC DEVICES AND CIRCUITS.pdf",
    )
    filter_page: Optional[int] = Field(
        default=None,
        description="Optional: Restrict retrieval to a specific page number (1-indexed).",
        example=5,
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded image attached by the user with the query.",
    )
    limit: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of context pages to retrieve from vector store.",
    )


class ContextSource(BaseModel):
    """Metadata and snippet of a retrieved context chunk."""
    key: str = Field(..., description="Unique vector ID in the collection.")
    distance: float = Field(..., description="Cosine distance score (lower is more similar).")
    document_name: str = Field(..., description="Source document file name.")
    page_number: int = Field(..., description="Page number of the source chunk (1-indexed).")
    total_pages: int = Field(default=0, description="Total pages in the source document.")
    text_snippet: str = Field(default="", description="Extracted text preview from the page.")
    image_b64: Optional[str] = Field(default=None, description="Base64-encoded rendered page image.")


class QueryResponse(BaseModel):
    """Response returned by POST /query endpoint."""
    answer: str = Field(..., description="Generated answer from the vision/text LLM.")
    sources: List[ContextSource] = Field(default_factory=list, description="Retrieved context sources.")
    model_used: str = Field(..., description="The model that produced the answer.")


class DocumentInfo(BaseModel):
    """Metadata summary of an indexed document."""
    document_name: str = Field(..., description="Name of the document file.")
    page_count: Optional[int] = Field(default=None, description="Number of indexed pages.")


class HealthResponse(BaseModel):
    """Response returned by GET /health endpoint."""
    status: str = Field(..., description="System health status ('healthy' or 'degraded').")
    api_key_configured: bool = Field(..., description="Whether GEMINI_API_KEY is configured.")
    total_vectors: int = Field(..., description="Total vector count in vector store.")
    indexed_documents: List[str] = Field(default_factory=list, description="List of unique indexed document names.")
    collection_name: str = Field(..., description="Current vector collection name.")
    persist_dir: str = Field(..., description="Path to local vector storage.")
