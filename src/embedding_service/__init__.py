"""
embedding_service
─────────────────
Document processing, multimodal embedding generation, vector store operations
(ChromaDB and Amazon S3 Vectors abstraction), and background ingestion worker.
"""

from src.embedding_service.document_processor import (
    SUPPORTED_EXTENSIONS,
    process_pdf,
    render_pdf_page_to_base64,
    scan_data_dir,
)
from src.embedding_service.embedder import GeminiEmbedder, embed_page, embed_query
from src.embedding_service.s3_vector_manager import S3VectorManager

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "process_pdf",
    "render_pdf_page_to_base64",
    "scan_data_dir",
    "GeminiEmbedder",
    "embed_page",
    "embed_query",
    "S3VectorManager",
]
