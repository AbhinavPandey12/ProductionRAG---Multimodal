"""
app.py — FastAPI Generation & API Service for Multimodal RAG
────────────────────────────────────────────────────────────
Serves:
  • Web UI static files at /ui/ (and redirects / to /ui/)
  • GET  /health                      — cluster & vector health
  • GET  /documents                   — list all indexed documents
  • POST /query                       — multimodal question answering
  • POST /upload                      — ingest uploaded PDF documents
  • DELETE /documents/{document_name} — remove document vectors
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.embedding_service.document_processor import (
    process_pdf,
    render_pdf_page_to_base64,
)
from src.embedding_service.embedder import embed_page, embed_query
from src.embedding_service.gemini_rag_llm import generate_answer as gemini_generate_answer
from src.embedding_service.s3_vector_manager import (
    collection_stats,
    delete_document_vectors,
    get_or_create_collection,
    list_indexed_documents,
    query_vectors,
    upsert_vectors,
)

logger = logging.getLogger("rag_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_REGISTRY = DATA_DIR / ".processed_files.json"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "rag_documents")
UI_DIR = PROJECT_ROOT / "ui"

app = FastAPI(
    title="Multimodal RAG API",
    description="Multimodal Document Intelligence with Gemini Embeddings & Vision",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Registry helpers ──────────────────────────────────────────────────────────

def _load_registry() -> set[str]:
    if PROCESSED_REGISTRY.exists() and PROCESSED_REGISTRY.stat().st_size > 0:
        try:
            return set(json.loads(PROCESSED_REGISTRY.read_text()))
        except json.JSONDecodeError:
            pass
    return set()


def _save_registry(processed: set[str]) -> None:
    PROCESSED_REGISTRY.write_text(json.dumps(sorted(processed), indent=2))


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    limit: int = Field(default=4, ge=1, le=20)
    model: Optional[str] = "gemini"
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    image_b64: Optional[str] = None


class SourceCitation(BaseModel):
    document_name: str
    page_number: int
    total_pages: int
    distance: float
    text_snippet: str
    image_b64: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    pages_retrieved: int
    sources: list[SourceCitation]
    images: list[str] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health_check():
    """Returns vector database status and list of indexed documents."""
    collection = get_or_create_collection(COLLECTION_NAME)
    stats = collection_stats(collection)
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    return {
        "status": "healthy" if has_gemini else "degraded",
        "api_key_configured": has_gemini,
        "total_vectors": stats["total_vectors"],
        "indexed_documents": stats["indexed_documents"],
        "collection_name": stats["collection_name"],
        "persist_dir": stats["persist_dir"],
    }


@app.get("/documents")
def list_documents():
    collection = get_or_create_collection(COLLECTION_NAME)
    docs = list_indexed_documents(collection)
    return {"documents": docs, "count": len(docs)}


@app.delete("/documents/{document_name}")
def delete_document(document_name: str):
    collection = get_or_create_collection(COLLECTION_NAME)
    deleted = delete_document_vectors(collection, document_name)

    registry = _load_registry()
    if document_name in registry:
        registry.discard(document_name)
        _save_registry(registry)

    return {
        "status": "success",
        "document_name": document_name,
        "vectors_deleted": deleted,
    }


@app.options("/query")
def options_query():
    return {}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Search vector collection for relevant pages and generate multimodal answer.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY is not configured in .env. Please set your Gemini API key.",
        )

    collection = get_or_create_collection(COLLECTION_NAME)
    stats = collection_stats(collection)
    if stats["total_vectors"] == 0:
        return QueryResponse(
            question=request.question,
            answer="No documents have been indexed yet. Please upload or ingest PDF files first.",
            model=request.model or "gemini",
            pages_retrieved=0,
            sources=[],
            images=[],
        )

    # 1. Embed user query (multimodal if user attached an image)
    try:
        query_vector = embed_query(request.question, image_b64=request.image_b64)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {exc}",
        )

    # 2. Retrieve top-K matching pages
    raw_results = query_vectors(
        collection=collection,
        query_embedding=query_vector,
        n_results=request.limit,
        filter_document=request.document_name,
        filter_page=request.page_number,
    )

    # 3. Enrich sources with on-demand rendered page base64 images
    context_chunks = []
    source_citations = []
    images = []

    for item in raw_results:
        meta = item.get("metadata", {})
        doc_name = meta.get("document_name", "")
        page_num = int(meta.get("page_number", 1))
        tot_pages = int(meta.get("total_pages", 1))

        # Check if local PDF exists to render visual page
        pdf_path = DATA_DIR / doc_name
        page_b64 = None
        if pdf_path.exists():
            page_b64 = render_pdf_page_to_base64(pdf_path, page_num)

        chunk_data = {
            "metadata": meta,
            "text_snippet": item.get("text_snippet", ""),
            "distance": item.get("distance", 0.0),
            "image_b64": page_b64 or "",
        }
        context_chunks.append(chunk_data)

        source_citations.append(
            SourceCitation(
                document_name=doc_name or "Unknown",
                page_number=page_num,
                total_pages=tot_pages,
                distance=item.get("distance", 0.0),
                text_snippet=item.get("text_snippet", ""),
                image_b64=page_b64,
            )
        )
        if page_b64:
            images.append(page_b64)

    # 4. Generate answer using LLM
    selected_model = (request.model or "gemini").lower()
    answer_text = ""

    if selected_model == "groq":
        try:
            from src.embedding_service.groq_rag_llm import generate_answer as groq_generate_answer
            answer_text = groq_generate_answer(request.question, context_chunks)
        except Exception as exc:
            logger.warning("Groq generation failed (%s), falling back to Gemini", exc)
            answer_text = gemini_generate_answer(request.question, context_chunks)
    elif selected_model == "bedrock":
        try:
            from src.generation_service.bedrock_rag_llm import generate_answer as bedrock_generate_answer
            answer_text = bedrock_generate_answer(request.question, context_chunks)
        except Exception as exc:
            logger.warning("Bedrock generation failed (%s), falling back to Gemini", exc)
            answer_text = gemini_generate_answer(request.question, context_chunks)
    else:
        try:
            answer_text = gemini_generate_answer(request.question, context_chunks)
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {exc}",
            )

    return QueryResponse(
        question=request.question,
        answer=answer_text,
        model=request.model or "gemini",
        pages_retrieved=len(source_citations),
        sources=source_citations,
        images=images,
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, extract pages, embed with Gemini, and index in ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported.",
        )

    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GEMINI_API_KEY is not set. Please add it to your .env file before ingesting.",
        )

    target_path = DATA_DIR / file.filename
    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # Ingest the PDF into ChromaDB
    collection = get_or_create_collection(COLLECTION_NAME)
    batch = []
    page_count = 0

    try:
        for chunk in process_pdf(target_path):
            embedding = embed_page(chunk["image_b64"], chunk["text_snippet"])
            chunk["embedding"] = embedding
            batch.append(chunk)
            page_count += 1

            if len(batch) >= 5:
                upsert_vectors(collection, batch)
                batch.clear()

        if batch:
            upsert_vectors(collection, batch)

        # Update registry
        registry = _load_registry()
        registry.add(file.filename)
        _save_registry(registry)

        return {
            "status": "success",
            "document_name": file.filename,
            "pages_ingested": page_count,
            "file_size": target_path.stat().st_size,
        }
    except Exception as exc:
        logger.error("Failed during ingestion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


# Mount UI static folder
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)