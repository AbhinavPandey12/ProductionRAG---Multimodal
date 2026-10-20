"""
query.py — Multimodal CLI Query Tool
────────────────────────────────────
Query indexed documents directly from terminal with optional metadata filters.
Usage:
    python src/generation_service/query.py "What are the main components of a circuit?"
    python src/generation_service/query.py "Explain viscosity" --filter-doc "PROPERTIES OF FLUIDS.pdf"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.embedding_service.document_processor import render_pdf_page_to_base64
from src.embedding_service.embedder import embed_query
from src.embedding_service.gemini_rag_llm import generate_answer
from src.embedding_service.s3_vector_manager import (
    collection_stats,
    get_or_create_collection,
    query_vectors,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Multimodal RAG pipeline."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Question you want to ask about your documents.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["gemini", "groq", "bedrock"],
        help="Inference model: 'gemini' (Gemini 2.0 Flash), 'groq' (Groq Llama), or 'bedrock' (Amazon Bedrock).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of context pages to retrieve (default: 4).",
    )
    parser.add_argument(
        "--filter-doc",
        type=str,
        default=None,
        help="Filter results to a specific document name.",
    )
    parser.add_argument(
        "--filter-page",
        type=int,
        default=None,
        help="Filter results to a specific page number.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    collection_name = os.environ.get("CHROMA_COLLECTION", "rag_documents")

    print("=" * 60)
    print("  Multimodal RAG CLI Query")
    print("=" * 60)
    print(f"Question:    {args.question}")
    print(f"Collection:  {collection_name}")
    if args.filter_doc:
        print(f"Filter Doc:  {args.filter_doc}")
    if args.filter_page:
        print(f"Filter Page: {args.filter_page}")
    print("=" * 60)

    # 1. Connect to Vector Store
    collection = get_or_create_collection(collection_name)
    stats = collection_stats(collection)

    if stats["total_vectors"] == 0:
        print("\n⚠️ No documents have been indexed yet.")
        print("Run ingestion first: python src/embedding_service/main.py")
        sys.exit(1)

    # 2. Embed Query
    print("\n🔍 Embedding query with Gemini Multimodal Embedder...")
    try:
        query_vector = embed_query(args.question)
    except Exception as exc:
        print(f"✗ Failed to embed query: {exc}")
        sys.exit(1)

    # 3. Retrieve Context
    print(f"🔍 Retrieving top {args.limit} relevant page(s)...")
    results = query_vectors(
        collection=collection,
        query_embedding=query_vector,
        n_results=args.limit,
        filter_document=args.filter_doc,
        filter_page=args.filter_page,
    )

    if not results:
        print("⚠️ No matching context found.")
        sys.exit(0)

    # 4. Resolve Context Images
    data_dir = PROJECT_ROOT / "data"
    context_chunks = []
    print("\n📄 Retrieved Context Pages:")
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        doc = meta.get("document_name", "Unknown")
        page = meta.get("page_number", 1)
        dist = r.get("distance", 0.0)
        snippet = r.get("text_snippet", "")

        pdf_path = data_dir / doc
        img_b64 = None
        if pdf_path.exists():
            img_b64 = render_pdf_page_to_base64(pdf_path, page)

        context_chunks.append({
            "metadata": meta,
            "text_snippet": snippet,
            "distance": dist,
            "image_b64": img_b64 or "",
        })

        print(f"  [{i}] {doc} — Page {page} (dist: {dist})")
        if snippet:
            print(f"      Text: {snippet[:100]}...")

    # 5. Generate Answer
    print(f"\n🤖 Generating answer with {args.model}...")
    print("-" * 60)
    try:
        if args.model == "groq":
            from src.embedding_service.groq_rag_llm import generate_answer as groq_gen
            answer = groq_gen(args.question, context_chunks)
            print(answer)
        elif args.model == "bedrock":
            from src.generation_service.bedrock_rag_llm import generate_answer as bedrock_gen
            answer = bedrock_gen(args.question, context_chunks)
            print(answer)
        else:
            generate_answer(args.question, context_chunks, stream=True)
    except Exception as exc:
        print(f"✗ Generation failed: {exc}")
        sys.exit(1)
    print("-" * 60 + "\n")


if __name__ == "__main__":
    main()