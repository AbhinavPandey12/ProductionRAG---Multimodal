"""
main.py — Multimodal RAG Root Entrypoint
────────────────────────────────────────
Usage:
    python main.py                    # Start FastAPI server & Web UI (default)
    python main.py --server           # Start FastAPI server & Web UI
    python main.py --ingest           # Ingest all PDFs in data/ into vector store
    python main.py --query "..."      # Run CLI query
    python main.py --stats            # Show vector store collection stats
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def show_stats():
    from src.embedding_service.s3_vector_manager import collection_stats, get_or_create_collection
    collection_name = os.environ.get("CHROMA_COLLECTION", "rag_documents")
    coll = get_or_create_collection(collection_name)
    stats = collection_stats(coll)
    print("\n" + "=" * 50)
    print("  Vector Store Status")
    print("=" * 50)
    print(f"  Collection Name  : {stats['collection_name']}")
    print(f"  Total Vectors    : {stats['total_vectors']}")
    print(f"  Indexed Documents: {', '.join(stats['indexed_documents']) or 'None'}")
    print(f"  Persist Directory: {stats['persist_dir']}")
    print("=" * 50 + "\n")


def run_ingest(force: bool = False):
    from src.embedding_service.main import ingest_documents
    ingest_documents(force=force)


def run_server(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    print("\n" + "=" * 60)
    print("  🚀 Starting Multimodal RAG Server & Web UI")
    print("=" * 60)
    print(f"  Web UI URL  : http://{host}:{port}/ui/")
    print(f"  API Docs    : http://{host}:{port}/docs")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    uvicorn.run("src.generation_service.app:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal RAG with Google Gemini & ChromaDB"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI backend server and Web UI (default).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest all PDFs in data/ into vector store.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of already processed documents.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display vector store stats and indexed documents.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a question directly from the CLI.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.ingest:
        run_ingest(force=args.force)
    elif args.query:
        from src.generation_service.query import main as run_query
        sys.argv = ["query.py", args.query]
        run_query()
    else:
        # Default behavior: run server
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()