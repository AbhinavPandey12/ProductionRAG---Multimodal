"""
delete_vectors.py
─────────────────
CLI utility to delete vectors from ChromaDB by document name.

Usage:
    python src/embedding_service/delete_vectors.py "ELECTRONIC DEVICES AND CIRCUITS.pdf"
    python src/embedding_service/delete_vectors.py --all   # drops the entire collection
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from src.embedding_service.s3_vector_manager import (
    collection_stats,
    delete_document_vectors,
    get_or_create_collection,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "rag_documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete vectors from ChromaDB.")
    parser.add_argument(
        "document_name",
        nargs="?",
        help="Document name to delete (e.g. 'report.pdf').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL vectors in the collection.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit.",
    )
    args = parser.parse_args()

    collection = get_or_create_collection(COLLECTION_NAME)

    if args.stats:
        stats = collection_stats(collection)
        print("\n📊 Collection Statistics")
        print(f"  Total vectors : {stats['total_vectors']}")
        print(f"  Collection    : {stats['collection_name']}")
        print(f"  Persist dir   : {stats['persist_dir']}")
        print(f"  Documents     : {', '.join(stats['indexed_documents']) or 'none'}")
        return

    if args.all:
        confirm = input(
            f"⚠️  Delete ALL vectors from '{COLLECTION_NAME}'? [yes/N] "
        ).strip()
        if confirm.lower() == "yes":
            # ChromaDB: delete all by getting all IDs
            all_ids = collection.get(include=[])["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
                print(f"✅ Deleted {len(all_ids)} vectors from '{COLLECTION_NAME}'.")
            else:
                print("Collection is already empty.")
        else:
            print("Aborted.")
        return

    if args.document_name:
        deleted = delete_document_vectors(collection, args.document_name)
        if deleted:
            print(f"✅ Deleted {deleted} vectors for '{args.document_name}'.")
        else:
            print(f"⚠️  No vectors found for '{args.document_name}'.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
