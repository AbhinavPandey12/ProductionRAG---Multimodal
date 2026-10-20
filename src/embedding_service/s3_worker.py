"""
s3_worker.py  (repurposed: local file-watcher ingestion daemon)
───────────────────────────────────────────────────────────────
Watches the data/ directory for new PDF files and automatically ingests them
into ChromaDB. Replaces the original SQS-based S3 event worker.

Usage:
    python src/embedding_service/s3_worker.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

from src.embedding_service.document_processor import process_pdf
from src.embedding_service.embedder import embed_page
from src.embedding_service.s3_vector_manager import (
    collection_stats,
    get_or_create_collection,
    upsert_vectors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("s3_worker")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "rag_documents")
WATCH_INTERVAL_SEC = 10  # poll every 10 seconds


def _load_processed_registry() -> set[str]:
    registry_path = DATA_DIR / ".processed_files.json"
    if registry_path.exists() and registry_path.stat().st_size > 0:
        try:
            return set(json.loads(registry_path.read_text()))
        except json.JSONDecodeError:
            pass
    return set()


def _save_processed_registry(processed: set[str]) -> None:
    registry_path = DATA_DIR / ".processed_files.json"
    registry_path.write_text(json.dumps(sorted(processed), indent=2))


def ingest_pdf(pdf_path: Path, collection) -> int:
    """Ingest a single PDF into ChromaDB. Returns number of pages upserted."""
    logger.info("Ingesting: %s", pdf_path.name)
    batch: list[dict] = []

    for chunk in process_pdf(pdf_path):
        embedding = embed_page(chunk["image_b64"], chunk["text_snippet"])
        chunk["embedding"] = embedding
        batch.append(chunk)

        # Upsert in batches of 10 to avoid memory spikes
        if len(batch) >= 10:
            upsert_vectors(collection, batch)
            batch.clear()

    if batch:
        upsert_vectors(collection, batch)

    logger.info("✅ Ingested %s", pdf_path.name)
    return 1


def run_watcher() -> None:
    """Main watch loop — polls data/ dir for new PDFs."""
    logger.info("🚀 File-watcher daemon started. Watching: %s", DATA_DIR)
    logger.info("   Press Ctrl+C to stop.")

    collection = get_or_create_collection(COLLECTION_NAME)
    processed: set[str] = _load_processed_registry()

    while True:
        new_pdfs = [
            p
            for p in DATA_DIR.glob("*.pdf")
            if p.name not in processed
        ]

        for pdf_path in new_pdfs:
            try:
                ingest_pdf(pdf_path, collection)
                processed.add(pdf_path.name)
                _save_processed_registry(processed)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", pdf_path.name, exc)

        if not new_pdfs:
            logger.debug("No new PDFs found. Sleeping %ds...", WATCH_INTERVAL_SEC)

        time.sleep(WATCH_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        run_watcher()
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user.")
