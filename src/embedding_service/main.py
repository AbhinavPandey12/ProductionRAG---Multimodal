import os
import sys
import glob
import json
import base64
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

from src.embedding_service.document_processor import (
    iter_pdf_pages,
    image_to_base64,
    render_pdf_page_to_base64,
)
from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.s3_vector_manager import S3VectorManager
from src.generation_service.gemini_rag_llm import GeminiRAG


def ingest_local_documents(
    data_dir: str,
    vector_manager: S3VectorManager,
    embedder: GeminiEmbedder,
    batch_size: int = 5,
    target_file: str | None = None,
    skip_existing: bool = False,
):
    """
    Scans the data directory for PDFs, extracts pages, computes Gemini embedding 2 vectors,
    and uploads vectors along with rich filterable metadata to Amazon S3 Vectors.
    Supports targeting a specific file and skipping already processed files.
    """
    processed_cache_path = os.path.join(data_dir, ".processed_files.json")
    processed_cache = {}
    if os.path.exists(processed_cache_path):
        try:
            with open(processed_cache_path, "r") as f:
                processed_cache = json.load(f)
        except Exception:
            processed_cache = {}

    if target_file:
        if os.path.isabs(target_file) and os.path.exists(target_file):
            pdf_files = [target_file]
        else:
            cand = os.path.join(data_dir, os.path.basename(target_file))
            if os.path.exists(cand):
                pdf_files = [cand]
            else:
                matches = glob.glob(os.path.join(data_dir, f"*{target_file}*"))
                pdf_files = sorted(matches)
        if not pdf_files:
            print(f"❌ Target file '{target_file}' not found in '{data_dir}'")
            return
    else:
        pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

    if not pdf_files:
        print(f"No PDF documents found in data directory: {data_dir}")
        return

    print(f"\n📂 Found {len(pdf_files)} PDF document(s) to process:")
    for pf in pdf_files:
        print(f"  - {os.path.basename(pf)} ({os.path.getsize(pf) / 1024:.1f} KB)")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        file_size = os.path.getsize(pdf_path)
        mtime = os.path.getmtime(pdf_path)

        if skip_existing and filename in processed_cache:
            prev = processed_cache[filename]
            if prev.get("size") == file_size:
                print(f"⏭️ Skipping '{filename}' (already ingested according to .processed_files.json).")
                continue

        print(f"\n" + "=" * 60)
        print(f"🚀 Processing: {filename}")
        print("=" * 60)

        batch_embeddings = []
        batch_images = []
        batch_metadata = []
        total_ingested = 0

        for page_num, total_pages, page_img, page_text in iter_pdf_pages(pdf_path, extract_text=True):
            print(f"Embedding page {page_num}/{total_pages} for {filename}...")
            b64_str = image_to_base64(page_img)
            emb = embedder.embed_image(page_img)

            meta = {
                "document_name": filename,
                "page_number": page_num,
                "total_pages": total_pages,
                "source_uri": pdf_path,
                "source_type": "local",
                "file_size": file_size,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "text_snippet": page_text[:1200] if page_text else "",
            }

            batch_embeddings.append(emb)
            batch_images.append(b64_str)
            batch_metadata.append(meta)
            total_ingested += 1

            if len(batch_embeddings) >= batch_size or page_num == total_pages:
                vector_manager.insert_image_embeddings(
                    embeddings=batch_embeddings,
                    base64_images=batch_images,
                    metadata_list=batch_metadata,
                    batch_size=batch_size,
                )
                print(f"✅ Upserted {len(batch_embeddings)} vectors to S3 Vectors (Progress: {page_num}/{total_pages})")
                batch_embeddings = []
                batch_images = []
                batch_metadata = []

        print(f"🎉 Completed ingestion for {filename} ({total_ingested} pages) into S3 Vectors.")

        # Update cache
        processed_cache[filename] = {
            "mtime": mtime,
            "size": file_size,
            "pages": total_ingested,
            "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            with open(processed_cache_path, "w") as f:
                json.dump(processed_cache, f, indent=4)
        except Exception as e:
            print(f"Note: Could not update .processed_files.json: {e}")



def interactive_query_loop(vector_manager: S3VectorManager, embedder: GeminiEmbedder, data_dir: str):
    """
    Interactive loop demonstrating similarity search against Amazon S3 Vectors
    with optional metadata filtering.
    """
    rag_llm = GeminiRAG()

    print("\n" + "=" * 60)
    print("✨ Amazon S3 Vectors RAG system is ready!")
    print("Commands:")
    print("  - Type your question directly")
    print("  - Type 'filter:<doc_name>' before question to filter by document name (e.g. 'filter:ELECTRONIC DEVICES AND CIRCUITS.pdf explain PN junction diode')")
    print("  - Type 'exit' or 'quit' to stop")
    print("=" * 60)

    while True:
        try:
            raw_input = input("\nAsk a question: ").strip()
            if not raw_input:
                continue
            if raw_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            filter_expr = None
            question = raw_input

            # Optional inline metadata filtering syntax
            if raw_input.startswith("filter:"):
                parts = raw_input.split(" ", 1)
                doc_filter = parts[0].replace("filter:", "").strip()
                filter_expr = {"document_name": doc_filter}
                question = parts[1] if len(parts) > 1 else ""
                if not question:
                    print("Please specify a question after the filter tag.")
                    continue
                print(f"Applying metadata filter: document_name == '{doc_filter}'")

            print("Embedding query with Gemini embedding 2...")
            query_embedding = embedder.embed_text(question)

            print("Querying Amazon S3 Vectors...")
            results = vector_manager.search(query_embedding, limit=2, filter_expr=filter_expr)

            if not results:
                print("No relevant context found in S3 Vectors.")
                continue

            # Resolve page images from matched metadata
            context_images = []
            for hit in results:
                meta = hit.get("metadata", {})
                doc_name = meta.get("document_name")
                page_num = meta.get("page_number", 1)
                distance = hit.get("distance")

                print(f"\n📌 Matched Document: {doc_name} (Page {page_num}) | Distance: {distance}")
                if meta.get("text_snippet"):
                    print(f"   Snippet: {meta['text_snippet'][:150]}...")

                # Resolve image from local data folder or source
                doc_path = os.path.join(data_dir, doc_name) if doc_name else ""
                b64_img = None
                if os.path.exists(doc_path):
                    b64_img = render_pdf_page_to_base64(doc_path, page_num)

                if b64_img:
                    context_images.append(b64_img)

            if not context_images:
                print("Could not resolve page images for the matched documents.")
                continue

            print("\nGenerating answer via Gemini LLM...")
            answer = rag_llm.answer_question(question, context_images)

            print("\n" + "=" * 60)
            print("Answer:")
            print(answer)
            print("=" * 60)

            # Save reference image
            reference_path = os.path.join(PROJECT_ROOT, "reference_output.jpg")
            with open(reference_path, "wb") as f:
                f.write(base64.b64decode(context_images[0]))
            print(f"Saved matched reference image to '{reference_path}'.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Amazon S3 Vectors Embedding & Ingestion Pipeline")
    parser.add_argument("--file", type=str, default=None, help="Process only a specific PDF file (name or path)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for vector upserts (default: 5)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that have already been ingested")
    parser.add_argument("--no-interactive", action="store_true", help="Do not start interactive QA prompt after ingestion")
    parser.add_argument("--query-only", action="store_true", help="Skip ingestion and start interactive QA immediately")
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data")

    print("=" * 60)
    print("Amazon S3 Vectors Embedding & Ingestion Pipeline")
    print(f"  Data Directory: {data_dir}")
    print("=" * 60)

    vector_manager = S3VectorManager()
    embedder = GeminiEmbedder(model_name="gemini-embedding-2")

    if not args.query_only:
        # Ingest local PDF documents
        ingest_local_documents(
            data_dir=data_dir,
            vector_manager=vector_manager,
            embedder=embedder,
            batch_size=args.batch_size,
            target_file=args.file,
            skip_existing=args.skip_existing,
        )

    # Interactive QA loop (unless --no-interactive was passed)
    if not args.no_interactive:
        interactive_query_loop(vector_manager, embedder, data_dir)
    else:
        print("\n✅ Ingestion complete. Exiting without interactive QA loop (--no-interactive set).")


if __name__ == "__main__":
    main()
