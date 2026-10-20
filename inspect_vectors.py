import os
import sys
import json
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.embedding_service.s3_vector_manager import S3VectorManager


def main():
    print("=" * 65)
    print("Amazon S3 / ChromaDB Vectors - Vector Inspector")
    print("=" * 65)


    try:
        manager = S3VectorManager()
        print(f"Connecting to S3 Vector Bucket: '{manager.vector_bucket_name}'")
        print(f"Connecting to S3 Vector Index:  '{manager.index_name}' in '{manager.region_name}'\n")

        vectors = manager.list_vectors(max_results=50, return_metadata=True)

        if not vectors:
            print("No vectors found in the index yet.")
            return

        print(f"✅ Found {len(vectors)} vector(s) in index '{manager.index_name}':\n")
        for idx, vec in enumerate(vectors, start=1):
            key = vec.get("key")
            metadata = vec.get("metadata", {})
            print(f"[{idx}] Vector Key: {key}")
            print(f"    Document Name : {metadata.get('document_name')}")
            print(f"    Page Number   : {metadata.get('page_number')} of {metadata.get('total_pages')}")
            print(f"    Source URI    : {metadata.get('source_uri')}")
            print(f"    Ingested At   : {metadata.get('ingested_at')}")
            if metadata.get("text_snippet"):
                snippet = metadata["text_snippet"].replace("\n", " ")[:90]
                print(f"    Snippet       : {snippet}...")
            print("-" * 65)

    except Exception as e:
        print(f"Error inspecting vectors: {e}")


if __name__ == "__main__":
    main()