"""
s3_vector_manager.py  (repurposed as: chroma_vector_manager.py logic inside this file)
────────────────────────────────────────────────────────────────────────────────────────
Local ChromaDB vector store manager — drop-in replacement for the original S3 Vectors
integration, keeping the same public API so nothing else needs to change.

Collection schema mirrors the README metadata spec:
    document_name, page_number, total_pages, file_size, ingested_at, text_snippet
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Where ChromaDB persists data (relative to project root)
CHROMA_PERSIST_DIR = str(
    Path(__file__).resolve().parents[2] / ".chroma_db"
)


def _get_client() -> chromadb.PersistentClient:
    """Return (or create) a persistent ChromaDB client."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """
    Return an existing ChromaDB collection or create it with cosine similarity.

    Args:
        collection_name: the name of the vector collection / index

    Returns:
        ChromaDB Collection object
    """
    client = _get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "ChromaDB collection '%s' ready (path: %s)", collection_name, CHROMA_PERSIST_DIR
    )
    return collection


def upsert_vectors(
    collection: chromadb.Collection,
    chunks: list[dict],
) -> None:
    """
    Upsert a batch of page-level vector chunks into the collection.

    Each chunk must have:
        key        : str  — unique ID
        embedding  : list[float]
        metadata   : dict (document_name, page_number, total_pages, …)

    Args:
        collection: ChromaDB Collection
        chunks:     list of chunk dicts (must include 'embedding' key)
    """
    if not chunks:
        return

    ids = [c["key"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = []
    documents = []

    for c in chunks:
        meta = {k: v for k, v in c["metadata"].items()}
        # ChromaDB only supports str / int / float / bool in metadata
        meta["page_number"] = int(meta.get("page_number", 0))
        meta["total_pages"] = int(meta.get("total_pages", 0))
        meta["file_size"] = int(meta.get("file_size", 0))
        metadatas.append(meta)
        documents.append(meta.get("text_snippet", ""))

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info("Upserted %d vectors into collection '%s'.", len(chunks), collection.name)


def query_vectors(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
    filter_document: Optional[str] = None,
    filter_page: Optional[int] = None,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Perform a similarity search against the collection.

    Args:
        collection:       ChromaDB Collection
        query_embedding:  query vector
        n_results:        number of top results to return
        filter_document:  optional — restrict to a specific document name
        filter_page:      optional — restrict to a specific page number
        where:            optional — raw ChromaDB where-filter (overrides above)

    Returns:
        List of result dicts with keys: key, distance, metadata, text_snippet
    """
    chroma_where: Optional[dict] = where

    if chroma_where is None:
        conditions: list[dict] = []
        if filter_document:
            conditions.append({"document_name": {"$eq": filter_document}})
        if filter_page is not None:
            conditions.append({"page_number": {"$eq": filter_page}})

        if len(conditions) == 1:
            chroma_where = conditions[0]
        elif len(conditions) > 1:
            chroma_where = {"$and": conditions}

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["distances", "metadatas", "documents"],
    }
    if chroma_where:
        query_kwargs["where"] = chroma_where

    results = collection.query(**query_kwargs)

    output = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    for vid, dist, meta, doc in zip(ids, distances, metadatas, documents):
        output.append(
            {
                "key": vid,
                "distance": round(float(dist), 4),
                "metadata": meta,
                "text_snippet": doc,
            }
        )

    return output


def delete_document_vectors(
    collection: chromadb.Collection,
    document_name: str,
) -> int:
    """
    Delete all vectors belonging to a specific document.

    Args:
        collection:    ChromaDB Collection
        document_name: the document_name metadata value to target

    Returns:
        Number of vectors deleted.
    """
    # First, find the IDs to delete
    existing = collection.get(
        where={"document_name": {"$eq": document_name}},
        include=[],
    )
    ids_to_delete = existing.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d vectors for document '%s'.", len(ids_to_delete), document_name
        )
    return len(ids_to_delete)


def list_indexed_documents(collection: chromadb.Collection) -> list[str]:
    """Return a sorted list of unique document names in the collection."""
    results = collection.get(include=["metadatas"])
    names = set()
    for meta in results.get("metadatas", []):
        if meta and "document_name" in meta:
            names.add(meta["document_name"])
    return sorted(names)


def collection_stats(collection: chromadb.Collection) -> dict:
    """Return basic stats about the collection."""
    count = collection.count()
    docs = list_indexed_documents(collection)
    return {
        "total_vectors": count,
        "indexed_documents": docs,
        "collection_name": collection.name,
        "persist_dir": CHROMA_PERSIST_DIR,
    }


class S3VectorManager:
    """
    Object-oriented vector manager wrapping local ChromaDB.
    Maintains compatibility with legacy S3VectorManager interface.
    """

    def __init__(
        self,
        vector_bucket_name: str = "rag-vectors",
        index_name: str = "rag_documents",
        region_name: str = "local",
    ):
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name or "rag_documents"
        self.region_name = region_name
        self.collection = get_or_create_collection(self.index_name)

    def ensure_index(self, dimension: int = 3072, similarity_metric: str = "cosine"):
        return self.collection

    def upsert_vectors(self, chunks: list[dict]):
        return upsert_vectors(self.collection, chunks)

    def query_vectors(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        n_results: Optional[int] = None,
        filter_document: Optional[str] = None,
        filter_page: Optional[int] = None,
        where: Optional[dict] = None,
    ):
        k = n_results if n_results is not None else top_k
        return query_vectors(
            collection=self.collection,
            query_embedding=query_embedding,
            n_results=k,
            filter_document=filter_document,
            filter_page=filter_page,
            where=where,
        )

    def delete_document(self, document_name: str) -> int:
        return delete_document_vectors(self.collection, document_name)

    def list_documents(self) -> list[str]:
        return list_indexed_documents(self.collection)

    def list_vectors(self, max_results: int = 50, return_metadata: bool = True) -> list[dict]:
        results = self.collection.get(
            limit=max_results,
            include=["metadatas", "documents"] if return_metadata else [],
        )
        vectors = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", []) or [{}] * len(ids)
        documents = results.get("documents", []) or [""] * len(ids)
        for vid, meta, doc in zip(ids, metadatas, documents):
            m = dict(meta) if meta else {}
            if "text_snippet" not in m and doc:
                m["text_snippet"] = doc
            vectors.append({"key": vid, "metadata": m})
        return vectors

    def get_stats(self) -> dict:
        return collection_stats(self.collection)


