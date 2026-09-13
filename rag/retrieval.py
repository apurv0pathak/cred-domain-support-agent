"""
Retrieval utilities for the Cred knowledge base.

Retrieval uses the two separately persisted ChromaDB collections and
returns parent-document metadata alongside each chunk.
"""

from __future__ import annotations

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from rag.indexing import (
    CHROMA_PERSIST_DIRECTORY,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION_NAME,
    SENTENCE_COLLECTION_NAME,
)


def _collection_name(strategy: str) -> str:
    if strategy == "fixed":
        return FIXED_COLLECTION_NAME

    if strategy == "sentence":
        return SENTENCE_COLLECTION_NAME

    raise ValueError(
        "strategy must be either 'fixed' or 'sentence'"
    )


def retrieve(
    query: str,
    strategy: str,
    top_k: int = 3,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k chunks for a query.

    The returned distance is ChromaDB's cosine distance for the normalized
    embeddings. Similarity is reported as:

        similarity = 1 - distance

    Every result includes its parent document_id.
    """

    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        _collection_name(strategy)
    )

    model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    local_files_only=True,
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "text": document,
                "distance": float(distance),
                "similarity": 1.0 - float(distance),
            }
        )

    return retrieved