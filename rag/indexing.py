"""
Persistent ChromaDB indexing for both RAG chunking strategies.

No network access or LLM calls are used by this module.
The SentenceTransformers embedding model runs locally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import build_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIRECTORY = "chroma_db"

FIXED_COLLECTION_NAME = "cred_kb_fixed"
SENTENCE_COLLECTION_NAME = "cred_kb_sentence"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_embedding_model() -> SentenceTransformer:
    """
    Load the already-cached local SentenceTransformers model.

    local_files_only=True is intentional: after the initial model setup,
    this project must not make network calls to Hugging Face or any other
    external service.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

def _collection_name(strategy: str) -> str:
    if strategy == "fixed":
        return FIXED_COLLECTION_NAME

    if strategy == "sentence":
        return SENTENCE_COLLECTION_NAME

    raise ValueError(
        "strategy must be either 'fixed' or 'sentence'"
    )


def build_index(
    strategy: str,
    knowledge_base_dir: str = "knowledge_base",
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> dict[str, Any]:
    """
    Build and persist one ChromaDB collection.
    Returns basic indexing statistics.
    """

    chunks = build_chunks(
        strategy=strategy,
        knowledge_base_dir=knowledge_base_dir,
    )

    if not chunks:
        raise RuntimeError(
            f"No chunks were produced for strategy={strategy}"
        )

    model = _load_embedding_model()

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection_name = _collection_name(strategy)

    # Rebuilding is deterministic and prevents stale chunks from an older run from remaining in the collection.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "strategy": strategy,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "strategy": chunk["strategy"],
            }
            for chunk in chunks
        ],
    )

    return {
        "strategy": strategy,
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "document_count": len(
            {chunk["document_id"] for chunk in chunks}
        ),
        "persist_directory": os.path.abspath(persist_directory),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


def build_all_indexes(
    knowledge_base_dir: str = "knowledge_base",
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> list[dict[str, Any]]:
    """Build both independent persisted collections."""

    return [
        build_index(
            strategy="fixed",
            knowledge_base_dir=knowledge_base_dir,
            persist_directory=persist_directory,
        ),
        build_index(
            strategy="sentence",
            knowledge_base_dir=knowledge_base_dir,
            persist_directory=persist_directory,
        ),
    ]


if __name__ == "__main__":
    results = build_all_indexes()

    for result in results:
        print(
            f"{result['strategy']}: "
            f"{result['collection_name']} | "
            f"{result['chunk_count']} chunks | "
            f"{result['document_count']} documents"
        )