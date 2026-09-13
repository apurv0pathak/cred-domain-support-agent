"""
Chunking strategies for the Cred knowledge base.

Two strategies are implemented:

1. Fixed-size chunks:
   400 characters with an 80-character overlap.

2. Sentence-based chunks:
   groups of 3 sentences with a 1-sentence overlap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


FIXED_CHUNK_SIZE = 400
FIXED_CHUNK_OVERLAP = 80

SENTENCES_PER_CHUNK = 3
SENTENCE_OVERLAP = 1


def _strip_markdown_heading(text: str) -> str:
    #Remove a leading Markdown heading from document text.

    lines = text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]

    return "\n".join(lines).strip()


def _split_sentences(text: str) -> list[str]:

    #Split policy text into sentences using deterministic punctuation rules.

    #The knowledge-base documents are intentionally simple prose, so a lightweight sentence splitter is sufficient and avoids another external NLP dependency.


    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def fixed_size_chunks(
    text: str,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    text = _strip_markdown_heading(text)

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def sentence_based_chunks(
    text: str,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
    sentence_overlap: int = SENTENCE_OVERLAP,
) -> list[str]:
    """
    Split text into groups of sentences with sentence-level overlap.
    """

    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be greater than zero")

    if sentence_overlap < 0 or sentence_overlap >= sentences_per_chunk:
        raise ValueError(
            "sentence_overlap must be >= 0 and smaller than "
            "sentences_per_chunk"
        )

    sentences = _split_sentences(
        _strip_markdown_heading(text)
    )

    if not sentences:
        return []

    chunks: list[str] = []
    step = sentences_per_chunk - sentence_overlap

    for start in range(0, len(sentences), step):
        group = sentences[start:start + sentences_per_chunk]

        if not group:
            break

        chunks.append(" ".join(group))

        if start + sentences_per_chunk >= len(sentences):
            break

    return chunks


def load_documents(
    knowledge_base_dir: str | Path = "knowledge_base",
) -> list[dict[str, Any]]:
    """
    Load every Markdown knowledge-base document.

    Returns dictionaries containing:
      - document_id
      - filename
      - topic
      - text
    """

    knowledge_base_dir = Path(knowledge_base_dir)
    documents: list[dict[str, Any]] = []

    for path in sorted(knowledge_base_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        document_id = path.stem

        documents.append(
            {
                "document_id": document_id,
                "filename": path.name,
                "text": text,
            }
        )

    if not documents:
        raise RuntimeError(
            f"No Markdown documents found in {knowledge_base_dir}"
        )

    return documents


def build_chunks(
    strategy: str,
    knowledge_base_dir: str | Path = "knowledge_base",
) -> list[dict[str, Any]]:
    """
    Build chunks for all knowledge-base documents.

    Every chunk retains its parent document_id and filename.
    """

    documents = load_documents(knowledge_base_dir)
    chunks: list[dict[str, Any]] = []

    if strategy not in {"fixed", "sentence"}:
        raise ValueError(
            "strategy must be either 'fixed' or 'sentence'"
        )

    for document in documents:
        if strategy == "fixed":
            document_chunks = fixed_size_chunks(document["text"])
        else:
            document_chunks = sentence_based_chunks(document["text"])

        for chunk_index, chunk_text in enumerate(document_chunks):
            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"__{strategy}__{chunk_index:03d}"
                    ),
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "strategy": strategy,
                }
            )

    return chunks