import json
import os

import faiss
import numpy as np

from src.chunker import semantic_chunk
from src.embedder import create_embeddings
from src.extractor import extract_text_from_pdf

_SOURCES_FILE = "chunk_sources.json"


def _load_sources(embeddings_dir: str) -> dict:
    path = os.path.join(embeddings_dir, _SOURCES_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_sources(sources: dict, embeddings_dir: str) -> None:
    with open(os.path.join(embeddings_dir, _SOURCES_FILE), "w") as f:
        json.dump(sources, f, indent=2)


def get_indexed_docs(embeddings_dir: str) -> list[str]:
    """Return sorted list of all source document names in the index."""
    return sorted(set(_load_sources(embeddings_dir).values()))


def add_pdf_to_index(pdf_path: str, chunks_dir: str, embeddings_dir: str, source_name: str) -> int:
    """
    Extract, chunk, and embed a new PDF, then merge it into the existing FAISS index.
    Returns the number of new chunks added.
    """
    # Extract text
    tmp_txt = os.path.join(embeddings_dir, "_tmp_extract.txt")
    extract_text_from_pdf(pdf_path, tmp_txt)
    with open(tmp_txt, "r", encoding="utf-8") as f:
        text = f.read()
    os.remove(tmp_txt)

    new_chunks = semantic_chunk(text)
    if not new_chunks:
        return 0

    # Continue chunk numbering from existing files
    existing = sorted(
        f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")
    )
    start_idx = len(existing)

    for i, chunk in enumerate(new_chunks):
        with open(os.path.join(chunks_dir, f"chunk_{start_idx + i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)

    # Embed and merge into existing FAISS index
    new_embeddings = create_embeddings(new_chunks)
    existing_emb = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    all_emb = np.vstack([existing_emb, new_embeddings]).astype(np.float32)

    index = faiss.IndexFlatL2(all_emb.shape[1])
    index.add(all_emb)

    np.save(os.path.join(embeddings_dir, "embeddings.npy"), all_emb)
    faiss.write_index(index, os.path.join(embeddings_dir, "faiss_index.bin"))

    # Track which source each chunk belongs to
    sources = _load_sources(embeddings_dir)
    for i in range(start_idx):
        sources.setdefault(str(i), "TN traffic rules.pdf")
    for i in range(len(new_chunks)):
        sources[str(start_idx + i)] = source_name
    _save_sources(sources, embeddings_dir)

    print(f"[indexer] +{len(new_chunks)} chunks from '{source_name}' → total {all_emb.shape[0]}")
    return len(new_chunks)
