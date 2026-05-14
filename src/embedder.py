import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_chunks(chunks_dir: str) -> list[str]:
    chunk_files = sorted(
        f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")
    )
    chunks = []
    for fname in chunk_files:
        with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
            chunks.append(f.read())
    return chunks


def create_embeddings(chunks: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    print(f"[embedder] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[embedder] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[embedder] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_artifacts(embeddings: np.ndarray, index: faiss.IndexFlatL2, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    print(f"[embedder] Saved embeddings and FAISS index to {output_dir}")
