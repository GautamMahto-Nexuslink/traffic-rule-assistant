import os
import re


def clean_text(text: str) -> str:
    # Collapse multiple blank lines and strip page markers for cleaner chunks
    text = re.sub(r"--- Page \d+ ---", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    text = clean_text(text)
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def save_chunks(chunks: list[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing chunks before saving new ones
    for f in os.listdir(output_dir):
        if f.startswith("chunk_") and f.endswith(".txt"):
            os.remove(os.path.join(output_dir, f))

    for i, chunk in enumerate(chunks):
        path = os.path.join(output_dir, f"chunk_{i:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk)

    print(f"[chunker] Saved {len(chunks)} chunks to {output_dir}")
