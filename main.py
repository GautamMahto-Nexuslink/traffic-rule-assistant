import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
EXTRACTED_DIR  = DATA_DIR / "extracted"   # one .txt per PDF
CHUNKS_DIR     = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def run_pipeline() -> None:
    from src.chunker import semantic_chunk
    from src.embedder import create_embeddings, build_faiss_index, save_artifacts
    from src.extractor import extract_text_from_pdf

    # ── Discover all PDFs in data/ ────────────────────────────────────────────
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/. Add at least one PDF and re-run.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"  • {p.name}")

    # ── Step 1 + 2: Extract and chunk each PDF ────────────────────────────────
    EXTRACTED_DIR.mkdir(exist_ok=True)
    all_chunks: list[str] = []
    chunk_sources: dict[str, str] = {}

    for pdf_path in pdf_files:
        print(f"\n{'=' * 50}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'=' * 50}")

        txt_path = EXTRACTED_DIR / f"{pdf_path.stem}.txt"
        extract_text_from_pdf(str(pdf_path), str(txt_path))

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = semantic_chunk(text)
        print(f"[chunker] {len(chunks)} semantic chunks from '{pdf_path.name}'")

        start = len(all_chunks)
        for i in range(len(chunks)):
            chunk_sources[str(start + i)] = pdf_path.name
        all_chunks.extend(chunks)

    # ── Step 2: Save all chunk files (fresh) ─────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Saving {len(all_chunks)} total chunks from {len(pdf_files)} PDF(s)")
    print(f"{'=' * 50}")

    CHUNKS_DIR.mkdir(exist_ok=True)
    for f in CHUNKS_DIR.iterdir():
        if f.name.startswith("chunk_") and f.name.endswith(".txt"):
            f.unlink()

    for i, chunk in enumerate(all_chunks):
        (CHUNKS_DIR / f"chunk_{i:04d}.txt").write_text(chunk, encoding="utf-8")
    print(f"[pipeline] Saved {len(all_chunks)} chunks → {CHUNKS_DIR}")

    # ── Step 3: Embed all chunks and build FAISS index ────────────────────────
    print(f"\n{'=' * 50}")
    print("Creating embeddings and FAISS index")
    print(f"{'=' * 50}")

    embeddings = create_embeddings(all_chunks)
    index      = build_faiss_index(embeddings)
    save_artifacts(embeddings, index, str(EMBEDDINGS_DIR))

    # Save source map so the UI and retriever know which chunk came from where
    sources_path = EMBEDDINGS_DIR / "chunk_sources.json"
    sources_path.write_text(json.dumps(chunk_sources, indent=2), encoding="utf-8")
    print(f"[pipeline] Saved chunk sources → {sources_path.name}")

    print(f"\nPipeline complete — {len(pdf_files)} PDF(s), {len(all_chunks)} chunks total.")
    print("Run `streamlit run app.py` or `python main.py chat` to start.")


def run_chat() -> None:
    from src.chatbot import run_chatbot

    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        groq_api_key = input("Enter your Groq API key: ").strip()
    if not groq_api_key:
        print("Error: Groq API key is required.")
        sys.exit(1)

    run_chatbot(str(EMBEDDINGS_DIR), str(CHUNKS_DIR), groq_api_key)


COMMANDS = {
    "pipeline": run_pipeline,
    "chat":     run_chat,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py [pipeline|chat]")
        print("  pipeline  — scan all PDFs in data/, extract → chunk → embed")
        print("  chat      — start the CLI chatbot (run pipeline first)")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
