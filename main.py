import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "TN traffic rules.pdf"
TEXT_PATH = DATA_DIR / "extracted_text.txt"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def run_pipeline() -> None:
    from src.extractor import extract_text_from_pdf
    from src.chunker import chunk_text, save_chunks
    from src.embedder import load_chunks, create_embeddings, build_faiss_index, save_artifacts

    print("=" * 50)
    print("Step 1: Extracting text from PDF")
    print("=" * 50)
    extract_text_from_pdf(str(PDF_PATH), str(TEXT_PATH))

    print("\n" + "=" * 50)
    print("Step 2: Chunking extracted text")
    print("=" * 50)
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    save_chunks(chunks, str(CHUNKS_DIR))

    print("\n" + "=" * 50)
    print("Step 3: Creating embeddings and FAISS index")
    print("=" * 50)
    chunks = load_chunks(str(CHUNKS_DIR))
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    save_artifacts(embeddings, index, str(EMBEDDINGS_DIR))

    print("\nPipeline complete! Run `python main.py chat` to start the chatbot.")


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
    "chat": run_chat,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py [pipeline|chat]")
        print("  pipeline  — run all 3 preprocessing steps (extract → chunk → embed)")
        print("  chat      — start the CLI chatbot (run pipeline first)")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
