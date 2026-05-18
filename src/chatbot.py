import os
from typing import Generator

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.retriever import HybridRetriever

EMBED_MODEL = "all-MiniLM-L6-v2"
# GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL = "llama-3.1-8b-instant"
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Tamil Nadu traffic rules. "
    "Use only the provided context to answer. If the answer is not in the context, say so clearly."
)


def load_resources(embeddings_dir: str, chunks_dir: str) -> HybridRetriever:
    index = faiss.read_index(os.path.join(embeddings_dir, "faiss_index.bin"))

    chunk_files = sorted(
        f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")
    )
    chunks = []
    for fname in chunk_files:
        with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
            chunks.append(f.read())

    model = SentenceTransformer(EMBED_MODEL)
    retriever = HybridRetriever(chunks, index, model, alpha=0.5)
    print(f"[chatbot] HybridRetriever ready — {len(chunks)} chunks, alpha=0.5")
    return retriever


def ask_groq_stream(query: str, context_chunks: list[str], client: Groq) -> Generator[str, None, None]:
    """Streams the answer token-by-token."""
    context = "\n\n---\n\n".join(context_chunks)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def ask_groq(query: str, context_chunks: list[str], client: Groq) -> str:
    """Non-streaming version — used by the CLI chatbot."""
    return "".join(ask_groq_stream(query, context_chunks, client))


def generate_followups(query: str, answer: str, client: Groq) -> list[str]:
    """Generate 3 short follow-up questions based on the Q&A exchange."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Based on this Q&A about Tamil Nadu traffic rules, suggest 3 short "
                    "follow-up questions the user might ask next.\n\n"
                    f"Q: {query}\nA: {answer}\n\n"
                    "Return only 3 questions, one per line, no numbering or bullets."
                ),
            }],
            temperature=0.7,
            max_tokens=150,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        return [l.strip().lstrip("0123456789.-) ") for l in lines if l.strip()][:3]
    except Exception:
        return []


def run_chatbot(embeddings_dir: str, chunks_dir: str, groq_api_key: str) -> None:
    retriever = load_resources(embeddings_dir, chunks_dir)
    client = Groq(api_key=groq_api_key)

    print("\n" + "=" * 50)
    print("  TN Traffic Rules Assistant (Hybrid Search)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        relevant = retriever.retrieve(query)
        answer = ask_groq(query, relevant, client)
        print(f"\nAssistant: {answer}\n")
