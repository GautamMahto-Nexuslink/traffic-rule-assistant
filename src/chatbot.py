import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL = "all-MiniLM-L6-v2"
# GROQ_MODEL = "llama3-8b-8192"
GROQ_MODEL='llama-3.1-8b-instant'
TOP_K = 5
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Tamil Nadu traffic rules. "
    "Use only the provided context to answer. If the answer is not in the context, say so clearly."
)


def load_resources(embeddings_dir: str, chunks_dir: str) -> tuple[faiss.IndexFlatL2, list[str], SentenceTransformer]:
    index = faiss.read_index(os.path.join(embeddings_dir, "faiss_index.bin"))

    chunk_files = sorted(
        f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".txt")
    )
    chunks = []
    for fname in chunk_files:
        with open(os.path.join(chunks_dir, fname), "r", encoding="utf-8") as f:
            chunks.append(f.read())

    model = SentenceTransformer(EMBED_MODEL)
    print(f"[chatbot] Loaded {index.ntotal} vectors, {len(chunks)} chunks.")
    return index, chunks, model


def retrieve(query: str, index: faiss.IndexFlatL2, chunks: list[str], model: SentenceTransformer, top_k: int = TOP_K) -> list[str]:
    query_vec = model.encode([query]).astype(np.float32)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask_groq(query: str, context_chunks: list[str], client: Groq) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def run_chatbot(embeddings_dir: str, chunks_dir: str, groq_api_key: str) -> None:
    print("\nLoading resources, please wait...")
    index, chunks, model = load_resources(embeddings_dir, chunks_dir)
    client = Groq(api_key=groq_api_key)

    print("\n" + "=" * 50)
    print("  TN Traffic Rules Assistant")
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

        relevant = retrieve(query, index, chunks, model)
        answer = ask_groq(query, relevant, client)
        print(f"\nAssistant: {answer}\n")
