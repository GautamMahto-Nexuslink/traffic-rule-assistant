# src/

All Python modules for the RAG pipeline, hybrid retrieval, LLM interaction, and web app backend. Each file has a single focused responsibility.

---

## Module Overview

```
src/
├── extractor.py   # Step 1 — PDF → plain text (direct + OCR fallback)
├── chunker.py     # Step 2 — text → semantic chunks on section headers
├── embedder.py    # Step 3 — chunks → sentence-transformer embeddings + FAISS index
├── retriever.py   # HybridRetriever — BM25 + FAISS combined at query time
├── chatbot.py     # Streaming LLM answers, follow-up generation, CLI loop
├── indexer.py     # Runtime: merge a new PDF into the existing index
├── auth.py        # User register / login with PBKDF2-SHA256 password hashing
├── database.py    # SQLite schema + all CRUD operations
└── __init__.py
```

---

## extractor.py

Extracts text from a PDF and saves it to a `.txt` file.

**Key function:**
```python
extract_text_from_pdf(pdf_path: str, output_path: str) -> str
```

**Two-stage extraction:**
1. **Direct** — PyMuPDF (`fitz`) reads embedded text objects. Fast, lossless.
2. **OCR fallback** — if direct extraction yields 0 characters (scanned PDF or vector-path PDF created in tools like CorelDRAW / Illustrator), pages are rendered to images at 200 DPI and passed through Tesseract OCR via `pytesseract`.

Each page is labelled `--- Page N ---` in the output. Returns an empty string and prints a warning if neither method yields text.

---

## chunker.py

Splits extracted text into semantically coherent chunks aligned with the document's own section structure.

**Key functions:**
```python
semantic_chunk(text: str) -> list[str]
save_chunks(chunks: list[str], output_dir: str) -> None
```

**Splitting strategy (in priority order):**
1. Detect numbered section headers using regex:
   - `4. Title` — main sections
   - `4.1 Sub-title` — sub-sections
   - `4.1.1 Detail` — sub-sub-sections
2. Each matched header + its body becomes one chunk.
3. Chunks exceeding **350 words** are split on paragraph (`\n\n`) boundaries.
4. Paragraphs still exceeding 350 words fall back to fixed word-splitting with **60-word overlap**.

Page markers (`--- Page N ---`) are stripped before chunking. `save_chunks` clears old chunk files before writing new ones.

---

## embedder.py

Converts text chunks into dense float32 vectors and builds a FAISS similarity index.

**Key functions:**
```python
load_chunks(chunks_dir: str) -> list[str]
create_embeddings(chunks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray
build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2
save_artifacts(embeddings, index, output_dir: str) -> None
```

- Model: `all-MiniLM-L6-v2` — 384-dimensional vectors, runs locally on CPU
- Index type: `IndexFlatL2` — exact L2 distance search (suitable for corpora up to ~10k chunks)
- Outputs: `embeddings.npy` (shape `N × 384`, float32) and `faiss_index.bin`

---

## retriever.py

Implements `HybridRetriever`, which combines BM25 keyword search with FAISS semantic search at query time.

**Class:**
```python
class HybridRetriever:
    def __init__(self, chunks, index, model, alpha=0.5): ...
    def retrieve(self, query: str, top_k: int = 5) -> list[str]: ...
```

**Scoring pipeline for each query:**
1. `BM25Okapi.get_scores(query_tokens)` → raw BM25 scores for all N chunks
2. `index.search(query_vec, N)` → L2 distances converted to similarity: `1 / (1 + dist)`
3. Both arrays are **min-max normalised** to `[0, 1]`
4. Combined: `score = alpha × bm25_norm + (1 − alpha) × faiss_norm`
5. Returns top-k chunks sorted by combined score

`alpha=0.5` balances keyword precision (BM25) with semantic recall (FAISS). Increasing alpha toward `1.0` favours exact term matches (e.g. section numbers, fine amounts); decreasing toward `0.0` favours conceptual similarity.

The BM25 index is rebuilt from chunks at load time — no separate serialisation needed.

---

## chatbot.py

Handles LLM interaction and exposes the API used by both the web UI and the CLI.

**Key functions:**
```python
load_resources(embeddings_dir, chunks_dir) -> HybridRetriever
ask_groq_stream(query, context_chunks, client) -> Generator[str, None, None]
ask_groq(query, context_chunks, client) -> str
generate_followups(query, answer, client) -> list[str]
run_chatbot(embeddings_dir, chunks_dir, groq_api_key) -> None
```

- `load_resources` loads the FAISS index, reads all chunk files, loads the embedding model, and returns a `HybridRetriever`. In the web UI, this is wrapped in `@st.cache_resource` so it runs only once.
- `ask_groq_stream` uses Groq's `stream=True` to yield tokens one by one — consumed by `st.write_stream()` in the UI for real-time output.
- `ask_groq` is the non-streaming version used by the CLI (`run_chatbot`).
- `generate_followups` makes a second Groq call asking for 3 short follow-up questions based on the Q&A exchange. Returns `[]` on failure (network error, rate limit) so it never breaks the main flow.
- Model: `llama-3.1-8b-instant`

---

## indexer.py

Merges a new PDF into the existing FAISS index at runtime (used by the web UI "Index PDF" button).

**Key functions:**
```python
add_pdf_to_index(pdf_path, chunks_dir, embeddings_dir, source_name) -> int
get_indexed_docs(embeddings_dir) -> list[str]
```

**`add_pdf_to_index` steps:**
1. Extract text from the PDF (with OCR fallback via `extractor.py`)
2. Semantic chunk the extracted text
3. Append new chunk files continuing the existing numbering (`chunk_XXXX.txt`)
4. Embed new chunks and stack onto the existing `embeddings.npy`
5. Rebuild `faiss_index.bin` from the combined embeddings
6. Update `chunk_sources.json` — maps every chunk index to its source PDF name

`get_indexed_docs` reads `chunk_sources.json` and returns a sorted list of unique document names, used by the sidebar to display what's in the knowledge base.

After `add_pdf_to_index` completes, `app.py` calls `load_rag.clear()` to invalidate the Streamlit cache and force the retriever to reload with the new chunks.

---

## auth.py

User authentication logic. Depends on `database.py` for persistence.

**Key functions:**
```python
register(username: str, password: str) -> tuple[bool, str]
login(username: str, password: str) -> tuple[bool, int | None, str]
```

- Passwords are hashed with **PBKDF2-HMAC-SHA256**: random 32-byte salt + 100,000 iterations. The salt is stored alongside the hash as `<salt_hex>:<key_hex>`.
- `register` enforces minimum lengths (username ≥ 3 chars, password ≥ 6 chars) and catches duplicate usernames.
- `login` returns `(True, user_id, message)` on success, `(False, None, message)` on failure.

---

## database.py

Single source of truth for all SQLite operations. All tables are created on first run via `init_db()`.

**Schema:**

```
users
  id INTEGER PK, username TEXT UNIQUE, password_hash TEXT, created_at TIMESTAMP

chat_sessions
  id INTEGER PK, user_id → users.id, title TEXT, created_at TIMESTAMP

messages
  id INTEGER PK, session_id → chat_sessions.id, role TEXT, content TEXT, created_at TIMESTAMP
```

**Key functions:**
```python
init_db()                                        # idempotent — safe to call on every startup
create_user(username, password_hash)
get_user(username) -> Row | None

create_session(user_id, title="New Chat") -> int
get_user_sessions(user_id) -> list[Row]
update_session_title(session_id, title)
delete_session(session_id, user_id)              # cascades — deletes messages too

add_message(session_id, role, content)
get_messages(session_id) -> list[Row]            # ordered by created_at ASC
```

All connections use `row_factory = sqlite3.Row` so results are accessible by column name. The database file lives at `data/app.db`.
