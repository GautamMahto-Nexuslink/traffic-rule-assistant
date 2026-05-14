# src/

This directory contains all the Python modules that make up the RAG pipeline and chatbot. Each file has a single, focused responsibility.

---

## Module Overview

```
src/
├── extractor.py   # Step 1 — PDF → plain text
├── chunker.py     # Step 2 — text → overlapping chunks
├── embedder.py    # Step 3 — chunks → embeddings + FAISS index
├── chatbot.py     # Step 4 — retrieval + Groq LLM answer generation
├── auth.py        # User registration and login (password hashing)
├── database.py    # SQLite schema + all DB operations
└── __init__.py
```

---

## extractor.py

Extracts raw text from the PDF using **PyMuPDF** (`fitz`).

**Key function:**
```python
extract_text_from_pdf(pdf_path: str, output_path: str) -> str
```

- Reads every page and labels it with `--- Page N ---`
- Skips empty pages
- Writes the full text to `data/extracted_text.txt`
- Returns the extracted string

---

## chunker.py

Splits the extracted text into overlapping word-based chunks suitable for embedding.

**Key functions:**
```python
chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]
save_chunks(chunks: list[str], output_dir: str) -> None
```

- Strips page markers before chunking for cleaner results
- Default: **400 words per chunk**, **80-word overlap** between consecutive chunks
- Saves each chunk as `data/chunks/chunk_XXXX.txt`
- Clears old chunk files before writing new ones

---

## embedder.py

Encodes chunks into dense vectors using `all-MiniLM-L6-v2` (384 dimensions) and builds a FAISS index for fast similarity search.

**Key functions:**
```python
load_chunks(chunks_dir: str) -> list[str]
create_embeddings(chunks: list[str], model_name: str) -> np.ndarray
build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2
save_artifacts(embeddings, index, output_dir: str) -> None
```

- Uses `IndexFlatL2` (exact L2 distance search — sufficient for small corpora)
- Saves `data/embeddings/embeddings.npy` and `data/embeddings/faiss_index.bin`

---

## chatbot.py

Implements the retrieval and generation loop used by both the CLI and the web UI.

**Key functions:**
```python
load_resources(embeddings_dir, chunks_dir) -> (index, chunks, model)
retrieve(query, index, chunks, model, top_k=5) -> list[str]
ask_groq(query, context_chunks, client) -> str
run_chatbot(embeddings_dir, chunks_dir, groq_api_key) -> None  # CLI only
```

- `load_resources` is called once and cached via `@st.cache_resource` in the web UI
- `retrieve` embeds the query and returns the **top-5** most similar chunks
- `ask_groq` sends the chunks as context to **Groq** (`llama-3.1-8b-instant`) with a system prompt restricting answers to the provided context
- `run_chatbot` runs an interactive CLI loop (used by `python main.py chat`)

---

## auth.py

Handles user account creation and login. Passwords are never stored in plain text.

**Key functions:**
```python
register(username: str, password: str) -> tuple[bool, str]
login(username: str, password: str) -> tuple[bool, int | None, str]
```

- Passwords hashed with **PBKDF2-HMAC-SHA256** using a random 32-byte salt and 100,000 iterations
- `register` validates minimum length (username ≥ 3, password ≥ 6) before writing to DB
- `login` returns `(success, user_id, message)` — `user_id` is `None` on failure

---

## database.py

Single source of truth for all SQLite operations. Initialises the schema on first run.

**Schema:**

```
users
  id, username (unique), password_hash, created_at

chat_sessions
  id, user_id → users.id, title, created_at

messages
  id, session_id → chat_sessions.id, role, content, created_at
```

**Key functions:**
```python
init_db()                                          # create tables if not exist
create_user(username, password_hash)
get_user(username) -> Row | None

create_session(user_id, title) -> int              # returns session id
get_user_sessions(user_id) -> list[Row]
update_session_title(session_id, title)
delete_session(session_id, user_id)                # cascades to messages

add_message(session_id, role, content)
get_messages(session_id) -> list[Row]
```

Database file is stored at `data/app.db`.
