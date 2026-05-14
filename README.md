# 🚦 TN Traffic Rules Assistant

A RAG (Retrieval-Augmented Generation) pipeline and web chatbot for answering questions about **Tamil Nadu traffic rules**, powered by **Groq LLM**, **FAISS**, and **Streamlit**.

---

## Features

- **PDF ingestion** — extracts and processes the official TN traffic rules document
- **Semantic search** — sentence-transformer embeddings + FAISS vector index for accurate retrieval
- **Groq LLM** — fast, free inference via `llama-3.1-8b-instant`
- **Web UI** — Streamlit chat interface with user login/signup
- **Chat history** — all conversations saved per user in a local SQLite database

---

## Project Structure

```
traffic-rule-assistant/
├── app.py                  # Streamlit web application (entry point for UI)
├── main.py                 # CLI pipeline runner (extract → chunk → embed)
├── pyproject.toml          # Project metadata and dependencies
├── .env                    # Environment variables (GROQ_API_KEY) — not committed
├── src/                    # All source modules
│   └── README.md
└── data/                   # All data artifacts
    └── README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Install dependencies

```bash
uv sync
```

### 3. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the RAG pipeline (one-time setup)

This extracts text from the PDF, chunks it, creates embeddings, and builds the FAISS index:

```bash
python main.py pipeline
```

Expected output:
```
Step 1: Extracting text from PDF
[extractor] Extracted 105,551 characters from 60 pages

Step 2: Chunking extracted text
[chunker] Saved 44 chunks

Step 3: Creating embeddings and FAISS index
[embedder] FAISS index built — 44 vectors, dim=384
```

### 5. Launch the web UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### (Optional) CLI chatbot

```bash
python main.py chat
```

---

## How It Works

```
PDF
 │
 ▼
[extractor.py]  →  extracted_text.txt
 │
 ▼
[chunker.py]    →  data/chunks/chunk_XXXX.txt  (400 words, 80-word overlap)
 │
 ▼
[embedder.py]   →  data/embeddings/embeddings.npy
                →  data/embeddings/faiss_index.bin
                    (all-MiniLM-L6-v2, 384-dim vectors)
 │
 ▼
[chatbot.py]    ←  User query
 │   1. Embed query with same model
 │   2. FAISS search → top-5 relevant chunks
 │   3. Send chunks + query to Groq
 └─► Answer
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | PDF text extraction |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Vector similarity search |
| `groq` | LLM inference (`llama-3.1-8b-instant`) |
| `streamlit` | Web UI |
| `python-dotenv` | Load `.env` variables |
| `numpy` | Array operations |

---

## Web UI Overview

| Screen | Description |
|--------|-------------|
| **Auth page** | Login and Sign Up tabs. Passwords hashed with PBKDF2-SHA256. |
| **Chat page** | Ask questions and get answers grounded in the TN traffic rules document. |
| **Sidebar** | List of all previous sessions per user. Click to reload, 🗑 to delete. |

Each chat session is automatically titled from the first user message and stored in `data/app.db`.
