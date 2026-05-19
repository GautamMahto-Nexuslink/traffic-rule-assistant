# 🚦 TN Traffic Rules Assistant

A multi-document RAG (Retrieval-Augmented Generation) pipeline and web chatbot for answering questions about **Tamil Nadu traffic rules**, powered by **Groq LLM**, **hybrid BM25 + FAISS search**, and **Streamlit**.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-PDF ingestion** | Automatically scans all PDFs in `data/` — drop a file there and re-run the pipeline |
| **OCR fallback** | Vector-path / scanned PDFs (e.g. from CorelDRAW) are processed automatically via Tesseract |
| **Semantic chunking** | Splits text on section/sub-section headers for coherent, self-contained chunks |
| **Hybrid search** | BM25 keyword search + FAISS semantic search combined for better retrieval |
| **Streaming responses** | Answers stream token-by-token via Groq's streaming API |
| **Follow-up suggestions** | Auto-generates 3 clickable follow-up questions after each answer |
| **Multi-user web UI** | Streamlit app with login / sign-up and per-user chat history |
| **Live PDF upload** | Add new documents through the sidebar without re-running the pipeline |
| **Export chat** | Download any conversation as a Markdown file |

---

## Project Structure

```
traffic-rule-assistant/
├── app.py                  # Streamlit web application (UI entry point)
├── main.py                 # CLI pipeline runner (extract → chunk → embed)
├── pyproject.toml          # Project metadata and dependencies
├── .env                    # GROQ_API_KEY — not committed
│
├── src/
│   ├── extractor.py        # PDF → plain text (PyMuPDF + Tesseract OCR fallback)
│   ├── chunker.py          # Text → semantic chunks (split on section headers)
│   ├── embedder.py         # Chunks → embeddings + FAISS index
│   ├── retriever.py        # HybridRetriever: BM25 + FAISS combined search
│   ├── chatbot.py          # Streaming LLM answers + follow-up generation
│   ├── indexer.py          # Runtime: add a new PDF to the existing index
│   ├── auth.py             # User register / login (PBKDF2-SHA256 hashing)
│   ├── database.py         # SQLite schema + all DB operations
│   └── README.md
│
└── data/
    ├── *.pdf               # Source PDFs (all files here are auto-indexed)
    ├── extracted/          # Per-PDF plain text files (pipeline output)
    ├── chunks/             # Semantic chunk files (pipeline output)
    ├── embeddings/         # embeddings.npy, faiss_index.bin, chunk_sources.json
    ├── uploads/            # PDFs uploaded at runtime via the web UI
    ├── app.db              # SQLite database — users + chat history
    └── README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide
- A free Groq API key from [console.groq.com](https://console.groq.com)

Install Tesseract (Ubuntu/Debian):
```bash
sudo apt install tesseract-ocr
```

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add PDFs and run the pipeline

Place one or more PDF files in `data/` then run:

```bash
python main.py pipeline
```

The pipeline auto-discovers every `*.pdf` in `data/`, processes them in alphabetical order, and builds a single unified index.

Expected output (2 PDFs example):
```
Found 2 PDF(s):
  • Road Safty (English) New Book.pdf
  • TN traffic rules.pdf

Processing: Road Safty (English) New Book.pdf
[extractor] Falling back to OCR (vector/scanned PDF detected)…
[extractor] Road Safty (English) New Book.pdf: 128,035 chars, 89 pages (OCR)
[chunker] 139 semantic chunks

Processing: TN traffic rules.pdf
[extractor] TN traffic rules.pdf: 105,551 chars, 60 pages (direct)
[chunker] 351 semantic chunks

Saving 490 total chunks from 2 PDF(s)
[embedder] FAISS index built — 490 vectors, dim=384

Pipeline complete — 2 PDF(s), 490 chunks total.
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
data/*.pdf  (all PDFs auto-discovered)
    │
    ▼
[extractor.py]
    ├─ PyMuPDF direct extraction  (text-based PDFs)
    └─ Tesseract OCR fallback     (scanned / vector-path PDFs)
    │
    ▼  data/extracted/<name>.txt  (one file per PDF)
    │
[chunker.py]
    └─ Split on numbered section headers (4. Title / 4.1 Sub / 4.1.1 Detail)
       Sections > 350 words → split on paragraphs → word-based as last resort
    │
    ▼  data/chunks/chunk_XXXX.txt
    │
[embedder.py]
    └─ all-MiniLM-L6-v2  →  384-dim float32 vectors
    │
    ▼  data/embeddings/embeddings.npy
       data/embeddings/faiss_index.bin
       data/embeddings/chunk_sources.json  (chunk → PDF name map)
    │
    ┌──────────────────┐
    │  HybridRetriever  │  (loaded once, cached)
    │  ┌─────────────┐ │
    │  │ BM25Okapi   │ │  ← keyword scores
    │  │ FAISS index │ │  ← semantic scores
    │  └──────┬──────┘ │
    │  score = 0.5 × bm25_norm + 0.5 × faiss_norm
    └──────────┬───────┘
               │  top-5 chunks
               ▼
[chatbot.py]  ask_groq_stream()  →  streamed answer
              generate_followups()  →  3 suggested questions
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | Direct PDF text extraction |
| `pytesseract` | OCR wrapper for Tesseract |
| `pdf2image` | Render PDF pages to images for OCR |
| `pillow` | Image processing for OCR |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Vector similarity search |
| `rank-bm25` | BM25 keyword search index |
| `groq` | LLM streaming inference (`llama-3.1-8b-instant`) |
| `streamlit` | Web UI |
| `python-dotenv` | Load `.env` variables |
| `numpy` | Array operations |

---

## Web UI Overview

| Area | Features |
|------|----------|
| **Auth page** | Login and Sign Up tabs. Passwords hashed with PBKDF2-SHA256 (100k iterations). |
| **Sidebar** | New Chat, previous session list (click to reload, 🗑 to delete), Export Chat button, Knowledge Base expander |
| **Knowledge Base expander** | Shows all indexed PDFs; upload a new PDF and click "Index PDF" to add it live (no pipeline re-run needed) |
| **Chat area** | Streaming token-by-token responses from Groq |
| **Follow-up chips** | 3 auto-generated follow-up questions appear after each answer; click any to send it |
| **Export** | Downloads the active conversation as a `.md` file |

Sessions are auto-titled from the first message and stored per user in `data/app.db`.

---

## Git Branch Layout

| Branch | Contents |
|--------|----------|
| `main` | Source code only (PDFs, chunks, embeddings are gitignored) |
| `chunks` | Committed chunk files from both PDFs (490 total) |
| `feature/add_multiplepdfs_followup_question` | Feature branch for latest additions |
| `feature/semantic-chunking-hybrid-search` | Earlier feature branch |
