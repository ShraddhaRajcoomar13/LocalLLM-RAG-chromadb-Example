
# Local RAG Assistant

A fully local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Chroma**, **HuggingFace embeddings**, and a **local Qwen LLM**. It retrieves relevant chunks from your PDFs and answers questions with **persistent SQLite memory** — no OpenAI API key required.

---

## Features

- 📄 **PDF ingestion** — Loads all PDFs from a folder, chunks them, and stores embeddings in Chroma.
- 🔍 **Semantic search** — Retrieves the top-k most relevant chunks per query.
- 🧠 **Local LLM** — Uses `Qwen/Qwen2.5-0.5B-Instruct` via HuggingFace `transformers`.
- 💾 **Persistent memory** — Stores facts and past Q&A in SQLite (`memory.db`).
- 🔒 **Fully offline** — No API keys, no external calls after the first model download.
- ⚡ **LangChain LCEL** — Clean, composable RAG chain.

---

## Project Structure

```
langchain/
├── documents/          # Place your PDF files here
├── chroma_db/          # Auto-created vector store (do not edit)
├── memory.db           # Auto-created SQLite memory
├── ingest.py           # Ingestion: PDFs → chunks → Chroma
├── rag.py              # RAG chat loop (retriever + Qwen + memory)
├── mem.py              # SQLite memory helpers
├── tes.py              # Quick memory test
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Requirements

- **Python** 3.10+
- **~2 GB free disk** (for the Qwen model + embedding model)
- Works on CPU. CUDA optional for speedup.

---

## Installation

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```powershell
pip install langchain langchain-community langchain-chroma langchain-huggingface transformers sentence-transformers chromadb python-dotenv
```

### 3. Install PyTorch (CPU build recommended)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU (CUDA 12.1):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **Note (PowerShell users):** Do **not** use `\` for line continuations — PowerShell uses the backtick `` ` ``. Keep installs on a single line to avoid errors.

---

## Usage

### Step 1 — Add your documents

Place one or more PDF files into the `documents/` folder:

```
documents/
├── paper1.pdf
├── paper2.pdf
└── notes.pdf
```

### Step 2 — Build the vector store

```powershell
python ingest.py
```

Expected output:
```
Loaded N pages.
Created M chunks.
Loading embedding model (first run downloads ~80MB)...
Successfully embedded and stored documents.
Vector database saved to: chroma_db
```

> ⚠️ **First run** downloads the embedding model (~80MB) and the Qwen model (~1GB). Subsequent runs are instant.

### Step 3 — Chat with your documents

```powershell
python rag.py
```

Example session:
```
======================================================================
LOCAL RAG (Qwen + Chroma + SQLite Memory)
======================================================================
LLM: Qwen/Qwen2.5-0.5B-Instruct
Embeddings: sentence-transformers/all-MiniLM-L6-v2

Type 'exit' to quit.

You: What is this document about?

Searching documents...
Generating answer with local Qwen model...

======================================================================
ANSWER
======================================================================
This document discusses ...

======================================================================
SOURCES
======================================================================
- documents/paper1.pdf, page 3
- documents/paper2.pdf, page 1
```

Type `exit`, `quit`, or `q` to stop.

### Step 4 (optional) — Test memory

```powershell
python tes.py
```

---

## Architecture

```
                ┌──────────────────┐
                │   PDF documents  │
                └────────┬─────────┘
                         │ PyPDFDirectoryLoader
                         ▼
                ┌──────────────────┐
                │  Text chunks     │  RecursiveCharacterTextSplitter
                └────────┬─────────┘
                         │ HuggingFaceEmbeddings
                         ▼
                ┌──────────────────┐
                │  Chroma vector   │  (chroma_db/)
                │      store       │
                └────────┬─────────┘
                         │ retriever (k=4)
                         ▼
   User question ──►┌──────────────────┐
                    │  LangChain RAG   │◄── SQLite memory (memory.db)
                    │      chain       │
                    └────────┬─────────┘
                             │ prompt
                             ▼
                    ┌──────────────────┐
                    │  Qwen 2.5 0.5B   │  (local, transformers)
                    └────────┬─────────┘
                             ▼
                        Final answer
```

---

## Configuration

Edit the top of `ingest.py` and `rag.py` to change defaults:

| Setting | Default | Where |
|---|---|---|
| `DOCUMENTS_DIR` | `"documents"` | `ingest.py` |
| `CHROMA_DIR` | `"chroma_db"` | both |
| `COLLECTION_NAME` | `"rag_documents"` | both |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | both |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | `rag.py` |
| `chunk_size` / `chunk_overlap` | `1000` / `200` | `ingest.py` |
| `k` (retrieved chunks) | `4` | `rag.py` |

---

## Important: Embedding Model Must Match

The embedding model used **at ingestion time** must be **identical** to the one used **at query time**. Otherwise, retrieval returns nonsense.

If you switch embedding models, **delete the old vector store first**:

```powershell
Remove-Item -Recurse -Force chroma_db
python ingest.py
```

---

## Upgrading the LLM

The default Qwen 0.5B model is tiny and fast but limited. For better quality, edit `rag.py`:

```python
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # ~3GB RAM
# or
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"     # ~6GB RAM, needs GPU ideally
```

For GPU acceleration, update the model load:

```python
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No PDF files found in the documents folder.` | Add PDFs to `documents/`. |
| `No relevant documents found.` | Re-run `ingest.py`. Check embedding model matches. |
| `Directory '\' is not installable` (PowerShell) | Don't use `\` for line continuation; use `` ` `` or one line. |
| Out of memory | Use `Qwen2.5-0.5B`, close other apps, or install CPU-only torch. |
| Slow generation | Expected on CPU. Use a smaller model or a GPU. |
| Nonsense answers | Embedding mismatch — delete `chroma_db/` and re-ingest. |

---

## Optional: Using OpenAI Instead

If you prefer OpenAI for embeddings and/or the LLM, add a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Then swap in `rag.py`:

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
```

And re-ingest with the matching OpenAI embeddings.

---

## .gitignore (recommended)

```gitignore
.venv/
__pycache__/
*.pyc
chroma_db/
memory.db
documents/
.env
```

---

## License

MIT — use freely for personal and commercial projects.

---

## Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [Qwen](https://huggingface.co/Qwen)
- [Sentence Transformers](https://www.sbert.net/)
```