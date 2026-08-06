---
title: DocBrain - Intelligent Multi-Domain Document Q&A System
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# DocBrain: Intelligent Multi-Domain Document Q&A System

![Overview](asset/Overview.jpg)

**DocBrain** is an Intelligent Question & Answer System (RAG - Retrieval-Augmented Generation) that supports accurate retrieval and responses from document files (`PDF`, `DOCX`, `TXT`) in isolated sessions.

---

## Key Features

- **Source Citations**: Accurately displays the original file name, page number, vector relevance score, and quoted text snippets.
- **Local LLM Engine**: Runs entirely offline on CPU (Qwen2.5-1.5B-Instruct GGUF) - no API keys required, no external inference services dependencies.
- **Hybrid Retrieval**: Combines `FAISS` (dense, `intfloat/multilingual-e5-small`) and `BM25` (sparse) using Reciprocal Rank Fusion and Cross-Encoder re-ranking.
- **Durable Storage via Hugging Face Dataset**: Sessions, FAISS/BM25 indices, and uploaded documents are synchronized to a Hugging Face Hub Dataset repo, preventing data loss upon Space restarts (local disk on HF Spaces is ephemeral).
- **Parallel Web UI & REST API**: Features a web interface (`src/frontend/templates/index.html`) alongside a FastAPI REST API (`/api/sessions/*`).

---

## Project Structure

```text
docbrain-rag/
├── config/
│   └── config.yaml           # Configuration for RAG, Embedder, Model, Chunking, Storage
├── data/                      # Local cache (ephemeral on HF Spaces) - real source is HF Dataset
├── src/
│   ├── core/                  # DocumentProcessor, Chunker, Embedder, VectorStore, BM25Store,
│   │                          #   Retriever, LLMReader, RAGPipeline, HFDatasetStorage
│   ├── main.py                 # FastAPI app: defines all /api/sessions/* endpoints
│   └── frontend/               # Web UI Frontend (Vanilla HTML/CSS/JS)
├── scripts/
│   ├── ingest.py               # Script for static data ingestion (data/raw) to FAISS index
│   ├── test_qa.py              # CLI testing script for QA
│   └── benchmark_rag.py        # Benchmarking utility
├── tests/                       # Unit / integration tests
├── app.py                       # FastAPI server entrypoint (used for HF Spaces Docker)
├── requirements.txt
└── Dockerfile
```

---

## Installation & Local Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Fill in HF_TOKEN (write scope) and HF_DATASET_REPO_ID to enable durable storage.
# Optional: if left blank, the app will still run but data is stored locally (lost on restart).
```

### 3. Start Server (Local Host: 7860)
```bash
python app.py
```
*Web Interface: `http://localhost:7860` — Swagger API docs: `http://localhost:7860/docs`*

---

## Deployment to Hugging Face Spaces

1. Create a new **Space** on Hugging Face (`https://huggingface.co/new-space`), selecting the **Docker** SDK.
2. Push the entire source code to the Space repository (do not push the `.env` file - it is already in `.gitignore`).
3. In **Settings -> Repository Secrets**, declare:
   - `HF_TOKEN`: access token (write scope) - used for reading/writing data to the HF Dataset repo, **not used for LLM inference** (LLM runs completely locally).
   - `HF_DATASET_REPO_ID`: the dataset repo ID used for storage, e.g., `your-username/docbrain-storage`.

## License & Author

- Developed for educational and production RAG purposes.
- License: MIT
