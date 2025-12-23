# Production‑Grade RAG Agent (NVIDIA LLMs)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)
![Docker](https://img.shields.io/badge/Docker-Containerized-informational)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-orange)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Integrate%20API-76B900)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **production-ready Retrieval‑Augmented Generation (RAG) system** built using **NVIDIA Integrate APIs**.

This repository is designed to be **easy to run, easy to understand, and easy to extend**.  
You can run it locally, via CLI, as an API, or fully containerized with Docker.

>  **Goal**: Give users a clear, step‑by‑step path to run a real RAG system without friction.

---

##  What This Project Does

This RAG system:
1. Reads your documents (TXT / MD / PDF)
2. Splits them into safe, meaningful chunks
3. Converts chunks into embeddings (cached locally)
4. Retrieves the most relevant chunks for a question
5. Generates an answer **only using retrieved sources**
6. Logs evaluation metrics for every query

No hidden magic. No black boxes.

---

##  Key Features

-  TXT / MD / PDF ingestion
-  Paragraph‑aware chunking with overlap
-  NVIDIA embeddings with on‑disk cache
-  Local cosine similarity retrieval
-  Citation‑strict answers (hallucination control)
-  Automatic evaluation metrics (JSONL)
-  CLI for daily usage
-  FastAPI server for production
-  Docker support

---

##  Project Structure (Simplified)

```
nvidia_rag/
├── app/            # Core RAG logic
├── data/           # Your documents
├── cache/          # Generated chunks & embeddings
├── metrics/        # Evaluation logs
├── scripts/        # CLI helpers
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

##  Requirements

- Python **3.11+**
- NVIDIA Integrate API key
- (Optional) Docker

---

##  Quick Start (Recommended Path)

### 1️. Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

### 3. Set NVIDIA API key

```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

>  Tip: Never commit API keys.  
> Use `.env.example` if needed.

---

##  Add Your Documents

Place your files here:

```
data/public_docs/
```

Supported formats:
- `.txt`
- `.md`
- `.pdf`

---

##  Step‑by‑Step Usage

### Step 1 - Ingest & chunk documents

```powershell
rag ingest
```

Expected output:
```
Loaded documents: 4
Created chunks: 120
Saved: cache/chunks.json
```

---

### Step 2 - Build embedding cache (one‑time)

```powershell
rag build
```

Expected output:
```
Cache missing — embedding chunks once...
Chunk vectors shape: (120, 1024)
```

>  This runs only once unless documents change.

---

### Step 3 - Ask a question

```powershell
rag ask "what is RAG?"
```

Example:
```
Answer:
RAG retrieves relevant chunks from a knowledge base and feeds them to an LLM [doc1#0].
```

---

### Step 4 - Interactive agent

```powershell
rag run
```

Ask multiple questions interactively.

---

##  Run as an API (FastAPI)

Start server:

```powershell
uvicorn app.server:app --reload
```

Available endpoints:
- `GET /health`
- `POST /ask`
- `GET /docs` (Swagger UI)

Example request:
```json
{
  "query": "what are the requirements for a GmbH?"
}
```

---

##  Run with Docker (Zero Setup)

```bash
docker compose up
```

Then open:
- http://127.0.0.1:8000/docs

---

##  Evaluation & Metrics

Every query is logged automatically:

```
metrics/rag_metrics.jsonl
```

Example:
```json
{
  "query": "what is RAG?",
  "top_score": 0.39,
  "has_citation": true,
  "sources": ["doc1#0"],
  "timestamp": "2025‑12‑22T13:01:45"
}
```

View metrics:
```powershell
rag metrics -n 5
```

---

##  How the System Works

```
User Question
   ↓
Query Embedding
   ↓
Cosine Similarity
   ↓
Top‑K Chunks
   ↓
Citation‑Strict Prompt
   ↓
NVIDIA LLM
   ↓
Answer + Metrics
```

---

##  Production Safeguards

- Chunk size enforcement (token‑safe)
- Content fingerprinted cache
- Strict citation validation
- Confidence thresholds
- Empty‑response retries
- Health checks

---

##  Design Philosophy

This project intentionally avoids vector databases to:
- Make RAG behavior transparent
- Keep debugging simple
- Teach core RAG concepts clearly

You can later swap retrieval for FAISS, Milvus, or Pinecone.

---

##  License

MIT

---

## 🙌 Acknowledgment

Built as a hands‑on production learning project using **NVIDIA Integrate LLM & Embedding APIs**.
