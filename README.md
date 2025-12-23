<<<<<<< HEAD
# Production-Grade RAG Agent using NVIDIA LLMs

A **production-style Retrieval-Augmented Generation (RAG) system** built from scratch using NVIDIA’s Integrate API.  
This project demonstrates **real-world RAG engineering practices**: document ingestion, robust chunking, embedding caching, semantic retrieval, citation-grounded answering, and automatic evaluation.

>  Goal: build a RAG system that is **explainable, reliable, and measurable** — not a demo script.

---

##  Key Features

-  Ingests **TXT / MD / PDF** documents
-  PDF-safe chunking with overlap (no broken paragraphs)
-  NVIDIA embeddings with **on-disk caching**
-  Cosine similarity retrieval (local, fast)
-  Strict citation-only answering (no hallucinations)
-  Confidence thresholds + clarification handling
-  Automatic RAG evaluation metrics (JSONL)
-  Modular, production-ready architecture

---

##  Project Structure

```
nvidia_rag/
│
├── app/
│   ├── agent.py        # Interactive production RAG agent
│   ├── cache.py        # Chunk fingerprinting + embedding cache
│   ├── chunk.py        # Robust chunking logic (PDF-safe)
│   ├── config.py       # Central configuration (models, paths, thresholds)
│   ├── embed.py        # NVIDIA embedding wrapper (batched)
│   ├── eval.py         # RAG evaluation + metrics logging
│   ├── ingest.py       # Document ingestion (TXT / MD / PDF)
│   ├── llm.py          # NVIDIA chat wrapper (retry-safe)
│   ├── prompt.py       # Strict citation prompt
│   └── retrieve.py     # Cosine similarity retrieval
│
├── data/
│   └── public_docs/    # Place source documents here
│
├── cache/              # Auto-generated chunks + embeddings
├── metrics/            # Auto-generated evaluation logs
│
├── scripts/
│   ├── ingest_and_chunk.py
│   ├── build_cache.py
│   ├── run_agent.py
│   └── show_metrics.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

##  Setup

### 1️⃣ Create & activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2️⃣ Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3️⃣ Set NVIDIA API key

```powershell
$env:NVIDIA_API_KEY="nvapi-your-key-here"
``` 
> Use `.env.example` as a template.

---

## ▶️ Running the System

### Step 1 — Add documents
Place your files in:

```
data/public_docs/
```

Supported formats:
- `.txt`
- `.md`
- `.pdf`

---

### Step 2 — Ingest & chunk documents

```powershell
python -m nvidia_rag.scripts.ingest_and_chunk
```

Example output:
```
Loaded documents: 4
Created chunks: 121
Saved: cache/chunks.json
```

---

### Step 3 — Build embedding cache (one-time per dataset)

```powershell
python -m nvidia_rag.scripts.build_cache
```

Example output:
```
Cache missing — embedding chunks once...
Chunk vectors shape: (121, 1024)
```

> Embeddings are cached using a fingerprint of content + model.  
> Cache is reused automatically unless documents change.

---

### Step 4 — Run the RAG agent

```powershell
python -m nvidia_rag.scripts.run_agent
```

Example interaction:

```
You: whats the procedure of Conversion from UG to GmbH?

Top sources:
- [Requirements_Company_founding#9] score=0.597
- [Requirements_Company_founding#2] score=0.588

Assistant:
A UG can convert to a GmbH once it accumulates €25,000 in share capital reserves [Requirements_Company_founding#9].
Additionally, the company must retain 25% of its annual profits until this threshold is reached [Requirements_Company_founding#2].
```

---

##  Evaluation & Metrics

Each query is logged automatically to:

```
metrics/rag_metrics.jsonl
```

Example record:

```json
{
  "query": "what is RAG?",
  "top_score": 0.39,
  "num_chunks": 3,
  "threshold_used": 0.25,
  "vague_query": false,
  "has_citation": true,
  "answer_len": 118,
  "sources": ["doc1#0"],
  "timestamp": "2025-12-22T13:01:45"
}
```

View metrics summary:

```powershell
python -m nvidia_rag.scripts.show_metrics
```

---

## 🧠 System Flow

```
User Query
   ↓
Query Embedding (NVIDIA)
   ↓
Cosine Similarity
   ↓
Top-K Relevant Chunks
   ↓
Citation-Strict Prompt
   ↓
NVIDIA LLM
   ↓
Answer + Evaluation Metrics
```

---

## 🛡️ Production Safeguards

- Content-based cache fingerprinting
- Chunk size enforcement (token-safe)
- Batched embedding requests
- Automatic retry on empty LLM responses
- Clarification prompts for low-confidence queries
- Strict citation enforcement (no hallucinations)

---

## 🧑‍💻 CLI Usage

```bash
rag ingest        # Ingest & chunk documents
rag build         # Build embedding cache
rag ask "..."     # Ask a single question
rag run           # Interactive agent


## 🧪 Example Questions

- `what is RAG?`
- `who is Abdul Rahman?`
- `what are the requirements for a GmbH?`
- `does the supervisory board required?`
- `who is he?` → agent requests clarification

---

## 📌 Design Philosophy

This project intentionally avoids vector databases to:
- demonstrate **core RAG mechanics**
- make retrieval transparent and debuggable
- focus on correctness before scale

The architecture can be extended to FAISS, Milvus, or Pinecone with minimal changes.

---

## 🏁 License

MIT

---

## 🙌 Acknowledgment

Built as a hands-on production learning project using NVIDIA’s LLM and embedding APIs.
=======
# production-rag-agent
A production-grade Retrieval-Augmented Generation (RAG) system using NVIDIA LLMs, featuring document ingestion, smart chunking, embedding caching, FastAPI backend, CLI tooling, Docker deployment, and automated evaluation metrics.
>>>>>>> 4d9f5603ae19c43863b41ec473d95d00dc7d1eff
