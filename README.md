# Production-Grade RAG Agent (NVIDIA LLMs)

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)
![Docker](https://img.shields.io/badge/Docker-Containerized-informational)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-orange)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Integrate%20API-76B900)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **production-grade Retrieval-Augmented Generation (RAG) system** built from scratch using **NVIDIA Integrate APIs**.

This project focuses on **real-world RAG engineering**, including document ingestion, robust chunking, embedding caching, semantic retrieval, citation-grounded answering, evaluation metrics, CLI tooling, API deployment, and Dockerization.

> **Goal:** Build a RAG system that is **explainable, reliable, and measurable** - not a demo or notebook experiment.

---

## Key Features

- Supports **TXT / MD / PDF** document ingestion  
- PDF-safe **paragraph-aware chunking with overlap**
- NVIDIA embeddings with **on-disk caching**
- Local **cosine similarity retrieval**
- **Strict citation-only answers**
- Automatic **RAG evaluation metrics**
- CLI + FastAPI + Docker support

---

## 📁 Project Structure

```
nvidia_rag/
├── app/
├── data/
├── cache/
├── metrics/
├── scripts/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:NVIDIA_API_KEY="nvapi-your-key-here"
```

---

## ▶ Run

```powershell
rag ingest
rag build
rag run
```

---

## 🌐 API

```powershell
uvicorn app.server:app --reload
```

Docs: http://127.0.0.1:8000/docs

---

## 🐳 Docker

```bash
docker compose up
```

---

## License

MIT
