from __future__ import annotations

from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from .config import settings
from .cache import load_chunks, build_or_load_chunk_vectors
from .retrieve import top_k_retrieve
from .prompt import build_prompt
from .llm import chat


app = FastAPI(title="NVIDIA RAG Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# These will be loaded once when the server starts
chunks = []
chunk_vecs = None


class AskRequest(BaseModel):
    query: str
    k: int = 3


class AskResponse(BaseModel):
    query: str
    answer: str
    top_sources: list[dict]
    top_score: float


@app.on_event("startup")
def startup() -> None:
    """
    This runs ONE time when the server starts.
    We load chunks + cached vectors here so we don't reload every request.
    """
    global chunks, chunk_vecs
    chunks = load_chunks()
    chunk_vecs = build_or_load_chunk_vectors(chunks)

@app.get("/")
def root():
    return {
        "message": "NVIDIA RAG API is running ✅",
        "try_docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health",
        "ask": "POST http://127.0.0.1:8000/ask",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Quick endpoint to check server is alive.
    """
    return {
        "status": "ok",
        "chunks": len(chunks),
        "cached_vectors": bool(chunk_vecs is not None),
        "embed_model": settings.embed_model,
        "gen_model": settings.gen_model,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    Main endpoint:
    - gets a question
    - retrieves top-k chunks
    - builds prompt
    - calls NVIDIA chat
    - returns answer + sources
    """
    query = req.query.strip()
    if not query:
        return AskResponse(query=req.query, answer="Query is empty.", top_sources=[], top_score=0.0)

    retrieved = top_k_retrieve(query, chunks, chunk_vecs, k=req.k)  # type: ignore[arg-type]
    if not retrieved:
        return AskResponse(query=query, answer="I don't know.", top_sources=[], top_score=0.0)

    prompt = build_prompt(query, retrieved)
    answer = chat(prompt)

    top_score = float(retrieved[0]["score"]) if retrieved else 0.0
    top_sources = [
        {"source": f"{r['doc_id']}#{r['chunk_id']}", "score": float(r["score"])}
        for r in retrieved
    ]

    return AskResponse(query=query, answer=answer, top_sources=top_sources, top_score=top_score)


    """
    End-to-end flow (important):
    
    HTTP POST /ask
            ↓
    FastAPI validates JSON
            ↓
    retrieve top chunks
            ↓
    build prompt
            ↓
    call NVIDIA LLM
            ↓
    return JSON answer + sources
    """