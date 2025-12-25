from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .retrieve import top_k_retrieve
from .prompt import build_prompt
from .llm import chat
from .embed import embed_texts  # <-- your NVIDIA embeddings wrapper


app = FastAPI(title="NVIDIA RAG Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Session storage (per-user)
# ----------------------------
SESSIONS_DIR = Path("cache/sessions")
ALLOWED_EXT = {".pdf", ".txt", ".md"}

# In-memory session cache: session_id -> (chunks, vectors)
SESSION_CACHE: Dict[str, Tuple[List[Dict[str, Any]], np.ndarray]] = {}


def session_dir(session_id: str) -> Path:
    if not session_id or len(session_id) < 8:
        raise HTTPException(status_code=400, detail="Missing or invalid x-session-id")
    d = SESSIONS_DIR / session_id
    (d / "docs").mkdir(parents=True, exist_ok=True)
    return d


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def read_file_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        # PyMuPDF required: pip install pymupdf
        import fitz  # type: ignore

        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_by_paragraphs(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    """
    Chunk by paragraphs (better for PDFs) + overlap.
    chunk_size is characters (simple & stable).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    # overlap: prepend tail of previous
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = out[-1][-overlap:]
            out.append((tail + "\n\n" + chunks[i]).strip())
        chunks = out

    return chunks


def build_chunks_from_docs(docs_dir: Path, chunk_size: int = 450, overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Reads all docs in docs_dir and returns chunks list.
    """
    files = []
    for ext in ("*.pdf", "*.txt", "*.md"):
        files.extend(docs_dir.glob(ext))

    if not files:
        raise HTTPException(status_code=400, detail="No documents uploaded for this session.")

    all_chunks: List[Dict[str, Any]] = []
    for file_path in sorted(files):
        text = clean_text(read_file_text(file_path))
        doc_id = file_path.stem
        pieces = chunk_by_paragraphs(text, chunk_size=chunk_size, overlap=overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "text": piece,
                    "source": str(file_path),
                }
            )

    return all_chunks


def load_session_state(session_id: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Load from memory cache first, otherwise from disk.
    """
    if session_id in SESSION_CACHE:
        return SESSION_CACHE[session_id]

    sdir = session_dir(session_id)
    chunks_path = sdir / "chunks.json"
    vecs_path = sdir / "vectors.npy"

    if not chunks_path.exists() or not vecs_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Session index not built yet. Upload docs then call /build.",
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vecs = np.load(vecs_path)

    SESSION_CACHE[session_id] = (chunks, vecs)
    return chunks, vecs


# ----------------------------
# Global dataset (optional)
# ----------------------------
global_chunks: List[Dict[str, Any]] = []
global_vecs: Optional[np.ndarray] = None


@app.on_event("startup")
def startup() -> None:
    """
    Optional: keep your old behavior for the default dataset
    (from settings.chunks_file / cache logic).
    If you don't want this, you can remove it safely.
    """
    global global_chunks, global_vecs
    try:
        from .cache import load_chunks, build_or_load_chunk_vectors

        global_chunks = load_chunks()
        global_vecs = build_or_load_chunk_vectors(global_chunks)
    except Exception:
        # Don't crash the server if global cache isn't present.
        global_chunks = []
        global_vecs = None


# ----------------------------
# API Models
# ----------------------------
class AskRequest(BaseModel):
    query: str
    k: int = 3


class AskResponse(BaseModel):
    query: str
    answer: str
    top_sources: list[dict]
    top_score: float


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "NVIDIA RAG API is running ",
        "docs": "/docs",
        "health": "/health",
        "upload": "POST /upload (x-session-id)",
        "build": "POST /build (x-session-id)",
        "ask": "POST /ask (x-session-id)",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "global_chunks": len(global_chunks),
        "global_cached_vectors": bool(global_vecs is not None),
        "sessions_cached_in_memory": len(SESSION_CACHE),
        "embed_model": settings.embed_model,
        "gen_model": settings.gen_model,
    }


@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    x_session_id: str = Header(default="", alias="x-session-id"),
):
    sdir = session_dir(x_session_id)
    docs_dir = sdir / "docs"
    saved = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")

        out_path = docs_dir / f.filename
        with out_path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    # If user uploads new docs, invalidate built index in memory + disk vectors
    SESSION_CACHE.pop(x_session_id, None)
    # (optional) keep old chunks/vectors, or delete to force rebuild:
    # for p in [sdir/"chunks.json", sdir/"vectors.npy"]:
    #     if p.exists(): p.unlink()

    return {"status": "ok", "session_id": x_session_id, "saved": saved}


@app.post("/build")
def build_session_index(
    x_session_id: str = Header(default="", alias="x-session-id"),
):
    """
    Creates:
      cache/sessions/<id>/chunks.json
      cache/sessions/<id>/vectors.npy
    """
    sdir = session_dir(x_session_id)
    docs_dir = sdir / "docs"

    chunks = build_chunks_from_docs(docs_dir, chunk_size=450, overlap=80)
    (sdir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    texts = [c["text"] for c in chunks]

    # Embedding API has token limits. Chunking above should keep it safe,
    # but we also embed in batches to be robust.
    vectors_list: List[np.ndarray] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = embed_texts(batch, input_type="passage")
        vectors_list.append(vecs)

    vectors = np.vstack(vectors_list).astype(np.float32)
    np.save(sdir / "vectors.npy", vectors)

    # Put into memory cache
    SESSION_CACHE[x_session_id] = (chunks, vectors)

    return {"status": "ok", "session_id": x_session_id, "chunks": len(chunks), "vectors_shape": list(vectors.shape)}


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    x_session_id: str = Header(default="", alias="x-session-id"),
) -> AskResponse:
    """
    Per-user RAG:
    - loads session chunks/vectors
    - retrieves top-k
    - calls NVIDIA LLM
    """
    query = req.query.strip()
    if not query:
        return AskResponse(query=req.query, answer="Query is empty.", top_sources=[], top_score=0.0)

    # If session provided, use session workspace; else fallback to global
    if x_session_id:
        chunks, vecs = load_session_state(x_session_id)
    else:
        if not global_chunks or global_vecs is None:
            return AskResponse(query=query, answer="No index available. Upload docs and build first.", top_sources=[], top_score=0.0)
        chunks, vecs = global_chunks, global_vecs

    retrieved = top_k_retrieve(query, chunks, vecs, k=req.k)  # type: ignore[arg-type]
    if not retrieved:
        return AskResponse(query=query, answer="I don't know.", top_sources=[], top_score=0.0)

    prompt = build_prompt(query, retrieved)
    answer = chat(prompt)

    # If NVIDIA returns empty sometimes, handle nicely
    if not answer:
        answer = "ERROR: Empty model response."

    top_score = float(retrieved[0]["score"]) if retrieved else 0.0
    top_sources = [{"source": f"{r['doc_id']}#{r['chunk_id']}", "score": float(r["score"])} for r in retrieved]

    return AskResponse(query=query, answer=answer, top_sources=top_sources, top_score=top_score)

@app.get("/status")
def status(x_session_id: str = Header(default="", alias="x-session-id")):
    sdir = session_dir(x_session_id)
    has_docs = (sdir / "docs").exists() and any((sdir / "docs").iterdir())
    has_chunks = (sdir / "chunks.json").exists()
    has_vecs = (sdir / "vectors.npy").exists()

    return {
        "session_id": x_session_id,
        "has_docs": bool(has_docs),
        "has_index": bool(has_chunks and has_vecs),
    }

