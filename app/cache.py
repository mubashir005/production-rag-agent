import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from .config import settings
from .embed import embed_texts


def chunks_fingerprint(chunks: List[Dict]) -> str:
    payload = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def cache_paths(chunks_fp: str) -> Tuple[Path, Path]:
    safe_model = settings.embed_model.replace("/", "__")
    vec_path = settings.cache_dir / f"chunk_vectors_{safe_model}_{chunks_fp}.npy"
    meta_path = settings.cache_dir / f"chunk_vectors_{safe_model}_{chunks_fp}.json"
    return vec_path, meta_path


def load_chunks() -> List[Dict]:
    return json.loads(settings.chunks_file.read_text(encoding="utf-8"))


def build_or_load_chunk_vectors(chunks: List[Dict]) -> np.ndarray:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    fp = chunks_fingerprint(chunks)
    vec_path, meta_path = cache_paths(fp)

    if vec_path.exists():
        return np.load(vec_path)

    print("Cache missing — embedding chunks once (passage mode)...")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, input_type="passage")
    np.save(vec_path, vectors)

    meta = {
        "fingerprint": fp,
        "num_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "embed_model": settings.embed_model,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return vectors
