from typing import List, Dict
import numpy as np

from .embed import embed_texts

def cosine_sim_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vecs / (np.linalg.norm(query_vecs, axis = 1, keepdims=True) + 1e-10)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    return (q @ d.T).ravel()

def top_k_retrieve(query: str, chunks: list[Dict], chunk_vecs: np.ndarray, k: int =  3)-> List[Dict]:
    query_vec = embed_texts([query], input_type="query")
    scores = cosine_sim_matrix(query_vec, chunk_vecs)
    
    ranked = np.argsort(scores)[::-1][:k]
    results: List[Dict] = []
    
    for idx in ranked:
        c= chunks[idx]
        results.append(
            {
                "score": float(scores[idx]),
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c["source"],
            }
        )
    return results