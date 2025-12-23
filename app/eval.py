import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def is_vague_query(q: str) -> bool:
    q = q.lower().strip()
    tokens = q.split()
    return any(t in tokens for t in ["he", "his", "she", "her", "him", "they", "their", "this", "that"])


def evaluate(query: str, retrieved: List[Dict], answer: str, threshold_used: float) -> Dict:
    top_score = float(retrieved[0]["score"]) if retrieved else 0.0
    answer_text = answer if isinstance(answer, str) else ""
    has_citation = ("[" in answer_text) and ("]" in answer_text)
    error = answer_text.startswith("ERROR:")

    return {
        "query": query,
        "top_score": top_score,
        "num_chunks": len(retrieved),
        "threshold_used": threshold_used,
        "vague_query": is_vague_query(query),
        "has_citation": has_citation,
        "error": error,
        "answer_len": len(answer_text),
        "sources": [f"{r['doc_id']}#{r['chunk_id']}" for r in retrieved],
    }


def log_metrics(metrics_dir: Path, record: Dict) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "rag_metrics.jsonl"
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path
