from typing import List, Dict

from .ingest import Document


def chunk_by_paragraphs(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    def split_hard(t: str) -> List[str]:
        # fallback split if one paragraph is huge (PDF case)
        out = []
        start = 0
        while start < len(t):
            out.append(t[start:start + chunk_size].strip())
            start += max(chunk_size - overlap, 1)
        return [x for x in out if x]

    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        
        if len(p) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_hard(p))
            continue

        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks



def make_chunks(docs: List[Document], chunk_size: int = 500, overlap: int = 80) -> List[Dict]:
    all_chunks: List[Dict] = []

    for doc in docs:
        pieces = chunk_by_paragraphs(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "doc_id": doc.doc_id,
                    "chunk_id": i,
                    "text": piece,
                    "source": doc.source,
                }
            )

    return all_chunks
