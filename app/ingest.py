import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import settings

@dataclass
class Document:
    doc_id: str
    source: str
    text: str   
    
def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}","\n\n", text)  # collapse multiple newlines
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse multiple spaces/tabs
    return text.strip()

def read_file_text(file_path: Path)-> str:
    ext = file_path.suffix.lower()
    if ext in [".txt",".md"]:
        return clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))
    
    if ext ==".pdf":
        import fitz
        doc = fitz.open(file_path)
        pages = []
        
        for page in doc:
            pages.append(page.get_text("text"))
        return clean_text("\n".join(pages))
    raise ValueError(f"Unsupported file type: {ext}")

def load_documents() -> List[Document]:
    data_dir = settings.data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    exts = {".txt", ".md", ".pdf"}
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    docs: List[Document] = []
    for p in sorted(files):
        text = read_file_text(p)
        if len(text) < 50:
            continue
        docs.append(Document(doc_id=p.stem, source=str(p), text=text))

    return docs