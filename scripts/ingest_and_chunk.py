import json
from nvidia_rag.app.config import settings
from nvidia_rag.app.ingest import load_documents
from nvidia_rag.app.chunk import make_chunks

def main():
    docs = load_documents()
    print(f"Loaded documents: {len(docs)}")
    chunks = make_chunks(docs, chunk_size=500, overlap=80)
    print(f"Created chunks: {len(chunks)}")
    
    settings.chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent = 2), encoding="utf-8")
    print(f"Saved: {settings.chunks_file}")
    
if __name__ == "__main__":
    main()