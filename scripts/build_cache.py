from nvidia_rag.app.cache import load_chunks, build_or_load_chunk_vectors

def main():
    chunks = load_chunks()
    vecs = build_or_load_chunk_vectors(chunks)
    print("Chunk vectors shape:", vecs.shape)

if __name__ == "__main__":
    main()
