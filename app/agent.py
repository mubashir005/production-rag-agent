from typing import List, Dict

from .config import settings
from .cache import load_chunks, build_or_load_chunk_vectors
from .retrieve import top_k_retrieve
from .prompt import build_prompt
from .llm import chat
from .eval import evaluate, log_metrics, is_vague_query

def threshold_for(query: str) -> float:
    return settings.confident_score_vague if is_vague_query(query) else settings.confident_score

def main():
    print("NVIDIA RAG Agent (type 'exit' to quit)\n")
    
    chunks = load_chunks()
    _ = build_or_load_chunk_vectors(chunks)
    
    chunk_vecs = build_or_load_chunk_vectors(chunks)
    
    chat_history: List[Dict[str, str]] = [] # list of {"role":..., "content":...}
    
    while True:
        query = input("\nyou: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        
        retrieved = top_k_retrieve(query, chunks, chunk_vecs, k=settings.top_k)
        
        print("\n=== TOP RESULTS (retrieval) ===")
        for r in retrieved:
            print(f"-[{r['doc_id']}#{r['chunk_id']}] score={r['score']: .3f}")
        
        thr = threshold_for(query)
        top_score = retrieved[0]['score'] if retrieved else 0.0
        # Clarify if retrieval weak
        if not retrieved or top_score < thr:
            msg = (
                "I’m not confident I found the right information.\n"
                "Please clarify (e.g., name, topic, or what exactly you want), or rephrase your question."
            )
            print("\nAssistant:")
            print(msg)

            metrics = evaluate(query, retrieved, answer=msg, threshold_used=thr)
            out = log_metrics(settings.metrics_dir, metrics)
            print(f"(metrics saved to {out})")
            continue
        
        prompt = build_prompt(query, retrieved)
        
        memory = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]])
        final_prompt = f"""
        Conversation context (for resolving references like he/his/that):
        {memory}

        {prompt}
        """.strip()

        
        answer = chat(final_prompt)
        
        print("\nAssistant:")
        print(answer)
        
        metrics = evaluate(query, retrieved, answer=answer, threshold_used=thr)
        out = log_metrics(settings.metrics_dir, metrics)
        print(f"(metrics saved to {out})")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        
if __name__ == "__main__":
    main()