from typing import List, Dict


def build_prompt(query: str, retrieved: List[Dict]) -> str:
    blocks = []
    for r in retrieved:
        blocks.append(f"[{r['doc_id']}#{r['chunk_id']}]\n{r['text']}")  

    context = "\n\n".join(blocks)

    return f"""
You are an AI assistant.
Answer ONLY using the Sources below.

Rules:
- Use ONLY these exact source tags: [Source: <doc_id>#<chunk_id>] shown below.
- End each sentence with the matching tag like [Requirements_Company_founding#4].
- If the answer is not in the Sources, say exactly: "I don't know."
- Do not use outside knowledge.

Sources:
{context}

Question:
{query}

Answer:
""".strip()
