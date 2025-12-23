# app/cli.py
# Production-grade CLI for your NVIDIA RAG project
# Commands:
#   rag ingest
#   rag build
#   rag ask "question" --k 5
#   rag run
#   rag metrics -n 10
#   rag doctor

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Dict

from .config import settings
from .cache import load_chunks, build_or_load_chunk_vectors
from .retrieve import top_k_retrieve
from .prompt import build_prompt
from .llm import chat


def cmd_ingest(_: argparse.Namespace) -> None:
    # Reuse your existing script entrypoint
    from nvidia_rag.scripts.ingest_and_chunk import main as ingest_main  # type: ignore
    ingest_main()


def cmd_build(_: argparse.Namespace) -> None:
    chunks = load_chunks()
    vecs = build_or_load_chunk_vectors(chunks)
    print(f"Cache ready. chunk_vectors shape = {vecs.shape}")


def cmd_ask(args: argparse.Namespace) -> None:
    query = args.query.strip()
    if not query:
        print("ERROR: Query is empty.")
        sys.exit(1)

    chunks = load_chunks()
    chunk_vecs = build_or_load_chunk_vectors(chunks)

    retrieved = top_k_retrieve(query, chunks, chunk_vecs, k=args.k)

    if not retrieved:
        print("No chunks retrieved.")
        sys.exit(0)

    print("\nTop sources:")
    for r in retrieved:
        print(f"- [{r['doc_id']}#{r['chunk_id']}] score={r['score']:.3f}")

    prompt = build_prompt(query, retrieved)
    answer = chat(prompt)

    print("\nAnswer:\n")
    print(answer)


def cmd_run(_: argparse.Namespace) -> None:
    from .agent import main as agent_main
    agent_main()


def cmd_metrics(args: argparse.Namespace) -> None:
    path = settings.metrics_dir / "rag_metrics.jsonl"
    if not path.exists():
        print("No metrics file found yet.")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    n = min(args.n, total)

    print(f"Total records: {total} | Showing last {n}\n")

    last = [json.loads(x) for x in lines[-n:]]
    for rec in last:
        ts = rec.get("timestamp", "?")
        score = float(rec.get("top_score", 0.0))
        cite = rec.get("has_citation", False)
        q = rec.get("query", "")
        print(f"- {ts} | score={score:.3f} | cite={cite} | q={q}")


def cmd_doctor(_: argparse.Namespace) -> None:
    print("== RAG Doctor ==")

    key = os.getenv("NVIDIA_API_KEY")
    print("NVIDIA_API_KEY:", "SET ✅" if key else "MISSING ❌")

    print("Docs folder:", settings.docs_dir, "OK ✅" if settings.docs_dir.exists() else "MISSING ❌")
    print("Chunks file:", settings.chunks_file, "OK ✅" if settings.chunks_file.exists() else "MISSING ❌")
    print("Cache dir:", settings.cache_dir, "OK ✅" if settings.cache_dir.exists() else "MISSING ❌")
    print("Metrics dir:", settings.metrics_dir, "OK ✅" if settings.metrics_dir.exists() else "MISSING ❌")

    print("\nModels / Endpoint")
    print("Base URL:", settings.base_url)
    print("Embed model:", settings.embed_model)
    print("Gen model:", settings.gen_model)

    if not key:
        print("\nFix:")
        print('  PowerShell:  $env:NVIDIA_API_KEY="nvapi-..."')
        print("  Then re-run: rag doctor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Production RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Ingest and chunk documents").set_defaults(func=cmd_ingest)
    sub.add_parser("build", help="Build embedding cache").set_defaults(func=cmd_build)

    ask = sub.add_parser("ask", help="Ask a question (single-shot)")
    ask.add_argument("query", type=str, help="Your question in quotes")
    ask.add_argument("--k", type=int, default=settings.top_k, help="Top-k chunks to retrieve (default: settings.top_k)")
    ask.set_defaults(func=cmd_ask)

    sub.add_parser("run", help="Interactive RAG agent").set_defaults(func=cmd_run)

    m = sub.add_parser("metrics", help="Show recent evaluation metrics")
    m.add_argument("-n", type=int, default=10, help="How many recent records to show")
    m.set_defaults(func=cmd_metrics)

    sub.add_parser("doctor", help="Check environment + files").set_defaults(func=cmd_doctor)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
