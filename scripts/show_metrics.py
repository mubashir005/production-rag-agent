import json
from nvidia_rag.app.config import settings

def main():
    path = settings.metrics_dir / "rag_metrics.jsonl"
    if not path.exists():
        print("No metrics file found yet.")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    print(f"Total records: {len(lines)}")

    last = json.loads(lines[-1])
    print("\nLast record:")
    for k, v in last.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
