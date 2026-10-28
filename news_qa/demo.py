"""Run a saved question list with a single model instance; retain every result."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .corpus import read_jsonl
from .qa import ExtractiveQA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--questions", default="examples/questions.json", type=Path)
    parser.add_argument("--output", default="examples/real_run/results.json", type=Path)
    args = parser.parse_args()
    passages = read_jsonl(args.corpus)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    model = "deepset/roberta-base-squad2"
    engine = ExtractiveQA(passages, model=model)
    results = []
    for item in questions:
        start = perf_counter()
        result = engine.ask(item["question"], item["source"], retrieve_k=5, answer_k=3)
        result["elapsed_seconds"] = round(perf_counter() - start, 3)
        result["reference_answers"] = item["reference_answers"]
        if "note" in item:
            result["note"] = item["note"]
        results.append(result)
        print(json.dumps({"source": item["source"], "question": item["question"],
                          "top_answer": result["answers"][0]["answer"] if result["answers"] else None}), flush=True)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "device": "CPU", "retrieve_k": 5, "answer_k": 3,
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("farm-haystack", "transformers", "torch")},
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "passage_count": len(passages),
        "sources": sorted({p["meta"]["source"] for p in passages}),
        "results": results,
        "scope": "Five illustrative, manually selected questions; not a representative benchmark. References were read from the collected articles before inference. All predictions are retained, including the unanswerable control.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
