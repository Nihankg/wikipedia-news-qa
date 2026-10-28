import argparse
import json
import sys
from pathlib import Path
from .corpus import collect, make_passages, read_jsonl, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wikipedia news-outlet extractive QA")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("collect", help="Collect Wikipedia extracts with provenance")
    fetch.add_argument("--titles", required=True, type=Path)
    fetch.add_argument("--output", default="data/articles.jsonl")
    fetch.add_argument("--user-agent", required=True, help="Application name and real contact URL/email")
    fetch.add_argument("--delay", type=float, default=1.0)
    split = sub.add_parser("prepare", help="Create overlapping passages")
    split.add_argument("--input", required=True)
    split.add_argument("--output", default="data/passages.jsonl")
    split.add_argument("--words", type=int, default=180)
    split.add_argument("--overlap", type=int, default=30)
    ask = sub.add_parser("ask", help="Retrieve passages and extract answer spans")
    ask.add_argument("--corpus", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--source")
    ask.add_argument("--retrieve-k", type=int, default=5)
    ask.add_argument("--answer-k", type=int, default=3)
    ask.add_argument("--output", default="results/answers.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            titles = [t for t in args.titles.read_text(encoding="utf-8").splitlines()
                      if t.strip() and not t.lstrip().startswith("#")]
            records, failures = collect(titles, args.user_agent, args.delay)
            write_jsonl(args.output, records)
            write_jsonl(str(args.output) + ".failures.jsonl", failures)
            print(f"Collected {len(records)} articles; failed: {len(failures)}")
            return 1 if failures else 0
        if args.command == "prepare":
            passages = make_passages(read_jsonl(args.input), args.words, args.overlap)
            if not passages:
                raise ValueError("No passages produced; check the input corpus")
            write_jsonl(args.output, passages)
            print(f"Wrote {len(passages)} passages to {args.output}")
            return 0
        from .qa import ExtractiveQA
        engine = ExtractiveQA(read_jsonl(args.corpus))
        result = engine.ask(args.question, args.source, args.retrieve_k, args.answer_k)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
