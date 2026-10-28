import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from news_qa.corpus import collect, fetch_article, make_passages, read_jsonl, write_jsonl
from news_qa.qa import ExtractiveQA


class CorpusTests(unittest.TestCase):
    def test_passage_coverage_and_source_boundaries(self):
        records = [{"source": "A", "text": " ".join(map(str, range(11)))},
                   {"source": "B", "text": "separate source"}]
        chunks = make_passages(records, words=5, overlap=2)
        a = [p for p in chunks if p["meta"]["source"] == "A"]
        self.assertEqual([(p["meta"]["word_start"], p["meta"]["word_end"]) for p in a],
                         [(0, 5), (3, 8), (6, 11)])
        self.assertEqual(set(" ".join(p["content"] for p in a).split()), set(map(str, range(11))))
        self.assertEqual(chunks[-1]["content"], "separate source")

    def test_failures_and_redirect_duplicates_are_explicit(self):
        def fetch(title, agent):
            if title == "Missing":
                raise ValueError("missing page")
            return {"page_id": 7, "source": "Canonical", "text": "text"}
        records, errors = collect(["Alias", "Canonical", "Missing"], "test", 0, fetch)
        self.assertEqual(len(records), 1)
        self.assertEqual(errors[0]["requested_title"], "Missing")

    def test_metadata_and_unicode_round_trip(self):
        payload = {"query": {"pages": [{"title": "News", "pageid": 1,
            "fullurl": "https://en.wikipedia.org/wiki/News", "extract": "İstanbul news",
            "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}]}]}}
        def opener(request, timeout):
            self.assertEqual(timeout, 30)
            self.assertIn("titles=News", request.full_url)
            return io.StringIO(json.dumps(payload))
        article = fetch_article("News", "test", opener)
        self.assertEqual(article["revision_id"], 42)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "articles.jsonl"
            write_jsonl(file, [article])
            self.assertEqual(read_jsonl(file), [article])

    def test_source_selects_separate_index_and_empty_answer_is_safe(self):
        class FakePipeline:
            def run(self, **kwargs):
                self.params = kwargs
                return {"answers": []}
        engine = object.__new__(ExtractiveQA)
        engine.sources = {"Forbes"}
        pipeline = FakePipeline()
        selected = []
        def choose(source):
            selected.append(source)
            return pipeline
        engine._pipeline = choose
        self.assertEqual(engine.ask("Where?", "Forbes")["answers"], [])
        self.assertEqual(selected, ["Forbes"])
        with self.assertRaises(ValueError):
            engine.ask("Where?", "Unknown")


if __name__ == "__main__":
    unittest.main()
