"""Optional real BM25 regression test; no model download or inference."""
import importlib.util
import unittest
from news_qa.corpus import make_passages
from news_qa.qa import ExtractiveQA


@unittest.skipUnless(importlib.util.find_spec("haystack"), "Optional Haystack backend is not installed")
class SourceIsolationTests(unittest.TestCase):
    def test_other_outlet_is_excluded_from_actual_index(self):
        from haystack.nodes.base import BaseComponent

        class DummyReader(BaseComponent):
            outgoing_edges = "output_1"

            def run(self, documents, **kwargs):
                return {"answers": [], "documents": documents}, "output_1"

            def run_batch(self, **kwargs):
                return {}, "output_1"

        engine = object.__new__(ExtractiveQA)
        engine.passages = make_passages([
            {"source": "A", "text": "The agency was founded in 1998."},
            {"source": "B", "text": "The agency was founded in 2005."},
        ])
        engine.sources = {"A", "B"}
        engine.reader = DummyReader()
        engine.pipelines = {}
        pipeline = engine._pipeline("A")
        documents = pipeline.get_node("Retriever").retrieve("When was the agency founded?")
        self.assertTrue(documents)
        self.assertEqual({document.meta["source"] for document in documents}, {"A"})


if __name__ == "__main__":
    unittest.main()
