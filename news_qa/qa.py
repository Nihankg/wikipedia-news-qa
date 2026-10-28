"""Optional Haystack 1.x backend, preserving the original research approach."""


class ExtractiveQA:
    def __init__(self, passages, model="deepset/roberta-base-squad2", gpu=False):
        if not passages:
            raise ValueError("The corpus contains no passages")
        try:
            from haystack.document_stores import InMemoryDocumentStore
            from haystack.nodes import BM25Retriever, FARMReader
            from haystack.pipelines import ExtractiveQAPipeline
            from haystack.schema import Document
        except ImportError as exc:
            raise RuntimeError(
                "QA requires the optional Haystack backend. In a separate Python 3.10 "
                "environment, install: pip install -e '.[qa]'"
            ) from exc
        self.sources = {p["meta"]["source"] for p in passages}
        self.passages = passages
        self.reader = FARMReader(model_name_or_path=model, use_gpu=gpu)
        self.pipelines = {}

    def _pipeline(self, source):
        # Haystack 1.x's in-memory BM25 ignores metadata filters. Restrict the
        # indexed documents instead, so source isolation is enforced before retrieval.
        from haystack.document_stores import InMemoryDocumentStore
        from haystack.nodes import BM25Retriever
        from haystack.pipelines import ExtractiveQAPipeline
        from haystack.schema import Document
        if source not in self.pipelines:
            passages = self.passages if source is None else [
                p for p in self.passages if p["meta"]["source"] == source
            ]
            store = InMemoryDocumentStore(use_bm25=True)
            store.write_documents([
                Document(id=p["id"], content=p["content"], meta=p["meta"])
                for p in passages
            ])
            self.pipelines[source] = ExtractiveQAPipeline(
                reader=self.reader, retriever=BM25Retriever(document_store=store)
            )
        return self.pipelines[source]

    def ask(self, question, source=None, retrieve_k=5, answer_k=3):
        if not question.strip():
            raise ValueError("Question must not be empty")
        if retrieve_k < 1 or answer_k < 1:
            raise ValueError("top-k values must be positive")
        if source is not None and source not in self.sources:
            raise ValueError(f"Unknown source {source!r}; available: {sorted(self.sources)}")
        retrieval = {"top_k": retrieve_k}
        result = self._pipeline(source).run(query=question, params={
            "Retriever": retrieval, "Reader": {"top_k": answer_k}
        })
        return {
            "question": question, "source_filter": source,
            "answers": [{
                "answer": answer.answer, "score": float(answer.score) if answer.score is not None else None,
                "context": answer.context, "document_id": answer.document_ids[0]
                if answer.document_ids else None,
                "meta": answer.meta,
                "offsets_in_context": [
                    {"start": span.start, "end": span.end}
                    for span in (answer.offsets_in_context or [])
                ],
            } for answer in result.get("answers", [])],
        }
