# Project history

## Original exploratory work

The project was assembled from Nihan Karagul's local files:

- `wikiscrapes.ipynb`: Wikipedia extraction and section/text processing.
- `WikiReader.ipynb`: collection of news-outlet article text, source metadata, an in-memory Haystack document store, and exploratory source-filtered questions.
- `haystack_qa.py`: BM25 retrieval and a pretrained `deepset/roberta-base-squad2` reader through `ExtractiveQAPipeline`.

The original files remain unchanged. They are not bundled in this upload package: notebooks contain environment-specific paths and exploratory state. Public results, execution success, and accuracy cannot be inferred merely from the presence of those cells.

## Packaging and changes added in October 2026

The following changes were prepared with AI assistance during application preparation, rather than represented as features of the original notebooks:

- A command-line interface and installable package layout.
- MediaWiki API extraction instead of BeautifulSoup HTML parsing.
- JSONL storage replacing the original Excel intermediates.
- Revision/source provenance, failure logs, redirect deduplication, and request pacing.
- Overlapping passages with stable IDs.
- A consistently initialized document store, retriever, reader, and pipeline.
- Explicit source validation, structured answer exports, a synthetic fixture, and regression tests.
- Source-specific indexes after a real-backend check showed that the in-memory BM25 retriever ignores metadata filters.

The original RoBERTa/BM25 extractive approach is retained. No model training, upstream contribution, or measured performance improvement is claimed.
