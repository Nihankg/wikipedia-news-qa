# Wikipedia News QA

A Python research project for building a corpus of Wikipedia articles about news outlets and asking source-specific factual questions using **BM25 retrieval and a RoBERTa extractive reader**.

The project grew out of exploratory Wikipedia collection and Haystack notebooks. It consolidates that workflow into separate collection, passage-preparation, and question-answering commands.

This is **extractive question answering**: the reader selects answer spans from retrieved text. It does not generate new answers, train a new model, or establish the truth of the source material.

## Workflow

```text
Wikipedia article titles
  -> plain-text extracts with source and revision metadata
  -> overlapping, source-labeled passages
  -> BM25 index containing only the selected outlet (or all outlets)
  -> pretrained RoBERTa reader
  -> answer spans, context, scores, and source metadata
```

## Features

- Collect English Wikipedia article extracts through the MediaWiki API.
- Preserve canonical article titles, URLs, revision IDs, and collection timestamps.
- Log failed requests separately and deduplicate redirected article titles.
- Split documents into overlapping passages without mixing sources.
- Restrict retrieval to a specific outlet using its canonical source name.
- Export answers with context and character offsets for inspection.
- Run corpus preparation and unit tests without installing model dependencies.

## Installation

### Collection and preparation

These commands use only Python's standard library. Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
news-qa --help
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Optional model backend

The QA code retains the original **Haystack 1.x** API. Use a separate environment for this legacy dependency stack; Python 3.10 is recommended. End-to-end inference was validated in an existing Python 3.11 environment; a fresh installation of the optional dependency set has not yet been validated. See the recorded versions in [the real run](examples/real_run/results.json).

```bash
pip install -e '.[qa]'
```

If the environment already contains TensorFlow/Keras and Haystack imports fail, run the commands with `USE_TF=0 USE_FLAX=0` (this project uses PyTorch). An isolated environment is preferable to modifying a shared research environment.

Do not install `haystack-ai` alongside `farm-haystack` in this environment: they use the same import namespace. Model weights must be available locally or downloaded from Hugging Face on the first QA run. The default reader is `deepset/roberta-base-squad2`; no paid API key is required. Inference defaults to CPU.

## Quick start

### 1. Prepare an offline example

The repository includes a short **synthetic** article so passage preparation can be checked without web requests or model downloads.

```bash
news-qa prepare --input examples/synthetic_articles.jsonl --output data/demo-passages.jsonl
```

With the optional backend installed, run:

```bash
news-qa ask --corpus data/demo-passages.jsonl --source "Example News Agency" --question "When was Example News Agency founded?"
```

The fixture contains the year `1998`. This is an illustrative expected answer, not a reported model prediction or benchmark result.

### 2. Collect a real corpus

Edit `examples/outlets.txt` to contain exact Wikipedia article titles. Supply an informative User-Agent with your real contact details, following [Wikimedia's User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).

```bash
news-qa collect --titles examples/outlets.txt --output data/articles.jsonl --user-agent "WikipediaNewsQA/0.1 (your real contact URL or email)"
news-qa prepare --input data/articles.jsonl --output data/passages.jsonl --words 180 --overlap 30
news-qa ask --corpus data/passages.jsonl --source "Associated Press" --question "When was the Associated Press founded?" --output results/ap-answer.json
```

Use the canonical `source` value in the collected JSONL when setting `--source`; Wikipedia may redirect a requested title. Omitting `--source` searches all indexed outlets.

Collection waits one second between requests by default and uses a 30-second request timeout. Failures are written to `data/articles.jsonl.failures.jsonl`; the command returns exit code 1 if any title fails. Review the log and corpus coverage before proceeding. No automatic retries or checkpoint resume are implemented.

## Real question-answering examples

The pipeline was run on October 28, 2026 using four Wikipedia articles, 148 passages, BM25 top-5 retrieval, and the pretrained RoBERTa reader on CPU.

| Source | Question | Actual top answer | Manual check against collected article |
| --- | --- | --- | --- |
| Associated Press | When was the Associated Press founded? | 1846 | Supported |
| Reuters | Who established Reuters? | Paul Reuter | Supported |
| Forbes | Where is Forbes headquartered? | Jersey City, New Jersey | Supported |
| The New York Times | In what year was The New York Times founded? | 1851 | Supported |
| Associated Press | What is the password for the Associated Press internal payroll system? | NFT | Unsupported: unanswerable control |

These are five manually selected demonstrations, **not a representative accuracy benchmark**. Reference answers were read from the collected articles before inference. All candidates, context excerpts, model scores, source revisions, and runtime versions are retained in [the raw output](examples/real_run/results.json). The unsupported control illustrates that this configuration can return an irrelevant span instead of abstaining; its low model score is not a calibrated abstention rule.

To repeat the batch after collecting and preparing the corpus:

```bash
python -m news_qa.demo --corpus data/passages.jsonl --questions examples/questions.json --output results/demo.json
```

See [run notes and provenance](examples/real_run/README.md). Later collection may produce different Wikipedia text.

## Files and outputs

| Path | Purpose |
| --- | --- |
| `news_qa/corpus.py` | Collection, provenance, JSONL storage, and passage splitting |
| `news_qa/qa.py` | Haystack document store, BM25 retriever, and extractive reader |
| `news_qa/cli.py` | Command-line interface |
| `examples/outlets.txt` | Small editable list of news-outlet article titles |
| `examples/synthetic_articles.jsonl` | Offline synthetic fixture |
| `tests/test_pipeline.py` | Corpus and source-filter regression checks |
| `tests/test_haystack_integration.py` | Optional real BM25 source-isolation check, without model weights |
| `docs/PROJECT_HISTORY.md` | Original work and subsequent packaging changes |
| `docs/VALIDATION.md` | Checks performed and remaining validation |

Collected articles contain `source`, `url`, revision metadata, `fetched_at`, and `text`. Prepared passages contain `id`, `content`, and a provenance `meta` object. Answer exports contain the question, source filter, answer candidates, model scores, context, document IDs, and context offsets. Scores are model outputs, not calibrated factual-confidence estimates.

## Testing

From the project root:

```bash
python -m unittest discover -s tests -v
```

Tests cover passage coverage and source separation, failed titles and redirect deduplication, API metadata parsing and Unicode storage, and source-specific index selection with a simulated QA backend. They do not measure model accuracy. See [validation notes](docs/VALIDATION.md).

**Source isolation:** Haystack 1.x's in-memory BM25 retriever ignores metadata filters. This project creates an index containing only the requested source before retrieval, rather than relying on that unsupported filter. Within one Python process, source-specific pipelines are cached and share a reader.

## Limitations and research use

- Wikipedia coverage and article content change. Save corpus snapshots and revision metadata when reporting results; this collector records a revision ID but does not pin later re-collection to that revision.
- Exact article titles need manual review to avoid ambiguous names and wrong entities.
- Extractive answers may be incorrect or drawn from irrelevant passages. Questions about absent facts may still produce an answer; abstention is not calibrated here.
- Word-based passage splitting can break sentences, and words are not model tokens. The reader's own token-window handling is separate from corpus chunking.
- The in-memory index is rebuilt for each `ask` command; this is a small research workflow, not a deployed search service.
- A small manually referenced demonstration is included, but no representative evaluation set or general accuracy estimate is claimed. Model scores should not be presented as evaluation results.
- This project demonstrates Python corpus engineering and pretrained-transformer integration. It does not claim generative-LLM training, security evaluation, or a contribution to the Haystack upstream repository.

## Attribution and reuse

Original exploratory work: Nihan Karagul. The model and framework were developed by their respective authors; this project integrates them.

- [Haystack 1.26.4](https://pypi.org/project/farm-haystack/1.26.4/)
- [deepset/roberta-base-squad2 model card](https://huggingface.co/deepset/roberta-base-squad2)
- [Wikipedia copyright and reuse information](https://en.wikipedia.org/wiki/Wikipedia:Copyrights)

Wikipedia content and model weights have their own licenses and attribution requirements. Collected Wikipedia text is not bundled here. A license for this project's own code has not yet been selected; choose one before advertising the repository as freely reusable open-source software.
