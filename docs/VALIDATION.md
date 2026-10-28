# Validation

Checked locally on October 28, 2026, using Python 3.12.14 for the dependency-free components.

## Passed

- Four unit tests: passage coverage/source boundaries; collection failures and alias deduplication; API provenance and Unicode JSONL round-trip; source-specific pipeline selection and an empty answer list.
- Command-line passage preparation using the bundled synthetic fixture: 30-word chunks with five-word overlap produced two passages.

The existing local Python 3.11 environment contains `farm-haystack` 1.26.4.post0, PyTorch 2.7.1, and Transformers 4.52.4. Its default import initially failed because of an unrelated Keras 3 compatibility issue. Disabling the unused TensorFlow/Flax integrations with `USE_TF=0 USE_FLAX=0` allowed the Haystack imports to succeed without changing installed packages.

The API parsing test uses a simulated response. The pipeline-selection test uses a simulated pipeline. Neither is a model accuracy measurement.

A real-backend check found that in-memory BM25 ignores metadata filters and can return documents from other sources. The implementation was changed to build source-specific indexes instead.

The corrected source-specific index passed a real Haystack BM25 integration check using two deliberately conflicting synthetic sources. A dummy reader was used, so this checks retrieval isolation, not neural-reader inference. This regression test is included in `tests/test_haystack_integration.py` and is skipped when Haystack is absent.

## Not yet validated

- Fresh installation of the optional legacy Haystack dependency stack.
- General accuracy, retrieval recall, or representative performance on answerable/unanswerable questions.

## End-to-end demonstration: October 28, 2026

- Live MediaWiki collection succeeded for Associated Press, Reuters, Forbes, and The New York Times; zero failed titles.
- Passage preparation produced 148 passages, with 180 words per passage and 30-word overlap.
- Downloaded and loaded `deepset/roberta-base-squad2` and ran actual CPU inference through Haystack.
- Ran the five questions in `examples/questions.json` with BM25 top-5 retrieval and reader top-3 answers, retaining all candidates.
- The top answer for each of four answerable questions matched a reference read from the collected article before inference.
- An intentionally unanswerable control returned the irrelevant span `NFT`, demonstrating the absence of a reliable abstention mechanism in this configuration.
- Run metadata and source revisions are recorded in `examples/real_run/`. This is a demonstration, not a benchmark.
