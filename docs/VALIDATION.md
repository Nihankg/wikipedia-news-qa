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

- Live Wikipedia collection.
- Fresh installation of the optional legacy Haystack dependency stack.
- End-to-end model download and inference for this packaged version.
- Accuracy, retrieval recall, or performance on answerable/unanswerable questions.

Before reporting model results, install the QA dependencies, run the demo and a real-source query, manually inspect source provenance, and save an evaluation set and corpus snapshot.
