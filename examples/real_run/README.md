# Real inference run

Run date: October 28, 2026. Model: `deepset/roberta-base-squad2`, CPU.

## Included files

- `results.json`: actual predictions for every question, including all three reader candidates, context, offsets, model scores, and runtime metadata.
- `sources.json`: source URLs, requested/canonical titles, page IDs, revision IDs and timestamps, and collection timestamps.
- `../questions.json`: questions and manually specified reference answers.

The model ran over 148 passages from four English Wikipedia articles. Passage length was 180 words, overlap 30 words, retrieval top-k 5, and reader top-k 3. Source-specific indexes restricted each question to its requested outlet.

The four answerable questions returned source-supported top answers. The unanswerable control returned `NFT`, an unrelated word from the AP article, at approximately 0.0701 model score. This is a failure to abstain, not an answer to the question.

These examples were manually selected to demonstrate the pipeline. They do not establish general accuracy, calibration, security robustness, or performance improvements. All five questions and their predictions are retained.

## Reproduction

Follow the collection/preparation instructions in the main README, then run:

```bash
python -m news_qa.demo --corpus data/passages.jsonl --questions examples/questions.json --output results/demo.json
```

The exact local prepared-corpus SHA-256 and installed package versions are recorded in `results.json`. A fresh collection uses current Wikipedia content and may differ from this run. Full article snapshots and model weights are not bundled. To inspect historical text, use the revision IDs in `sources.json` with Wikipedia's `oldid` URL parameter.

## Source attribution

Context excerpts in `results.json` are derived from English Wikipedia articles by their Wikipedia contributors. Every candidate includes its source URL and revision ID. Source articles and their revision histories:

- [Associated Press](https://en.wikipedia.org/w/index.php?title=Associated_Press&oldid=1378087166)
- [Reuters](https://en.wikipedia.org/w/index.php?title=Reuters&oldid=1381602016)
- [Forbes](https://en.wikipedia.org/w/index.php?title=Forbes&oldid=1375833728)
- [The New York Times](https://en.wikipedia.org/w/index.php?title=The_New_York_Times&oldid=1379161752)

The text was converted to plain text, split into passages, and excerpted by the reader. Wikipedia text is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) subject to applicable source-page notices. This attribution applies to the Wikipedia-derived text, independently of any license for the project's code or model weights.
