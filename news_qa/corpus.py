"""Collection and passage preparation, using Python's standard library."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fetch_article(title, user_agent, opener=urlopen):
    """Fetch a plain-text extract and revision metadata from English Wikipedia."""
    query = urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "extracts|info|revisions", "explaintext": 1,
        "inprop": "url", "rvprop": "ids|timestamp", "redirects": 1,
        "titles": title,
    })
    request = Request("https://en.wikipedia.org/w/api.php?" + query,
                      headers={"User-Agent": user_agent})
    with opener(request, timeout=30) as response:
        payload = json.load(response)
    if "error" in payload:
        raise ValueError(f"Wikipedia API error: {payload['error']}")
    page = payload["query"]["pages"][0]
    if page.get("missing") or not page.get("extract", "").strip():
        raise ValueError(f"No article text found for {title!r}")
    revision = page.get("revisions", [{}])[0]
    return {
        "requested_title": title, "source": page["title"],
        "url": page["fullurl"], "page_id": page["pageid"],
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "text": page["extract"],
    }


def collect(titles, user_agent, delay=1.0, fetcher=fetch_article):
    """Continue past failed titles and return records plus an explicit failure log."""
    if delay < 0:
        raise ValueError("Delay must be nonnegative")
    records, failures, seen = [], [], set()
    titles = list(dict.fromkeys(t.strip() for t in titles if t.strip()))
    for index, title in enumerate(titles):
        if index:
            time.sleep(delay)
        try:
            record = fetcher(title, user_agent)
            if record["page_id"] not in seen:
                records.append(record)
                seen.add(record["page_id"])
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append({"requested_title": title, "error": str(exc)})
    return records, failures


def make_passages(records, words=180, overlap=30):
    """Split each source independently; retain provenance on every passage."""
    if words < 1 or not 0 <= overlap < words:
        raise ValueError("Require words > 0 and 0 <= overlap < words")
    passages = []
    for record in records:
        tokens = record["text"].split()
        # A stable ID identifies the source snapshot and the passage position.
        digest = hashlib.sha256(record["text"].encode()).hexdigest()[:12]
        for start in range(0, len(tokens), words - overlap):
            end = min(start + words, len(tokens))
            passages.append({
                "id": f"{record['source']}:{digest}:{start}",
                "content": " ".join(tokens[start:end]),
                "meta": {key: record[key] for key in (
                    "source", "url", "revision_id", "revision_timestamp", "fetched_at"
                ) if key in record} | {"word_start": start, "word_end": end},
            })
            if end == len(tokens):
                break
    return passages
