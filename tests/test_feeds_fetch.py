"""
Network-level feed parsing diagnostics.

Run all feeds (slow, ~2-5 min):
    python3 tests/test_feeds_fetch.py

Run as pytest (skipped by default unless --network flag passed):
    python3 -m pytest tests/test_feeds_fetch.py -m network -v -s

The script probes every source in feeds.toml and reports:
  ✓ OK      – feed reachable, at least 1 entry parsed
  ~ STALE   – feed reachable but 0 articles within the lookback window
  ✗ FAIL    – HTTP error or connection error
"""

from __future__ import annotations

import sys
import time
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests
import feedparser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feeds import load_feeds, fetch_feed  # noqa: E402

LOOKBACK_DAYS = 7          # wider window so recently inactive feeds aren't penalised
TIMEOUT_SECS  = 20
MAX_WORKERS   = 8
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── core probe ────────────────────────────────────────────────────────────────

def probe_feed(feed: dict) -> dict:
    """Return a result dict for a single feed."""
    result = {
        "name":        feed["name"],
        "url":         feed["url"],
        "lang":        feed["lang"],
        "status":      None,   # "ok" | "stale" | "fail"
        "http_code":   None,
        "total_entries": 0,
        "articles_in_window": 0,
        "sample_title": "",
        "error":       "",
        "elapsed_ms":  0,
    }
    t0 = time.monotonic()
    try:
        resp = requests.get(feed["url"], headers=HEADERS, timeout=TIMEOUT_SECS)
        result["http_code"] = resp.status_code
        resp.raise_for_status()

        parsed = feedparser.parse(resp.content)
        result["total_entries"] = len(parsed.entries)

        # grab sample title from first entry regardless of date
        if parsed.entries:
            result["sample_title"] = parsed.entries[0].get("title", "").strip()[:80]

        articles = fetch_feed(feed, lookback_days=LOOKBACK_DAYS)
        result["articles_in_window"] = len(articles)
        result["status"] = "ok" if articles else "stale"

    except Exception as exc:
        result["status"] = "fail"
        result["error"] = str(exc)[:120]
    finally:
        result["elapsed_ms"] = int((time.monotonic() - t0) * 1000)

    return result


# ── standalone runner ─────────────────────────────────────────────────────────

def _icon(status: str) -> str:
    return {"ok": "✓", "stale": "~", "fail": "✗"}.get(status, "?")


def run_all() -> list[dict]:
    feeds = load_feeds(str(ROOT / "feeds.toml"))
    results: list[dict] = []

    print(f"\nProbing {len(feeds)} feeds  (lookback={LOOKBACK_DAYS}d, workers={MAX_WORKERS})\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(probe_feed, f): f for f in feeds}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            icon = _icon(r["status"])
            extra = (
                f"{r['articles_in_window']} articles  [{r['elapsed_ms']}ms]"
                if r["status"] != "fail"
                else f"HTTP {r['http_code']}  {r['error']}"
            )
            print(f"  {icon} {r['name']:<35} {extra}")

    # ── summary ──────────────────────────────────────────────────────────────
    ok     = [r for r in results if r["status"] == "ok"]
    stale  = [r for r in results if r["status"] == "stale"]
    failed = [r for r in results if r["status"] == "fail"]

    print(f"\n{'─'*60}")
    print(f"  Total : {len(results)}  |  OK: {len(ok)}  Stale: {len(stale)}  Fail: {len(failed)}")

    if stale:
        print("\n  ~ Stale (reachable but no articles in window):")
        for r in stale:
            sample = f'  "{r["sample_title"]}"' if r["sample_title"] else ""
            print(f"      {r['name']}{sample}")

    if failed:
        print("\n  ✗ Failed:")
        for r in failed:
            print(f"      {r['name']}")
            print(f"        {textwrap.shorten(r['error'], 100)}")

    return results


# ── pytest entry points ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def all_results():
    return run_all()


@pytest.mark.network
def test_no_feeds_fail(all_results):
    failed = [r for r in all_results if r["status"] == "fail"]
    assert not failed, (
        f"{len(failed)} feed(s) failed:\n"
        + "\n".join(f"  {r['name']}: {r['error']}" for r in failed)
    )


@pytest.mark.network
def test_majority_feeds_return_articles(all_results):
    ok = [r for r in all_results if r["status"] == "ok"]
    total = len(all_results)
    pct = len(ok) / total * 100
    assert pct >= 50, (
        f"Only {len(ok)}/{total} ({pct:.0f}%) feeds returned articles – expected ≥50 %"
    )


@pytest.mark.network
@pytest.mark.parametrize("feed", load_feeds(str(ROOT / "feeds.toml")))
def test_feed_is_parseable(feed):
    """Each feed must at least return a valid HTTP response and a parseable RSS/Atom document."""
    resp = requests.get(feed["url"], headers=HEADERS, timeout=TIMEOUT_SECS)
    assert resp.status_code == 200, (
        f"{feed['name']} returned HTTP {resp.status_code}"
    )
    parsed = feedparser.parse(resp.content)
    assert parsed.entries or parsed.feed, (
        f"{feed['name']} returned an empty/invalid feed document"
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all()
    failed = [r for r in results if r["status"] == "fail"]
    sys.exit(1 if failed else 0)
