import tomllib
from pathlib import Path

REQUIRED_KEYS = {"name", "url", "lang"}
VALID_LANGS = {"en", "zh"}


def _load_feeds() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "feeds.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)["feeds"]


def test_all_feeds_have_required_keys():
    for feed in _load_feeds():
        missing = REQUIRED_KEYS - feed.keys()
        assert not missing, f"'{feed.get('name')}' 缺少字段: {missing}"


def test_all_langs_are_valid():
    for feed in _load_feeds():
        assert feed["lang"] in VALID_LANGS, (
            f"'{feed['name']}' 的 lang 值非法: '{feed['lang']}' (允许值: {VALID_LANGS})"
        )


def test_all_urls_start_with_http():
    for feed in _load_feeds():
        assert feed["url"].startswith("http"), (
            f"'{feed['name']}' URL 格式不对: '{feed['url']}'"
        )


def test_no_duplicate_urls():
    feeds = _load_feeds()
    urls = [f["url"] for f in feeds]
    seen, duplicates = set(), set()
    for url in urls:
        if url in seen:
            duplicates.add(url)
        seen.add(url)
    assert not duplicates, f"存在重复 URL: {duplicates}"


def test_no_duplicate_names():
    feeds = _load_feeds()
    names = [f["name"] for f in feeds]
    seen, duplicates = set(), set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    assert not duplicates, f"存在重复 name: {duplicates}"


def test_names_are_non_empty():
    for feed in _load_feeds():
        assert feed.get("name", "").strip(), "存在空的 name 字段"


def test_feed_count_is_reasonable():
    feeds = _load_feeds()
    assert len(feeds) >= 1, "feeds.toml 中没有任何订阅源"
    assert len(feeds) <= 200, f"订阅源数量过多 ({len(feeds)}), 请检查是否误添加"
