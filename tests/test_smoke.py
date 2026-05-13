import json

import importlib.util
from pathlib import Path


def test_pytest_is_working():
    assert 1 + 1 == 2


def test_send_feishu_post_md_payload_shape():
    root = Path(__file__).resolve().parents[1]
    path = root / "sender" / "send_feishu.py"
    spec = importlib.util.spec_from_file_location("send_feishu_cli", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = json.loads(mod._post_md_content_json("**x**", title="hi"))  # noqa: SLF001
    assert raw["zh_cn"]["title"] == "hi"
    assert raw["zh_cn"]["content"] == [[{"tag": "md", "text": "**x**"}]]
