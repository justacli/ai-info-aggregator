#!/usr/bin/env python3
"""
Send messages to Feishu (Lark) via the official Python SDK (im/v1/message).

The ``sender/`` directory holds small CLIs for pushing messages to different platforms
(Feishu, and others you add later). This code is separate from ``src/`` application logic.

Credentials and target come from the environment — do not hardcode app secrets.

If a ``.env`` file exists at the **repository root**, it is loaded automatically (only keys
that are not already set in the environment). You can still use ``set -a; source .env; set +a``.

Required env:
  FEISHU_APP_ID
  FEISHU_APP_SECRET
  FEISHU_RECEIVE_ID       # e.g. user_id or chat_id value (see FEISHU_RECEIVE_ID_TYPE)

Optional env:
  FEISHU_RECEIVE_ID_TYPE  # default: user_id  (also: chat_id, open_id, union_id, email)
  FEISHU_MESSAGE          # body (overrides --file / default digest file)
  FEISHU_LOG_LEVEL        # DEBUG or INFO (default INFO)
  FEISHU_MSG_FORMAT       # auto | text | post-md (default auto; see --format)
  FEISHU_POST_TITLE       # optional title when using post-md
  FEISHU_MAX_CHARS        # truncate limit for msg_type=text (default 8000)
  FEISHU_POST_MD_MAX_CHARS  # truncate limit inside md tag (default 28000; rich text ~30KB cap)
  FEISHU_RAW_ANGLE_BRACKETS  # if "1"/"true", do not rewrite < > for **text** mode only
  FEISHU_SKIP_FLATTEN      # if "1"/"true", skip Markdown flatten (**text** mode only)

**Markdown digests** default to ``msg_type=post`` with a single ``{"tag":"md","text":...}`` node; see
https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json
That avoids ``text`` lightweight-markup validation (**230001**) on arbitrary ``*``, ``@``, ``[]``, etc.

``msg_type=text`` remains available (``--format text``): angle brackets / ``*`` / ``@`` are sanitized,
Markdown is flattened — use for short plain notices.

CLI:
  python sender/send_feishu.py
  python sender/send_feishu.py --file output/AI\\ Daily\\ -\\ 2026-05-13.md
  python sender/send_feishu.py --format text --file note.md
  python sender/send_feishu.py --text "hello"

Docs: https://open.feishu.cn/document/server-docs/im-v1/message/create
Content: https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json
SDK: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/server-side-sdk/python--sdk/preparations-before-development
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    CreateMessageResponse,
)

# Feishu text: keep inner "text" field modest; huge bodies risk API errors.
_DEFAULT_MAX_CHARS = 8_000
# post + md: stay under rich-text ~30KB guidance in open docs
_DEFAULT_POST_MD_MAX_CHARS = 28_000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_repo_dotenv() -> None:
    """Load repo-root .env into os.environ (does not override existing variables)."""
    path = _repo_root() / ".env"
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or " " in key or "#" in key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val


def _latest_digest_md(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    md_files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return md_files[0] if md_files else None


def _use_post_md(args: argparse.Namespace, fmt: str) -> bool:
    """Whether to send as ``post`` + ``md`` (see Feishu create_json)."""
    if fmt == "post-md":
        return True
    if fmt == "text":
        return False
    # auto
    if args.text is not None:
        return False
    if os.environ.get("FEISHU_MESSAGE") is not None and args.file is None:
        return False
    if args.file is not None:
        return Path(args.file).suffix.lower() == ".md"
    latest = _latest_digest_md(_repo_root() / "output")
    return bool(latest and latest.suffix.lower() == ".md")


def _strip_yaml_frontmatter(text: str) -> str:
    """
    Remove leading YAML ``--- ... ---`` block if present.

    Digest files often include ``tags: [ai-daily]``; Feishu ``text`` validates bracket-heavy
    fragments and may return **230001** for patterns that look like invalid markup.
    """
    if not text.startswith("---"):
        return text
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[i + 1 :]).lstrip("\n")
    return text


def _read_message_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    env_msg = os.environ.get("FEISHU_MESSAGE")
    if env_msg is not None:
        return env_msg
    if args.file is not None:
        path = Path(args.file).expanduser()
        if not path.is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8", errors="replace")
    latest = _latest_digest_md(_repo_root() / "output")
    if latest is None:
        print(
            "Error: no --file/--text and no FEISHU_MESSAGE; also no .md files under output/.",
            file=sys.stderr,
        )
        sys.exit(1)
    return latest.read_text(encoding="utf-8", errors="replace")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = max_chars
    snap = text.rfind("\n", 0, cut + 1)
    if snap >= cut // 2:
        cut = snap
    base = text[:cut].rstrip()
    return base + "\n\n…(truncated)"


def _flatten_markdown_for_feishu(text: str) -> str:
    """
    Feishu text msg validates ** / ` / []() patterns. Truncation leaves broken pairs → 230001.
    Strip common Markdown so the body is mostly plain text.
    """
    # **bold** (repeat: non-greedy inner without *)
    for _ in range(200):
        new, n = re.subn(r"\*\*([^*]+)\*\*", r"\1", text, count=1)
        if n == 0:
            break
        text = new
    text = text.replace("**", "")
    # `` `inline` `` (repeat)
    for _ in range(500):
        new, n = re.subn(r"`([^`]+)`", r"\1", text, count=1)
        if n == 0:
            break
        text = new
    text = text.replace("`", "'")
    # [title](url)
    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r"\1 \2", text)
    return text


def _minimal_control_strip(text: str) -> str:
    """Drop NUL / C0 controls; keep newlines (for post/md)."""
    text = text.replace("\x00", "")
    return "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")


def _post_md_content_json(body: str, *, title: str) -> str:
    """
    Build ``content`` string for ``msg_type=post`` with one ``md`` paragraph.

    See: https://open.feishu.cn/document/server-docs/im-v1/message-content-description/create_json
    """
    payload = {
        "zh_cn": {
            "title": title,
            "content": [[{"tag": "md", "text": body}]],
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def _sanitize_feishu_text(text: str, *, preserve_angle_brackets: bool) -> str:
    """
    Make content safe for msg_type=text (avoid 230001).

    Feishu validates ``*…*`` (italic) and ``@`` (mentions) in ``text`` bodies. Digests often
    contain multiplication like ``2344 * 5252`` or prose like ``@mention``, which can fail
    validation after Markdown flattening. Replace with fullwidth homoglyphs that read the same.
    """
    text = _minimal_control_strip(text)
    if not preserve_angle_brackets:
        # U+FF1C / U+FF1E: fullwidth less-than / greater-than (readable, not parsed as tags)
        text = text.replace("<", "\uff1c").replace(">", "\uff1e")
    # U+FF0A FULLWIDTH ASTERISK, U+FF20 FULLWIDTH COMMERCIAL AT
    text = text.replace("*", "\uff0a").replace("@", "\uff20")
    return text


def main() -> None:
    _load_repo_dotenv()

    parser = argparse.ArgumentParser(description="Send messages to Feishu via lark-oapi SDK.")
    parser.add_argument("--file", "-f", help="Markdown or text file to send as the message body.")
    parser.add_argument("--text", "-t", help="Literal message string (small payloads only).")
    parser.add_argument(
        "--format",
        choices=("auto", "text", "post-md"),
        default=None,
        help="auto: .md file / latest digest → post+md; else text. Override: env FEISHU_MSG_FORMAT.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate body (text default 8000, post-md default 28000; see FEISHU_* env).",
    )
    parser.add_argument(
        "--preserve-angle-brackets",
        action="store_true",
        help="Do not rewrite < > (same as FEISHU_RAW_ANGLE_BRACKETS=1). Text mode only.",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Skip Markdown flattening (same as FEISHU_SKIP_FLATTEN=1). Text mode only.",
    )
    args = parser.parse_args()

    app_id = os.environ.get("FEISHU_APP_ID", "").strip()
    app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
    receive_id = os.environ.get("FEISHU_RECEIVE_ID", "").strip()
    receive_id_type = os.environ.get("FEISHU_RECEIVE_ID_TYPE", "user_id").strip()

    if not app_id or not app_secret:
        print("Error: set FEISHU_APP_ID and FEISHU_APP_SECRET in the environment.", file=sys.stderr)
        sys.exit(1)
    if not receive_id:
        print("Error: set FEISHU_RECEIVE_ID (user_id, chat_id, etc.).", file=sys.stderr)
        sys.exit(1)

    fmt = (args.format or os.environ.get("FEISHU_MSG_FORMAT", "auto") or "auto").strip().lower().replace(
        "_", "-"
    )
    if fmt not in ("auto", "text", "post-md"):
        fmt = "auto"
    use_post_md = _use_post_md(args, fmt)

    if args.max_chars is not None:
        limit = max(1, args.max_chars)
    elif use_post_md:
        limit = max(1, int(os.environ.get("FEISHU_POST_MD_MAX_CHARS", str(_DEFAULT_POST_MD_MAX_CHARS))))
    else:
        limit = max(1, int(os.environ.get("FEISHU_MAX_CHARS", str(_DEFAULT_MAX_CHARS))))

    body = _read_message_text(args)
    body = _strip_yaml_frontmatter(body)
    body = _truncate(body, limit)

    if use_post_md:
        body = _minimal_control_strip(body)
        title = os.environ.get("FEISHU_POST_TITLE", "").strip()
        content_json = _post_md_content_json(body, title=title)
        msg_type = "post"
    else:
        skip_flatten = args.no_flatten or os.environ.get("FEISHU_SKIP_FLATTEN", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not skip_flatten:
            body = _flatten_markdown_for_feishu(body)

        raw_brackets = args.preserve_angle_brackets or os.environ.get(
            "FEISHU_RAW_ANGLE_BRACKETS", ""
        ).strip().lower() in ("1", "true", "yes")
        body = _sanitize_feishu_text(body, preserve_angle_brackets=raw_brackets)
        content_json = json.dumps({"text": body}, ensure_ascii=False)
        msg_type = "text"

    log_level = os.environ.get("FEISHU_LOG_LEVEL", "INFO").upper()
    level = lark.LogLevel.DEBUG if log_level == "DEBUG" else lark.LogLevel.INFO

    client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(level)
        .build()
    )

    request = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(receive_id)
            .msg_type(msg_type)
            .content(content_json)
            .build()
        )
        .build()
    )

    response: CreateMessageResponse = client.im.v1.message.create(request)

    if not response.success():
        try:
            detail = json.dumps(json.loads(response.raw.content), indent=2, ensure_ascii=False)
        except Exception:
            detail = response.raw.content.decode("utf-8", errors="replace")
        print(
            f"Feishu API error: code={response.code} msg={response.msg} log_id={response.get_log_id()}\n{detail}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(lark.JSON.marshal(response.data, indent=2))


if __name__ == "__main__":
    main()
