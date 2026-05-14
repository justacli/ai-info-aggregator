import json
import os
from openai import OpenAI

SCORE_PROMPT = """你是一个 AI 信息策展人。请分析以下文章，判断它是否值得推送给一个关注 AI 趋势和一人公司创业的读者。

文章标题：{title}
文章内容：{content}

请按以下 JSON 格式输出：
{{
  "topic": "主题分类（从以下选一个）：OPC/AI赚钱案例 | AI+电商 | AI工具实操/Agent工作流 | AI新技术/新模型 | 无关",
  "score": 评分(0-10，整数),
  "tags": ["标签1", "标签2"],
  "keep": true或false
}}

评分标准（信息密度 × 可操作性）：
有具体事实（数字/产品名/技术名）才有信息密度；读完之后能做某件事或做更好的判断，才有可操作性。
泛泛观点、营销软文、无数据的预测，直接 ≤4 分。

按主题的高分门槛：

▸ AI 新技术 / 新模型（得 8+ 分须满足）：
  - 有具体 benchmark 数据或与现有模型的对比
  - 有 API 可用性、定价或开源状态
  - 有实际能力演示或用户反馈
  缺以上任意两项 → 降 2 分

▸ OPC / AI 赚钱案例（得 8+ 分须满足）：
  - 有具体收入数字（月收入/年收入/ARR）
  - 有产品/服务形态描述
  - 有获客方式或冷启动路径
  - 小团队或个人可复制（非大公司赚钱案例）
  - 有清楚目标用户和真实痛点
  缺收入数字 → 最高 6 分；缺其余任意一项 → 降 1 分

▸ AI 工具实操 / Agent 工作流（得 8+ 分须满足）：
  - 工具实操类：有可直接复用的 prompt 或操作步骤，有效果对比
  - Agent/工作流类：有具体工具组合或架构描述，有实际效果或成本数据
  - GitHub 开源项目：有明确功能描述和使用场景即可
  只有概念介绍、无落地内容 → 最高 5 分


▸ AI + 电商（得 8+ 分须满足）：
  - 内容涵盖：AI 工具用于电商选品/广告/客服/独立站/内容生成/工作流/提示词等任意环节
  - 有具体工具名、操作步骤、效果数据或案例
  - 信息层面（行业动态）、实用技巧、新产品、工作流、提示词均视为高价值
  泛泛的"AI 将改变电商"无数据 → 最高 4 分


通用扣分项：
- 明显是 PR 稿 / 官方宣传：直接 ≤3 分
- 无原创内容，纯转载/聚合：-2 分
- 内容超过 14 天：-1 分
- 标题党，正文与标题严重不符：-3 分

keep 规则：score >= 7 且 topic != 无关 时为 true。
只输出 JSON，不要其他文字。"""

SUMMARY_PROMPT = """请为以下文章生成一段中文摘要，2-3 句话。

要求：
- 包含文章中最关键的具体事实（数字、产品名、技术名）
- 如果是 OPC/赚钱案例，必须包含：收入规模、实现方式、获客路径
- 如果是技术/模型，必须包含：核心能力提升、与现有方案的对比
- 如果是工具实操，必须包含：具体操作步骤或可复用的 prompt
- 不写"本文介绍了……"这类套话，直接陈述事实

文章标题：{title}
文章内容：{content}

只输出摘要文本，不要其他内容。"""


SCORING_MODEL = "deepseek-v4-flash"
SUMMARY_MODEL = "deepseek-v4-flash"

# DeepSeek official pricing for V4-Flash ($/M tokens), cache-miss list price.
# Actual cost may be lower: cache hits 1/10 input, night discount (UTC 16:30-00:30) -50%.
PRICE_IN_PER_M = 0.14
PRICE_OUT_PER_M = 0.28

USAGE = {"input": 0, "output": 0}


def _track(response) -> None:
    """Accumulate token usage from a chat completion response."""
    if getattr(response, "usage", None):
        USAGE["input"] += response.usage.prompt_tokens
        USAGE["output"] += response.usage.completion_tokens


def _reset_usage() -> None:
    USAGE["input"] = 0
    USAGE["output"] = 0


def _print_usage_summary() -> None:
    cost = USAGE["input"] / 1_000_000 * PRICE_IN_PER_M + USAGE["output"] / 1_000_000 * PRICE_OUT_PER_M
    print(f"\n[USAGE] tokens in={USAGE['input']:,} out={USAGE['output']:,} | est. cost ${cost:.4f}")




def _call_with_retry(client: OpenAI, model: str, prompt: str, max_tokens: int,
                     json_mode: bool = False, attempts: int = 2):
    """Call chat completion with retry on empty content or invalid JSON.
    Returns parsed dict (json_mode=True) or stripped text (json_mode=False).
    Raises the last exception if all attempts fail."""
    last_err: Exception | None = None
    for _ in range(attempts):
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        _track(response)
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            last_err = ValueError("empty response content")
            continue
        if json_mode:
            # Defensive: strip markdown fences if model still wraps despite json_mode
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                continue
        return raw
    raise last_err  # type: ignore[misc]


def score_article(article: dict, client: OpenAI) -> dict:
    """Score and classify an article. Returns article enriched with topic/score/tags/keep."""
    prompt = SCORE_PROMPT.format(
        title=article["title"],
        content=article["content"],
    )
    try:
        result = _call_with_retry(client, SCORING_MODEL, prompt, max_tokens=512, json_mode=True)
        topic = result.get("topic", "无关")
        score = int(result.get("score", 0))
        # Threshold 6 (5 for GitHub Trending) — calibrated for V4-Flash which scores
        # ~1.5 points lower than Haiku on identical input.
        is_github_trending = article.get("source") == "GitHub Trending"
        threshold = 6 if is_github_trending else 6
        keep = score >= threshold and topic != "无关"
        article.update({
            "topic": topic,
            "score": score,
            "tags": result.get("tags", []),
            "keep": keep,
        })
    except Exception as e:
        print(f"  [WARN] Scoring failed for '{article['title']}': {e}")
        article.update({"topic": "无关", "score": 0, "tags": [], "keep": False})
    return article


def summarize_article(article: dict, client: OpenAI) -> str:
    """Generate a 2-3 sentence Chinese summary for an article."""
    prompt = SUMMARY_PROMPT.format(
        title=article["title"],
        content=article["content"],
    )
    try:
        return _call_with_retry(client, SUMMARY_MODEL, prompt, max_tokens=512, json_mode=False)
    except Exception as e:
        print(f"  [WARN] Summary failed for '{article['title']}': {e}")
        return ""


DEDUP_PROMPT = """以下是一批通过质量筛选的文章列表，格式为 [序号] 标题 (来源)。

请找出其中报道同一件事/同一产品/同一发布的文章组。
同一件事的判断标准：核心主题相同（如都在报道 Gemma 4 发布、都在报道某工具上线）。

对于每个重复组，只保留序号最小的那篇（通常是评分最高的，因为列表已按评分降序排列）。

{articles}

请按以下 JSON 格式输出需要删除的文章序号，如果没有重复则 to_remove 为空数组：
{{"to_remove": [2, 5, 8]}}
只输出 JSON，不要其他文字。"""


def dedup_articles(articles: list[dict], client: OpenAI) -> tuple[list[dict], list[dict]]:
    """Remove duplicate articles covering the same event. Returns (deduped, dupes)."""
    if len(articles) <= 1:
        return articles, []

    # Sort by score descending so we always keep the highest-scoring version
    sorted_articles = sorted(articles, key=lambda x: x["score"], reverse=True)

    lines = "\n".join(
        f"[{i}] {a['title']} (来源: {a['source']}, 评分: {a['score']})"
        for i, a in enumerate(sorted_articles)
    )
    prompt = DEDUP_PROMPT.format(articles=lines)

    try:
        result = _call_with_retry(client, SCORING_MODEL, prompt, max_tokens=512, json_mode=True)
        to_remove = set(result.get("to_remove", []))
    except Exception as e:
        print(f"  [WARN] Dedup failed: {e}, skipping dedup")
        return articles, []

    deduped = [a for i, a in enumerate(sorted_articles) if i not in to_remove]
    dupes = [a for i, a in enumerate(sorted_articles) if i in to_remove]
    print(f"  Dedup: removed {len(dupes)} duplicate(s)")
    return deduped, dupes


def process_articles(articles: list[dict], api_key: str) -> tuple[list[dict], list[dict]]:
    """Score all articles, dedup, then summarize the ones worth keeping.
    Returns (kept, rejected) tuple.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    _reset_usage()

    print(f"\n[1/3] Scoring {len(articles)} articles...")
    scored = [score_article(a, client) for a in articles]

    kept = [a for a in scored if a["keep"]]
    rejected = [a for a in scored if not a["keep"]]
    print(f"  Kept {len(kept)} / {len(scored)} articles (score ≥ 6, GitHub Trending ≥ 5)")

    print(f"\n[2/3] Deduplicating {len(kept)} articles...")
    kept, dupes = dedup_articles(kept, client)
    rejected.extend(dupes)

    print(f"\n[3/3] Summarizing {len(kept)} articles...")
    for article in kept:
        article["summary"] = summarize_article(article, client)

    _print_usage_summary()
    return kept, rejected
