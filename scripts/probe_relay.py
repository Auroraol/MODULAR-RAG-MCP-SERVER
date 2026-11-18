"""探测第三方中转站（new-api）对项目所需能力的支持情况。

用法（用项目的 .venv 跑）：
    .venv/Scripts/python.exe probe_relay.py
"""

import base64
import json
import os
import sys
from pathlib import Path

import urllib.error
import urllib.request

BASE = "https://new-api.xiaoduoai.com/v1"
TOKEN = os.environ.get("RELAY_TOKEN", "")
IMG = Path(
    r"C:/Users/jindao/Documents/GitHub/MODULAR-RAG-MCP-SERVER"
    r"/tests/fixtures/sample_documents/test_vision_llm.jpg"
)


def post(path: str, payload: dict, timeout: int = 60) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body[:300]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def summarize(status: int, body) -> str:
    if status != 200:
        if isinstance(body, dict) and "error" in body:
            return f"HTTP {status}  {str(body['error'].get('message', ''))[:90]}"
        return f"HTTP {status}  {str(body)[:90]}"
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return (
        f"HTTP 200  content={content[:60]!r}"
        f"  reasoning_tokens={len(reasoning)}"
    )


def main() -> int:
    if not TOKEN:
        print("请先设置 RELAY_TOKEN 环境变量")
        return 1

    print("=" * 78)
    print("1) Anthropic 原生格式 /v1/messages（Claude Code 走这个）")
    print("=" * 78)
    req = urllib.request.Request(
        "https://new-api.xiaoduoai.com/v1/messages",
        data=json.dumps(
            {
                "model": "deepseek-v4.1-flash-kingsoft",
                "max_tokens": 40,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            }
        ).encode(),
        headers={
            "x-api-key": TOKEN,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
            blocks = [b.get("type") for b in d.get("content", [])]
            text = "".join(
                b.get("text", "") for b in d.get("content", []) if b.get("type") == "text"
            )
            print(f"  HTTP 200  blocks={blocks}  text={text[:60]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {type(exc).__name__}: {exc}")

    print()
    print("=" * 78)
    print("2) Chat 补全（项目 LLM 走这个）—— 检查 reasoner 是否会把 content 挤空")
    print("=" * 78)
    for label, mt in (("max_tokens=16", 16), ("max_tokens=300", 300)):
        status, body = post(
            "/chat/completions",
            {
                "model": "deepseek-v4.1-flash-kingsoft",
                "max_tokens": mt,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            },
        )
        print(f"  {label:18s} {summarize(status, body)}")

    print()
    print("=" * 78)
    print("3) Embeddings（项目 Embedding 必需）")
    print("=" * 78)
    for model in ("text-embedding-ada-002", "text-embedding-3-small"):
        status, body = post("/embeddings", {"model": model, "input": "hello world"})
        if status == 200 and isinstance(body, dict):
            vec = body["data"][0]["embedding"]
            print(f"  {model:26s} HTTP 200  dim={len(vec)}")
        else:
            msg = body.get("error", {}).get("message", "") if isinstance(body, dict) else body
            print(f"  {model:26s} HTTP {status}  {str(msg)[:80]}")

    print()
    print("=" * 78)
    print("4) Vision 多模态（PDF 图片描述必需）")
    print("=" * 78)
    if not IMG.exists():
        print(f"  测试图片不存在: {IMG}")
    else:
        b64 = base64.b64encode(IMG.read_bytes()).decode()
        print(f"  测试图片: {IMG.name} ({IMG.stat().st_size // 1024} KB)")
        for model in (
            "deepseek-v4.1-flash-kingsoft",
            "qwen3.8-max-kingsoft",
            "glm-5.3-kingsoft",
            "kimi-k3",
        ):
            status, body = post(
                "/chat/completions",
                {
                    "model": model,
                    "max_tokens": 60,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "用一句话描述这张图。"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                },
                timeout=120,
            )
            print(f"  {model:34s} {summarize(status, body)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
