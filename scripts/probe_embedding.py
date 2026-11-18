"""穷举探测中转站是否有任何可用的 embedding 模型。"""

import json
import os

import urllib.error
import urllib.request

BASE = "https://new-api.xiaoduoai.com/v1"
TOKEN = os.environ.get("RELAY_TOKEN", "")

CANDIDATES = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-v3",
    "text-embedding-v4",
    "embedding-3",
    "bge-m3",
    "bge-large-zh-v1.5",
    "text-embedding-004",
    "nomic-embed-text",
    "qwen3-embedding-8b",
    "jina-embeddings-v3",
    "m3e-base",
]


def probe(model: str) -> str:
    payload = json.dumps({"model": model, "input": "hello"}).encode()
    req = urllib.request.Request(
        BASE + "/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode())
            dim = len(body["data"][0]["embedding"])
            return f"OK  dim={dim}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            msg = json.loads(body)["error"]["message"]
        except Exception:  # noqa: BLE001
            msg = body[:80]
        if "No available channel" in msg:
            return "NO_CHANNEL"
        return f"HTTP {exc.code}  {msg[:80]}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    print(f"{'model':30s} result")
    print("-" * 70)
    ok = []
    for m in CANDIDATES:
        r = probe(m)
        print(f"{m:30s} {r}")
        if r.startswith("OK"):
            ok.append(m)
    print()
    print("可用 embedding 模型:", ok if ok else "无 —— 中转站未开通任何 embedding 通道")
