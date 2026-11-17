#!/usr/bin/env python3
"""QA Config — 切换 settings.yaml 配置档，以测试不同的配置。

用法:
    python .github/skills/qa-tester/scripts/qa_config.py show          # 列出配置档
    python .github/skills/qa-tester/scripts/qa_config.py check         # 校验凭据
    python .github/skills/qa-tester/scripts/qa_config.py apply <name>  # 应用配置档
    python .github/skills/qa-tester/scripts/qa_config.py restore       # 恢复备份

配置档:
    deepseek         LLM → DeepSeek + Vision 关闭（需要 test_credentials.yaml）
    rerank_llm       Rerank → 基于 LLM
    no_vision        关闭 Vision LLM
    invalid_llm_key  LLM API key 设为无效值
    invalid_embed_key  Embedding API key 设为无效值
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_FILE = REPO_ROOT / "config" / "settings.yaml"
SETTINGS_BACKUP = REPO_ROOT / "config" / "settings.yaml.bak"
CREDENTIALS_FILE = REPO_ROOT / "config" / "test_credentials.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _save_yaml(path: Path, data: dict) -> None:
    # 通过定向编辑保留注释 —— 但为简单起见，
    # 使用 default_flow_style=False 的 dump
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def _backup() -> None:
    """若尚未备份则创建备份。"""
    if not SETTINGS_BACKUP.exists():
        shutil.copy2(SETTINGS_FILE, SETTINGS_BACKUP)
        print(f"   📋 Backup created: {SETTINGS_BACKUP.name}")
    else:
        print(f"   📋 Backup already exists")


def _load_credentials() -> dict | None:
    """加载 test_credentials.yaml。"""
    if not CREDENTIALS_FILE.exists():
        return None
    return _load_yaml(CREDENTIALS_FILE)


def apply_deepseek(settings: dict) -> dict:
    """将 LLM 切换为 DeepSeek，关闭 Vision。"""
    creds = _load_credentials()
    if not creds or "deepseek" not in creds:
        print("   ❌ DeepSeek credentials not found in test_credentials.yaml")
        sys.exit(1)

    ds = creds["deepseek"]
    api_key = ds.get("api_key", "")
    if not api_key or api_key.startswith("<"):
        print("   ❌ DeepSeek API key is not configured (still placeholder)")
        sys.exit(1)

    settings["llm"]["provider"] = "deepseek"
    settings["llm"]["model"] = ds.get("model", "deepseek-chat")
    settings["llm"]["api_key"] = api_key
    if "base_url" in ds:
        settings["llm"]["base_url"] = ds["base_url"]
    # 移除 Azure 专用字段
    for key in ["deployment_name", "azure_endpoint", "api_version"]:
        settings["llm"].pop(key, None)

    # 关闭 Vision LLM（DeepSeek 没有 Vision API）
    settings["vision_llm"]["enabled"] = False

    print("   LLM -> deepseek / " + ds.get("model", "deepseek-chat"))
    print("   Vision -> disabled")
    return settings


def apply_rerank_llm(settings: dict) -> dict:
    """启用基于 LLM 的 rerank。"""
    settings["rerank"]["enabled"] = True
    settings["rerank"]["provider"] = "llm"
    settings["rerank"]["top_k"] = 5
    print("   Rerank -> llm (top_k=5)")
    return settings


def apply_no_vision(settings: dict) -> dict:
    """关闭 Vision LLM。"""
    settings["vision_llm"]["enabled"] = False
    print("   Vision -> disabled")
    return settings


def apply_invalid_llm_key(settings: dict) -> dict:
    """将 LLM API key 设为无效值。"""
    settings["llm"]["api_key"] = "invalid_key_12345"
    print("   LLM api_key -> invalid_key_12345")
    return settings


def apply_invalid_embed_key(settings: dict) -> dict:
    """将 Embedding API key 设为无效值。"""
    settings["embedding"]["api_key"] = "invalid_key_12345"
    print("   Embedding api_key -> invalid_key_12345")
    return settings


PROFILES = {
    "deepseek": apply_deepseek,
    "rerank_llm": apply_rerank_llm,
    "no_vision": apply_no_vision,
    "invalid_llm_key": apply_invalid_llm_key,
    "invalid_embed_key": apply_invalid_embed_key,
}


def cmd_show() -> None:
    """显示可用的配置档。"""
    print("📋 Available Configuration Profiles")
    print("=" * 50)
    for name, fn in PROFILES.items():
        doc = fn.__doc__ or ""
        print(f"   {name:25s} {doc.strip()}")
    print()
    print(f"Current settings: {SETTINGS_FILE}")
    print(f"Backup exists: {SETTINGS_BACKUP.exists()}")


def cmd_check() -> None:
    """校验凭据文件。"""
    print("🔑 Credential Check")
    print("=" * 50)
    if not CREDENTIALS_FILE.exists():
        print(f"   ❌ {CREDENTIALS_FILE} not found")
        print(f"   → Copy from {CREDENTIALS_FILE.with_suffix('.yaml.example')}")
        return

    creds = _load_credentials()
    if creds and "deepseek" in creds:
        key = creds["deepseek"].get("api_key", "")
        if key and not key.startswith("<"):
            print(f"   ✅ DeepSeek API key: configured ({key[:8]}...)")
        else:
            print(f"   ⚠️  DeepSeek API key: placeholder (not configured)")
    else:
        print(f"   ⚠️  DeepSeek section: missing")


def cmd_apply(profile_name: str) -> None:
    """应用一个配置档。"""
    if profile_name not in PROFILES:
        print(f"❌ Unknown profile: {profile_name}")
        print(f"   Available: {', '.join(PROFILES.keys())}")
        sys.exit(1)

    print(f"🔄 Applying profile: {profile_name}")
    _backup()

    settings = _load_yaml(SETTINGS_FILE)
    settings = PROFILES[profile_name](settings)
    _save_yaml(SETTINGS_FILE, settings)
    print(f"   ✅ settings.yaml updated")


def cmd_restore() -> None:
    """从备份恢复 settings.yaml。"""
    if not SETTINGS_BACKUP.exists():
        print("❌ No backup found. Nothing to restore.")
        return

    shutil.copy2(SETTINGS_BACKUP, SETTINGS_FILE)
    SETTINGS_BACKUP.unlink()
    print("✅ settings.yaml restored from backup")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA Config — manage settings profiles")
    parser.add_argument(
        "command",
        choices=["show", "check", "apply", "restore"],
        help="Command to execute",
    )
    parser.add_argument(
        "profile",
        nargs="?",
        help="Profile name (for 'apply' command)",
    )
    args = parser.parse_args()

    if args.command == "show":
        cmd_show()
    elif args.command == "check":
        cmd_check()
    elif args.command == "apply":
        if not args.profile:
            print("❌ Profile name required. Usage: qa_config.py apply <profile>")
            sys.exit(1)
        cmd_apply(args.profile)
    elif args.command == "restore":
        cmd_restore()


if __name__ == "__main__":
    main()
