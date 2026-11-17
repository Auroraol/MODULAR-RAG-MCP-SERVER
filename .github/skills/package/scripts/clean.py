"""清理项目以便分发 — 删除缓存、venv、构建产物、数据、日志。

用法:
    python .github/skills/package/scripts/clean.py          # 试运行（显示将要删除的内容）
    python .github/skills/package/scripts/clean.py --execute # 实际执行删除
    python .github/skills/package/scripts/clean.py --execute --keep-data  # 保留 data/ 与 logs/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 确保 Windows 下 stdout 使用 UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[4]  # .github/skills/package/scripts → 仓库根目录

# ── 待删除的目录 ───────────────────────────────────────────────────
REMOVE_DIRS: list[str] = [
    # Python 缓存
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.hypothesis",
    # 虚拟环境
    ".venv",
    "venv",
    "env",
    # 构建 / 分发产物
    "build",
    "dist",
    "*.egg-info",
    ".eggs",
    "sdist",
    "wheels",
    # IDE 配置
    ".idea",
    ".vscode",
    # Jupyter
    ".ipynb_checkpoints",
    # 覆盖率
    "htmlcov",
    ".tox",
    ".nox",
    # 杂项缓存
    "cache",
    ".cache",
    # 过期产物
    "nonexistent_traces.jsonl",
    # 测试数据缓存
    "test_data/chroma",
    # Claude skill 测试产物
    ".claude/skills/test-skill",
]

# 仅在未设置 --keep-data 时删除的目录
DATA_DIRS: list[str] = [
    "data",
    "logs",
]

# ── 待删除的文件 ───────────────────────────────────────────────────
REMOVE_FILES: list[str] = [
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    "**/*.so",
    "**/*.egg",
    "**/*.cover",
    "**/*.py,cover",
    ".coverage",
    ".coverage.*",
    "coverage.xml",
    "nosetests.xml",
    "pip-log.txt",
    "pip-delete-this-directory.txt",
    "*.log",
    # 绝不应随包发布的敏感信息
    ".env",
    ".env.local",
    ".env.*.local",
    "secrets.yaml",
    # Skill 缓存
    ".github/skills/auto-coder/.spec_hash",
    ".claude/skills/auto-coder/.spec_hash",
    # 配置备份
    "config/settings.yaml.bak",
    "config/settings.yaml.qa_backup",
    "config/test_credentials.yaml",
]

# ── 待脱敏的文件（将 API key 替换为占位符） ────────────────────────
SANITIZE_FILES: list[str] = [
    "config/settings.yaml",
]


# 整体删除的目录 — 跳过其中的 glob 匹配项
_WHOLE_REMOVE_DIRS = {".venv", "venv", "env", "data", "build", "dist", "cache", ".cache"}


def _glob_resolve(patterns: list[str]) -> list[Path]:
    """相对于 REPO_ROOT 展开 glob 模式，返回已存在的路径。"""
    found: list[Path] = []
    for pat in patterns:
        if "**" in pat or "*" in pat:
            found.extend(REPO_ROOT.glob(pat))
        else:
            p = REPO_ROOT / pat
            if p.exists():
                found.append(p)
    # 去重，并过滤掉位于整体删除目录下的路径
    result: list[Path] = []
    for p in sorted(set(found)):
        rel = p.relative_to(REPO_ROOT)
        parts = rel.parts
        # 若该路径位于将被整体删除的目录内，则跳过
        # （例如 .venv 已列出时，.venv/__pycache__ 是多余的）
        if len(parts) > 1 and parts[0] in _WHOLE_REMOVE_DIRS:
            continue
        result.append(p)
    return result


# 目录过大，不扫描其大小（仅显示 "large"）
_SKIP_SIZE_SCAN = {".venv", "venv", "env", "data"}


def _is_symlink(p: Path) -> bool:
    try:
        return p.is_symlink()
    except (OSError, PermissionError):
        return True


def _dir_size(d: Path) -> int | None:
    """返回文件总大小（字节），若目录过大无法扫描则返回 None。"""
    if d.name in _SKIP_SIZE_SCAN:
        return None
    try:
        return sum(
            f.stat().st_size
            for f in d.rglob("*")
            if f.is_file() and not _is_symlink(f)
        )
    except (OSError, PermissionError):
        return None


def _sanitize_settings(path: Path, *, dry_run: bool) -> list[str]:
    """将 settings.yaml 中真实的 API key / endpoint 替换为占位符。"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    import re
    actions: list[str] = []

    # 匹配 api_key 值（带引号或不带引号）
    def _replace_key(m: re.Match) -> str:
        prefix = m.group(1)
        actions.append(f"  Sanitize api_key in {path.relative_to(REPO_ROOT)}")
        return f'{prefix}"YOUR_API_KEY_HERE"'

    new_text = re.sub(
        r'(api_key:\s*)("[^"]*"|\'[^\']*\'|\S+)',
        _replace_key,
        text,
    )

    # 替换 azure_endpoint 的值
    def _replace_endpoint(m: re.Match) -> str:
        prefix = m.group(1)
        actions.append(f"  Sanitize azure_endpoint in {path.relative_to(REPO_ROOT)}")
        return f'{prefix}"https://YOUR_ENDPOINT.openai.azure.com/"'

    new_text = re.sub(
        r'(azure_endpoint:\s*)("[^"]*"|\'[^\']*\'|\S+)',
        _replace_endpoint,
        new_text,
    )

    if new_text != text and not dry_run:
        path.write_text(new_text, encoding="utf-8")

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean project for packaging")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    parser.add_argument("--keep-data", action="store_true", help="Keep data/ and logs/ directories")
    parser.add_argument("--no-sanitize", action="store_true", help="Skip API key sanitization")
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("🔍 DRY RUN — showing what would be removed (use --execute to delete)\n")
    else:
        print("🗑️  EXECUTING cleanup...\n")

    # ── 收集目标 ────────────────────────────────────────────────────
    dir_patterns = list(REMOVE_DIRS)
    if not args.keep_data:
        dir_patterns.extend(DATA_DIRS)

    dirs_to_remove = _glob_resolve(dir_patterns)
    files_to_remove = _glob_resolve(REMOVE_FILES)

    # ── 报告并执行：目录 ────────────────────────────────────────────
    total_size = 0
    if dirs_to_remove:
        print("📁 Directories to remove:")
        for d in dirs_to_remove:
            if d.is_dir():
                size = _dir_size(d)
                size_str = _human_size(size) if size is not None else "large"
                if size is not None:
                    total_size += size
                label = f"  {d.relative_to(REPO_ROOT)}/  ({size_str})"
                print(label)
                if not dry_run:
                    shutil.rmtree(d, ignore_errors=True)
        print()

    # ── 报告并执行：文件 ────────────────────────────────────────────
    if files_to_remove:
        print("📄 Files to remove:")
        for f in files_to_remove:
            if f.is_file():
                try:
                    size = f.stat().st_size
                except (OSError, PermissionError):
                    size = 0
                total_size += size
                print(f"  {f.relative_to(REPO_ROOT)}  ({_human_size(size)})")
                if not dry_run:
                    f.unlink(missing_ok=True)
        print()

    # ── 配置文件脱敏 ────────────────────────────────────────────────
    if not args.no_sanitize:
        print("🔐 Sanitize secrets in config files:")
        for pat in SANITIZE_FILES:
            for p in _glob_resolve([pat]):
                actions = _sanitize_settings(p, dry_run=dry_run)
                for a in actions:
                    print(a)
        print()

    # ── 汇总 ────────────────────────────────────────────────────────
    action = "Would free" if dry_run else "Freed"
    print(f"✅ {action} ~{_human_size(total_size)}")
    if dry_run:
        print("\nRe-run with --execute to apply.")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


if __name__ == "__main__":
    main()
