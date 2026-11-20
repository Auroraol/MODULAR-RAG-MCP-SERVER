"""Modular RAG Dashboard 的便捷启动器。

用法::

    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
"""

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """从脚本位置向上查找，定位包含 src/observability/dashboard/app.py 的项目根。"""
    for p in [start, *start.parents]:
        if (p / "src" / "observability" / "dashboard" / "app.py").exists():
            return p
    raise SystemExit(f"Error: 未找到 dashboard app.py（从 {start} 向上查找）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Modular RAG Dashboard")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    root = find_project_root(Path(__file__).resolve().parent)
    app_path = root / "src" / "observability" / "dashboard" / "app.py"

    cmd = [
        sys.executable,
        "-Xfrozen_modules=off",          # 消掉调试器 frozen modules 警告
        "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",      # 跳过首次邮箱向导
        "--browser.gatherUsageStats", "false",
    ]
    print(f"Starting Dashboard: {' '.join(cmd)}")

    # 清掉调试器注入的环境变量，避免子进程被 pydevd 干扰
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYDEVD_")}

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)  # headless 模式不自动开，这里补上

    try:
        result = subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

    if result.returncode != 0:
        print(f"Streamlit exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()