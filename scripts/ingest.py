#!/usr/bin/env python
"""Modular RAG MCP Server 的摄取脚本。

本脚本提供命令行接口，用于将文档摄取到知识中心。支持处理单个文件或整个目录。

用法:
    # 处理单个 PDF 文件
    python scripts/ingest.py --path documents/report.pdf --collection contracts

    # 处理目录中的所有 PDF
    python scripts/ingest.py --path documents/ --collection technical_docs

    # 强制重新处理（忽略之前的摄取记录）
    python scripts/ingest.py --path documents/report.pdf --collection contracts --force

    # 使用自定义配置文件
    python scripts/ingest.py --path documents/ --collection contracts --config custom_settings.yaml

退出码:
    0 - 成功（所有文件均已处理）
    1 - 部分失败（部分文件失败）
    2 - 完全失败（所有文件均失败或配置错误）
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

# 确保项目根目录位于 sys.path 中
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

# 为 Windows 控制台设置 UTF-8 编码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保项目根目录位于导入路径中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import load_settings, Settings
from src.core.trace import TraceContext, TraceCollector
from src.ingestion.pipeline import IngestionPipeline, PipelineResult
from src.observability.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to file or directory to ingest. "
             "If directory, processes all PDF files recursively."
    )
    
    parser.add_argument(
        "--collection", "-c",
        default="default",
        help="Collection name for organizing documents (default: 'default')"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-processing even if file was previously ingested"
    )
    
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without actually processing"
    )
    
    return parser.parse_args()


def discover_files(path: str, extensions: List[str] = None) -> List[Path]:
    """从路径中发现待处理的文件。

    参数:
        path: 文件或目录路径
        extensions: 要包含的文件扩展名列表 (default: ['.pdf'])

    返回:
        待处理的文件路径列表
    """
    if extensions is None:
        extensions = ['.pdf']
    
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {extensions}")
    
    # 目录：递归查找所有匹配的文件
    files = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))
        files.extend(path.rglob(f"*{ext.upper()}"))

    # 去重并排序
    files = sorted(set(files))
    
    return files


def print_summary(results: List[PipelineResult], verbose: bool = False) -> None:
    """打印处理摘要。

    参数:
        results: Pipeline 结果列表
        verbose: 是否打印详细信息
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    total_chunks = sum(r.chunk_count for r in results if r.success)
    total_images = sum(r.image_count for r in results if r.success)
    
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total}")
    print(f"  [OK] Successful: {successful}")
    print(f"  [FAIL] Failed: {failed}")
    print(f"\nTotal chunks generated: {total_chunks}")
    print(f"Total images processed: {total_images}")
    
    if verbose and failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  [FAIL] {r.file_path}: {r.error}")
    
    if verbose and successful > 0:
        print("\nSuccessful files:")
        for r in results:
            if r.success:
                skipped = r.stages.get("integrity", {}).get("skipped", False)
                status = "[SKIP] skipped" if skipped else f"[OK] {r.chunk_count} chunks"
                print(f"  {status}: {r.file_path}")
    
    print("=" * 60)


def main() -> int:
    """摄取脚本的主入口点。

    返回:
        退出码（0=成功，1=部分失败，2=完全失败）
    """
    args = parse_args()

    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("[*] Modular RAG Ingestion Script")
    print("=" * 60)
    
    # 加载配置
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[FAIL] Configuration file not found: {config_path}")
            return 2
        
        settings = load_settings(str(config_path))
        print(f"[OK] Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"[FAIL] Failed to load configuration: {e}")
        return 2
    
    # 发现文件
    try:
        files = discover_files(args.path)
        print(f"[INFO] Found {len(files)} file(s) to process")
        
        if len(files) == 0:
            print("[WARN] No files found to process")
            return 0
        
        for f in files:
            print(f"   - {f}")
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return 2
    except ValueError as e:
        print(f"[FAIL] {e}")
        return 2
    
    # 试运行模式
    if args.dry_run:
        print("\n[INFO] Dry run mode - no files were processed")
        return 0
    
    # 初始化 Pipeline
    print(f"\n[INFO] Initializing pipeline...")
    print(f"   Collection: {args.collection}")
    print(f"   Force: {args.force}")
    
    try:
        pipeline = IngestionPipeline(
            settings=settings,
            collection=args.collection,
            force=args.force
        )
    except Exception as e:
        print(f"[FAIL] Failed to initialize pipeline: {e}")
        logger.exception("Pipeline initialization failed")
        return 2
    
    # 处理文件
    print(f"\n[INFO] Processing files...")
    results: List[PipelineResult] = []
    
    collector = TraceCollector()

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path}")
        
        try:
            trace = TraceContext(trace_type="ingestion")
            trace.metadata["source_path"] = str(file_path)
            result = pipeline.run(str(file_path), trace=trace)
            collector.collect(trace)
            results.append(result)
            
            if result.success:
                skipped = result.stages.get("integrity", {}).get("skipped", False)
                if skipped:
                    print(f"   [SKIP] Skipped (already processed)")
                else:
                    print(f"   [OK] Success: {result.chunk_count} chunks, {result.image_count} images")
            else:
                print(f"   [FAIL] Failed: {result.error}")
        
        except Exception as e:
            logger.exception(f"Unexpected error processing {file_path}")
            results.append(PipelineResult(
                success=False,
                file_path=str(file_path),
                error=str(e)
            ))
            print(f"   [FAIL] Error: {e}")
    
    # 打印摘要
    print_summary(results, args.verbose)

    # 确定退出码
    successful = sum(1 for r in results if r.success)
    if successful == len(results):
        return 0  # 全部成功
    elif successful > 0:
        return 1  # 部分失败
    else:
        return 2  # 完全失败


if __name__ == "__main__":
    sys.exit(main())
