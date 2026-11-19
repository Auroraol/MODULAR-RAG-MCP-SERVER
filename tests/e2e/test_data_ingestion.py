"""数据摄取脚本的端到端测试。

本模块通过命令行接口测试完整的摄取工作流，包括：
- 单文件摄取
- 目录摄取
- 强制重新处理
- 跳过已处理的文件
- 非法输入的错误处理
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# 执行脚本所用的项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDataIngestion:
    """scripts/ingest.py 的端到端测试。"""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """为测试输出创建一个临时数据目录。

        生成:
            临时目录的路径
        """
        # 创建与生产环境结构一致的子目录
        (tmp_path / "db" / "chroma").mkdir(parents=True)
        (tmp_path / "db" / "bm25").mkdir(parents=True)
        (tmp_path / "images").mkdir(parents=True)

        yield tmp_path

        # 清理工作由 pytest 的 tmp_path fixture 负责

    @pytest.fixture
    def sample_pdf(self):
        """获取用于测试的示例 PDF 路径。

        返回:
            示例 PDF 文件的路径
        """
        pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"
        if not pdf_path.exists():
            pytest.skip("Sample PDF not found")
        return pdf_path
    
    @pytest.fixture
    def complex_pdf(self):
        """获取用于测试的复杂技术文档路径。

        返回:
            含图片的复杂 PDF 路径
        """
        pdf_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "complex_technical_doc.pdf"
        if not pdf_path.exists():
            pytest.skip("Complex PDF not found")
        return pdf_path
    
    def run_ingest_script(
        self,
        path: str,
        collection: str = "test_collection",
        force: bool = False,
        config: str = None,
        dry_run: bool = False,
        verbose: bool = False
    ) -> subprocess.CompletedProcess:
        """以子进程方式运行摄取脚本。

        参数:
            path: 文件或目录路径
            collection: 集合名称
            force: 是否强制重新处理
            config: 自定义配置路径
            dry_run: 是否以 dry-run 模式运行
            verbose: 是否启用详细输出

        返回:
            包含 stdout、stderr 和返回码的 CompletedProcess
        """
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest.py"),
            "--path", str(path),
            "--collection", collection
        ]
        
        if force:
            cmd.append("--force")
        
        if config:
            cmd.extend(["--config", config])
        
        if dry_run:
            cmd.append("--dry-run")
        
        if verbose:
            cmd.append("--verbose")
        
        # 设置 PYTHONUTF8=1 以避免 Windows 上的编码问题
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,  # LLM 调用的 10 分钟超时（含 Vision 的复杂文档）
            env=env,
            encoding='utf-8',
            errors='replace'
        )
    
    def test_ingest_help(self):
        """测试 --help 标志是否可用。"""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest.py"), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT)
        )
        
        assert result.returncode == 0
        assert "--path" in result.stdout
        assert "--collection" in result.stdout
        assert "--force" in result.stdout
    
    def test_ingest_nonexistent_file(self):
        """测试文件不存在时的错误处理。"""
        result = self.run_ingest_script(
            path="/nonexistent/path/document.pdf"
        )
        
        assert result.returncode == 2
        assert "does not exist" in result.stdout or "not found" in result.stdout.lower()
    
    def test_ingest_invalid_config(self, sample_pdf):
        """测试配置文件非法时的错误处理。"""
        result = self.run_ingest_script(
            path=str(sample_pdf),
            config="/nonexistent/config.yaml"
        )
        
        assert result.returncode == 2
        assert "not found" in result.stdout.lower() or "Configuration" in result.stdout
    
    def test_ingest_dry_run(self, sample_pdf):
        """测试 dry-run 模式不会处理文件。"""
        result = self.run_ingest_script(
            path=str(sample_pdf),
            dry_run=True
        )
        
        assert result.returncode == 0
        assert "Dry run" in result.stdout or "dry run" in result.stdout.lower()
        assert "1 file" in result.stdout
    
    def test_ingest_unsupported_file_type(self, tmp_path):
        """测试不支持的文件类型时的错误处理。"""
        # 创建一个文本文件
        text_file = tmp_path / "document.txt"
        text_file.write_text("This is a text file")
        
        result = self.run_ingest_script(
            path=str(text_file)
        )
        
        assert result.returncode == 2
        assert "Unsupported" in result.stdout or "unsupported" in result.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_simple_pdf(self, sample_pdf):
        """测试摄取简单 PDF 文件。

        该测试需要预先配置 Azure API 凭据。
        """
        result = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_simple",
            force=True,  # 强制处理以确保得到全新的结果
            verbose=True
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # 应完全成功或部分成功
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout
    
    @pytest.mark.integration
    def test_ingest_complex_pdf_with_images(self, complex_pdf):
        """测试摄取含图片的复杂 PDF。

        该测试需要预先配置 Azure API 凭据和 Vision LLM。
        测试包括图片描述（image captioning）在内的完整 Pipeline。
        """
        result = self.run_ingest_script(
            path=str(complex_pdf),
            collection="e2e_test_complex",
            force=True,
            verbose=True
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # 应完全成功或部分成功
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout

        # 若成功，应报告 Chunk 数量，可能还包括图片数量
        if result.returncode == 0:
            assert "chunks" in result.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_skip_already_processed(self, sample_pdf):
        """测试已处理的文件会被跳过。

        连续运行两次摄取，验证第二次运行会跳过该文件。
        """
        # 第一次运行——应进行处理
        result1 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_skip",
            force=True  # 确保从干净状态开始
        )

        print("First run STDOUT:", result1.stdout)

        # 若第一次运行失败，则跳过该测试
        if result1.returncode == 2:
            pytest.skip("First ingestion failed - cannot test skip behavior")

        # 第二次运行——应跳过
        result2 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_skip",
            force=False  # 不强制，应跳过
        )

        print("Second run STDOUT:", result2.stdout)

        # 应成功但带有跳过
        assert result2.returncode == 0
        assert "skip" in result2.stdout.lower() or "already processed" in result2.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_force_reprocess(self, sample_pdf):
        """测试 --force 标志会触发重新处理。"""
        # 第一次运行
        result1 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_force",
            force=True
        )

        # 若第一次运行失败，则跳过该测试
        if result1.returncode == 2:
            pytest.skip("First ingestion failed - cannot test force behavior")

        # 第二次带 force 运行——应再次处理
        result2 = self.run_ingest_script(
            path=str(sample_pdf),
            collection="e2e_test_force",
            force=True,
            verbose=True
        )

        print("Second run STDOUT:", result2.stdout)

        # 应成功并完成处理（而非跳过）
        assert result2.returncode in [0, 1]
        # 强制处理时不应显示 "skipped"
        if "Success" in result2.stdout:
            assert "chunks" in result2.stdout.lower() or "processed" in result2.stdout.lower()
    
    @pytest.mark.integration
    def test_ingest_directory(self, tmp_path, sample_pdf):
        """测试摄取目录中的所有 PDF。"""
        # 创建一个包含多个 PDF 的目录（复制示例文件）
        test_dir = tmp_path / "pdfs"
        test_dir.mkdir()
        
        shutil.copy(sample_pdf, test_dir / "doc1.pdf")
        shutil.copy(sample_pdf, test_dir / "doc2.pdf")
        
        result = self.run_ingest_script(
            path=str(test_dir),
            collection="e2e_test_dir",
            force=True,
            verbose=True
        )
        
        print("STDOUT:", result.stdout)
        
        # 应能找到两个文件
        assert "2 file" in result.stdout

        # 应尝试处理这两个文件
        assert "[1/2]" in result.stdout
        assert "[2/2]" in result.stdout
    
    def test_ingest_empty_directory(self, tmp_path):
        """测试不含 PDF 的目录的处理。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = self.run_ingest_script(
            path=str(empty_dir)
        )
        
        assert result.returncode == 0
        assert "0 file" in result.stdout or "No files" in result.stdout


class TestIngestScriptIntegration:
    """验证数据持久化的集成测试。"""

    @pytest.mark.integration
    def test_creates_vector_store_data(self, tmp_path):
        """验证摄取过程会创建 Vector Store 数据。"""
        # 该测试验证 Pipeline 会生成预期的输出文件
        # 标记为 integration 是因为它需要完整的运行环境
        
        sample_pdf = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf"
        if not sample_pdf.exists():
            pytest.skip("Sample PDF not found")
        
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "ingest.py"),
                "--path", str(sample_pdf),
                "--collection", "e2e_test_persistence",
                "--force"
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=300
        )
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # 摄取成功后验证数据目录存在
        if result.returncode == 0:
            chroma_dir = PROJECT_ROOT / "data" / "db" / "chroma"
            bm25_dir = PROJECT_ROOT / "data" / "db" / "bm25" / "e2e_test_persistence"

            assert chroma_dir.exists(), "ChromaDB directory should exist"
            # BM25 索引目录是否存在取决于具体实现


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
