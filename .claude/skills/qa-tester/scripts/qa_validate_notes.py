#!/usr/bin/env python3
"""QA Note Validator — 检测 QA_TEST_PROGRESS.md 中基于推断的备注。

扫描所有 ✅ 测试条目，标记 Note 列中包含交叉引用、推断或缺少执行证据的条目。

用法:
    python .github/skills/qa-tester/scripts/qa_validate_notes.py

退出码:
    0 — 所有备注均通过校验
    1 — 一个或多个备注被标记为可能为推断
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROGRESS_FILE = REPO_ROOT / "QA_TEST_PROGRESS.md"

# ── 表明是 INFERRED 而非 EXECUTED 结果的模式 ──

INFERENCE_PATTERNS = [
    # 交叉引用另一个测试
    (r"(?i)\bverified\s+(in|via|across|through)\s+[A-Z]-\d", "Cross-references another test ID"),
    (r"(?i)\balready\s+verified\b", "Claims 'already verified' elsewhere"),
    (r"(?i)\bsame\s+(as|code\s+path|pattern)\b", "Infers from 'same code path'"),
    (r"(?i)\bwill\s+be\s+tested\b", "Defers to future test"),
    (r"(?i)\bverified\s+(via|through)\s+[A-Z]-\d", "Cross-references test ID"),

    # 结构性 / 读代码推断
    (r"(?i)\bcode\s+uses\b", "Infers from reading code, not execution"),
    (r"(?i)\bdataclass\s+validates\b", "Infers from code structure"),
    (r"(?i)\bauto-creates\b", "Infers behavior, not observed"),
    (r"(?i)\bwould\s+(raise|fail|crash|similarly)\b", "Speculative 'would' language"),
    (r"(?i)\bshould\s+(work|succeed|fail)\b", "Speculative 'should' language"),
    (r"(?i)\b(expected|expect)\b(?!.*\bactual\b)", "Says 'expected' without 'actual'"),

    # 过于模糊 —— 没有具体数值
    (r"(?i)\bparameter\s+accepted\b$", "Vague: 'parameter accepted' without observed result"),
    (r"(?i)\bconfig\s+.*\bcontrols\s+behavior\b", "Vague: 'controls behavior' without evidence"),
    (r"(?i)\bsimilar\s+to\s+[A-Z]-\d", "Cross-references another test ID"),

    # 缺少执行证据关键词
    # 有效的 ✅ 备注应至少包含以下之一：exit=、stdout、AppTest:、output、returned、=
]

# 有效的 PASS 备注必须至少包含以下证据标记之一
EVIDENCE_MARKERS = [
    r"exit[_=]",
    r"stdout",
    r"AppTest:",
    r"output",
    r"returned",
    r"score[=s]",
    r"\d+\s*(chunks|results|items|ms|chars|files)",
    r"exit_code",
    r"PASS",
    r"FAIL",
    r"VERDICT",
    r"source_path",
    r"source_file",
]


def validate():
    content = PROGRESS_FILE.read_text(encoding="utf-8")
    lines = content.splitlines()

    issues = []
    total_pass = 0

    for lineno, line in enumerate(lines, 1):
        # 匹配 ✅ 行：| ✅ | ID | Title | Note |
        m = re.match(r"\|\s*✅\s*\|\s*(\S+)\s*\|[^|]*\|\s*(.*?)\s*\|$", line)
        if not m:
            continue

        test_id = m.group(1)
        note = m.group(2).strip()
        total_pass += 1

        # 检查推断模式
        for pattern, reason in INFERENCE_PATTERNS:
            if re.search(pattern, note):
                issues.append((test_id, lineno, reason, note[:80]))

        # 检查证据标记（至少需要一个）
        has_evidence = any(re.search(p, note, re.IGNORECASE) for p in EVIDENCE_MARKERS)
        if not has_evidence and note:
            issues.append((test_id, lineno, "No execution evidence (missing exit=, stdout, score, etc.)", note[:80]))

    # 打印报告
    print("QA Note Validation Report")
    print(f"   Total PASS entries: {total_pass}")
    print(f"   Flagged entries:    {len(issues)}")
    print()

    if issues:
        print("FLAGGED ENTRIES (likely inferred, not executed):")
        print("-" * 70)
        for test_id, lineno, reason, note_preview in issues:
            print(f"  [{test_id}] Line {lineno}")
            print(f"    Reason: {reason}")
            print(f"    Note:   {note_preview}...")
            print()
        print(f"FAIL: {len(issues)} notes need re-execution or re-writing with actual evidence.")
    else:
        print("PASS: All notes contain execution evidence. No inference detected.")

    return len(issues) == 0


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
