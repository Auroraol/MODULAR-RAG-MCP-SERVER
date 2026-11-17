#!/usr/bin/env python3
"""
Skill Packager - 将 skill 目录打包为可分发的 .skill 文件

用法:
    python package_skill.py <path/to/skill-folder> [output-directory]

示例:
    python package_skill.py .github/skills/my-skill
    python package_skill.py .github/skills/my-skill ./dist
"""

import sys
import zipfile
from pathlib import Path

# 确保在任意工作目录下运行时都能导入同级模块
sys.path.insert(0, str(Path(__file__).parent))
from quick_validate import validate_skill


def package_skill(skill_path, output_dir=None):
    """
    将 skill 目录打包为 .skill 文件。

    参数:
        skill_path: skill 目录的路径
        output_dir: .skill 文件的可选输出目录（默认为当前目录）

    返回:
        已创建的 .skill 文件路径；若出错则返回 None
    """
    skill_path = Path(skill_path).resolve()

    # 校验 skill 目录是否存在
    if not skill_path.exists():
        print(f"❌ Error: Skill folder not found: {skill_path}")
        return None

    if not skill_path.is_dir():
        print(f"❌ Error: Path is not a directory: {skill_path}")
        return None

    # 校验 SKILL.md 是否存在
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        print(f"❌ Error: SKILL.md not found in {skill_path}")
        return None

    # 打包前先校验
    print("🔍 Validating skill...")
    valid, message = validate_skill(skill_path)
    if not valid:
        print(f"❌ Validation failed: {message}")
        print("   Please fix the validation errors before packaging.")
        return None
    print(f"✅ {message}\n")

    # 确定输出位置
    skill_name = skill_path.name
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    skill_filename = output_path / f"{skill_name}.skill"

    # 创建 .skill 文件（zip 格式）
    try:
        with zipfile.ZipFile(skill_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 遍历 skill 目录
            for file_path in skill_path.rglob('*'):
                if file_path.is_file():
                    # 计算文件在 zip 内的相对路径
                    arcname = file_path.relative_to(skill_path.parent)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

        print(f"\n✅ Successfully packaged skill to: {skill_filename}")
        return skill_filename

    except Exception as e:
        print(f"❌ Error creating .skill file: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python package_skill.py <path/to/skill-folder> [output-directory]")
        print("\nExample:")
        print("  python package_skill.py .github/skills/my-skill")
        print("  python package_skill.py .github/skills/my-skill ./dist")
        sys.exit(1)

    skill_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"📦 Packaging skill: {skill_path}")
    if output_dir:
        print(f"   Output directory: {output_dir}")
    print()

    result = package_skill(skill_path, output_dir)

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
