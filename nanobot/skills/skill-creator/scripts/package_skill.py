#!/usr/bin/env python3
"""
技能打包工具 - 将技能文件夹打包为可分发的 .skill 文件（ZIP 格式）。

用法：
    python package_skill.py <技能文件夹路径> [输出目录]

示例：
    python package_skill.py skills/public/my-skill
    python package_skill.py skills/public/my-skill ./dist

说明：本脚本会执行基本的安全与格式校验，确保生成的 .skill 包不包含符号链接、
也不会将输出压缩包自身包含进压缩文件中，适用于本地打包与发布前的检查。
"""

# 导入 sys 模块用于命令行参数解析和程序退出
import sys
# 导入 zipfile 用于创建 ZIP 压缩文件
import zipfile
# 导入 Path 用于跨平台的文件路径处理
from pathlib import Path

# 导入技能校验工具（同一目录下的 quick_validate.py）
from quick_validate import validate_skill


def _is_within(path: Path, root: Path) -> bool:
    """判断文件路径是否位于指定根目录内部，防止路径穿越攻击。

    路径穿越攻击是指通过 ../ 等相对路径逃逸到目标目录之外。
    此函数通过尝试计算相对路径来检测：如果 path 在 root 内部，
    则 relative_to() 成功；否则抛出 ValueError 异常。

    参数：
        path: 待检查的文件路径
        root: 根目录路径

    返回：
        True 表示 path 在 root 内部；False 表示超出范围
    """
    try:
        # 尝试获取 path 相对于 root 的相对路径
        # 如果成功，说明 path 是 root 的子路径
        path.relative_to(root)
        return True
    except ValueError:
        # 抛出 ValueError 说明 path 不在 root 的子树中
        # 即 path 超出了 root 目录范围
        return False


def _cleanup_partial_archive(skill_filename: Path) -> None:
    """删除打包失败时遗留的不完整压缩文件。

    当打包过程中发生错误时，可能会留下一个未完成的 .skill 文件。
    此函数负责清理这些半成品文件，避免混淆用户。

    参数：
        skill_filename: 要删除的文件路径
    """
    try:
        # 检查文件是否存在，存在则删除
        if skill_filename.exists():
            skill_filename.unlink()  # 删除文件
    except OSError:
        # 忽略删除失败的异常（如权限不足、文件被占用等）
        pass  # 静默处理，不影响主流程


def package_skill(skill_path, output_dir=None):
    """将技能目录打包为 `.skill` 文件（ZIP 格式），并返回生成的文件路径。

    打包流程：
    1. 校验技能目录是否存在且包含 SKILL.md
    2. 执行技能格式校验（quick_validate）
    3. 遍历目录收集文件（排除符号链接和噪声目录）
    4. 创建 ZIP 压缩包

    参数：
        skill_path: 技能目录路径（字符串或 Path 对象）
        output_dir: 可选的输出目录路径，未指定则使用当前工作目录

    返回：
        打包成功返回 Path 对象（.skill 文件路径）
        失败返回 None
    """
    # 标准化路径为绝对路径（解析 .. 和符号链接）
    skill_path = Path(skill_path).resolve()

    # ========== 校验阶段 ==========
    # 校验 1：技能文件夹是否存在
    if not skill_path.exists():
        print(f"[错误] 技能文件夹不存在：{skill_path}")
        return None

    # 校验 2：指定路径是否为文件夹（不能是文件）
    if not skill_path.is_dir():
        print(f"[错误] 指定路径不是文件夹：{skill_path}")
        return None

    # 校验 3：必须包含 SKILL.md 核心文件（技能的定义文件）
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        print(f"[错误] 技能文件夹内未找到 SKILL.md：{skill_path}")
        return None

    # 打包前执行技能格式校验（检查名称、描述、目录结构等）
    print("正在校验技能格式...")
    valid, message = validate_skill(skill_path)
    if not valid:
        # 校验失败则中止打包
        print(f"[错误] 技能校验失败：{message}")
        print("   请修复校验错误后重新打包。")
        return None
    print(f"[成功] {message}\n")

    # ========== 准备输出路径 ==========
    skill_name = skill_path.name  # 技能目录名作为压缩包名
    if output_dir:
        # 用户指定了输出目录，创建目录（支持多级）
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # 默认输出到当前工作目录
        output_path = Path.cwd()

    # 最终打包文件名：技能名.skill（如 my-skill.skill）
    skill_filename = output_path / f"{skill_name}.skill"

    # 打包时需要排除的目录名称（版本控制、缓存、依赖文件夹等）
    EXCLUDED_DIRS = {".git", ".svn", ".hg", "__pycache__", "node_modules"}

    # 待打包的文件列表
    files_to_package = []
    # 打包文件的绝对路径（用于自引用判断，避免把压缩包自身打包进去）
    resolved_archive = skill_filename.resolve()

    # ========== 遍历收集文件 ==========
    # rglob("*") 递归遍历技能文件夹下所有文件和子目录
    for file_path in skill_path.rglob("*"):
        # 安全校验 1：禁止打包符号链接（防止路径漏洞）
        # 符号链接可能指向外部文件，打包后解压会造成安全隐患
        if file_path.is_symlink():
            print(f"[错误] 技能包中不允许包含符号链接：{file_path}")
            _cleanup_partial_archive(skill_filename)  # 清理半成品
            return None

        # 获取文件相对路径的各部分，检查是否在排除目录中
        rel_parts = file_path.relative_to(skill_path).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue  # 跳过排除目录中的文件

        # 只处理普通文件，跳过目录
        if file_path.is_file():
            resolved_file = file_path.resolve()
            # 安全校验 2：文件不能超出技能根目录范围（防路径穿越）
            if not _is_within(resolved_file, skill_path):
                print(f"[错误] 文件超出技能根目录范围：{file_path}")
                _cleanup_partial_archive(skill_filename)
                return None
            # 安全校验 3：避免将压缩包自身加入压缩包（自引用会导致无限循环）
            if resolved_file == resolved_archive:
                print(f"[警告] 跳过输出压缩包自身：{file_path}")
                continue
            # 所有校验通过，加入打包列表
            files_to_package.append(file_path)

    # ========== 创建 ZIP 压缩包 ==========
    try:
        # 创建 ZIP 文件，ZIP_DEFLATED 表示启用压缩（减小文件体积）
        with zipfile.ZipFile(skill_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_package:
                # 计算压缩包内的相对路径（保持原有目录结构）
                # arcname 是文件在压缩包内的名称
                arcname = Path(skill_name) / file_path.relative_to(skill_path)
                # 将文件写入压缩包
                zipf.write(file_path, arcname)
                print(f"  已添加：{arcname}")

        # 打包成功
        print(f"\n[成功] 技能打包完成，文件路径：{skill_filename}")
        return skill_filename

    except Exception as e:
        # 打包过程中发生异常（如磁盘空间不足、权限问题等）
        _cleanup_partial_archive(skill_filename)  # 清理不完整的压缩包
        print(f"[错误] 创建 .skill 文件失败：{e}")
        return None


def main():
    """主函数：处理命令行参数，调用打包逻辑。

    此函数是脚本的入口点，负责：
    1. 解析命令行参数
    2. 校验参数合法性
    3. 调用 package_skill 执行打包
    4. 根据结果设置程序退出码
    """
    # 校验命令行参数数量（至少需要技能路径参数）
    if len(sys.argv) < 2:
        # 参数不足时打印使用说明
        print("用法：python package_skill.py <技能文件夹路径> [输出目录]")
        print("\n示例：")
        print("  python package_skill.py skills/public/my-skill")
        print("  python package_skill.py skills/public/my-skill ./dist")
        sys.exit(1)  # 退出码 1 表示错误退出

    # 解析参数
    skill_path = sys.argv[1]  # 第一个参数：技能文件夹路径
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None  # 第二个参数（可选）：输出目录

    # 打印打包信息（让用户知道正在处理什么）
    print(f"正在打包技能：{skill_path}")
    if output_dir:
        print(f"   输出目录：{output_dir}")
    print()  # 空行分隔

    # 执行打包
    result = package_skill(skill_path, output_dir)

    # 根据结果设置退出码：成功返回 0，失败返回 1
    sys.exit(0 if result else 1)


# 程序入口
if __name__ == "__main__":
    main()