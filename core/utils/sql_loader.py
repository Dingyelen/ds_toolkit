"""
core.utils.sql_loader
----------------------

SQL 文件读取工具：
- 按目录列出 .sql 文件；
- 读取单个 .sql 文件内容。

注意：
- 不关心具体业务目录结构，仅处理传入路径；
- 禁止引用 modules 下的任何内容。
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def list_sql_files(sql_dir: str | Path) -> List[Path]:
    """
    列出指定目录下的所有 .sql 文件（不递归）。

    输入：
        sql_dir: 含 .sql 文件的目录路径（字符串或 Path）。
    输出：
        List[Path]: 按文件名排序的 .sql 文件 Path 列表。
    异常：
        FileNotFoundError: 目录不存在时抛出；
        NotADirectoryError: 路径存在但不是目录时抛出。
    """
    dir_path = Path(sql_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"SQL 目录不存在：{dir_path.absolute()}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"传入路径不是目录：{dir_path.absolute()}")

    files = sorted(dir_path.glob("*.sql"))
    return files


def read_sql_file(path: str | Path, encoding: str = "utf-8") -> str:
    """
    读取单个 .sql 文件内容并返回字符串。

    输入：
        path: .sql 文件路径（字符串或 Path）；
        encoding: 文本编码，默认 utf-8。
    输出：
        str: SQL 文本内容。
    异常：
        FileNotFoundError: 文件不存在时抛出；
        IsADirectoryError: 传入路径为目录时抛出。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"SQL 文件不存在：{file_path.absolute()}")
    if file_path.is_dir():
        raise IsADirectoryError(f"期望为文件但得到目录：{file_path.absolute()}")

    return file_path.read_text(encoding=encoding)

