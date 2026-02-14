# core/utils/file_io.py
# 统一文件读取入口：按后缀自动选择 CSV/Excel，支持编码回退与 kwargs 透传

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# 若已创建 core/logger.py 则用于记录错误，否则仅抛异常
try:
    from core.logger import get_logger
    _logger = get_logger(__name__)
except ImportError:
    _logger = None

# 文本类后缀用 read_csv，Excel 用 read_excel
_CSV_LIKE_SUFFIXES = {".csv", ".txt"}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}


class DataLoader:
    """
    统一数据加载器：根据文件后缀自动选择 pandas 读取方式，并对 CSV 做编码回退。

    Input:
        encoding_list: 尝试的编码顺序，仅对 CSV/TXT 生效。默认 ["utf-8", "gbk", "gb18030"]。
                      可由 configs 中 file_io.encodings 注入。
    Output:
        无（构造器）。实际数据通过 read_data() 返回 DataFrame。
    """

    def __init__(
        self,
        encoding_list: Optional[List[str]] = None,
    ) -> None:
        """
        初始化 DataLoader。

        参数:
            encoding_list: 读取文本文件时的编码尝试列表，None 时使用默认列表。
        """
        self._encoding_list: List[str] = encoding_list or [
            "utf-8",
            "gbk",
            "gb18030",
        ]

    def read_data(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        根据文件后缀自动选择读取方式，并支持编码回退与 Pandas 原生参数透传。

        Input:
            file_path: 文件路径（字符串或 Path）。
            **kwargs: 透传给 pd.read_csv 或 pd.read_excel 的参数，
                      如 sheet_name, sep, header, usecols 等。
        Output:
            pd.DataFrame: 读取后的数据表。
        逻辑:
            1. 校验文件存在性，不存在则记录日志（若有）并抛出中文异常。
            2. 根据后缀选择读取方式：.csv/.txt -> read_csv；.xlsx/.xls -> read_excel。
            3. 对 CSV/TXT：按 encoding_list 依次尝试编码，直到成功。
            4. 其余参数通过 **kwargs 原样传给 Pandas。
        """
        path = Path(file_path)
        if not path.exists():
            _msg = f"文件不存在，请检查路径：{path.absolute()}"
            if _logger is not None:
                _logger.error(_msg)
            raise FileNotFoundError(_msg)

        suffix = path.suffix.lower()
        if suffix in _CSV_LIKE_SUFFIXES:
            return self._read_csv_with_encoding(path, **kwargs)
        if suffix in _EXCEL_SUFFIXES:
            return pd.read_excel(path, **kwargs)

        raise ValueError(
            f"不支持的文件格式：{suffix}。"
            f"当前支持：{', '.join(_CSV_LIKE_SUFFIXES | _EXCEL_SUFFIXES)}。"
        )

    def _read_csv_with_encoding(
        self,
        path: Path,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        按 encoding_list 顺序尝试编码读取 CSV/TXT，直到成功。

        Input:
            path: 文件 Path 对象。
            **kwargs: 透传给 pd.read_csv 的参数（若已含 encoding 则首次尝试使用）。
        Output:
            pd.DataFrame: 读取成功后的数据。
        """
        # 若调用方在 kwargs 里传了 encoding，优先用该编码列表：先试传入的，再试默认列表
        encodings_to_try: List[str] = []
        if "encoding" in kwargs:
            encodings_to_try.append(kwargs.pop("encoding"))
        encodings_to_try.extend(self._encoding_list)

        last_error: Optional[Exception] = None
        for enc in encodings_to_try:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue

        _msg = (
            f"使用编码 {encodings_to_try} 均无法正确解码文件：{path.absolute()}。"
            f"最后错误：{last_error!s}"
        )
        if _logger is not None:
            _logger.error(_msg)
        raise ValueError(_msg) from last_error
