from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from modules.schedule.parser import parse_schedule_file as _parse_schedule_file


def parse_schedule_file(file_path: str | Path, config: Dict[str, Any]) -> pd.DataFrame:
    """
    解析活动排期 Excel 表，输出标准化的活动明细表。

    输入：
    - file_path: 排期表 .xlsx 文件路径，可以是绝对路径或相对项目根目录的路径。
    - config: 从 configs/schedule_parser.yaml 读取的解析配置字典，
              包含 sheet 名称、结构信息、日期格式、文本解析规则等。

    输出：
    - pandas.DataFrame，包含以下字段：
      - project: 项目名称
      - category: 活动类别
      - activity_name: 活动名称
      - owner: 负责人
      - start_date: 活动开始日期（date）
      - end_date: 活动结束日期（date）
      - activity_days: 活动天数（end_date - start_date + 1）
      - （可选）sheet_name: 来源工作表名称，当配置中要求保留时才会出现。
      - （可选）time_note: 单元格中的时间文本备注，如 "1/4 12:00 - 1/9 12:00"。

    核心逻辑：
    - 由 modules.schedule.parser.parse_schedule_file 完成实际解析。
    - 此函数仅作为对外统一入口，方便在 scripts 层引用。
    """
    path = Path(file_path)
    return _parse_schedule_file(path, config)


__all__ = ["parse_schedule_file"]

