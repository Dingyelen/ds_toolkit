"""
scripts/run_schedule_parse.py
-----------------------------

根据 configs/schedule_parser.yaml 中的配置，解析指定的活动排期 Excel（可包含多个 sheet），
抽取项目、活动类别、活动名称、起止时间、负责人等信息，并将结果导出为 CSV。

使用方式（示例）：
1. 在 configs/schedule_parser.yaml 中配置 input_file、输出路径与表结构等参数；
2. 确保排期表 .xlsx 文件已就绪；
3. 在项目根目录下运行：
   python -m scripts.run_schedule_parse
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.utils import load_yaml  # noqa: E402
from modules.schedule import parse_schedule_file  # noqa: E402


def _load_schedule_config() -> Dict[str, Any]:
    """
    读取活动排期解析配置 configs/schedule_parser.yaml。

    输入：
        无。
    输出：
        dict：解析后的配置字典。
    异常：
        FileNotFoundError: 配置文件不存在时抛出。
    """
    config_path = _project_root / "configs" / "schedule_parser.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    return load_yaml(config_path)


def main() -> None:
    """
    解析活动排期 Excel 并导出合并后的活动明细表。

    输入：
        无（参数由 configs/schedule_parser.yaml 控制）。
    输出：
        无。结果以 CSV 文件形式写入配置指定路径。
    """
    cfg = _load_schedule_config()

    input_file = cfg.get("input_file")
    if not input_file:
        raise ValueError("schedule_parser.yaml 中未配置 input_file。")

    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = _project_root / input_path

    output_cfg = cfg.get("output") or {}
    output_file = output_cfg.get("file_path", "outputs/schedule_activities.csv")
    output_encoding = output_cfg.get("encoding", "utf-8-sig")
    output_encoding_errors = output_cfg.get("encoding_errors")
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = _project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[开始] 解析排期表：{input_path}")
    df: pd.DataFrame = parse_schedule_file(input_path, cfg)
    print(f"[信息] 共解析得到活动记录数：{len(df)}")

    to_csv_kwargs: Dict[str, Any] = {
        "index": False,
        "encoding": output_encoding,
    }
    # pandas.to_csv 支持 errors 参数，用于控制编码错误行为
    if output_encoding_errors:
        to_csv_kwargs["errors"] = output_encoding_errors

    df.to_csv(output_path, **to_csv_kwargs)
    print(f"[完成] 已将结果导出到：{output_path}")


if __name__ == "__main__":
    main()

