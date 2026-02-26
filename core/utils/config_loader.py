"""
core.utils.config_loader: 配置文件读取（当前为 YAML）。

禁止引用 modules 下的任何内容。
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取 YAML 文件并返回字典。

    输入：
    - path: 文件路径，可为 str 或 Path。

    输出：
    - 解析得到的字典；若文件为空或仅包含空文档，返回空字典。

    异常：
    - FileNotFoundError: 路径不存在；
    - yaml.YAMLError: 解析失败时由 PyYAML 抛出。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML 文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
