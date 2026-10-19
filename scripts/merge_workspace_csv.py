"""
scripts/merge_workspace_csv.py
--------------------------------

根据配置与指定的需求目录名称，读取该目录下的 CSV 文件并纵向合并，
另存为指定名称的 CSV 文件。

使用方式（示例）：
1. 确保已在 configs/default_settings.yaml 中配置 workspace_settings.base_path；
2. 修改本脚本中的 REQURE_DIR_NAME、FILE_LIST、OUTPUT_FILE_NAME；
3. 在项目根目录下运行：
   python scripts/merge_workspace_csv.py --mode=list
   python scripts/merge_workspace_csv.py --mode=union
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

MergeMode = Literal["list", "union"]

import pandas as pd

# 将项目根目录加入 sys.path，保证 from core.xxx 可被解析（无论从何处执行脚本）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.utils import DataLoader, load_yaml  # noqa: E402


def _load_default_config() -> Dict[str, Any]:
    """
    从 configs/default_settings.yaml 读取全局默认配置。

    输入：无。
    输出：dict，解析后的配置字典。
    """
    config_path = _project_root / "configs" / "default_settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    return load_yaml(config_path)


def resolve_workspace_dir(requre_dir_name: str) -> Path:
    """
    根据需求目录名称解析绝对工作目录路径。

    输入：requre_dir_name，需求目录名称（不含 base_path）。
    输出：需求工作目录 Path。
    """
    default_cfg = _load_default_config()
    ws = default_cfg.get("workspace_settings") or {}
    base_path = ws.get("base_path")
    if not base_path:
        raise ValueError(
            "未在 configs/default_settings.yaml 中配置 workspace_settings.base_path，"
            "无法定位需求工作目录。"
        )

    workspace_dir = Path(base_path) / requre_dir_name
    if not workspace_dir.is_dir():
        raise FileNotFoundError(f"需求工作目录不存在或不是目录：{workspace_dir}")
    return workspace_dir


def list_workspace_csv_names(requre_dir_name: str) -> List[str]:
    """
    列出需求目录根目录下全部 CSV 文件名（不递归子目录）。

    输入：requre_dir_name，需求目录名称。
    输出：按文件名排序的 CSV 文件名列表。
    """
    workspace_dir = resolve_workspace_dir(requre_dir_name)
    return sorted(
        p.name
        for p in workspace_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
    )


def _normalize_csv_names(names: Sequence[str]) -> List[str]:
    """
    将文件名规范为带 .csv 后缀的形式。

    输入：names，文件名或路径片段序列。
    输出：仅保留文件名的列表，且均以 .csv 结尾。
    """
    result: List[str] = []
    for raw in names:
        name = Path(raw).name
        if not name.lower().endswith(".csv"):
            name = f"{name}.csv"
        result.append(name)
    return result


def _list_root_csv_files(workspace_dir: Path, exclude_names: set[str]) -> List[Path]:
    """
    列出需求目录根目录下的所有 CSV 文件（不含子目录），并排除指定文件名。

    输入：
        workspace_dir: 需求工作目录。
        exclude_names: 需要排除的文件名集合（如合并输出文件）。
    输出：按文件名排序的 Path 列表。
    """
    files = sorted(
        p for p in workspace_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.name not in exclude_names
    )
    return files


def _resolve_csv_paths(
    workspace_dir: Path,
    file_list: Optional[Union[str, List[str]]],
    output_file_name: str,
) -> List[Path]:
    """
    根据 file_list 解析待合并的 CSV 路径列表。

    输入：
        workspace_dir: 需求工作目录。
        file_list: 为 None 或空列表时，扫描根目录全部 CSV（排除输出文件）；
                   非空时按给定文件名读取。
        output_file_name: 输出文件名，用于自动发现时排除自身。
    输出：待读取的 CSV 路径列表。
    """
    out_name = Path(output_file_name).name
    if not out_name.lower().endswith(".csv"):
        out_name = f"{out_name}.csv"

    if file_list is None or (isinstance(file_list, list) and len(file_list) == 0):
        return _list_root_csv_files(workspace_dir, exclude_names={out_name})

    if isinstance(file_list, str):
        names = _normalize_csv_names([file_list])
    else:
        names = _normalize_csv_names(file_list)

    paths: List[Path] = []
    for name in names:
        p = workspace_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"指定的 CSV 文件不存在：{p}")
        if p.suffix.lower() != ".csv":
            raise ValueError(f"不是 CSV 文件：{name}")
        paths.append(p)
    return paths


def _assert_columns_match_strict(reference: pd.DataFrame, current: pd.DataFrame, file_name: str) -> None:
    """
    严格校验当前 DataFrame 与参考表的列名及顺序完全一致。

    输入：
        reference: 首个文件读入的 DataFrame。
        current: 当前待合并的 DataFrame。
        file_name: 当前文件名，用于错误信息。
    输出：无。不一致时抛出 ValueError。
    """
    ref_cols = list(reference.columns)
    cur_cols = list(current.columns)
    if ref_cols == cur_cols:
        return

    ref_set = set(ref_cols)
    cur_set = set(cur_cols)
    only_ref = sorted(ref_set - cur_set)
    only_cur = sorted(cur_set - ref_set)
    msg_parts = [f"文件「{file_name}」的列结构与首个文件不一致。"]
    if only_ref:
        msg_parts.append(f"首个文件独有列：{only_ref}")
    if only_cur:
        msg_parts.append(f"当前文件独有列：{only_cur}")
    if not only_ref and not only_cur and ref_cols != cur_cols:
        msg_parts.append("列名相同但顺序不同，请统一列顺序后再合并。")
    raise ValueError(" ".join(msg_parts))


def merge_workspace_csv(
    requre_dir_name: str,
    output_file_name: str,
    file_list: Optional[Union[str, List[str]]] = None,
) -> Path:
    """
    在需求工作目录下纵向合并 CSV 并保存。

    输入：
        requre_dir_name: 需求目录名称（不含 base_path）。
        output_file_name: 合并结果文件名（保存于需求目录根目录）。
        file_list: 待合并文件名；None 或 [] 表示合并根目录下全部 CSV（排除输出文件）。
    输出：
        Path：写入的合并结果文件绝对路径。
    """
    workspace_dir = resolve_workspace_dir(requre_dir_name)

    out_name = Path(output_file_name).name
    if not out_name.lower().endswith(".csv"):
        out_name = f"{out_name}.csv"
    output_path = workspace_dir / out_name

    csv_paths = _resolve_csv_paths(workspace_dir, file_list, out_name)
    if not csv_paths:
        raise FileNotFoundError(
            f"未找到可合并的 CSV 文件。目录：{workspace_dir}，"
            f"已排除输出文件：{out_name}"
        )

    loader = DataLoader()
    frames: List[pd.DataFrame] = []
    reference: Optional[pd.DataFrame] = None

    print(f"[开始] 工作目录：{workspace_dir}")
    print(f"[信息] 待合并文件数：{len(csv_paths)}")
    for idx, csv_path in enumerate(csv_paths, start=1):
        print(f"[读取] 第 {idx}/{len(csv_paths)} 个：{csv_path.name}")
        df = loader.read_data(csv_path)
        if reference is None:
            reference = df
        else:
            _assert_columns_match_strict(reference, df, csv_path.name)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"[完成] 已合并 {len(csv_paths)} 个文件，"
        f"共 {len(merged)} 行、{len(merged.columns)} 列"
    )
    print(f"[输出] {output_path.resolve()}")
    return output_path.resolve()


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="需求目录 CSV 列表 / 纵向合并")
    parser.add_argument(
        "--mode",
        choices=["list", "union"],
        default="union",
        help="list：输出目录下全部 CSV 文件名；union：纵向合并 CSV",
    )
    parser.add_argument(
        "--requre-dir",
        default=None,
        help="需求目录名称，不传则使用脚本内 REQURE_DIR_NAME",
    )
    return parser


def main(
    mode: MergeMode = "union",
    requre_dir_name: str = "",
    output_file_name: str = "",
    file_list: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    脚本主入口：按 mode 列出 CSV 或执行纵向合并。

    输入：
        mode: list 或 union。
        requre_dir_name: 需求目录名称。
        output_file_name: union 模式下的输出文件名。
        file_list: union 模式下的待合并文件列表。
    """
    if not requre_dir_name:
        raise ValueError("请设置需求目录名称（脚本内 REQURE_DIR_NAME 或 --requre-dir）。")

    if mode == "list":
        names = list_workspace_csv_names(requre_dir_name)
        workspace_dir = resolve_workspace_dir(requre_dir_name)
        print(f"[目录] {workspace_dir}")
        print(f"[信息] 根目录 CSV 文件数：{len(names)}")
        if not names:
            print("[提示] 未找到任何 CSV 文件。")
            return
        for name in names:
            print(name)
        return

    if not output_file_name:
        raise ValueError("union 模式请设置 OUTPUT_FILE_NAME。")

    merge_workspace_csv(
        requre_dir_name=requre_dir_name,
        output_file_name=output_file_name,
        file_list=file_list,
    )


if __name__ == "__main__":
    # ---------- 使用前请修改下面变量 ----------
    REQURE_DIR_NAME = "2026-10-15 TP1-20261013-102 YXY 商业化实验分析 da_abtest_pay_layer0910"

    # 待合并的 CSV 文件名；[] 或 None 表示合并需求目录根目录下全部 CSV（自动排除输出文件）
    # FILE_LIST: Optional[Union[str, List[str]]] = ["sql01-love_hearts_res.csv", "sql01-raindrop_bounty_res.csv", "sql01-rewards_trail_v1_res.csv","sql01-wild_dash_res.csv"]
    FILE_LIST: Optional[Union[str, List[str]]] = ["sql02-love_hearts_res.csv", "sql02-raindrop_bounty_res.csv", "sql02-rewards_trail_v1_res.csv","sql02-wild_dash_res.csv"]
    # 例如：FILE_LIST = ["sql01_res.csv", "sql02_res.csv"]

    # OUTPUT_FILE_NAME = "sql01_res.csv"
    OUTPUT_FILE_NAME = "sql02_res.csv"

    args = _build_arg_parser().parse_args()
    dir_name = args.requre_dir or REQURE_DIR_NAME

    main(
        mode=args.mode,
        requre_dir_name=dir_name,
        output_file_name=OUTPUT_FILE_NAME,
        file_list=FILE_LIST,
    )
