"""
scripts.abtest_report

AB 实验报表生成入口：读取聚合数据与配置，运行 AB/DID 分析，生成带图片嵌入的 Excel 报表。
对外提供调用函数 run_abtest_report(data_path, output_path, base_group, report_config?, stratify_by?) -> dict。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# 将项目根目录加入 sys.path，保证从命令行直接运行脚本时可以 import modules/core
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import DataLoader, load_yaml
from modules.abtest import compute_did, load_abtest_config, run_ab_test
from modules.reporter import generate_abtest_report


def run_abtest_report(
    data_path: str,
    output_path: str,
    base_group: str,
    report_config: Optional[Mapping[str, Any]] = None,
    ab_config_path: Optional[str] = None,
    stratify_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    生成 AB 实验报表并保存到指定位置，返回预览信息。

    输入：
    - data_path: 聚合宽表数据文件路径（支持 parquet、csv、xlsx，由 core.utils.DataLoader 读取）；
    - output_path: 报表 Excel 的保存路径（.xlsx）；
    - base_group: 对照组名称，与数据中组名列的取值一致（如 "control"、"A"、"对照组"）；
    - report_config: 可选，报表配置字典；未传则从 configs/abtest_report.yaml 读取；
    - ab_config_path: 可选，AB 分析配置 YAML 路径；未传则使用 configs/abtest_demo.yaml；
    - stratify_by: 可选，分层维度列名列表（如 ["country", "level"]）；不传或 None 表示不分层。

    输出：
    - 字典，包含：
      - "excel_path": 生成的 Excel 文件路径；
      - "sheets": 报表内 sheet 名称列表；
      - "n_metric_figures": 嵌入的 metric 柱状图数量；
      - "n_did_figures": 嵌入的 DID 图数量；
      - "metric_figures": [(title, file_path), ...]；
      - "did_figures": [(title, file_path), ...]。
    """
    project_root = _PROJECT_ROOT
    data_file = Path(data_path)
    out_file = Path(output_path)

    if report_config is None:
        report_cfg_path = project_root / "configs" / "abtest_report.yaml"
        report_config = load_yaml(report_cfg_path)
    if ab_config_path is None:
        ab_cfg_path = project_root / "configs" / "abtest_demo.yaml"
    else:
        ab_cfg_path = Path(ab_config_path)
    ab_cfg_raw = load_yaml(ab_cfg_path)
    ab_config = load_abtest_config(ab_cfg_raw)

    # 可视化配置（非必需；若缺失则使用 core.visualizer 默认样式）
    visual_config_path = project_root / "configs" / "visualizer.yaml"
    visual_config: Optional[Mapping[str, Any]] = None
    if visual_config_path.exists():
        visual_config = load_yaml(visual_config_path)

    data_loader = DataLoader()
    df = data_loader.read_data(data_file)

    ab_result = run_ab_test(
        df=df,
        config=ab_config,
        base_group=base_group,
        target_phases=["after"],
        stratify_by=stratify_by,
    )
    aa_result = run_ab_test(
        df=df,
        config=ab_config,
        base_group=base_group,
        target_phases=["before"],
        stratify_by=stratify_by,
    )
    did_result = compute_did(
        aa_result=aa_result,
        ab_result=ab_result,
        stratify_by=stratify_by,
    )

    outputs = generate_abtest_report(
        ab_result=ab_result,
        did_result=did_result,
        report_config=report_config,
        output_path=str(out_file),
        visual_config=visual_config,
    )

    metric_figures = outputs.get("metric_figures", [])
    did_figures = outputs.get("did_figures", [])
    sheets = ["Overview", "DID", "Metric_Charts", "DID_Charts"]

    return {
        "excel_path": outputs["excel_path"],
        "sheets": sheets,
        "n_metric_figures": len(metric_figures),
        "n_did_figures": len(did_figures),
        "metric_figures": metric_figures,
        "did_figures": did_figures,
    }


def main() -> None:
    """命令行入口：接收数据路径、报表输出路径、对照组名称，可选分层列，调用 run_abtest_report 并打印预览。"""
    if len(sys.argv) < 4:
        print("用法: python scripts/abtest_report.py <数据路径> <报表输出路径> <对照组名称> [分层列,逗号分隔]")
        print("示例: python scripts/abtest_report.py data/abtest_demo.csv outputs/report.xlsx control")
        print("示例: python scripts/abtest_report.py data/abtest_demo.csv outputs/report.xlsx control country,level")
        sys.exit(1)
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    base_group = sys.argv[3]
    stratify_by: Optional[List[str]] = None
    if len(sys.argv) > 4 and sys.argv[4].strip():
        stratify_by = [s.strip() for s in sys.argv[4].split(",") if s.strip()]

    preview = run_abtest_report(
        data_path=data_path,
        output_path=output_path,
        base_group=base_group,
        stratify_by=stratify_by,
    )
    print("AB 实验报表已生成，预览：")
    print(f"  excel_path: {preview['excel_path']}")
    print(f"  sheets: {preview['sheets']}")
    print(f"  n_metric_figures: {preview['n_metric_figures']}")
    print(f"  n_did_figures: {preview['n_did_figures']}")


if __name__ == "__main__":
    main()
