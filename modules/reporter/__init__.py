"""
modules.reporter: 报表输出模块。

当前提供：
- generate_abtest_report: 针对 AB 实验分析结果生成带图片的 Excel 报表；
- distribution 诊断：Notebook 友好的 Markdown + 静态分布图。
"""

from .abtest_reporter import generate_abtest_report
from .distribution_reporter import (
    DistributionReportBundle,
    build_distribution_report,
    display_distribution_report,
)

__all__ = [
    "generate_abtest_report",
    "DistributionReportBundle",
    "build_distribution_report",
    "display_distribution_report",
]
