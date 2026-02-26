"""
modules.reporter: 报表输出模块。

当前提供：
- generate_abtest_report: 针对 AB 实验分析结果生成带图片的 Excel 报表。
"""

from .abtest_reporter import generate_abtest_report

__all__ = ["generate_abtest_report"]
