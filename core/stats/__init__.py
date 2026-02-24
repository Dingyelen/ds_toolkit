"""
core.stats: 统计学底层（T 检验、卡方、正态性检验等）

本模块仅提供通用统计学算法，不包含任何业务逻辑。
"""

from .distribution_diagnostics import diagnose_continuous_distribution
from .hypothesis_test import HypothesisTest
from .sample_size_calculation import SampleSizeCalculation

__all__ = [
    "HypothesisTest",
    "SampleSizeCalculation",
    "diagnose_continuous_distribution",
]

