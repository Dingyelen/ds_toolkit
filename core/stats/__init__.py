"""
core.stats: 统计学底层（T 检验、卡方、正态性检验、留存曲线拟合等）

本模块仅提供通用统计学算法，不包含任何业务逻辑。
"""

from .distribution_diagnostics import diagnose_continuous_distribution
from .hypothesis_test import HypothesisTest
from .sample_size_calculation import SampleSizeCalculation
from .retention_fitting import (
    fit_exponential_decay,
    predict_exponential_decay,
    fit_weibull_curve,
    predict_weibull_curve,
    fit_powerlaw_curve,
    predict_powerlaw_curve,
)

__all__ = [
    "HypothesisTest",
    "SampleSizeCalculation",
    "diagnose_continuous_distribution",
    "fit_exponential_decay",
    "predict_exponential_decay",
    "fit_weibull_curve",
    "predict_weibull_curve",
    "fit_powerlaw_curve",
    "predict_powerlaw_curve",
]

