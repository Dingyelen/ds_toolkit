"""
modules.abtest: AB 实验分析模块。

对外提供：
- load_abtest_config: 从 YAML 字典构建 ABTestConfig；
- run_ab_test: 执行 AB 实验显著性分析。
"""

from .abtest_engine import run_ab_test
from .config_schema import ABTestConfig, MetricConfig, load_abtest_config

__all__ = ["ABTestConfig", "MetricConfig", "load_abtest_config", "run_ab_test"]
