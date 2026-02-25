"""
modules.abtest: AB 实验分析模块。

对外提供：
- load_abtest_config: 从 YAML 字典构建 ABTestConfig；
- run_ab_test: 执行 AB 实验显著性分析；
- compute_did: 基于 AA/AB 两阶段结果计算双重差分（DID）点估计。
"""

from .abtest_engine import compute_did, run_ab_test
from .config_schema import ABTestConfig, MetricConfig, load_abtest_config

__all__ = [
    "ABTestConfig",
    "MetricConfig",
    "compute_did",
    "load_abtest_config",
    "run_ab_test",
]
