"""
基于 SciPy 的正态性检验封装。

仅依赖数值序列，不包含业务逻辑；大样本下检验易显著，需结合分位数与图形判断。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from scipy import stats


def _to_finite_array(values: Sequence[float], *, name: str = "values") -> np.ndarray:
    """
    转为 float64 数组并剔除 inf/nan。

    输入：
    - values: 数值序列；
    - name: 参数名，用于异常信息。

    输出：
    - 一维 ndarray。

    异常：
    - ValueError: 全为缺失或空序列。
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空。")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} 在剔除 inf/nan 后为空。")
    return arr.ravel()


def _anderson_rejection_index(alpha: float) -> int:
    """
    Anderson-Darling（dist='norm'）与 SciPy 内置临界值行的对应关系。

    significance_level 顺序为 [15%, 10%, 5%, 2.5%, 1%]。
    将常用 alpha 映射到「最接近且不宽于该 alpha 的显著性档位」对应的行下标。
    """
    if alpha >= 0.15:
        return 0
    if alpha >= 0.10:
        return 1
    if alpha >= 0.05:
        return 2
    if alpha >= 0.025:
        return 3
    return 4


def test_normality(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    methods: Optional[Sequence[str]] = None,
    shapiro_max_n: int = 5000,
    random_seed: Optional[int] = None,
    method_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """
    对单样本连续数值做正态性检验（多方法可配置）。

    输入：
    - values: 原始样本（可含 nan/inf，会先剔除）；
    - alpha: 显著性水平；
    - methods: 方法标识序列，支持：
        * "shapiro" — Shapiro-Wilk；
        * "dagostino" — D'Agostino–Pearson，对应 scipy.stats.normaltest；
        * "anderson" — Anderson-Darling（相对「标准正态」外推至估参正态的近似理解，见 SciPy 文档）；
      若为 None，默认 ("shapiro", "dagostino")。
    - shapiro_max_n: Shapiro-Wilk 推荐最大样本量；超出时无放回随机抽样至该规模并记录说明。
    - random_seed: 抽样用的随机种子；
    - method_aliases: 可选，将配置中的名称映射到上述标识（如 {"normaltest": "dagostino"}）。

    输出：
    - 字典：alpha、n、effective_n、tests（各方法 statistic、p_value、rejected、notes）、warnings。
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0, 1) 内。")

    raw = np.asarray(values, dtype=float)
    n_raw = int(raw.size)
    data = _to_finite_array(values, name="values")
    n = int(data.size)

    alias_map: MutableMapping[str, str] = {
        "normaltest": "dagostino",
        "pearson": "dagostino",
    }
    if method_aliases:
        alias_map.update({str(k): str(v) for k, v in method_aliases.items()})

    if methods is None:
        resolved_methods: List[str] = ["shapiro", "dagostino"]
    else:
        resolved_methods = []
        for m in methods:
            key = str(m).strip().lower()
            resolved_methods.append(alias_map.get(key, key))

    warnings: List[str] = []
    tests_out: List[Dict[str, Any]] = []

    for method in resolved_methods:
        if method == "shapiro":
            note_parts: List[str] = []
            sample = data
            if n > shapiro_max_n:
                rng = np.random.default_rng(random_seed)
                ix = rng.choice(n, size=shapiro_max_n, replace=False)
                sample = data[ix]
                warnings.append(
                    f"Shapiro：原始样本量 {n} 大于 shapiro_max_n={shapiro_max_n}，"
                    f"已随机抽样 {shapiro_max_n} 条参与检验。"
                )
                note_parts.append(f"检验子样本量={shapiro_max_n}")
            if sample.size < 3:
                tests_out.append(
                    {
                        "method": "shapiro",
                        "statistic": math.nan,
                        "p_value": math.nan,
                        "rejected": None,
                        "notes": "样本量不足（<3），已跳过。",
                    }
                )
                continue
            stat, p = stats.shapiro(sample)
            rejected = bool(p < alpha)
            tests_out.append(
                {
                    "method": "shapiro",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "rejected": rejected,
                    "notes": "；".join(note_parts) if note_parts else "",
                }
            )

        elif method == "dagostino":
            if n < 8:
                tests_out.append(
                    {
                        "method": "dagostino",
                        "statistic": math.nan,
                        "p_value": math.nan,
                        "rejected": None,
                        "notes": "normaltest 建议样本量不少于 8，已跳过。",
                    }
                )
                continue
            stat, p = stats.normaltest(data)
            tests_out.append(
                {
                    "method": "dagostino",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "rejected": bool(p < alpha),
                    "notes": "scipy.stats.normaltest（D'Agostino–Pearson）",
                }
            )

        elif method == "anderson":
            res = stats.anderson(data, dist="norm")
            idx = _anderson_rejection_index(alpha)
            sig_level = float(res.significance_level[idx])
            crit = float(res.critical_values[idx])
            stat = float(res.statistic)
            rejected = stat > crit
            tests_out.append(
                {
                    "method": "anderson",
                    "statistic": stat,
                    "p_value": None,
                    "rejected": bool(rejected),
                    "notes": (
                        f"Anderson-Darling 与正态性；采用显著性档位 {sig_level:.2f}% "
                        f"的临界值 {crit:.6f}（非经典 p 值）。"
                    ),
                }
            )
        else:
            warnings.append(f"未知方法 '{method}'，已忽略。")

    return {
        "alpha": alpha,
        "n_input": n_raw,
        "n": n,
        "tests": tests_out,
        "warnings": warnings,
    }
