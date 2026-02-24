import math
from statistics import mean
from typing import Dict, Iterable, List, Sequence


def _to_float_list(values: Iterable[float], name: str) -> List[float]:
    """
    将任意可迭代对象转换为 float 列表，并做基础校验。

    参数：
    - values: 输入序列；
    - name: 参数名称，用于报错信息。
    """
    try:
        data = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} 必须是可迭代的数值序列。") from exc
    except ValueError as exc:
        raise ValueError(f"{name} 中存在无法转换为浮点数的元素。") from exc

    if not data:
        raise ValueError(f"{name} 不能为空。")
    return data


def diagnose_continuous_distribution(values: Sequence[float]) -> Dict[str, object]:
    """
    对连续型指标做简单的分布诊断，帮助后续选择合适的显著性检验方法。

    该函数适用于：
    - ARPU / ARPPU、在线时长、人均次数等连续型业务指标；
    - 样本量中等及以上（如 n >= 30），结果更稳定。

    诊断内容（近似，非严格统计检验）：
    - 样本规模：n；
    - 基本统计量：均值、标准差、最小值、最大值、中位数；
    - 偏度 / 峰度（以 0 为参考，绝对值越大越偏离正态）；
    - 零占比：zero_ratio，用于识别 ARPU 这类零膨胀分布；
    - 经验标签：
      * is_approximately_normal: 是否“大致接近正态”（|skew| < 1 且 kurtosis_excess 在合理范围内）；
      * is_heavy_tailed: 是否重尾（|skew| >= 1 或峰度过高）；
      * is_zero_inflated: 是否存在大量 0；
    - 推荐检验策略（recommended_tests）：
      * "mean_z_test"：基于均值的 z/t 检验；
      * "log_mean_test"：对数变换后再做均值检验，适合正偏分布且值 > 0；
      * "mann_whitney_u"：Mann–Whitney U 非参数检验，适合强偏态或极端重尾。

    返回：
    - 字典，既包含数值指标，也包含布尔标签和建议列表。
    """
    data = _to_float_list(values, "values")
    n = len(data)

    data_sorted = sorted(data)
    data_mean = mean(data)

    # 使用无偏估计计算样本标准差
    if n > 1:
        var = sum((x - data_mean) ** 2 for x in data) / (n - 1)
        std = math.sqrt(var)
    else:
        var = 0.0
        std = 0.0

    # 中位数
    if n % 2 == 1:
        median = data_sorted[n // 2]
    else:
        median = 0.5 * (data_sorted[n // 2 - 1] + data_sorted[n // 2])

    # 偏度和峰度（样本版本，简化实现）
    if n > 2 and std > 0:
        m3 = sum((x - data_mean) ** 3 for x in data) / n
        m4 = sum((x - data_mean) ** 4 for x in data) / n
        skewness = m3 / (std**3)
        kurtosis = m4 / (var**2) if var > 0 else 0.0
        kurtosis_excess = kurtosis - 3.0
    else:
        skewness = 0.0
        kurtosis_excess = 0.0

    # 零占比
    zero_count = sum(1 for x in data if x == 0.0)
    zero_ratio = zero_count / n

    # 经验规则阈值（可根据经验调整）
    is_approximately_normal = (
        abs(skewness) < 1.0 and abs(kurtosis_excess) < 3.0 and zero_ratio < 0.3
    )
    is_heavy_tailed = abs(skewness) >= 1.0 or abs(kurtosis_excess) >= 3.0
    is_zero_inflated = zero_ratio >= 0.3

    recommended_tests = []
    # 近似正态且样本量不太小：均值检验优先
    if is_approximately_normal:
        recommended_tests.append("mean_z_test")
    else:
        # 明显右偏 + 全部为正值：推荐 log 变换 + 均值检验
        if all(x > 0.0 for x in data) and skewness > 0.5:
            recommended_tests.append("log_mean_test")
        # 强偏态 / 重尾：推荐非参数检验
        if is_heavy_tailed or is_zero_inflated:
            recommended_tests.append("mann_whitney_u")

    # 如果什么都没推荐（如样本极小），至少给出 mean_z_test 作为兜底
    if not recommended_tests:
        recommended_tests.append("mean_z_test")

    return {
        "n": n,
        "mean": data_mean,
        "std": std,
        "min": data_sorted[0],
        "max": data_sorted[-1],
        "median": median,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "zero_ratio": zero_ratio,
        "is_approximately_normal": is_approximately_normal,
        "is_heavy_tailed": is_heavy_tailed,
        "is_zero_inflated": is_zero_inflated,
        "recommended_tests": recommended_tests,
    }

