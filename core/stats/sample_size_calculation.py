import math
from typing import Dict


def _normal_cdf(x: float) -> float:
    """
    标准正态分布的累积分布函数 Φ(x)。

    说明：
    - 使用误差函数 `erf` 实现，无需依赖 SciPy；
    - 精度对样本量计算已足够。
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float, tol: float = 1e-6) -> float:
    """
    标准正态分布的分位数函数（近似实现）。

    参数：
    - p: 分位点，0 < p < 1；
    - tol: 二分法的精度要求。

    说明：
    - 通过在区间 [-10, 10] 上使用二分搜索反解 CDF；
    - 对于 AB 测试中常用的 0.8 / 0.9 / 0.95 等统计功效足够准确。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 必须在 (0, 1) 区间内，当前为: {p}")

    low, high = -10.0, 10.0
    while high - low > tol:
        mid = (low + high) / 2.0
        if _normal_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


class SampleSizeCalculation:
    """
    AB 测试样本量计算工具类。

    设计目标：
    - 提供常见场景下的“每组样本量”估算方法；
    - 仅包含数学公式实现，不涉及任何业务逻辑。

    主要支持两类场景：
    1. 连续型指标（如人均消费、人均时长）——使用标准化效应大小 Cohen's d；
    2. 比率型指标（如转化率 / 留存率）——基于基线转化率和可检测的绝对差值。
    """

    @staticmethod
    def calculate_for_means(
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> int:
        """
        计算连续型指标（两独立样本均值检验）的每组所需样本量。

        统计假设：
        - H0: μ_variant - μ_control = 0
        - H1: μ_variant - μ_control = d * σ （双侧检验）

        数学公式（基于正态近似的双样本 t 检验）：
        - 双侧检验(two-sided)：n_per_group = 2 * (Z_{1-α/2} + Z_{power})^2 / d^2
        - 单侧检验(one-sided)：n_per_group = 2 * (Z_{1-α}   + Z_{power})^2 / d^2

        参数说明：
        - mde: 最小可检测效应（Minimum Detectable Effect），
          这里指 Cohen's d，即“均值差 / 标准差”，例如：
          * 希望检测到 0.1 个标准差的提升，则 mde = 0.1；
        - alpha: 显著性水平，一般取 0.05；
        - power: 统计功效（1 - β），一般取 0.8 或 0.9；
        - sided:
          * "two-sided"：双侧检验（默认，较保守，样本量更大）；
          * "one-sided"：单侧检验（只关心一个方向的提升，样本量更小）。

        返回：
        - 每组所需的最小样本量（向上取整）。
        """
        SampleSizeCalculation._validate_common_params(mde, alpha, power)

        if sided not in {"two-sided", "one-sided"}:
            raise ValueError(
                "sided 仅支持 'two-sided' 或 'one-sided'。"
            )

        if sided == "two-sided":
            z_alpha = _norm_ppf(1.0 - alpha / 2.0)
        else:
            z_alpha = _norm_ppf(1.0 - alpha)
        z_power = _norm_ppf(power)

        n_per_group = 2.0 * (z_alpha + z_power) ** 2 / (mde**2)
        return int(math.ceil(n_per_group))

    @staticmethod
    def calculate_for_means_nonparametric(
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> int:
        """
        针对非参数检验（如 Mann–Whitney U）的连续型指标样本量近似计算。

        思路说明（大白话）：
        - Mann–Whitney U 检验在“正态分布”场景下的渐近效率
          相比 t 检验大约是 0.955（即稍微“弱”一点点）；
        - 这意味着：如果你用 Mann–Whitney U 替代 t 检验，
          在同样的效应量和显著性水平/统计功效下，
          理论上大约需要 1 / 0.955^2 ≈ 1.1 倍的样本量；
        - 为了给数据分析打工人一个简单易记的规则，
          这里采用“在 t 检验样本量基础上乘以 1.1”的近似方式。

        参数说明：
        - mde / alpha / power / sided: 与 calculate_for_means 完全一致；
        - 返回值：每组所需的最小样本量（向上取整）。

        注意：
        - 这是一个实用主义近似公式，而非严格推导的精确样本量；
        - 在分布极端偏态、极重尾时，建议在此结果基础上进一步放大样本量。
        """
        base_n = SampleSizeCalculation.calculate_for_means(
            mde=mde,
            alpha=alpha,
            power=power,
            sided=sided,
        )
        # 渐近相对效率 ~0.955，对应样本量放大系数约为 1 / 0.955^2 ≈ 1.095
        adjusted_n = base_n * 1.1
        return int(math.ceil(adjusted_n))

    @staticmethod
    def calculate_for_proportions(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> int:
        """
        计算比率型指标（两比例检验）的每组所需样本量。

        典型场景：
        - 转化率、付费率、留存率等 AB 测试设计；
        - 需要在给定显著性水平和统计功效下，估算“每组最少需要多少用户”。

        统计假设：
        - 基线比例为 p1（baseline_rate），实验组目标比例为 p2 = p1 + mde；
        - H0: p2 - p1 = 0
        - H1: p2 - p1 = mde

        常用近似公式（两比例 z 检验）：
        - 双侧检验(two-sided)：
          n_per_group = [ Z_{1-α/2} * √(2 * p_bar * (1 - p_bar)) +
                          Z_{power} * √(p1 * (1 - p1) + p2 * (1 - p2)) ]^2 / (p2 - p1)^2
        - 单侧检验(one-sided)：
          n_per_group = [ Z_{1-α}   * √(2 * p_bar * (1 - p_bar)) +
                          Z_{power} * √(p1 * (1 - p1) + p2 * (1 - p2)) ]^2 / (p2 - p1)^2
          其中 p_bar = (p1 + p2) / 2。

        参数说明：
        - baseline_rate: 基线比例 p1（如当前付费率 0.1）；
        - mde: 希望检测到的“绝对差值”，即 p2 - p1，如希望从 0.10 提升到 0.12，则 mde = 0.02；
        - alpha: 显著性水平；
        - power: 统计功效；
        - sided: 同 calculate_for_means。

        返回：
        - 每组所需最小样本量（向上取整）。
        """
        if not 0.0 < baseline_rate < 1.0:
            raise ValueError(
                f"baseline_rate 必须在 (0, 1) 区间内，当前为: {baseline_rate}"
            )
        SampleSizeCalculation._validate_common_params(mde, alpha, power)

        if sided not in {"two-sided", "one-sided"}:
            raise ValueError(
                "sided 仅支持 'two-sided' 或 'one-sided'。"
            )

        p1 = baseline_rate
        p2 = baseline_rate + mde

        if not 0.0 < p2 < 1.0:
            raise ValueError(
                f"p2 = baseline_rate + mde 必须在 (0, 1) 区间内，当前 p2={p2}"
            )

        p_bar = (p1 + p2) / 2.0
        if sided == "two-sided":
            z_alpha = _norm_ppf(1.0 - alpha / 2.0)
        else:
            z_alpha = _norm_ppf(1.0 - alpha)
        z_power = _norm_ppf(power)

        se1 = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        se2 = math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))

        numerator = (z_alpha * se1 + z_power * se2) ** 2
        denominator = (p2 - p1) ** 2

        n_per_group = numerator / denominator
        return int(math.ceil(n_per_group))

    @staticmethod
    def summarize_for_means(
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> Dict[str, float]:
        """
        计算并输出连续型指标样本量的配置摘要，便于在上层模块中直接使用。

        返回字段示例：
        - {
            "alpha": 0.05,
            "power": 0.8,
            "mde": 0.1,
            "n_per_group": 784
          }
        """
        n_per_group = SampleSizeCalculation.calculate_for_means(
            mde=mde,
            alpha=alpha,
            power=power,
            sided=sided,
        )
        return {
            "alpha": alpha,
            "power": power,
            "mde": mde,
            "n_per_group": float(n_per_group),
        }

    @staticmethod
    def summarize_for_means_nonparametric(
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> Dict[str, float]:
        """
        连续型指标基于非参数检验（Mann–Whitney U 等）的样本量配置摘要。

        使用场景：
        - 已决定在显著性检验中使用秩和类非参数方法；
        - 希望预估在给定 mde / alpha / power / sided 下的“每组推荐样本量”。

        返回示例：
        - {
            "alpha": 0.05,
            "power": 0.8,
            "mde": 0.1,
            "n_per_group": 870.0,
          }
        """
        n_per_group = SampleSizeCalculation.calculate_for_means_nonparametric(
            mde=mde,
            alpha=alpha,
            power=power,
            sided=sided,
        )
        return {
            "alpha": alpha,
            "power": power,
            "mde": mde,
            "n_per_group": float(n_per_group),
        }

    @staticmethod
    def summarize_for_proportions(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        sided: str = "two-sided",
    ) -> Dict[str, float]:
        """
        计算并输出比率型指标样本量的配置摘要。

        返回字段示例：
        - {
            "alpha": 0.05,
            "power": 0.8,
            "baseline_rate": 0.1,
            "mde": 0.02,
            "n_per_group": 5233
          }
        """
        n_per_group = SampleSizeCalculation.calculate_for_proportions(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            sided=sided,
        )
        return {
            "alpha": alpha,
            "power": power,
            "baseline_rate": baseline_rate,
            "mde": mde,
            "n_per_group": float(n_per_group),
        }

    @staticmethod
    def _validate_common_params(mde: float, alpha: float, power: float) -> None:
        """
        校验 mde / alpha / power 的共用约束条件。

        参数：
        - mde: 最小可检测效应，必须 > 0；
        - alpha: 显著性水平，0 < alpha < 1；
        - power: 统计功效，0 < power < 1。
        """
        if mde <= 0.0:
            raise ValueError(f"mde 必须大于 0，当前为: {mde}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha 必须在 (0, 1) 区间内，当前为: {alpha}")
        if not 0.0 < power < 1.0:
            raise ValueError(f"power 必须在 (0, 1) 区间内，当前为: {power}")

