import math
from statistics import mean, variance
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _normal_cdf(x: float) -> float:
    """
    标准正态分布的累积分布函数 Φ(x)。

    说明：
    - 使用误差函数 `erf` 实现，无需依赖 SciPy；
    - 在 AB 实验常见的大样本场景下，该近似足够精确。
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _get_z_by_confidence(confidence_level: float) -> float:
    """
    根据置信水平获取常用的 Z 临界值。

    支持：
    - 0.90 -> 1.6449
    - 0.95 -> 1.96
    - 0.99 -> 2.5758
    """
    mapping = {
        0.90: 1.6449,
        0.95: 1.96,
        0.99: 2.5758,
    }
    if confidence_level not in mapping:
        raise ValueError(
            f"当前仅支持 0.90/0.95/0.99 三种置信水平，收到: {confidence_level}"
        )
    return mapping[confidence_level]


class HypothesisTest:
    """
    AB 测试常用假设检验工具类。

    设计目标：
    - 仅实现纯统计学算法，不包含任何业务逻辑；
    - 主要面向“游戏 / 互联网”常见指标：
      * 连续型指标：如 ARPU（人均消费）、时长、人均付费次数等；
      * 比率型指标：如付费率、留存率、活跃率、转化率（成功用户数 / 总用户数）等；
    - 支持 2 组样本的显著性分析（对照组 vs 单个实验组）；
      多实验组场景可在业务层循环调用本类方法。

    统计方法（大样本近似）：
    - 连续型指标：基于中心极限定理，将“均值差”近似为正态分布，使用 z 检验；
    - 比率型指标：使用两比例 z 检验；
    """

    def mean_test(
        self,
        control_values: Sequence[float],
        variant_values: Sequence[float],
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        对连续型指标（如人均消费、平均在线时长等）做两组均值的显著性检验。

        参数说明：
        - control_values: 对照组样本值列表（如 A 组的人均消费，每个元素为一个用户的数值）；
        - variant_values: 实验组样本值列表（如 B 组的人均消费）；
        - alternative:
            * "two-sided"：双侧检验（默认）；
            * "larger"：备择假设为 实验组 > 对照组；
            * "smaller"：备择假设为 实验组 < 对照组；
        - confidence_level: 置信水平，常用 0.90 / 0.95 / 0.99。

        返回：
        - 字典，包含以下键：
          * "metric_type"：指标类型（"continuous"）；
          * "control_mean" / "variant_mean"：两组均值；
          * "effect"：实验组 - 对照组 的均值差；
          * "statistic"：近似 z 统计量；
          * "p_value"：p 值；
          * "is_significant"：在给定置信水平下是否显著；
          * "confidence_interval"：均值差的置信区间 (low, high)；
          * "alpha"：显著性水平；
          * "alternative"：备择假设类型。

        适用场景（示例）：
        - A/B 方案对比人均充值金额是否显著提升；
        - 不同引导流程下的人均完成任务数是否有显著差异。
        """
        control_list = self._to_float_list(control_values, "control_values")
        variant_list = self._to_float_list(variant_values, "variant_values")

        (
            control_mean,
            variant_mean,
            diff,
            se,
            z_stat,
            ci_low,
            ci_high,
        ) = self._two_sample_mean_z(
            control_list,
            variant_list,
            confidence_level=confidence_level,
        )

        p_value = self._p_value_from_z(z_stat, alternative)
        alpha = 1.0 - confidence_level

        # 合并标准差，供 DID 等场景计算 Cohen's d 与所需样本量
        pooled_std = self._pooled_std(control_list, variant_list)

        result = {
            "metric_type": "continuous",
            "control_mean": control_mean,
            "variant_mean": variant_mean,
            "effect": diff,
            "statistic": z_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "confidence_interval": (ci_low, ci_high),
            "alpha": alpha,
            "alternative": alternative,
        }
        if pooled_std is not None:
            result["pooled_std"] = pooled_std
        return result

    def log_mean_test(
        self,
        control_values: Sequence[float],
        variant_values: Sequence[float],
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        对数变换后的连续型指标两组均值检验。

        适用场景：
        - 指标明显正偏、重尾，且数值全部大于 0（如 ARPPU、停留时长等）；
        - 希望对“相对变化（倍率变化）”做显著性分析，例如：
          * 关注实验组 ARPPU 是否在对照组基础上提升 10%~20%；
        - 在对数空间做均值检验，本质上是在比较“对数均值差”，
          对应到原始空间是比较“几何均值之比”。

        参数说明：
        - control_values / variant_values: 两组原始指标值，要求全部 > 0；
        - alternative / confidence_level: 同 mean_test。

        返回：
        - 字典，包含：
          * "metric_type": "continuous_log"；
          * "control_mean_log" / "variant_mean_log"：log 空间均值；
          * "effect_log": 对数均值差（variant - control）；
          * "effect_ratio": 原始空间的相对变化倍数 exp(effect_log)；
          * "statistic": z 统计量（在 log 空间计算）；
          * "p_value" / "is_significant" / "confidence_interval_log" /
            "confidence_interval_ratio" / "alpha" / "alternative" 等。
        """
        control_list = self._to_float_list(control_values, "control_values")
        variant_list = self._to_float_list(variant_values, "variant_values")

        if any(x <= 0.0 for x in control_list + variant_list):
            raise ValueError("log 均值检验要求所有样本值严格大于 0。")

        control_log = [math.log(x) for x in control_list]
        variant_log = [math.log(x) for x in variant_list]

        (
            control_mean_log,
            variant_mean_log,
            diff_log,
            se_log,
            z_stat,
            ci_low_log,
            ci_high_log,
        ) = self._two_sample_mean_z(
            control_log,
            variant_log,
            confidence_level=confidence_level,
        )

        p_value = self._p_value_from_z(z_stat, alternative)
        alpha = 1.0 - confidence_level

        effect_ratio = math.exp(diff_log)
        ci_low_ratio = math.exp(ci_low_log)
        ci_high_ratio = math.exp(ci_high_log)

        # 原始空间的合并标准差，供 DID 计算 Cohen's d 与所需样本量
        pooled_std = self._pooled_std(control_list, variant_list)

        result = {
            "metric_type": "continuous_log",
            "control_mean_log": control_mean_log,
            "variant_mean_log": variant_mean_log,
            "effect_log": diff_log,
            "effect_ratio": effect_ratio,
            "statistic": z_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "confidence_interval_log": (ci_low_log, ci_high_log),
            "confidence_interval_ratio": (ci_low_ratio, ci_high_ratio),
            "alpha": alpha,
            "alternative": alternative,
        }
        if pooled_std is not None:
            result["pooled_std"] = pooled_std
        return result

    def proportion_test(
        self,
        control_success: int,
        control_total: int,
        variant_success: int,
        variant_total: int,
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        对比率型指标做两比例显著性检验。

        典型业务场景：
        - 付费率：付费用户数 / 总用户数；
        - 留存率：留存用户数 / 注册用户数；
        - 活跃率：活跃用户数 / 当日登陆用户数；
        - 转化率：完成某行为的用户数 / 曝光用户数。

        参数说明：
        - control_success: 对照组成功用户数（如 A 组付费用户数）；
        - control_total: 对照组总用户数；
        - variant_success: 实验组成功用户数（如 B 组付费用户数）；
        - variant_total: 实验组总用户数；
        - alternative: 同 mean_test；
        - confidence_level: 置信水平。

        返回：
        - 字典，包含：
          * "metric_type"：指标类型（"proportion"）；
          * "control_rate" / "variant_rate"：两组比例；
          * "effect"：实验组 - 对照组 的比例差；
          * "statistic"：z 统计量；
          * "p_value"：p 值；
          * "is_significant"：是否显著；
          * "confidence_interval"：比例差的置信区间；
          * "alpha" / "alternative"：同上。
        """
        self._validate_proportion_input(control_success, control_total, "control")
        self._validate_proportion_input(variant_success, variant_total, "variant")

        control_rate = control_success / control_total
        variant_rate = variant_success / variant_total

        se = math.sqrt(
            control_rate * (1.0 - control_rate) / control_total
            + variant_rate * (1.0 - variant_rate) / variant_total
        )
        if se == 0:
            raise ValueError("两组比例的标准误为 0，无法进行显著性检验。")

        diff = variant_rate - control_rate
        z_stat = diff / se

        p_value = self._p_value_from_z(z_stat, alternative)
        alpha = 1.0 - confidence_level
        z_crit = _get_z_by_confidence(confidence_level)
        ci_low = diff - z_crit * se
        ci_high = diff + z_crit * se

        return {
            "metric_type": "proportion",
            "control_rate": control_rate,
            "variant_rate": variant_rate,
            "effect": diff,
            "statistic": z_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "confidence_interval": (ci_low, ci_high),
            "alpha": alpha,
            "alternative": alternative,
        }

    def batch_mean_test(
        self,
        control_values: Sequence[float],
        variants: Mapping[str, Sequence[float]],
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """
        批量对多个实验组的连续型指标做显著性检验。

        说明：
        - 在多实验组 ABN 测试场景下，经常需要“对照组 vs 多个实验组”；
        - 该方法简单封装成循环调用，返回一个以实验组名称为 key 的结果字典。

        参数说明：
        - control_values: 对照组样本值列表；
        - variants: 映射，key 为实验组名称，value 为该实验组样本值列表；
        - alternative / confidence_level: 同 mean_test。

        返回：
        - 字典，形如：
          {
              "B": { ... mean_test 结果 ... },
              "C": { ... mean_test 结果 ... },
          }
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, data in variants.items():
            results[name] = self.mean_test(
                control_values=control_values,
                variant_values=data,
                alternative=alternative,
                confidence_level=confidence_level,
            )
        return results

    def batch_proportion_test(
        self,
        control_success: int,
        control_total: int,
        variants: Mapping[str, Tuple[int, int]],
        alternative: str = "two-sided",
        confidence_level: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """
        批量对多个实验组的比率型指标做显著性检验。

        参数说明：
        - control_success / control_total: 对照组成功数 & 总数；
        - variants: 映射，key 为实验组名称，value 为 (success, total) 元组；
        - alternative / confidence_level: 同 proportion_test。

        返回：
        - 字典，key 为实验组名称，value 为 proportion_test 的结果。
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, (success, total) in variants.items():
            results[name] = self.proportion_test(
                control_success=control_success,
                control_total=control_total,
                variant_success=success,
                variant_total=total,
                alternative=alternative,
                confidence_level=confidence_level,
            )
        return results

    def mann_whitney_u_test(
        self,
        control_values: Sequence[float],
        variant_values: Sequence[float],
        alternative: str = "two-sided",
    ) -> Dict[str, float]:
        """
        Mann–Whitney U（Wilcoxon 秩和）非参数检验，两独立样本。

        适用场景：
        - 连续型指标分布严重偏态、存在极端重尾或异常值较多；
        - 更关注“整体分布/中位数是否有系统性差异”，而不是严格的均值差；
        - 指标可以含 0 或负值，对分布形状要求较宽松。

        方法要点：
        - 将两组样本合并后做整体排序，计算各自秩和；
        - 进而得到 U 统计量，并在大样本下使用正态近似给出 z 统计量和 p 值；
        - 该实现使用大样本近似，不做精确 p 值计算。

        参数说明：
        - control_values / variant_values: 两组样本值；
        - alternative:
          * "two-sided"：检验两组分布是否存在差异；
          * "larger"：备择为“实验组整体偏大”；
          * "smaller"：备择为“实验组整体偏小”。

        返回：
        - 字典，包含：
          * "metric_type": "continuous_rank"；
          * "u_statistic": U 统计量（以实验组为基准）；
          * "z_statistic": z 统计量；
          * "p_value": p 值；
          * "is_significant": 是否显著；
          * "n_control" / "n_variant"：样本量。
        """
        control_list = self._to_float_list(control_values, "control_values")
        variant_list = self._to_float_list(variant_values, "variant_values")

        n1 = len(control_list)
        n2 = len(variant_list)
        if n1 < 1 or n2 < 1:
            raise ValueError("Mann–Whitney U 检验要求两组样本量均至少为 1。")

        # 合并数据并打上组别标签：0=control, 1=variant
        combined = [(x, 0) for x in control_list] + [(x, 1) for x in variant_list]
        # 按数值排序，相同数值在后续会被赋予平均秩
        combined.sort(key=lambda t: t[0])

        # 计算平均秩，处理 ties
        ranks: List[float] = [0.0] * (n1 + n2)
        i = 0
        while i < n1 + n2:
            j = i + 1
            while j < n1 + n2 and combined[j][0] == combined[i][0]:
                j += 1
            # 区间 [i, j) 为相同数值，赋予平均秩（1-based）
            avg_rank = 0.5 * (i + 1 + j)
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        # 计算两组的秩和
        rank_sum_control = 0.0
        rank_sum_variant = 0.0
        for (value, group), r in zip(combined, ranks):
            if group == 0:
                rank_sum_control += r
            else:
                rank_sum_variant += r

        # 按“实验组”为基准计算 U2
        u2 = rank_sum_variant - n2 * (n2 + 1) / 2.0
        u1 = rank_sum_control - n1 * (n1 + 1) / 2.0

        # 在大样本下，U 的期望和方差
        mean_u = n1 * n2 / 2.0
        var_u = n1 * n2 * (n1 + n2 + 1) / 12.0

        if var_u == 0:
            raise ValueError("Mann–Whitney U 检验无法在当前样本下计算有效方差。")

        # 使用实验组 U2 与其期望的差异来构造 z 统计量
        z_stat = (u2 - mean_u) / math.sqrt(var_u)
        p_value = self._p_value_from_z(z_stat, alternative)

        # 这里的 alpha 仅用于输出方便，具体显著性水平仍由上层控制
        alpha = 0.05

        # 原始空间的合并标准差（需每组至少 2 个样本），供 DID 计算所需样本量
        pooled_std = self._pooled_std(control_list, variant_list)

        result = {
            "metric_type": "continuous_rank",
            "u_statistic": u2,
            "z_statistic": z_stat,
            "p_value": p_value,
            "is_significant": p_value < alpha,
            "n_control": n1,
            "n_variant": n2,
            "alternative": alternative,
        }
        if pooled_std is not None:
            result["pooled_std"] = pooled_std
        return result

    @staticmethod
    def _pooled_std(
        control_list: Sequence[float],
        variant_list: Sequence[float],
    ) -> Optional[float]:
        """
        计算两独立样本的合并标准差，用于 Cohen's d 与样本量估算。

        公式：pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        要求每组至少 2 个样本，否则返回 None。
        """
        n1, n2 = len(control_list), len(variant_list)
        if n1 < 2 or n2 < 2:
            return None
        try:
            var1 = variance(control_list)
            var2 = variance(variant_list)
        except Exception:
            return None
        df = n1 + n2 - 2
        if df <= 0:
            return None
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        return math.sqrt(pooled_var)

    @staticmethod
    def _to_float_list(values: Iterable[float], name: str) -> List[float]:
        """
        将任意可迭代对象转为 float 列表，并进行基础校验。

        参数：
        - values: 输入数据；
        - name: 参数名称，用于报错提示。
        """
        try:
            result = [float(v) for v in values]
        except TypeError as exc:
            raise ValueError(f"{name} 必须是可迭代的数值序列。") from exc
        except ValueError as exc:
            raise ValueError(f"{name} 中存在无法转换为浮点数的元素。") from exc

        if not result:
            raise ValueError(f"{name} 不能为空。")
        return result

    @staticmethod
    def _two_sample_mean_z(
        control_list: Sequence[float],
        variant_list: Sequence[float],
        confidence_level: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        两独立样本均值差的 z 检验核心计算逻辑。

        参数：
        - control_list / variant_list: 两组数值序列；
        - confidence_level: 置信水平。

        返回：
        - (control_mean, variant_mean, diff, se, z_stat, ci_low, ci_high)
        """
        if len(control_list) < 2 or len(variant_list) < 2:
            raise ValueError("均值检验要求每组至少包含 2 个样本。")

        control_mean = mean(control_list)
        variant_mean = mean(variant_list)

        control_var = variance(control_list)
        variant_var = variance(variant_list)

        se = math.sqrt(
            control_var / len(control_list) + variant_var / len(variant_list)
        )
        if se == 0:
            raise ValueError("两组数据方差为 0，无法进行显著性检验。")

        diff = variant_mean - control_mean
        z_stat = diff / se

        z_crit = _get_z_by_confidence(confidence_level)
        ci_low = diff - z_crit * se
        ci_high = diff + z_crit * se

        return (
            control_mean,
            variant_mean,
            diff,
            se,
            z_stat,
            ci_low,
            ci_high,
        )

    @staticmethod
    def _validate_proportion_input(success: int, total: int, prefix: str) -> None:
        """
        校验比率型指标输入。

        参数：
        - success: 成功用户数；
        - total: 总用户数；
        - prefix: 前缀，用于区分 control / variant。
        """
        if total <= 0:
            raise ValueError(f"{prefix}_total 必须为正整数。当前为: {total}")
        if success < 0:
            raise ValueError(f"{prefix}_success 不能为负数。当前为: {success}")
        if success > total:
            raise ValueError(
                f"{prefix}_success 不能大于 {prefix}_total，当前 success={success}, total={total}"
            )

    @staticmethod
    def _p_value_from_z(z_stat: float, alternative: str) -> float:
        """
        根据 z 统计量和备择假设类型计算 p 值。

        参数：
        - z_stat: z 统计量；
        - alternative:
            * "two-sided"：双侧检验；
            * "larger"：实验组 > 对照组；
            * "smaller"：实验组 < 对照组。
        """
        alternative = alternative.lower()
        if alternative not in {"two-sided", "larger", "smaller"}:
            raise ValueError(
                f"alternative 仅支持 'two-sided' / 'larger' / 'smaller'，当前为: {alternative}"
            )

        if alternative == "two-sided":
            p_value = 2.0 * (1.0 - _normal_cdf(abs(z_stat)))
        elif alternative == "larger":
            p_value = 1.0 - _normal_cdf(z_stat)
        else:  # "smaller"
            p_value = _normal_cdf(z_stat)

        # 数值上可能出现极小的负数，这里做一下截断保护
        return max(min(p_value, 1.0), 0.0)

