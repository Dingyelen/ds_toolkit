from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class MetricTestConfig:
    """
    指标的检验配置。

    说明：
    - metric_type: 指标类型，"continuous" 表示连续型（如 ARPPU、在线时长），
      "proportion" 表示比例型（如付费率、转化率）；
    - column: 输入数据中，该指标所在的列名（列内通常是一个 list/array 样本集合）；
    - alternative: 备择假设类型，"two-sided" / "larger" / "smaller"；
    - prefer: 连续型指标时偏好的检验方法，"auto" / "mean" / "log_mean" / "mann_whitney"；
      * "auto" 表示后续 engine 可结合分布诊断自动选择；
    - sided: 双侧或单侧检验，"two-sided" / "one-sided"。
    """

    metric_type: str
    column: str
    alternative: str = "two-sided"
    prefer: str = "auto"
    sided: str = "two-sided"


@dataclass
class MetricSampleSizeConfig:
    """
    指标的样本量配置（可选）。

    说明：
    - mde: 最小可检测效应（连续型时通常是标准化效应大小；比例型时是绝对差值）；
    - alpha: 显著性水平；
    - power: 统计功效；
    - method: 连续型指标样本量方法，"means" / "nonparametric" 等；
    - baseline_rate: 比例型指标时的基线比例，用于样本量计算。
    """

    mde: Optional[float] = None
    alpha: float = 0.05
    power: float = 0.8
    method: str = "means"
    baseline_rate: Optional[float] = None


@dataclass
class MetricConfig:
    """
    单个指标的完整配置。

    说明：
    - name: 指标名称（如 "arppu_7d"、"pay_rate_7d"）；
    - test: 检验相关配置；
    - sample_size: 样本量相关配置，可为空。
    """

    name: str
    test: MetricTestConfig
    sample_size: Optional[MetricSampleSizeConfig] = None


@dataclass
class ABTestConfig:
    """
    AB 实验的整体配置对象。

    说明：
    - group_col: 组名列名（如 "group" / "variant"）；
    - phase_col: 实验阶段列名（如 "phase"，值为 "before" / "after" 等）；
    - metrics: 需要分析的指标列表。
    """

    group_col: str
    phase_col: str
    metrics: List[MetricConfig] = field(default_factory=list)


def load_abtest_config(raw_config: Mapping[str, Any]) -> ABTestConfig:
    """
    从字典（通常由 YAML 解析而来）构建 ABTestConfig 对象，并做基础校验与默认值填充。

    期望的配置结构大致为（示例）：

    - data:
        group_col: "ab_group"
        phase_col: "phase"
    - metrics:
        - name: "arppu_7d"
          type: "continuous"
          column: "arppu"
          test:
            alternative: "larger"
            prefer: "auto"
            sided: "one-sided"
          sample_size:
            mde: 0.1
            alpha: 0.05
            power: 0.8
            method: "nonparametric"

    如果缺少必要字段或取值非法，将抛出 ValueError，错误信息为中文，方便排查。
    """
    data_cfg = raw_config.get("data", {})
    group_col = data_cfg.get("group_col")
    phase_col = data_cfg.get("phase_col")

    if not group_col:
        raise ValueError("AB 实验配置缺少 data.group_col。")
    if not phase_col:
        raise ValueError("AB 实验配置缺少 data.phase_col。")

    metrics_cfg = raw_config.get("metrics", [])
    if not metrics_cfg:
        raise ValueError("AB 实验配置中 metrics 列表不能为空。")

    metrics: List[MetricConfig] = []
    for item in metrics_cfg:
        name = item.get("name")
        metric_type = item.get("type")
        column = item.get("column")

        if not name:
            raise ValueError("metrics 中存在缺少 name 的配置。")
        if metric_type not in {"continuous", "proportion"}:
            raise ValueError(
                f"指标 {name} 的 type 必须为 'continuous' 或 'proportion'，当前为: {metric_type}"
            )
        if not column:
            raise ValueError(f"指标 {name} 缺少 column 字段。")

        test_raw: Dict[str, Any] = item.get("test", {}) or {}
        alternative = test_raw.get("alternative", "two-sided")
        prefer = test_raw.get("prefer", "auto")
        sided = test_raw.get("sided", "two-sided")

        if alternative not in {"two-sided", "larger", "smaller"}:
            raise ValueError(
                f"指标 {name} 的 alternative 必须为 'two-sided'/'larger'/'smaller'，当前为: {alternative}"
            )
        if prefer not in {"auto", "mean", "log_mean", "mann_whitney"}:
            raise ValueError(
                f"指标 {name} 的 prefer 必须为 "
                f"'auto'/'mean'/'log_mean'/'mann_whitney'，当前为: {prefer}"
            )
        if sided not in {"two-sided", "one-sided"}:
            raise ValueError(
                f"指标 {name} 的 sided 必须为 'two-sided' 或 'one-sided'，当前为: {sided}"
            )

        test_cfg = MetricTestConfig(
            metric_type=metric_type,
            column=column,
            alternative=alternative,
            prefer=prefer,
            sided=sided,
        )

        sample_raw: Optional[Dict[str, Any]] = item.get("sample_size")
        sample_cfg: Optional[MetricSampleSizeConfig] = None
        if sample_raw is not None:
            mde = sample_raw.get("mde")
            alpha = sample_raw.get("alpha", 0.05)
            power = sample_raw.get("power", 0.8)
            method = sample_raw.get("method", "means")
            baseline_rate = sample_raw.get("baseline_rate")

            if mde is not None and mde <= 0:
                raise ValueError(f"指标 {name} 的 sample_size.mde 必须大于 0。当前为: {mde}")
            if not 0 < alpha < 1:
                raise ValueError(
                    f"指标 {name} 的 sample_size.alpha 必须在 (0,1) 区间内，当前为: {alpha}"
                )
            if not 0 < power < 1:
                raise ValueError(
                    f"指标 {name} 的 sample_size.power 必须在 (0,1) 区间内，当前为: {power}"
                )

            sample_cfg = MetricSampleSizeConfig(
                mde=mde,
                alpha=alpha,
                power=power,
                method=method,
                baseline_rate=baseline_rate,
            )

        metrics.append(MetricConfig(name=name, test=test_cfg, sample_size=sample_cfg))

    return ABTestConfig(group_col=group_col, phase_col=phase_col, metrics=metrics)

