import ast
import math
from collections.abc import Iterable as IterableABC
from dataclasses import asdict, dataclass
from statistics import mean as stat_mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import warnings

import pandas as pd

from core.stats import (
    HypothesisTest,
    SampleSizeCalculation,
    diagnose_continuous_distribution,
)
from .config_schema import ABTestConfig, MetricConfig


def _to_float_list(values: Iterable[float], metric_name: str) -> List[float]:
    """
    将任意可迭代对象转换为 float 列表，并做基础校验。

    参数：
    - values: 输入样本集合（通常为 list/tuple/Series）；
    - metric_name: 指标名称，用于报错信息。
    """
    try:
        data = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"指标 {metric_name} 的样本值不可迭代，请检查数据结构。") from exc
    except ValueError as exc:
        raise ValueError(f"指标 {metric_name} 的样本中存在无法转换为浮点数的元素。") from exc

    if not data:
        raise ValueError(f"指标 {metric_name} 的样本列表不能为空。")
    return data


def parse_array_cell(cell: Any, metric_name: str) -> List[float]:
    """
    将单元格中的“数组形式”数据统一转换为 float 列表。

    背景：
    - 上游通常在 SQL 中使用 array_agg / JSON 等方式聚合样本；
    - 经过 CSV/Excel + pandas 读取后，单元格可能是：
      * Python list/tuple；
      * pandas.Series / numpy.ndarray 等可迭代对象；
      * 字符串表示的数组（如 "[1, 2, 3]" 或 "{1,2,3}" 或 "1,2,3"）；
      * 单个标量。

    处理策略：
    1. 若为字符串：
       - 优先尝试 ast.literal_eval（支持 "[1,2,3]" / "(1,2,3)" 等）；
       - 若失败，则尝试按逗号分隔解析；
    2. 若为非字符串的可迭代对象（如 list/tuple/Series/ndarray）：
       - 直接遍历并转换为 float；
    3. 其余情况视为单个标量，封装为长度为 1 的列表。
    """
    # 字符串：尝试解析为数组
    if isinstance(cell, str):
        text = cell.strip()
        # 优先尝试按 Python 字面量解析（如 "[1, 2, 3]")
        try:
            if text and text[0] in "[({":
                parsed = ast.literal_eval(text)
                if isinstance(parsed, IterableABC) and not isinstance(
                    parsed, (str, bytes)
                ):
                    return _to_float_list(parsed, metric_name)
        except (SyntaxError, ValueError):
            pass

        # 兜底：按逗号分隔，去掉常见括号符号
        inner = text.strip("[]{}()")
        if not inner:
            raise ValueError(f"指标 {metric_name} 的样本字符串内容为空，无法解析。")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return _to_float_list(parts, metric_name)

    # 非字符串的通用可迭代对象（如 list/tuple/Series/ndarray 等）
    if isinstance(cell, IterableABC) and not isinstance(cell, (str, bytes)):
        return _to_float_list(cell, metric_name)

    # 单个标量
    return _to_float_list([cell], metric_name)


def _extract_samples_for_metric_phase(
    df: pd.DataFrame,
    group_col: str,
    phase_col: str,
    metric_column: str,
    metric_name: str,
    phase: str,
) -> Dict[str, List[float]]:
    """
    从 DataFrame 中抽取某个指标在指定 phase 下，各 group 的样本列表。

    参数：
    - df: 聚合后的宽表；
    - group_col: 组名列名；
    - phase_col: 阶段列名；
    - metric_column: 指标所在列名；
    - metric_name: 指标名称；
    - phase: 目标阶段值（如 "before" / "after"）。

    返回：
    - 字典，key 为 group 名称，value 为该 group × phase 下的样本列表。
    """
    required_cols = {group_col, phase_col, metric_column}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"指标 {metric_name} 抽取样本时，DataFrame 缺少以下列：{', '.join(sorted(missing))}"
        )

    subset = df[df[phase_col] == phase]
    if subset.empty:
        raise ValueError(f"在 phase='{phase}' 下，未找到任何用于指标 {metric_name} 的数据行。")

    samples_by_group: Dict[str, List[float]] = {}
    for _, row in subset[[group_col, metric_column]].iterrows():
        group = str(row[group_col])
        raw_values = row[metric_column]
        samples_by_group[group] = parse_array_cell(raw_values, metric_name=metric_name)

    return samples_by_group


def _run_continuous_test(
    control_values: List[float],
    variant_values: List[float],
    metric_cfg: MetricConfig,
    confidence_level: float,
) -> Dict[str, Any]:
    """
    对连续型指标执行显著性检验。

    说明：
    - 不在此处对样本做 0 值过滤，是否包含 0 完全由上游聚合逻辑决定；
    - prefer == "auto" 时，会结合分布诊断推荐方法。
    """
    tester = HypothesisTest()
    test_cfg = metric_cfg.test

    # 默认使用控制组分布做诊断
    diagnosis = diagnose_continuous_distribution(control_values)
    recommended = diagnosis.get("recommended_tests", []) or []

    method = test_cfg.prefer
    if method == "auto":
        # 简单优先级：log_mean > mann_whitney > mean
        if "log_mean_test" in recommended and all(v > 0.0 for v in control_values + variant_values):
            method = "log_mean"
        elif "mann_whitney_u" in recommended:
            method = "mann_whitney"
        else:
            method = "mean"

    if method == "mean":
        stat_result = tester.mean_test(
            control_values=control_values,
            variant_values=variant_values,
            alternative=test_cfg.alternative,
            confidence_level=confidence_level,
        )
    elif method == "log_mean":
        stat_result = tester.log_mean_test(
            control_values=control_values,
            variant_values=variant_values,
            alternative=test_cfg.alternative,
            confidence_level=confidence_level,
        )
    elif method == "mann_whitney":
        stat_result = tester.mann_whitney_u_test(
            control_values=control_values,
            variant_values=variant_values,
            alternative=test_cfg.alternative,
        )
    else:
        raise ValueError(
            f"连续型指标 {metric_cfg.name} 的检验方法 {method} 非法，应为 "
            f"'auto'/'mean'/'log_mean'/'mann_whitney' 之一。"
        )

    return {
        "diagnosis": diagnosis,
        "stat_result": stat_result,
        "used_method": method,
    }


def _report_values_from_test(
    stat_result: Dict[str, Any],
    used_method: str,
    control_samples: List[float],
    variant_samples: List[float],
) -> Tuple[int, int, float, float]:
    """
    从检验结果与样本中抽取报表用：n_base、n_variant、base_value、variant_value。

    参数：
    - stat_result: core.stats 返回的检验结果字典；
    - used_method: 实际使用的检验方法（mean / log_mean / mann_whitney / proportion_z_test）；
    - control_samples / variant_samples: 对照组与实验组的样本列表。

    返回：
    - (n_base, n_variant, base_value, variant_value)。
      base_value/variant_value：连续型为均值（或几何均值），比例型为比例。
    """
    n_base = len(control_samples)
    n_variant = len(variant_samples)

    if used_method == "mean":
        base_value = float(stat_result["control_mean"])
        variant_value = float(stat_result["variant_mean"])
    elif used_method == "log_mean":
        base_value = math.exp(float(stat_result["control_mean_log"]))
        variant_value = math.exp(float(stat_result["variant_mean_log"]))
    elif used_method == "mann_whitney":
        base_value = stat_mean(control_samples)
        variant_value = stat_mean(variant_samples)
    elif used_method == "proportion_z_test":
        base_value = float(stat_result["control_rate"])
        variant_value = float(stat_result["variant_rate"])
    else:
        base_value = float("nan")
        variant_value = float("nan")

    return (n_base, n_variant, base_value, variant_value)


def _run_proportion_test(
    control_values: List[float],
    variant_values: List[float],
    metric_cfg: MetricConfig,
    confidence_level: float,
) -> Dict[str, Any]:
    """
    对比例型指标执行显著性检验。

    说明：
    - 输入样本为 0/1 列表，表示每个用户是否满足事件（如是否付费）；
    - 本函数会自动将样本转换为 success/total 形式再调用核心检验函数。
    """
    tester = HypothesisTest()
    test_cfg = metric_cfg.test

    control_success = int(sum(1 for v in control_values if v != 0))
    control_total = len(control_values)
    variant_success = int(sum(1 for v in variant_values if v != 0))
    variant_total = len(variant_values)

    stat_result = tester.proportion_test(
        control_success=control_success,
        control_total=control_total,
        variant_success=variant_success,
        variant_total=variant_total,
        alternative=test_cfg.alternative,
        confidence_level=confidence_level,
    )

    diagnosis = {
        "n_control": control_total,
        "n_variant": variant_total,
        "control_success": control_success,
        "variant_success": variant_success,
    }

    return {
        "diagnosis": diagnosis,
        "stat_result": stat_result,
        "used_method": "proportion_z_test",
    }


def _plan_sample_size_for_metric(
    metric_cfg: MetricConfig,
) -> Optional[Dict[str, Any]]:
    """
    根据指标配置中的 sample_size 字段，给出样本量规划结果。

    说明：
    - 若未配置 sample_size，则返回 None；
    - 连续型指标：
      * method == "means"        → 使用 calculate_for_means
      * method == "nonparametric"→ 使用 calculate_for_means_nonparametric
    - 比例型指标：
      * 使用 calculate_for_proportions（需要 baseline_rate 与 mde）。
    """
    sample_cfg = metric_cfg.sample_size
    if sample_cfg is None:
        return None

    test_cfg = metric_cfg.test
    planner = SampleSizeCalculation()

    if metric_cfg.test.metric_type == "continuous":
        if sample_cfg.method == "nonparametric":
            n_per_group = planner.calculate_for_means_nonparametric(
                mde=sample_cfg.mde or 0.0,
                alpha=sample_cfg.alpha,
                power=sample_cfg.power,
                sided=test_cfg.sided,
            )
        else:
            n_per_group = planner.calculate_for_means(
                mde=sample_cfg.mde or 0.0,
                alpha=sample_cfg.alpha,
                power=sample_cfg.power,
                sided=test_cfg.sided,
            )
        return {
            "type": "continuous",
            "n_per_group": float(n_per_group),
            "alpha": sample_cfg.alpha,
            "power": sample_cfg.power,
            "mde": sample_cfg.mde,
            "method": sample_cfg.method,
        }

    if metric_cfg.test.metric_type == "proportion":
        if sample_cfg.baseline_rate is None or sample_cfg.mde is None:
            return None
        n_per_group = planner.calculate_for_proportions(
            baseline_rate=sample_cfg.baseline_rate,
            mde=sample_cfg.mde,
            alpha=sample_cfg.alpha,
            power=sample_cfg.power,
            sided=test_cfg.sided,
        )
        return {
            "type": "proportion",
            "n_per_group": float(n_per_group),
            "alpha": sample_cfg.alpha,
            "power": sample_cfg.power,
            "mde": sample_cfg.mde,
            "baseline_rate": sample_cfg.baseline_rate,
        }

    return None


@dataclass
class ABTestResultRow:
    """
    单条 AB 实验结果行的数据结构。

    字段说明：
    - metric: 指标名称；
    - metric_type: 指标类型（continuous/proportion）；
    - phase: 实验阶段（如 before/after）；
    - base_group: 对照组名称；
    - variant_group: 实验组名称；
    - used_method: 实际采用的检验方法；
    - p_value: p 值；
    - is_significant: 是否显著；
    - effect: 效应值（若 stat_result 中存在 effect 字段）；
    - extra: 其他辅助信息（诊断结果、置信区间、样本量计划等）。
    """

    metric: str
    metric_type: str
    phase: str
    base_group: str
    variant_group: str
    used_method: str
    p_value: float
    is_significant: bool
    effect: Optional[float]
    extra: Dict[str, Any]


def run_ab_test(
    df: pd.DataFrame,
    config: ABTestConfig,
    base_group: str = "control",
    target_phases: Optional[List[str]] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    运行 AB 实验分析的主入口。

    设计目标：
    - 只依赖已经聚合好的宽表 df（每行一个 (group, phase)）；
    - 根据 ABTestConfig 中的指标配置，调用 core.stats 完成分布诊断、显著性检验和样本量规划；
    - 返回统一结构的结果 DataFrame，方便后续报表与可视化。

    参数：
    - df: 聚合后的 DataFrame；
      * 必须至少包含 config.group_col / config.phase_col，以及各指标配置中声明的 column；
      * 每行对应一个 (group, phase)，指标列中为该组该阶段下的样本集合（list/array）。
    - config: ABTestConfig 配置对象；
    - base_group: 对照组名称（默认 "control"），用于与其他组做对比；
    - target_phases: 需要分析的阶段列表（如 ["after"]），
      * 若为 None，则使用数据中出现的所有 phase；
    - confidence_level: 置信水平，默认 0.95。

    返回：
    - DataFrame，每行对应一个 (metric, phase, variant_group) 与 base_group 的对比结果。
      若某些指标在配置中声明但数据集中缺少对应列，会静默跳过该指标。
      若数据集中存在未在 metrics 中声明的列，会给出提示，方便你补充配置。
    """
    if config.group_col not in df.columns:
        raise ValueError(f"DataFrame 缺少组名列 {config.group_col}。")
    if config.phase_col not in df.columns:
        raise ValueError(f"DataFrame 缺少阶段列 {config.phase_col}。")

    # 提示：数据中有哪些列尚未在 metrics 中声明为指标（不报错，仅提醒）
    configured_metric_columns = {m.test.column for m in config.metrics}
    reserved_cols = {config.group_col, config.phase_col}
    unused_columns = set(df.columns) - reserved_cols - configured_metric_columns
    if unused_columns:
        warnings.warn(
            "AB 实验数据中存在未在 metrics 配置中声明的列："
            f"{', '.join(sorted(map(str, unused_columns)))}。"
            "如需对这些指标做显著性分析，请在配置中补充相应的 metrics 项。",
            UserWarning,
        )

    unique_phases = sorted(str(p) for p in df[config.phase_col].unique())
    if target_phases is None:
        phases_to_use = unique_phases
    else:
        phases_to_use = [str(p) for p in target_phases]

    if base_group not in set(str(g) for g in df[config.group_col].unique()):
        raise ValueError(
            f"DataFrame 中未找到 base_group='{base_group}'，"
            f"请检查 {config.group_col} 列的取值。"
        )

    results: List[ABTestResultRow] = []

    for metric_cfg in config.metrics:
        metric_name = metric_cfg.name
        metric_type = metric_cfg.test.metric_type
        metric_column = metric_cfg.test.column

        # 若 DataFrame 中缺少该指标列，则直接跳过（视为本次实验未涉及该指标）
        if metric_column not in df.columns:
            continue

        # 样本量规划（与 phase 无关），若配置了则只算一次
        sample_plan = _plan_sample_size_for_metric(metric_cfg)

        for phase in phases_to_use:
            samples_by_group = _extract_samples_for_metric_phase(
                df=df,
                group_col=config.group_col,
                phase_col=config.phase_col,
                metric_column=metric_column,
                metric_name=metric_name,
                phase=phase,
            )

            if base_group not in samples_by_group:
                # 当前 phase 下缺少对照组，跳过该 phase
                continue

            control_samples = samples_by_group[base_group]

            for group, variant_samples in samples_by_group.items():
                if group == base_group:
                    continue

                if metric_type == "continuous":
                    test_info = _run_continuous_test(
                        control_values=control_samples,
                        variant_values=variant_samples,
                        metric_cfg=metric_cfg,
                        confidence_level=confidence_level,
                    )
                elif metric_type == "proportion":
                    test_info = _run_proportion_test(
                        control_values=control_samples,
                        variant_values=variant_samples,
                        metric_cfg=metric_cfg,
                        confidence_level=confidence_level,
                    )
                else:
                    raise ValueError(
                        f"指标 {metric_name} 的类型 {metric_type} 非法，应为 "
                        f"'continuous' 或 'proportion'。"
                    )

                stat_result = test_info["stat_result"]
                used_method = test_info["used_method"]
                p_value = float(stat_result.get("p_value", 1.0))
                is_significant = bool(stat_result.get("is_significant", False))

                n_base, n_variant, base_value, variant_value = _report_values_from_test(
                    stat_result=stat_result,
                    used_method=used_method,
                    control_samples=control_samples,
                    variant_samples=variant_samples,
                )

                # 效应值：无论是否显著都展示。优先用 stat_result 的 effect；log_mean 用 effect_ratio；否则用差值兜底
                effect_raw = stat_result.get("effect")
                if effect_raw is not None:
                    effect_display = float(effect_raw)
                elif stat_result.get("effect_ratio") is not None:
                    effect_display = float(stat_result["effect_ratio"])
                else:
                    effect_display = variant_value - base_value

                extra: Dict[str, Any] = {
                    "diagnosis": test_info.get("diagnosis"),
                    "stat_result": stat_result,
                    "sample_size_plan": sample_plan,
                    "n_base": n_base,
                    "n_variant": n_variant,
                    "base_value": base_value,
                    "variant_value": variant_value,
                }

                results.append(
                    ABTestResultRow(
                        metric=metric_name,
                        metric_type=metric_type,
                        phase=phase,
                        base_group=base_group,
                        variant_group=group,
                        used_method=used_method,
                        p_value=p_value,
                        is_significant=is_significant,
                        effect=effect_display,
                        extra=extra,
                    )
                )

    if not results:
        raise ValueError("run_ab_test 未生成任何结果，请检查输入数据与配置是否匹配。")

    # 将结果行转换为 DataFrame；报表列优先，diagnosis/stat_detail/sample_size_plan 放最后三列
    rows: List[Dict[str, Any]] = []
    for row in results:
        base = asdict(row)
        extra = base.pop("extra", {}) or {}
        # 报表用：对照组/实验组样本量与指标数值作为顶层列
        base["n_base"] = extra.get("n_base")
        base["n_variant"] = extra.get("n_variant")
        base["base_value"] = extra.get("base_value")
        base["variant_value"] = extra.get("variant_value")
        # 详细诊断与统计结果放最后三列，便于需要时查看、不影响主表浏览
        base["diagnosis"] = extra.get("diagnosis")
        base["stat_detail"] = extra.get("stat_result")
        base["sample_size_plan"] = extra.get("sample_size_plan")
        rows.append(base)

    out = pd.DataFrame(rows)
    # 确保 diagnosis、stat_detail、sample_size_plan 为最后三列
    last_three = ["diagnosis", "stat_detail", "sample_size_plan"]
    other_cols = [c for c in out.columns if c not in last_three]
    final_cols = other_cols + [c for c in last_three if c in out.columns]
    return out[final_cols]


def _n_required_for_did_row(
    row: Dict[str, Any],
    alpha: float,
    power: float,
    sided: str,
) -> Optional[int]:
    """
    根据 DID 合并行（含 AB 阶段 stat_detail）计算检测当前效应所需每组最小样本量。

    比例型：baseline_rate=base_value_ab，mde=abs(effect_ab)，调用 calculate_for_proportions；
    连续型：从 stat_detail_ab 取出 pooled_std，Cohen's d = effect_ab / pooled_std，调用 calculate_for_means。
    无法计算时返回 None。
    """
    metric_type = row.get("metric_type_ab") or row.get("metric_type_aa")
    effect_ab = row.get("effect_ab")

    if metric_type == "proportion":
        base_value = row.get("base_value_ab")
        if base_value is None or effect_ab is None:
            return None
        try:
            baseline = float(base_value)
            mde = abs(float(effect_ab))
            if not 0.0 < baseline < 1.0 or mde <= 0 or baseline + mde >= 1.0:
                return None
            return SampleSizeCalculation.calculate_for_proportions(
                baseline_rate=baseline,
                mde=mde,
                alpha=alpha,
                power=power,
                sided=sided,
            )
        except (TypeError, ValueError):
            return None

    # 连续型：从 stat_detail 提出 pooled_std
    stat = row.get("stat_detail_ab")
    if not isinstance(stat, dict):
        return None
    pooled_std = stat.get("pooled_std")
    if pooled_std is None or effect_ab is None:
        return None
    try:
        ps = float(pooled_std)
        if ps <= 0:
            return None
        d = abs(float(effect_ab) / ps)
        if d <= 0:
            return None
        return SampleSizeCalculation.calculate_for_means(
            mde=d,
            alpha=alpha,
            power=power,
            sided=sided,
        )
    except (TypeError, ValueError):
        return None


def compute_did(
    aa_result: pd.DataFrame,
    ab_result: pd.DataFrame,
    merge_on: Optional[List[str]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    sided: str = "two-sided",
) -> pd.DataFrame:
    """
    基于 AA / AB 两阶段 run_ab_test 结果，计算双重差分（DID）点估计，并评估样本量是否充足。

    数据层面：
    - AA：run_ab_test(..., target_phases=["before"]) → 每 (metric, variant_group) 的 effect、p_value 等；
    - AB：run_ab_test(..., target_phases=["after"]) → 同样结构。
    两表按 (metric, variant_group) 对齐，DID 数值为 did_effect = effect_ab - effect_aa。

    参数：
    - aa_result: AA 阶段 run_ab_test 返回的 DataFrame（如 target_phases=["before"]）；
    - ab_result: AB 阶段 run_ab_test 返回的 DataFrame（如 target_phases=["after"]）；
    - merge_on: 对齐键列，默认 ["metric", "variant_group"]；
    - alpha: 样本量计算用显著性水平，默认 0.05；
    - power: 样本量计算用统计功效，默认 0.8；
    - sided: 样本量计算用双侧/单侧，默认 "two-sided"。

    返回：
    - DataFrame，仅包含以下列（不包含统计学过程参数）：
      * 标识：metric, variant_group；
      * 效应与 DID：effect_aa, effect_ab, did_effect；
      * 显著性（AB 阶段）：p_value, is_significant；
      * 报表用（AB 阶段）：n_base, n_variant, base_value, variant_value；
      * 样本量双保险：n_required（检测当前效应所需每组最小样本量）、n_actual（min(n_base,n_variant)）、sample_sufficient（是否充足）。
    仅保留在两表中均能对齐成功的行（inner join）。
    """
    key = merge_on if merge_on is not None else ["metric", "variant_group"]
    required = set(key) | {"effect", "p_value", "is_significant", "metric_type"}
    for name, frame in [("aa_result", aa_result), ("ab_result", ab_result)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} 缺少列 {sorted(missing)}，无法计算 DID。"
                "请确保为 run_ab_test 返回的 DataFrame，且包含 metric、variant_group、effect、p_value、is_significant、metric_type。"
            )

    merged = aa_result.merge(
        ab_result,
        on=key,
        how="inner",
        suffixes=("_aa", "_ab"),
    )
    merged["did_effect"] = merged["effect_ab"] - merged["effect_aa"]

    # 根据效应值计算最小样本量（从 stat_detail_ab 提出 pooled_std 用于连续型）
    merged["n_required"] = merged.apply(
        lambda r: _n_required_for_did_row(r, alpha, power, sided),
        axis=1,
    )
    if "n_base_ab" in merged.columns and "n_variant_ab" in merged.columns:
        merged["n_actual"] = merged[["n_base_ab", "n_variant_ab"]].min(axis=1)
    else:
        merged["n_actual"] = pd.NA
    merged["sample_sufficient"] = (
        merged["n_required"].notna()
        & merged["n_actual"].notna()
        & (merged["n_actual"] >= merged["n_required"])
    )

    # 按约定字段布局输出：variant_group 后紧跟 AB 阶段对照组/实验组数值，再为效应与显著性等
    out_cols: List[str] = list(key) + [
        "base_value_ab",
        "variant_value_ab",
        "effect_aa",
        "effect_ab",
        "did_effect",
        "p_value_ab",
        "is_significant_ab",
        "n_base_ab",
        "n_variant_ab",
        "n_required",
        "n_actual",
        "sample_sufficient",
    ]
    renames = {
        "p_value_ab": "p_value",
        "is_significant_ab": "is_significant",
        "n_base_ab": "n_base",
        "n_variant_ab": "n_variant",
        "base_value_ab": "base_value",
        "variant_value_ab": "variant_value",
    }
    out = merged[[c for c in out_cols if c in merged.columns]].copy()
    out = out.rename(columns=renames)
    return out

