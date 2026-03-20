from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.stats import (
    diagnose_continuous_distribution,
    quantile_summary,
    test_normality,
)
from core.visualizer import plot_distribution_diagnostic

# --- 分布报表内置默认（无独立 YAML；需在 Notebook 中改的请改常量或后续加函数参数） ---

_REPORT_NORM_ALPHA = 0.05
_REPORT_NORM_METHODS: Sequence[str] = ("shapiro", "dagostino", "anderson")
_REPORT_SHAPIRO_MAX_N = 5000
_REPORT_SHAPIRO_RANDOM_SEED = 42
_REPORT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
_REPORT_HIST_BINS = 40
_REPORT_LOG_PANEL_MIN_ABS_SKEW = 0.5


def _finite_series(values: Sequence[float]) -> pd.Series:
    s = pd.Series(values, dtype="float64")
    return s[np.isfinite(s)]


def _format_diagnostic_cell(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _md_table_from_records(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """简单 Markdown 表（无外部 tabulate 依赖）。"""
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [headers, sep]
    for row in rows:
        cells = []
        for c in columns:
            v = row.get(c, "")
            if v is None:
                v = ""
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_analyst_notes(
    diagnostic: Mapping[str, Any],
    normality: Mapping[str, Any],
) -> str:
    """供分析师参考的固定说明（非自动结论）。"""
    lines: List[str] = ["## 分析师提示", ""]
    n = int(diagnostic.get("n", 0))
    if n >= 1000:
        lines.append(
            "- 样本量较大时，正态性检验容易在数学意义上「拒绝正态」，但仍可结合 QQ 图与业务判断"
            "是否采用基于均值的渐近方法。"
        )
    if diagnostic.get("is_zero_inflated"):
        lines.append(
            "- 零占比较高：全样本均值/正态假定可能失真，可关注非零子总体或零膨胀类模型。"
        )
    if diagnostic.get("is_heavy_tailed"):
        lines.append("- 偏度/峰度提示重尾或明显偏态：秩检验或对数变换可能更稳妥（视取值是否为正而定）。")

    rej = [
        t.get("method")
        for t in normality.get("tests", [])
        if t.get("rejected") is True
    ]
    acc = [
        t.get("method")
        for t in normality.get("tests", [])
        if t.get("rejected") is False
    ]
    if rej:
        lines.append(f"- 在设定 α 下拒绝正态假定的检验：{', '.join(str(x) for x in rej)}。")
    if acc:
        lines.append(f"- 在设定 α 下未拒绝正态假定的检验：{', '.join(str(x) for x in acc)}。")
    lines.append("- **是否采用参数检验/是否变换尺度，请以业务口径与图形为准，勿仅依赖 p 值。**")
    return "\n".join(lines)


@dataclass
class DistributionReportBundle:
    """
    分布诊断报表聚合结果（面向 Notebook）。

    字段：
    - metric_name: 指标名称；
    - diagnostic: diagnose_continuous_distribution 输出；
    - normality: test_normality 输出；
    - quantiles: 分位数字典；
    - figure: Matplotlib Figure；
    - markdown_body: 完整 Markdown 正文；
    - extra_markdown_prefix: 开头可追加的前缀块（例如数据源说明）。
    """

    metric_name: str
    diagnostic: Dict[str, Any]
    normality: Dict[str, Any]
    quantiles: Dict[str, float]
    figure: plt.Figure
    markdown_body: str
    extra_markdown_prefix: str = ""

    def full_markdown(self) -> str:
        """合并前缀与正文（图表另外通过 `display_distribution_report` 展示）。"""
        parts = [self.extra_markdown_prefix, self.markdown_body]
        return "\n\n".join(p for p in parts if p and p.strip())


def build_distribution_report(
    values: Sequence[float],
    metric_name: str,
    *,
    visual_cfg: Optional[Mapping[str, Any]] = None,
    extra_markdown_prefix: str = "",
) -> DistributionReportBundle:
    """
    生成分布诊断报表数据与 Figure（参数写死在模块常量中，不读 YAML）。

    输入：
    - values: 原始样本（inf/nan 会在报表内剔除）；
    - metric_name: 指标展示名；
    - visual_cfg: 可选，通常来自 `load_yaml(\"configs/visualizer.yaml\")`，仅控制字体/负号等 matplotlib 样式；
    - extra_markdown_prefix: 可选，报告开头追加的 Markdown。
    """
    s = _finite_series(values)
    if s.empty:
        raise ValueError("有效样本为空，无法生成分布报表。")
    arr = s.to_numpy()

    diag = diagnose_continuous_distribution(arr)
    norm = test_normality(
        arr,
        alpha=_REPORT_NORM_ALPHA,
        methods=_REPORT_NORM_METHODS,
        shapiro_max_n=_REPORT_SHAPIRO_MAX_N,
        random_seed=_REPORT_SHAPIRO_RANDOM_SEED,
    )
    quantiles = quantile_summary(arr, _REPORT_PERCENTILES)

    show_log = True
    min_skew = _REPORT_LOG_PANEL_MIN_ABS_SKEW
    all_positive = bool((arr > 0).all())
    skew = float(diag.get("skewness", 0.0))
    if not all_positive or abs(skew) < min_skew:
        show_log = False

    fig = plot_distribution_diagnostic(
        arr,
        title=metric_name,
        bins=_REPORT_HIST_BINS,
        show_kde=True,
        show_qq=True,
        show_log_panel=show_log,
        visual_config=visual_cfg,
    )

    sections: List[str] = [f"# 指标分布诊断：{metric_name}", f"有效样本量 n={len(arr)}"]
    if norm.get("warnings"):
        sections.append("## 检验警告\n" + "\n".join(f"- {w}" for w in norm["warnings"]))

    diag_order = [
        "n",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skewness",
        "kurtosis_excess",
        "zero_ratio",
        "is_approximately_normal",
        "is_heavy_tailed",
        "is_zero_inflated",
    ]
    diag_rows = [{"项": k, "值": _format_diagnostic_cell(diag[k])} for k in diag_order if k in diag]
    sections.append(
        "## 描述性诊断（启发式）\n"
        + _md_table_from_records(diag_rows, ["项", "值"])
        + f"\n\n**recommended_tests**（仅供检验策略参考）: `{diag.get('recommended_tests')}`"
    )

    test_rows: List[Dict[str, Any]] = []
    for t in norm.get("tests", []):
        p = t.get("p_value")
        test_rows.append(
            {
                "方法": t.get("method", ""),
                "统计量": ""
                if t.get("statistic") is None or pd.isna(t.get("statistic"))
                else round(float(t["statistic"]), 6),
                "p 值": ""
                if p is None or (isinstance(p, float) and pd.isna(p))
                else round(float(p), 6),
                "拒绝正态(α)": t.get("rejected"),
                "备注": t.get("notes") or "",
            }
        )
    sections.append(
        "## 正态性检验（SciPy）\n"
        + _md_table_from_records(test_rows, ["方法", "统计量", "p 值", "拒绝正态(α)", "备注"])
    )

    qrows = [{"分位键": k, "值": round(v, 6)} for k, v in sorted(quantiles.items())]
    sections.append("## 分位数\n" + _md_table_from_records(qrows, ["分位键", "值"]))

    sections.append(_build_analyst_notes(diag, norm))

    body = "\n\n".join(sections)
    return DistributionReportBundle(
        metric_name=metric_name,
        diagnostic=diag,
        normality=norm,
        quantiles=quantiles,
        figure=fig,
        markdown_body=body,
        extra_markdown_prefix=extra_markdown_prefix.strip(),
    )


def display_distribution_report(bundle: DistributionReportBundle) -> None:
    """
    在 Jupyter Notebook 中展示 Markdown 与 Figure。

    按需安装 IPython。
    """
    try:
        from IPython.display import Markdown, display
    except ImportError as exc:
        raise ImportError("display_distribution_report 需要 IPython（如在 Jupyter 中运行）。") from exc
    text = bundle.full_markdown()
    display(Markdown(text))
    display(bundle.figure)
