from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def _setup_matplotlib_style(
    style_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    初始化 Matplotlib 的字体与负号显示设置。

    说明：
    - 默认统一使用 Arial Unicode MS 作为优先字体，尽可能覆盖中文与全角符号；
      将 axes.unicode_minus 设为 False，保证负号显示为 "-" 而不是方块；
    - 若传入 style_config（通常来自 visualizer.yaml["matplotlib"]），则优先使用配置，
      缺失字段回退到上述默认行为。
    """
    # --- 基础默认值 ---
    default_unicode_minus = False
    default_font_family = "sans-serif"
    default_sans_serif = [
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "DejaVu Sans",
    ]

    # 后端与全局 style 可以通过配置注入，但不是强制
    if style_config is not None:
        backend = style_config.get("backend")
        if isinstance(backend, str) and backend.strip():
            try:
                matplotlib.use(backend)
            except Exception:
                # 后端设置失败时静默回退，避免在 notebook/GUI 环境报错
                pass

        style_name = style_config.get("style")
        if isinstance(style_name, str) and style_name.strip():
            try:
                plt.style.use(style_name)
            except Exception:
                # 不识别的 style 名称时同样静默回退
                pass

        axes_cfg = style_config.get("axes") or {}
        unicode_minus = bool(axes_cfg.get("unicode_minus", default_unicode_minus))
    else:
        unicode_minus = default_unicode_minus

    matplotlib.rcParams["axes.unicode_minus"] = unicode_minus

    # 字体相关
    if style_config is not None:
        font_cfg = style_config.get("font") or {}
        font_family = font_cfg.get("family", default_font_family)
        sans_serif_cfg = font_cfg.get("sans_serif")
        if isinstance(sans_serif_cfg, (list, tuple)) and sans_serif_cfg:
            sans_serif_list = [str(x) for x in sans_serif_cfg]
        else:
            sans_serif_list = default_sans_serif
    else:
        font_family = default_font_family
        sans_serif_list = default_sans_serif

    # 合并配置字体与当前 rcParams 中已有的 sans-serif，避免覆盖用户其他设置
    current: Sequence[str] | str = matplotlib.rcParams.get("font.sans-serif", [])
    if isinstance(current, str):
        existing_fonts: List[str] = [current]
    else:
        existing_fonts = list(current)

    for name in reversed(sans_serif_list):
        if name not in existing_fonts:
            existing_fonts.insert(0, name)

    matplotlib.rcParams["font.sans-serif"] = existing_fonts
    matplotlib.rcParams["font.family"] = font_family


def _detect_stratify_cols(df: pd.DataFrame) -> List[str]:
    """
    从 AB 实验结果表中识别分层列（即 run_ab_test / compute_did 中由 stratify_by 带来的列）。

    约定：
    - 去除 metric、phase、base_group、variant_group 等核心字段及统计字段；
    - 剩余的维度型列视为分层列，在可视化中用于标签或过滤。
    """
    core_cols = {
        "metric",
        "metric_type",
        "phase",
        "base_group",
        "variant_group",
        "used_method",
        "p_value",
        "is_significant",
        "effect",
        "effect_ratio",
        "n_base",
        "n_variant",
        "base_value",
        "variant_value",
        "diagnosis",
        "stat_detail",
        "sample_size_plan",
        # compute_did 的核心列
        "effect_aa",
        "effect_ab",
        "did_effect",
        "aa_significant",
        "ab_significant",
        "n_required",
        "effect_n_required",
        "sample_sufficient",
    }
    return [c for c in df.columns if c not in core_cols]


def style_ab_overview_table(
    result_df: pd.DataFrame,
    sort_by: Optional[Sequence[str]] = None,
    p_value_col: str = "p_value",
    effect_col: str = "effect",
    sample_sufficient_col: str = "sample_sufficient",
    max_rows: Optional[int] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> "pd.io.formats.style.Styler":
    """
    根据 AB 实验结果 DataFrame 生成带条件格式的总览表（Styler）。

    典型用途：
    - 在 JupyterLab 中快速总览“哪些指标在哪些实验组显著、效应方向如何、样本量是否充足”；
    - 作为后续在 notebook 中展示或导出为 Excel 的基础表格。

    输入：
    - result_df:
        run_ab_test 返回的结果表，至少包含：
        metric, variant_group, p_value, effect 以及样本量相关列；
    - sort_by:
        可选，排序列名列表；若为空，则按 (metric, phase, variant_group, p_value) 排序；
    - p_value_col / effect_col / sample_sufficient_col:
        对应列名，可根据实际列名在调用时调整；
    - max_rows:
        可选，限制总览表最多展示多少行，用于指标/分层非常多时的截断。

    输出：
    - pandas Styler 对象，可在 notebook 中直接显示，也可导出为 Excel。
    """
    if result_df.empty:
        raise ValueError("style_ab_overview_table 收到空的 result_df，无法生成总览表。")

    df = result_df.copy()

    # 默认排序：按 metric / phase / variant_group / p_value
    if sort_by:
        sort_cols = [c for c in sort_by if c in df.columns]
    else:
        candidate = ["metric", "phase", "variant_group", p_value_col]
        sort_cols = [c for c in candidate if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    # 识别分层列：在总览表中放在 variant_group 之后，方便看到完整的 (metric, phase, variant_group, stratify...) 组合
    stratify_cols = _detect_stratify_cols(df)

    # 只保留在汇总视角下最常用的几列；其余列依然保留在原始 result_df 中，便于深入分析时使用
    ordered_cols = [
        "metric",
        "metric_type",
        "phase",
        "base_group",
        "variant_group",
        *stratify_cols,
        "n_base",
        "n_variant",
        "base_value",
        "variant_value",
        effect_col,
        p_value_col,
        "is_significant",
        "n_required",
        sample_sufficient_col,
    ]
    display_cols = [c for c in ordered_cols if c in df.columns]
    if display_cols:
        df = df[display_cols]

    styler = df.style

    table_cfg: Mapping[str, Any] = {}
    if visual_config is not None:
        abtest_cfg = visual_config.get("abtest") or {}
        table_cfg = abtest_cfg.get("table") or {}

    # p_value 条件格式：p 越小背景越红
    if p_value_col in df.columns:

        p_cfg: Mapping[str, Any] = table_cfg.get("p_value") or {}
        thr_cfg: Mapping[str, Any] = p_cfg.get("thresholds") or {}
        color_cfg: Mapping[str, Any] = p_cfg.get("colors") or {}

        strong_thr = float(thr_cfg.get("strong", 0.01))
        medium_thr = float(thr_cfg.get("medium", 0.05))
        weak_thr = float(thr_cfg.get("weak", 0.1))

        strong_color = str(color_cfg.get("strong", "#d73027"))
        medium_color = str(color_cfg.get("medium", "#fc8d59"))
        weak_color = str(color_cfg.get("weak", "#fee08b"))

        def _color_p_value(val: Any) -> str:
            if pd.isna(val):
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v < strong_thr:
                return f"background-color:{strong_color};color:white"
            if v < medium_thr:
                return f"background-color:{medium_color}"
            if v < weak_thr:
                return f"background-color:{weak_color}"
            return ""

        styler = styler.applymap(_color_p_value, subset=[p_value_col])

    # 效应方向用颜色区分：正为绿色，负为红色
    if effect_col in df.columns:

        effect_cfg: Mapping[str, Any] = table_cfg.get("effect") or {}
        effect_colors: Mapping[str, Any] = effect_cfg.get("colors") or {}

        pos_color = str(effect_colors.get("positive", "#1a9850"))
        neg_color = str(effect_colors.get("negative", "#d73027"))

        def _color_effect(val: Any) -> str:
            if pd.isna(val):
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > 0:
                return f"color:{pos_color}"
            if v < 0:
                return f"color:{neg_color}"
            return ""

        styler = styler.applymap(_color_effect, subset=[effect_col])

    # 样本量是否充足：用浅绿/浅橙背景标记
    if sample_sufficient_col in df.columns:

        ss_cfg: Mapping[str, Any] = table_cfg.get("sample_sufficient") or {}
        ss_colors: Mapping[str, Any] = ss_cfg.get("colors") or {}

        enough_color = str(ss_colors.get("enough", "#c7e9c0"))
        not_enough_color = str(ss_colors.get("not_enough", "#fee0b6"))

        def _color_sample_sufficient(val: Any) -> str:
            if pd.isna(val):
                return ""
            if bool(val):
                return f"background-color:{enough_color}"
            return f"background-color:{not_enough_color}"

        styler = styler.applymap(_color_sample_sufficient, subset=[sample_sufficient_col])

    return styler


def plot_metric_effect_bars(
    result_df: pd.DataFrame,
    metric: str,
    phase: Optional[str] = None,
    title: Optional[str] = None,
    stratify_filters: Optional[Mapping[str, Any]] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制单个指标在各实验组上的“基准值 vs 实验值”柱状图，并标注效应方向与数值标签。

    输入：
    - result_df:
        run_ab_test 返回的结果表；
    - metric:
        指标名称，对应 result_df["metric"] 的取值；
    - phase:
        可选，仅绘制某个阶段（如 "after"）；若为空且存在 phase 列，则会自动选取全部 phase，并在标签中体现；
    - title:
        可选，图标题；为空时自动生成。
    - stratify_filters:
        可选，按分层列做精确筛选的条件，如 {"country": "US", "level": "VIP"}；
        若为空，则保留所有分层组合。

    输出：
    - Matplotlib Figure 对象，可在 Jupyter 中直接显示或保存为图片。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    plot_cfg: Mapping[str, Any] = {}
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None
        abtest_cfg = visual_config.get("abtest") or {}
        plot_cfg = abtest_cfg.get("plot") or {}

    _setup_matplotlib_style(matplotlib_cfg)

    if "metric" not in result_df.columns:
        raise ValueError("result_df 中缺少 'metric' 列，无法根据指标筛选。")

    data = result_df[result_df["metric"] == metric].copy()
    if data.empty:
        raise ValueError(f"在 result_df 中未找到指标 '{metric}' 的结果行。")

    if phase is not None and "phase" in data.columns:
        data = data[data["phase"] == phase]
        if data.empty:
            raise ValueError(f"在 phase='{phase}' 下未找到指标 '{metric}' 的结果行。")

    # 分层过滤：精确锁定某个 segment
    if stratify_filters:
        for col, val in stratify_filters.items():
            if col not in data.columns:
                raise ValueError(f"stratify_filters 中的列 '{col}' 不在 result_df 中。")
            data = data[data[col] == val]
        if data.empty:
            raise ValueError("在给定的 stratify_filters 条件下未找到任何结果行。")

    # 构造 x 轴标签：variant_group / (phase, variant_group) + 分层信息
    if "variant_group" not in data.columns or "base_group" not in data.columns:
        raise ValueError("result_df 需要包含 'base_group' 与 'variant_group' 列。")

    base_group = str(data["base_group"].iloc[0])

    stratify_cols = _detect_stratify_cols(data)

    labels: List[str] = []
    for _, row in data.iterrows():
        # 若存在分层列，则横坐标仅展示分层信息（使用换行避免过长）
        if stratify_cols:
            seg_parts = [f"{col}={row[col]}" for col in stratify_cols]
            full_label = "\n".join(seg_parts)
        else:
            # 无分层列时退化为原有逻辑：phase-variant_group 或 variant_group
            if "phase" in data.columns and phase is None:
                full_label = f"{row['phase']}-{row['variant_group']}"
            else:
                full_label = str(row["variant_group"])

        labels.append(full_label)

    # 数值列
    if "base_value" not in data.columns or "variant_value" not in data.columns:
        raise ValueError("result_df 需要包含 'base_value' 与 'variant_value' 列。")

    base_values = data["base_value"].astype(float).tolist()
    variant_values = data["variant_value"].astype(float).tolist()

    x = range(len(labels))
    width = 0.35

    # 宽度保持不变，长度随样本数量适度增加，避免拥挤
    fig_width = max(6, len(labels) * 0.8)
    fig_height = max(5.0, 3.0 + 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    base_positions = [i - width / 2 for i in x]
    variant_positions = [i + width / 2 for i in x]

    metric_bars_cfg: Mapping[str, Any] = plot_cfg.get("metric_effect_bars") or {}
    base_color = str(metric_bars_cfg.get("base_color", "#cccccc"))
    variant_color = str(metric_bars_cfg.get("variant_color", "#3182bd"))

    base_bars = ax.bar(
        base_positions,
        base_values,
        width=width,
        label=f"{base_group}（对照组）",
        color=base_color,
    )
    variant_bars = ax.bar(
        variant_positions,
        variant_values,
        width=width,
        label="实验组",
        color=variant_color,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("实验组 / 分层组合")
    ax.set_ylabel("指标数值")

    if title:
        ax.set_title(title)
    else:
        if phase is not None:
            ax.set_title(f"{metric}（{phase} 阶段）对照组 vs 实验组")
        else:
            ax.set_title(f"{metric} 对照组 vs 实验组")

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 在柱子顶部增加数值标签
    def _annotate_bars(container, values: List[float]) -> None:
        for rect, val in zip(container, values):
            height = rect.get_height()
            if pd.isna(val):
                continue
            text = f"{float(val):.3g}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate_bars(base_bars, base_values)
    _annotate_bars(variant_bars, variant_values)

    fig.tight_layout()
    return fig


def _extract_effect_and_ci_for_forest_row(
    row: Mapping[str, Any],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    从 run_ab_test 的单行结果中抽取用于森林图的效应及置信区间。

    返回：
    - (effect, ci_low, ci_high)，若缺少置信区间则后两者为 None。
    """
    effect = row.get("effect")
    if effect is None and "effect_ratio" in row:
        effect = row.get("effect_ratio")

    stat_detail = row.get("stat_detail")
    if not isinstance(stat_detail, Mapping):
        return (None if effect is None else float(effect), None, None)

    metric_type = row.get("metric_type")
    ci_key = "confidence_interval"
    if metric_type == "continuous_log" and "confidence_interval_ratio" in stat_detail:
        ci_key = "confidence_interval_ratio"

    ci_val = stat_detail.get(ci_key)
    if (
        isinstance(ci_val, (list, tuple))
        and len(ci_val) == 2
        and all(val is not None for val in ci_val)
    ):
        ci_low, ci_high = float(ci_val[0]), float(ci_val[1])
    else:
        ci_low, ci_high = None, None

    return (None if effect is None else float(effect), ci_low, ci_high)


def plot_forest_for_metric(
    result_df: pd.DataFrame,
    metric: str,
    phase: Optional[str] = None,
    title: Optional[str] = None,
    stratify_filters: Optional[Mapping[str, Any]] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制单个指标的“效应值 + 置信区间”森林图（火柴人图）。

    设计目标：
    - x 轴为效应值（差值或倍数），y 轴为各实验组（及分层组合）；
    - 通过误差线展示置信区间，并用颜色区分是否显著。

    输入：
    - result_df:
        run_ab_test 返回的结果表；
    - metric:
        指标名称；
    - phase:
        可选，仅绘制某个阶段；为空时若存在 phase 列则自动包含所有阶段；
    - title:
        可选，自定义标题。
    - stratify_filters:
        可选，按分层列做精确筛选的条件，如 {"country": "US", "level": "VIP"}；
        若为空，则保留所有分层组合。

    输出：
    - Matplotlib Figure 对象。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None

    _setup_matplotlib_style(matplotlib_cfg)

    if "metric" not in result_df.columns:
        raise ValueError("result_df 中缺少 'metric' 列，无法根据指标筛选。")

    data = result_df[result_df["metric"] == metric].copy()
    if data.empty:
        raise ValueError(f"在 result_df 中未找到指标 '{metric}' 的结果行。")

    if phase is not None and "phase" in data.columns:
        data = data[data["phase"] == phase]
        if data.empty:
            raise ValueError(f"在 phase='{phase}' 下未找到指标 '{metric}' 的结果行。")

    # 分层过滤：精确锁定某个 segment
    if stratify_filters:
        for col, val in stratify_filters.items():
            if col not in data.columns:
                raise ValueError(f"stratify_filters 中的列 '{col}' 不在 result_df 中。")
            data = data[data[col] == val]
        if data.empty:
            raise ValueError("在给定的 stratify_filters 条件下未找到任何结果行。")

    # 构造 y 轴标签：variant_group + 分层信息
    if "variant_group" not in data.columns:
        raise ValueError("result_df 需要包含 'variant_group' 列。")

    stratify_cols: List[str] = _detect_stratify_cols(data)

    labels: List[str] = []
    effects: List[float] = []
    ci_lows: List[Optional[float]] = []
    ci_highs: List[Optional[float]] = []
    sig_flags: List[bool] = []

    for _, row in data.iterrows():
        parts = [str(row["variant_group"])]
        if "phase" in data.columns and phase is None:
            parts.insert(0, f"{row['phase']}")
        for col in stratify_cols:
            parts.append(f"{col}={row[col]}")
        label = " | ".join(parts)

        effect, ci_low, ci_high = _extract_effect_and_ci_for_forest_row(row)
        if effect is None:
            continue

        labels.append(label)
        effects.append(effect)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        sig_flags.append(bool(row.get("is_significant", False)))

    if not labels:
        raise ValueError("在指定条件下未能解析出有效的效应与置信区间，无法绘制森林图。")

    # y 轴从上到下对应 labels
    y_pos = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 0.6 * len(labels) + 2))

    for idx, (y, eff, ci_low, ci_high, is_sig) in enumerate(
        zip(y_pos, effects, ci_lows, ci_highs, sig_flags)
    ):
        color = "#1a9850" if is_sig and eff > 0 else "#d73027" if is_sig else "#636363"
        if ci_low is not None and ci_high is not None:
            ax.errorbar(
                x=eff,
                y=y,
                xerr=[[eff - ci_low], [ci_high - eff]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
            )
        else:
            ax.plot(eff, y, "o", color=color)

    ax.axvline(0.0, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("效应值")

    if title:
        ax.set_title(title)
    else:
        if phase is not None:
            ax.set_title(f"{metric}（phase={phase}）效应森林图")
        else:
            ax.set_title(f"{metric} 效应森林图")

    fig.tight_layout()
    return fig


def plot_segment_heatmap(
    result_df: pd.DataFrame,
    metric: str,
    stratify_col: str,
    value_col: str = "effect",
    phase: Optional[str] = None,
    title: Optional[str] = None,
    stratify_filters: Optional[Mapping[str, Any]] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制单个指标在不同分层维度与实验组上的热力图。

    典型用法：
    - 查看某个指标在不同国家/等级/渠道下，各实验组 uplift 的“好坏分布”；
    - 快速识别“在哪些人群里实验效果明显好/明显差”。

    输入：
    - result_df:
        run_ab_test 或 compute_did 返回的结果表；
    - metric:
        指标名称；
    - stratify_col:
        作为行索引的分层列名，如 "country"、"level" 等；
    - value_col:
        作为热力值的列名，默认使用 effect（可根据需要传入 did_effect 等）；
    - phase:
        可选，仅绘制某个阶段；
    - title:
        可选，自定义标题。
    - stratify_filters:
        可选，其它分层列的固定条件（如 stratify_by=["country","level"] 时，传 {"level": "VIP"}，
        则此图只展示 level=VIP 这一层的 country × variant_group 热力图）。

    输出：
    - Matplotlib Figure 对象。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None

    _setup_matplotlib_style(matplotlib_cfg)

    required_cols = {"metric", stratify_col, "variant_group", value_col}
    missing = required_cols.difference(result_df.columns)
    if missing:
        raise ValueError(
            f"result_df 缺少绘制分层热力图所需的列：{', '.join(sorted(missing))}。"
        )

    data = result_df[result_df["metric"] == metric].copy()
    if phase is not None and "phase" in data.columns:
        data = data[data["phase"] == phase]

    # 对除 stratify_col 以外的分层列应用过滤条件
    if stratify_filters:
        for col, val in stratify_filters.items():
            if col == stratify_col:
                continue
            if col not in data.columns:
                raise ValueError(f"stratify_filters 中的列 '{col}' 不在 result_df 中。")
            data = data[data[col] == val]

    if data.empty:
        raise ValueError("在给定的 metric/phase/stratify_filters 条件下未找到任何结果行，无法绘制热力图。")

    pivot = data.pivot_table(
        index=stratify_col,
        columns="variant_group",
        values=value_col,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(
        figsize=(
            max(4, len(pivot.columns) * 0.8),
            max(3, len(pivot.index) * 0.5),
        )
    )
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    if title:
        ax.set_title(title)
    else:
        base_title = f"{metric} 分层热力图（按 {stratify_col} × variant_group）"
        if phase is not None:
            base_title += f"｜phase={phase}"
        ax.set_title(base_title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    fig.tight_layout()
    return fig


def plot_did_effect(
    did_df: pd.DataFrame,
    metric: str,
    variant_group: Optional[str] = None,
    title: Optional[str] = None,
    stratify_filters: Optional[Mapping[str, Any]] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制单个指标的 DID 前后对比图：effect_aa / effect_ab / did_effect。

    输入：
    - did_df:
        compute_did 返回的结果表；
    - metric:
        指标名称；
    - variant_group:
        可选，指定某个实验组；为空时若存在多个实验组会全部绘制，每组三根柱；
    - title:
        可选，自定义标题。
    - stratify_filters:
        可选，按分层列做精确筛选的条件，如 {"country": "US", "level": "VIP"}；
        若为空，则保留所有分层组合。

    输出：
    - Matplotlib Figure 对象。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    plot_cfg: Mapping[str, Any] = {}
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None
        abtest_cfg = visual_config.get("abtest") or {}
        plot_cfg = abtest_cfg.get("plot") or {}

    _setup_matplotlib_style(matplotlib_cfg)

    required_cols = {
        "metric",
        "variant_group",
        "effect_aa",
        "effect_ab",
        "did_effect",
    }
    missing = required_cols.difference(did_df.columns)
    if missing:
        raise ValueError(
            f"did_df 缺少绘制 DID 图所需的列：{', '.join(sorted(missing))}。"
        )

    data = did_df[did_df["metric"] == metric].copy()
    if variant_group is not None:
        data = data[data["variant_group"] == variant_group]

    # 分层过滤：精确锁定某个 segment
    if stratify_filters:
        for col, val in stratify_filters.items():
            if col not in data.columns:
                raise ValueError(f"stratify_filters 中的列 '{col}' 不在 did_df 中。")
            data = data[data[col] == val]

    if data.empty:
        raise ValueError("在给定的 metric/variant_group/stratify_filters 条件下未找到 DID 结果行。")

    # 横坐标标签：优先使用分层信息，仅在无分层列时退化为 variant_group
    stratify_cols = _detect_stratify_cols(data)

    labels: List[str] = []
    for _, row in data.iterrows():
        if stratify_cols:
            seg_parts = [f"{col}={row[col]}" for col in stratify_cols]
            label = "\n".join(seg_parts)
        else:
            label = str(row["variant_group"])
        labels.append(label)
    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))

    effect_aa = data["effect_aa"].astype(float).tolist()
    effect_ab = data["effect_ab"].astype(float).tolist()
    did_effect = data["did_effect"].astype(float).tolist()

    did_plot_cfg: Mapping[str, Any] = plot_cfg.get("did_effect") or {}
    aa_color = str(did_plot_cfg.get("effect_aa_color", "#9ecae1"))
    ab_color = str(did_plot_cfg.get("effect_ab_color", "#3182bd"))
    did_color = str(did_plot_cfg.get("did_effect_color", "#de2d26"))

    bars_aa = ax.bar(
        [i - width for i in x],
        effect_aa,
        width=width,
        label="effect_aa",
        color=aa_color,
    )
    bars_ab = ax.bar(
        x,
        effect_ab,
        width=width,
        label="effect_ab",
        color=ab_color,
    )
    bars_did = ax.bar(
        [i + width for i in x],
        did_effect,
        width=width,
        label="did_effect",
        color=did_color,
    )

    ax.axhline(0.0, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("效应")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric} DID 前后效应对比图")

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 在三组柱子顶部增加数值标签
    def _annotate_bars(container, values: List[float]) -> None:
        for rect, val in zip(container, values):
            height = rect.get_height()
            if pd.isna(val):
                continue
            text = f"{float(val):.3g}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height,
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    _annotate_bars(bars_aa, effect_aa)
    _annotate_bars(bars_ab, effect_ab)
    _annotate_bars(bars_did, did_effect)

    fig.tight_layout()
    return fig


def plot_metric_distribution(
    control_samples: Iterable[float],
    variant_samples: Iterable[float],
    metric_name: str,
    bins: int = 40,
    density: bool = True,
    title: Optional[str] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制连续型指标在对照组与实验组上的分布诊断图（直方图叠加）。

    设计目标：
    - 辅助解释为何选择 mean/log_mean/mann_whitney 等不同检验方法；
    - 快速感知分布形态（偏态、重尾、零膨胀等）。

    输入：
    - control_samples:
        对照组样本序列；
    - variant_samples:
        实验组样本序列；
    - metric_name:
        指标名称，用于图例/标题；
    - bins:
        直方图分箱数，默认 40；
    - density:
        是否使用密度直方图（面积为 1），默认 True；
    - title:
        可选，自定义标题。

    输出：
    - Matplotlib Figure 对象。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None

    _setup_matplotlib_style(matplotlib_cfg)

    control_series = pd.Series(list(control_samples), dtype="float64")
    variant_series = pd.Series(list(variant_samples), dtype="float64")

    if control_series.empty or variant_series.empty:
        raise ValueError("control_samples 与 variant_samples 不能为空。")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        control_series,
        bins=bins,
        alpha=0.5,
        density=density,
        label="control",
        color="#9ecae1",
    )
    ax.hist(
        variant_series,
        bins=bins,
        alpha=0.5,
        density=density,
        label="variant",
        color="#fc9272",
    )

    ax.set_xlabel(metric_name)
    ax.set_ylabel("密度" if density else "频数")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric_name} 分布诊断图")

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig

