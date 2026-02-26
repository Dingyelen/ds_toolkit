from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from core.visualizer.abtest_visualizer import (
    plot_did_effect,
    plot_metric_effect_bars,
    style_ab_overview_table,
)


@dataclass
class MetricEffectBarsConfig:
    """
    单个指标在报表中绘制“基准 vs 实验”柱状图的配置。

    输入字段说明：
    - name: 指标名称，对应 ab_result["metric"] 的取值；
    - phase: 可选，仅针对某个 phase 绘图；若为 None 则沿用可视化函数的默认行为；
    - stratify_filters: 可选，分层过滤条件，如 {"country": "US", "level": "VIP"}。
    """

    name: str
    phase: Optional[str] = None
    stratify_filters: Optional[Mapping[str, Any]] = None


@dataclass
class DidEffectConfig:
    """
    单个指标 DID 图的配置。

    输入字段说明：
    - metric: 指标名称；
    - variant_group: 可选，指定某一实验组；为 None 时由 plot_did_effect 自行决定。
    """

    metric: str
    variant_group: Optional[str] = None


@dataclass
class MetricEffectBarsVisualConfig:
    """
    报表中“metric effect bars” 图的整体配置。

    字段说明：
    - enabled: 是否生成此类图并写入 Excel；
    - sheet_name: 图所在的 sheet 名；
    - charts_per_row: 每行安排多少张图，用于计算插入位置；
    - items: 需要绘制的指标列表配置。
    """

    enabled: bool = True
    sheet_name: str = "Metric_Charts"
    charts_per_row: int = 2
    items: Sequence[MetricEffectBarsConfig] = ()


@dataclass
class DidEffectVisualConfig:
    """
    报表中 DID 效应图的整体配置。

    字段说明：
    - enabled: 是否生成 DID 图并写入 Excel；
    - sheet_name: 图所在的 sheet 名；
    - charts_per_row: 每行安排多少张图；
    - items: 需要绘制的 (metric, variant_group) 组合。
    """

    enabled: bool = True
    sheet_name: str = "DID_Charts"
    charts_per_row: int = 1
    items: Sequence[DidEffectConfig] = ()


@dataclass
class ReportOutputConfig:
    """
    报表输出相关配置。

    字段说明：
    - excel_path: 最终输出的 xlsx 文件路径；
    - figure_temp_dir: 图像 PNG 的临时输出目录，后续会被插入 Excel。
    """

    excel_path: Path
    figure_temp_dir: Path


@dataclass
class AbTestReportVisualConfig:
    """
    AB 实验报表的可视化配置封装。

    字段说明：
    - metric_effect_bars: 单指标柱状图配置；
    - did_effect: DID 图配置。
    """

    metric_effect_bars: MetricEffectBarsVisualConfig
    did_effect: DidEffectVisualConfig


def _ensure_directory(path: Path) -> None:
    """
    确保目录存在，如不存在则递归创建。

    输入：
    - path: 目标目录路径。
    """
    path.mkdir(parents=True, exist_ok=True)


def _load_output_config(
    raw_cfg: Mapping[str, Any],
    output_path: Optional[str] = None,
) -> ReportOutputConfig:
    """
    从原始字典中解析报表输出相关配置；若传入 output_path 则优先使用，覆盖配置中的路径。

    预期结构示例：
    report:
      output:
        excel_path: "outputs/abtest_reports/report.xlsx"
        figure_temp_dir: "outputs/abtest_reports/figures"

    当 output_path 非空时：excel_path 使用该值，figure_temp_dir 使用其同目录下的 figures 子目录。
    """
    if output_path is not None and str(output_path).strip():
        excel_path = Path(output_path)
        figure_temp_dir = excel_path.parent / "figures"
        _ensure_directory(excel_path.parent)
        _ensure_directory(figure_temp_dir)
        return ReportOutputConfig(excel_path=excel_path, figure_temp_dir=figure_temp_dir)

    report_cfg = raw_cfg.get("report", {})
    output_cfg = report_cfg.get("output", {})
    excel_path_raw = output_cfg.get("excel_path")
    figure_dir_raw = output_cfg.get("figure_temp_dir")

    if not excel_path_raw:
        raise ValueError("abtest 报表配置缺少 report.output.excel_path；或传入 output_path 参数。")
    if not figure_dir_raw:
        raise ValueError("abtest 报表配置缺少 report.output.figure_temp_dir；或传入 output_path 参数。")

    excel_path = Path(excel_path_raw)
    figure_temp_dir = Path(figure_dir_raw)
    _ensure_directory(excel_path.parent)
    _ensure_directory(figure_temp_dir)

    return ReportOutputConfig(excel_path=excel_path, figure_temp_dir=figure_temp_dir)


def _load_metric_effect_bars_visual(raw_cfg: Mapping[str, Any]) -> MetricEffectBarsVisualConfig:
    """
    从原始字典中解析 metric effect bars 相关可视化配置。

    预期结构示例：
    report:
      visuals:
        metric_effect_bars:
          enabled: true
          sheet_name: "Metric_Charts"
          charts_per_row: 2
          metrics:
            - name: "pay_rate"
              phase: "after"
              stratify_filters: {}
    """
    report_cfg = raw_cfg.get("report", {})
    visuals_cfg = report_cfg.get("visuals", {})
    metric_cfg_raw = visuals_cfg.get("metric_effect_bars", {})

    enabled = bool(metric_cfg_raw.get("enabled", True))
    sheet_name = metric_cfg_raw.get("sheet_name", "Metric_Charts")
    charts_per_row = int(metric_cfg_raw.get("charts_per_row", 2)) or 1

    items_raw: Iterable[Mapping[str, Any]] = metric_cfg_raw.get("metrics", []) or []
    items: List[MetricEffectBarsConfig] = []
    for item in items_raw:
        name = item.get("name")
        if not name:
            continue
        config = MetricEffectBarsConfig(
            name=str(name),
            phase=item.get("phase"),
            stratify_filters=item.get("stratify_filters") or None,
        )
        items.append(config)

    return MetricEffectBarsVisualConfig(
    enabled=enabled,
        sheet_name=str(sheet_name),
        charts_per_row=charts_per_row,
        items=items,
    )


def _load_did_effect_visual(raw_cfg: Mapping[str, Any]) -> DidEffectVisualConfig:
    """
    从原始字典中解析 DID 图相关可视化配置。

    预期结构示例：
    report:
      visuals:
        did_effect:
          enabled: true
          sheet_name: "DID_Charts"
          charts_per_row: 1
          items:
            - metric: "pay_rate"
              variant_group: null
    """
    report_cfg = raw_cfg.get("report", {})
    visuals_cfg = report_cfg.get("visuals", {})
    did_cfg_raw = visuals_cfg.get("did_effect", {})

    enabled = bool(did_cfg_raw.get("enabled", True))
    sheet_name = did_cfg_raw.get("sheet_name", "DID_Charts")
    charts_per_row = int(did_cfg_raw.get("charts_per_row", 1)) or 1

    items_raw: Iterable[Mapping[str, Any]] = did_cfg_raw.get("items", []) or []
    items: List[DidEffectConfig] = []
    for item in items_raw:
        metric = item.get("metric")
        if not metric:
            continue
        config = DidEffectConfig(
            metric=str(metric),
            variant_group=item.get("variant_group"),
        )
        items.append(config)

    return DidEffectVisualConfig(
        enabled=enabled,
        sheet_name=str(sheet_name),
        charts_per_row=charts_per_row,
        items=items,
    )


def _load_visual_config(raw_cfg: Mapping[str, Any]) -> AbTestReportVisualConfig:
    """
    从原始报表配置字典中构造可视化配置对象。
    """
    metric_cfg = _load_metric_effect_bars_visual(raw_cfg)
    did_cfg = _load_did_effect_visual(raw_cfg)
    return AbTestReportVisualConfig(
        metric_effect_bars=metric_cfg,
        did_effect=did_cfg,
    )


def _build_overview_table(
    ab_result: pd.DataFrame,
    max_rows: Optional[int] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> "pd.io.formats.style.Styler":
    """
    基于 AB 实验结果构建总览表 Styler。

    说明：
    - 内部直接复用 core.visualizer.abtest_visualizer.style_ab_overview_table，
      这里仅加上一层包装，便于后续扩展排序规则等。
    """
    return style_ab_overview_table(
        result_df=ab_result,
        max_rows=max_rows,
        visual_config=visual_config,
    )


def _generate_metric_effect_bars_images(
    ab_result: pd.DataFrame,
    visual_cfg: MetricEffectBarsVisualConfig,
    figure_dir: Path,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[str, Path]]:
    """
    根据配置生成各指标的“基准 vs 实验”柱状图，并保存为 PNG。

    输入：
    - ab_result: run_ab_test 返回的结果 DataFrame；
    - visual_cfg: 指标柱状图的可视化配置；
    - figure_dir: 图像输出目录。

    输出：
    - 列表，每个元素为 (title, file_path)，供后续写入 Excel 使用。
    """
    if not visual_cfg.enabled:
        return []

    metric_dir = figure_dir / "metric_effect_bars"
    _ensure_directory(metric_dir)

    outputs: List[Tuple[str, Path]] = []

    savefig_cfg: Mapping[str, Any] = {}
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or {}
        savefig_cfg = matplotlib_cfg.get("savefig") or {}

    dpi = int(savefig_cfg.get("dpi", 150))
    bbox_inches = savefig_cfg.get("bbox_inches", "tight")
    for cfg in visual_cfg.items:
        fig = plot_metric_effect_bars(
            result_df=ab_result,
            metric=cfg.name,
            phase=cfg.phase,
            title=None,
            stratify_filters=cfg.stratify_filters,
            visual_config=visual_config,
        )
        title = cfg.name
        if cfg.phase is not None:
            title = f"{cfg.name} | phase={cfg.phase}"

        file_name = f"{cfg.name}"
        if cfg.phase is not None:
            file_name += f"__{cfg.phase}"
        file_path = metric_dir / f"{file_name}.png"

        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        outputs.append((title, file_path))

    return outputs


def _generate_did_effect_images(
    did_result: Optional[pd.DataFrame],
    visual_cfg: DidEffectVisualConfig,
    figure_dir: Path,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[str, Path]]:
    """
    根据配置生成各指标的 DID 效应图，并保存为 PNG。

    输入：
    - did_result: compute_did 返回的 DataFrame；若为 None 或为空，将直接返回空列表；
    - visual_cfg: DID 图的可视化配置；
    - figure_dir: 图像输出目录。

    输出：
    - 列表，每个元素为 (title, file_path)，供后续写入 Excel 使用。
    """
    if not visual_cfg.enabled:
        return []
    if did_result is None or did_result.empty:
        return []

    did_dir = figure_dir / "did_effect"
    _ensure_directory(did_dir)

    outputs: List[Tuple[str, Path]] = []

    savefig_cfg: Mapping[str, Any] = {}
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or {}
        savefig_cfg = matplotlib_cfg.get("savefig") or {}

    dpi = int(savefig_cfg.get("dpi", 150))
    bbox_inches = savefig_cfg.get("bbox_inches", "tight")
    for cfg in visual_cfg.items:
        fig = plot_did_effect(
            did_df=did_result,
            metric=cfg.metric,
            variant_group=cfg.variant_group,
            title=None,
            stratify_filters=None,
            visual_config=visual_config,
        )
        if cfg.variant_group is None:
            title = cfg.metric
            file_name = cfg.metric
        else:
            title = f"{cfg.metric} | {cfg.variant_group}"
            file_name = f"{cfg.metric}__{cfg.variant_group}"

        file_path = did_dir / f"{file_name}.png"
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        outputs.append((title, file_path))

    return outputs


def _write_excel_with_images(
    output_cfg: ReportOutputConfig,
    overview_styler: "pd.io.formats.style.Styler",
    did_result: Optional[pd.DataFrame],
    visual_cfg: AbTestReportVisualConfig,
    metric_figs: Sequence[Tuple[str, Path]],
    did_figs: Sequence[Tuple[str, Path]],
) -> Path:
    """
    将总览表、DID 结果与图像嵌入到同一个 Excel 文件中。

    说明：
    - 使用 xlsxwriter 作为 Excel 引擎，以支持 insert_image；
    - 总览表写入 "Overview" sheet；
    - 如有 DID 结果，则写入 "DID" sheet；
    - 图像根据配置分别写入 metric_effect_bars.sheet_name 与 did_effect.sheet_name。
    """
    excel_path = output_cfg.excel_path

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # 写入总览表
        overview_sheet_name = "Overview"
        overview_styler.to_excel(writer, sheet_name=overview_sheet_name)

        workbook = writer.book

        # 写入 DID 结果表
        if did_result is not None and not did_result.empty:
            did_sheet_name = "DID"
            did_result.to_excel(writer, sheet_name=did_sheet_name, index=False)

        # 写入 metric effect bars 图表
        if visual_cfg.metric_effect_bars.enabled and metric_figs:
            metric_sheet_name = visual_cfg.metric_effect_bars.sheet_name
            metric_sheet = workbook.add_worksheet(metric_sheet_name)
            charts_per_row = max(1, visual_cfg.metric_effect_bars.charts_per_row)

            for idx, (title, file_path) in enumerate(metric_figs):
                row_block = 20
                col_block = 8
                row = (idx // charts_per_row) * row_block
                col = (idx % charts_per_row) * col_block

                # 标题文本
                metric_sheet.write(row, col, title)
                # 图片放在标题下一行
                metric_sheet.insert_image(
                    row + 1,
                    col,
                    str(file_path),
                    {
                        "x_scale": 0.9,
                        "y_scale": 0.9,
                    },
                )

        # 写入 DID 效应图表
        if visual_cfg.did_effect.enabled and did_figs:
            did_chart_sheet_name = visual_cfg.did_effect.sheet_name
            did_chart_sheet = workbook.add_worksheet(did_chart_sheet_name)
            charts_per_row = max(1, visual_cfg.did_effect.charts_per_row)

            for idx, (title, file_path) in enumerate(did_figs):
                row_block = 25
                col_block = 8
                row = (idx // charts_per_row) * row_block
                col = (idx % charts_per_row) * col_block

                did_chart_sheet.write(row, col, title)
                did_chart_sheet.insert_image(
                    row + 1,
                    col,
                    str(file_path),
                    {
                        "x_scale": 0.9,
                        "y_scale": 0.9,
                    },
                )

    return excel_path


def generate_abtest_report(
    ab_result: pd.DataFrame,
    did_result: Optional[pd.DataFrame],
    report_config: Mapping[str, Any],
    output_path: Optional[str] = None,
    visual_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    生成 AB 实验报表（含总览表 + DID 表 + 嵌入图片的 Excel）。

    输入：
    - ab_result: run_ab_test 返回的结果 DataFrame；
    - did_result: 可选，compute_did 返回的 DataFrame；
    - report_config: 从 YAML 解析得到的报表配置字典；
    - output_path: 可选，指定报表 Excel 的保存路径；若传入则优先使用，临时图目录为其同目录下的 figures 子目录。

    输出：
    - 字典，包含：
      - "excel_path": 生成的 Excel 文件路径（字符串）；
      - "metric_figures": [(title, file_path), ...] 形式的列表（便于调试或日志）；
      - "did_figures": 同上，用于记录 DID 图列表。
    """
    output_cfg = _load_output_config(report_config, output_path=output_path)
    visual_cfg = _load_visual_config(report_config)

    overview_styler = _build_overview_table(
        ab_result=ab_result,
        max_rows=None,
        visual_config=visual_config,
    )

    metric_figs = _generate_metric_effect_bars_images(
        ab_result=ab_result,
        visual_cfg=visual_cfg.metric_effect_bars,
        figure_dir=output_cfg.figure_temp_dir,
        visual_config=visual_config,
    )
    did_figs = _generate_did_effect_images(
        did_result=did_result,
        visual_cfg=visual_cfg.did_effect,
        figure_dir=output_cfg.figure_temp_dir,
        visual_config=visual_config,
    )

    excel_path = _write_excel_with_images(
        output_cfg=output_cfg,
        overview_styler=overview_styler,
        did_result=did_result,
        visual_cfg=visual_cfg,
        metric_figs=metric_figs,
        did_figs=did_figs,
    )

    return {
        "excel_path": str(excel_path),
        "metric_figures": [(title, str(path)) for title, path in metric_figs],
        "did_figures": [(title, str(path)) for title, path in did_figs],
    }

