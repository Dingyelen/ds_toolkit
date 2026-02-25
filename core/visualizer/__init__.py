"""
core.visualizer

基于 Plotly/Matplotlib 的标准化绘图模板。
约定：
- 在 mac 环境下正常显示中文，负号显示为 "-";
- 仅依赖 core 与第三方库，不引用 modules 下的业务逻辑。
"""

from .abtest_visualizer import (
    plot_did_effect,
    plot_forest_for_metric,
    plot_metric_distribution,
    plot_metric_effect_bars,
    plot_segment_heatmap,
    style_ab_overview_table,
)

__all__ = [
    "style_ab_overview_table",
    "plot_metric_effect_bars",
    "plot_forest_for_metric",
    "plot_segment_heatmap",
    "plot_did_effect",
    "plot_metric_distribution",
]
