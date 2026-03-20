"""
单样本连续指标的分布诊断图（Matplotlib 静态、Notebook 友好）。

返回 Figure 对象，不依赖业务模块。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from core.visualizer.abtest_visualizer import _setup_matplotlib_style


def plot_distribution_diagnostic(
    values: Sequence[float],
    title: str = "分布诊断",
    *,
    bins: int = 40,
    show_kde: bool = True,
    show_qq: bool = True,
    show_log_panel: bool = False,
    figsize: Tuple[float, float] = (10.0, 4.0),
    figsize_with_log: Tuple[float, float] = (10.0, 7.5),
    visual_config: Optional[Mapping[str, Any]] = None,
) -> plt.Figure:
    """
    绘制直方图（可选 KDE）与正态 QQ 图；可选第二行「log10 尺度」直方图（仅全为正时建议）。

    输入：
    - values: 一维数值序列（inf/nan 会剔除）；
    - title: 总标题；
    - bins: 直方图分箱数；
    - show_kde: 是否叠加高斯 KDE；
    - show_qq: 是否绘制 QQ 图；
    - show_log_panel: 是否在第二行绘制 log10(x) 的直方图（自动过滤 x<=0）；
    - figsize / figsize_with_log: 无 log / 有 log 时的画布尺寸；
    - visual_config: 与 abtest 相同，读取 visualizer.yaml 的 matplotlib 段。

    输出：
    - matplotlib.figure.Figure。
    """
    matplotlib_cfg: Optional[Mapping[str, Any]] = None
    if visual_config is not None:
        matplotlib_cfg = visual_config.get("matplotlib") or None
    _setup_matplotlib_style(matplotlib_cfg)

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("values 在剔除 inf/nan 后为空，无法绘图。")

    if show_log_panel:
        fig, axes = plt.subplots(2, 2, figsize=figsize_with_log, constrained_layout=True)
        ax_hist = axes[0, 0]
        ax_qq = axes[0, 1]
        ax_log = axes[1, 0]
        axes[1, 1].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        ax_hist = axes[0]
        ax_qq = axes[1]
        ax_log = None

    # --- 直方图 + KDE ---
    ax_hist.hist(
        arr,
        bins=bins,
        density=True,
        color="#9ecae1",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    if show_kde and arr.size >= 2:
        try:
            kde = stats.gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 200)
            ax_hist.plot(xs, kde(xs), color="#3182bd", linewidth=2.0, label="KDE")
            ax_hist.legend(loc="upper right", framealpha=0.9)
        except Exception:
            pass
    ax_hist.set_title("直方图 + KDE（原始尺度）")
    ax_hist.set_xlabel("值")
    ax_hist.set_ylabel("密度")

    # --- QQ 图 ---
    if show_qq:
        stats.probplot(arr, dist="norm", plot=ax_qq)
        ax_qq.set_title("正态 QQ 图")
        ax_qq.get_lines()[0].set_markerfacecolor("#3182bd")
        ax_qq.get_lines()[0].set_markeredgecolor("#3182bd")
    else:
        ax_qq.axis("off")
        ax_qq.text(0.5, 0.5, "QQ 图已关闭", ha="center", va="center")

    # --- log 面板 ---
    if show_log_panel and ax_log is not None:
        pos = arr[arr > 0]
        if pos.size >= 3:
            log_x = np.log10(pos)
            ax_log.hist(
                log_x,
                bins=bins,
                density=True,
                color="#fc8d59",
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            if show_kde:
                try:
                    lkde = stats.gaussian_kde(log_x)
                    lxs = np.linspace(log_x.min(), log_x.max(), 200)
                    ax_log.plot(lxs, lkde(lxs), color="#d73027", linewidth=2.0, label="KDE")
                    ax_log.legend(loc="upper right", framealpha=0.9)
                except Exception:
                    pass
            ax_log.set_title("直方图 + KDE（log10，仅 x>0）")
            ax_log.set_xlabel("log10(x)")
            ax_log.set_ylabel("密度")
        else:
            ax_log.text(0.5, 0.5, "正值过少，未绘制 log10 面板", ha="center", va="center")
            ax_log.axis("off")

    fig.suptitle(f"{title}（n={arr.size}）", fontsize=13)
    return fig
