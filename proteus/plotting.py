"""Figures in the style of the paper (matplotlib, optional dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .roadmap import RoadMap


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_roadmap(ax, road_map: RoadMap, color="0.75", lw=1.0, nodes=False, **kw):
    from matplotlib.collections import LineCollection

    segs = road_map.segments()
    if len(segs):
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, **kw))
    if nodes and road_map.graph.number_of_nodes():
        p = np.array(list(road_map.node_positions().values()))
        ax.scatter(p[:, 0], p[:, 1], s=4, color=color, zorder=3)
    ax.autoscale()
    ax.set_aspect("equal")
    return ax


def plot_result(result, pre_map: RoadMap | None = None, truth: RoadMap | None = None,
                traces: pd.DataFrame | None = None, path=None, title: str | None = None):
    """Generated map over the pre-disaster map (cf. Figs 7-8)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 8))
    if pre_map is not None:
        plot_roadmap(ax, pre_map, color="0.85", lw=3, label="pre-disaster map")
    if traces is not None and len(traces):
        ax.scatter(traces["x"], traces["y"], s=0.2, color="tab:blue", alpha=0.15)
    plot_roadmap(ax, result.final_map, color="tab:red", lw=1.5)
    inter = result.intersections.intersections
    if inter:
        p = np.array(inter)
        ax.scatter(p[:, 0], p[:, 1], s=12, color="tab:orange", zorder=4, label="intersections")
    ax.set_title(title or "Proteus reconstruction")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def anova_diagnostics(model, path=None, title: str | None = None):
    """Residuals vs fitted, Q-Q, scale-location and Cook's distance (Figs 9-10)."""
    plt = _plt()
    from scipy import stats

    infl = model.get_influence()
    fitted = model.fittedvalues
    resid = model.resid
    std_resid = infl.resid_studentized_internal
    cooks = infl.cooks_distance[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(fitted, resid, s=4, facecolors="none", edgecolors="k")
    axes[0, 0].axhline(0, color="0.6", ls=":")
    axes[0, 0].set(title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")
    (osm, osr), (slope, inter, _) = stats.probplot(std_resid, dist="norm")
    axes[0, 1].scatter(osm, osr, s=4, facecolors="none", edgecolors="k")
    axes[0, 1].plot(osm, slope * np.asarray(osm) + inter, color="0.5", ls=":")
    axes[0, 1].set(title="Q-Q Residuals", xlabel="Theoretical Quantiles", ylabel="Standardized residuals")
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), s=4, facecolors="none", edgecolors="k")
    axes[1, 0].set(title="Scale-Location", xlabel="Fitted values", ylabel="sqrt(|Standardized residuals|)")
    axes[1, 1].vlines(np.arange(len(cooks)), 0, cooks, color="k", lw=0.5)
    axes[1, 1].set(title="Cook's distance", xlabel="Obs. number", ylabel="Cook's distance")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def jitter_boxplots(df: pd.DataFrame, models=(2, 3, 5), path=None, factor: str = "Jitter"):
    """Accuracy by jitter level for each model (Fig 11)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    levels = sorted(df[factor].unique())
    for ax, m in zip(axes, models):
        col = f"{m} edge accuracy"
        ax.boxplot([df.loc[df[factor] == lv, col] for lv in levels])
        ax.set_xticks(range(1, len(levels) + 1), [str(lv) for lv in levels])
        ax.set(title=f"{m}-Edge model", xlabel=factor)
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def factor_comparison(df: pd.DataFrame, other: str, model: int = 5, path=None):
    """Jitter vs another factor for one model (Figs 12-15)."""
    plt = _plt()
    col = f"{model} edge accuracy"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, f in zip(axes, ("Jitter", other)):
        levels = sorted(df[f].unique())
        ax.boxplot([df.loc[df[f] == lv, col] for lv in levels])
        ax.set_xticks(range(1, len(levels) + 1), [str(lv) for lv in levels])
        ax.set(xlabel=f)
    axes[0].set_ylabel(f"{model} Edge Accuracy")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
