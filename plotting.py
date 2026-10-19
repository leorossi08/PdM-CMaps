"""Publication-friendly static figures for the script-based analyses."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(figure, path: Path) -> Path:
    """Write a figure after ensuring its destination directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_baseline(summary: pd.DataFrame, path: Path) -> Path:
    """Plot pooled grouped-CV R² with fold standard deviation bars."""
    ordered = summary.sort_values("R2_pooled_OOF", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ordered["Subpart"], ordered["R2_pooled_OOF"], yerr=ordered["R2_fold_std"], capsize=4, color="#2a7a8c")
    for index, value in enumerate(ordered["R2_pooled_OOF"]):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled out-of-fold R² (error bars: fold SD)")
    ax.set_title("C-MAPSS capped-RUL grouped-CV baseline")
    return save_figure(fig, path)


def plot_regime_scores(scores: pd.DataFrame, path: Path) -> Path:
    """Plot healthy-row virtual-sensor R² for each learned regime."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(scores["Regime_ID"].astype(str), scores["R2"], color="#c9742a")
    ax.set_xlabel("Learned flight regime")
    ax.set_ylabel("Held-out healthy-row R²")
    ax.set_title("FD004 healthy T50 virtual sensor by regime")
    return save_figure(fig, path)


def plot_learning_curves(summary: pd.DataFrame, path: Path) -> Path:
    """Plot repeated grouped-holdout R² as training-engine count grows."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subset, group in summary.groupby("Subpart"):
        ax.errorbar(group["Train_engine_fraction"], group["Mean_R2"], yerr=group["Std_R2"], marker="o", capsize=3, label=subset)
    ax.set_xlabel("Fraction of training engines")
    ax.set_ylabel("Held-out R²")
    ax.set_title("Learning curves by subset")
    ax.legend()
    return save_figure(fig, path)


def plot_sensor_variance(results: pd.DataFrame, path: Path) -> Path:
    """Plot between- and within-regime sensor variance shares."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(results))
    ax.bar(positions, results["Between_regime_pct"], label="Between regime", color="#2a7a8c")
    ax.bar(positions, results["Within_regime_pct"], bottom=results["Between_regime_pct"], label="Within regime", color="#c9742a")
    ax.set_xticks(positions, results["Sensor"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of total variance (%)")
    ax.set_title("FD004 sensor variance by operating regime")
    ax.legend(frameon=False)
    return save_figure(fig, path)
