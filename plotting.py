"""Publication-friendly static figures for the script-based analyses."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score


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


def plot_failure_detection(scored: pd.DataFrame, engines: pd.DataFrame, path: Path, healthy_cycle_limit: int = 50) -> Path:
    """Plot a representative engine's sensor residual alarm and first threshold crossing."""
    detected = engines.loc[engines["detected"]]
    if detected.empty:
        engine_id = int(engines.iloc[0]["unit_id"])
    else:
        engine_id = int(detected.sort_values("lead_time_cycles", ascending=False).iloc[0]["unit_id"])
    trace = scored.loc[scored["unit_id"] == engine_id].sort_values("cycle")
    summary = engines.loc[engines["unit_id"] == engine_id].iloc[0]
    figure, (sensor_ax, alarm_ax) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for axis in (sensor_ax, alarm_ax):
        axis.axvspan(trace["cycle"].min(), healthy_cycle_limit, color="#55b982", alpha=0.14, label="Healthy calibration")
        axis.axvspan(healthy_cycle_limit, trace["cycle"].max(), color="#e46f70", alpha=0.10, label="Post-calibration")
    sensor_ax.plot(trace["cycle"], trace["T50"], color="#26364a", linewidth=1.6, label="Measured T50")
    sensor_ax.plot(trace["cycle"], trace["prediction"], color="#326bff", linewidth=1.4, linestyle="--", label="Virtual sensor")
    sensor_ax.set_ylabel("T50")
    sensor_ax.set_title(f"Held-out engine {engine_id}: measured and virtual T50")
    sensor_ax.legend(loc="upper left", ncol=2, frameon=True)

    alarm_ax.plot(trace["cycle"], trace["ewma_abs_z"], color="#7444ed", linewidth=1.6, label="EWMA absolute Z-score")
    alarm_ax.axhline(3.0, color="#d62728", linestyle=":", linewidth=1.8, label="Alarm threshold (3σ)")
    alarm_cycle = summary["first_alarm_cycle"]
    if pd.notna(alarm_cycle):
        alarm_row = trace.loc[trace["cycle"] == alarm_cycle]
        if not alarm_row.empty:
            alarm_ax.scatter([alarm_cycle], alarm_row["ewma_abs_z"], marker="*", s=150, color="#bd1e1e", zorder=4, label="First alarm")
    alarm_ax.set_xlabel("Operational cycle")
    alarm_ax.set_ylabel("EWMA |Z|")
    alarm_ax.set_title(
        f"First alarm: cycle {int(alarm_cycle)}; lead time {int(summary['lead_time_cycles'])} cycles"
        if pd.notna(alarm_cycle) else "No alarm crossed the threshold"
    )
    alarm_ax.legend(loc="upper left", ncol=2, frameon=True)
    return save_figure(figure, path)


def plot_actual_vs_predicted(y_true, y_pred, path: Path, title: str, value_label: str) -> Path:
    """Plot held-out actual-versus-predicted values with identity line and R²."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    score = float(r2_score(truth, prediction))
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    padding = (upper - lower) * 0.04 or 1.0
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(truth, prediction, s=10, alpha=0.25, color="#326bff", edgecolors="none")
    axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.5, label="Perfect prediction")
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel(f"Actual {value_label}")
    axis.set_ylabel(f"Predicted {value_label}")
    axis.set_title(f"{title}\nHeld-out R² = {score:.3f}")
    axis.legend(frameon=False)
    return save_figure(figure, path)


def plot_model_r2(summary: pd.DataFrame, path: Path, score_column: str = "R2") -> Path:
    """Rank model holdout R² values and mark the top-scoring model."""
    ordered = summary.sort_values(score_column, ascending=True)
    colors = ["#2a7a8c" if name == ordered.iloc[-1]["Model"] else "#91b6bb" for name in ordered["Model"]]
    figure, axis = plt.subplots(figsize=(8, max(4.0, 0.55 * len(ordered))))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered[score_column], color=colors)
    axis.axvline(0.0, color="#26364a", linewidth=0.8)
    axis.set_yticks(positions, ordered["Model"])
    axis.set_xlabel("Held-out R²")
    axis.set_title("RUL model comparison (best model highlighted)")
    for position, value in zip(positions, ordered[score_column]):
        axis.text(value + (0.01 if value >= 0 else -0.01), position, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right")
    return save_figure(figure, path)


def plot_sensor_r2(results: pd.DataFrame, path: Path) -> Path:
    """Rank held-out virtual-sensor R² scores across physical channels."""
    ordered = results.sort_values("R2", ascending=True)
    defined_scores = results.dropna(subset=["R2"])
    best_sensor = defined_scores.loc[defined_scores["R2"].idxmax(), "Sensor"] if not defined_scores.empty else None
    colors = ["#c9742a" if sensor == best_sensor else "#ddb18a" for sensor in ordered["Sensor"]]
    figure, axis = plt.subplots(figsize=(8, max(6.0, 0.36 * len(ordered))))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered["R2"], color=colors)
    axis.axvline(0.0, color="#26364a", linewidth=0.8)
    axis.set_yticks(positions, ordered["Sensor"])
    axis.set_xlabel("Held-out R²")
    axis.set_title("Virtual-sensor prediction by physical channel")
    return save_figure(figure, path)


def plot_sensor_parity_grid(predictions: pd.DataFrame, path: Path) -> Path:
    """Plot actual-vs-predicted held-out values, colored by operating regime."""
    sensors = list(dict.fromkeys(predictions["Sensor"]))
    columns = 4
    rows = int(np.ceil(len(sensors) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), squeeze=False)
    sensor_scores = {
        sensor: float(r2_score(rows["Actual"], rows["Predicted"]))
        for sensor, rows in predictions.groupby("Sensor", sort=False)
        if rows["Actual"].var() > 1e-12
    }
    best_sensor = max(sensor_scores, key=sensor_scores.get) if sensor_scores else None
    for axis, sensor in zip(axes.flat, sensors):
        rows_for_sensor = predictions.loc[predictions["Sensor"] == sensor]
        truth = rows_for_sensor["Actual"].to_numpy(dtype=float)
        estimate = rows_for_sensor["Predicted"].to_numpy(dtype=float)
        score = float(r2_score(truth, estimate)) if np.var(truth) > 1e-12 else np.nan
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        padding = (upper - lower) * 0.04 or 1.0
        if "regime_id" in rows_for_sensor:
            for regime_id, regime_rows in rows_for_sensor.groupby("regime_id", sort=True):
                axis.scatter(
                    regime_rows["Actual"], regime_rows["Predicted"], s=5, alpha=0.35,
                    color=plt.get_cmap("tab10")(int(regime_id) % 10), edgecolors="none",
                    rasterized=True,
                )
        else:
            axis.scatter(truth, estimate, s=5, alpha=0.2, color="#326bff", edgecolors="none", rasterized=True)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        score_label = f"{score:.3f}" if np.isfinite(score) else "n/a"
        axis.set_title(f"{sensor}{' (best)' if sensor == best_sensor else ''}: R² {score_label}")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
    for axis in axes.flat[len(sensors):]:
        axis.set_visible(False)
    regime_ids = sorted(predictions["regime_id"].dropna().unique()) if "regime_id" in predictions else []
    if regime_ids:
        handles = [Patch(facecolor=plt.get_cmap("tab10")(int(regime_id) % 10), label=f"Regime {regime_id}") for regime_id in regime_ids]
        figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
    figure.suptitle("Held-out actual vs predicted sensor values by operating regime", y=1.0)
    figure.tight_layout(rect=(0, 0.04 if regime_ids else 0, 1, 0.98))
    return save_figure(figure, path)


def plot_sensor_parity_by_channel(predictions: pd.DataFrame, directory: Path) -> list[Path]:
    """Save per-sensor parity plots and held-out engine value distributions."""
    paths = []
    for sensor, rows in predictions.groupby("Sensor", sort=False):
        figure, (parity_axis, distribution_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
        truth = rows["Actual"].to_numpy(dtype=float)
        estimate = rows["Predicted"].to_numpy(dtype=float)
        score = float(r2_score(truth, estimate)) if np.var(truth) > 1e-12 else np.nan
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        padding = (upper - lower) * 0.04 or 1.0
        if "regime_id" in rows:
            for regime_id, regime_rows in rows.groupby("regime_id", sort=True):
                parity_axis.scatter(
                    regime_rows["Actual"], regime_rows["Predicted"], s=8, alpha=0.4,
                    color=plt.get_cmap("tab10")(int(regime_id) % 10), edgecolors="none",
                    rasterized=True, label=f"Regime {regime_id}",
                )
            parity_axis.legend(title="Operating regime", frameon=False, fontsize=8)
        else:
            parity_axis.scatter(truth, estimate, s=8, alpha=0.3, color="#326bff", edgecolors="none", rasterized=True)
        parity_axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.2)
        parity_axis.set_xlim(lower - padding, upper + padding)
        parity_axis.set_ylim(lower - padding, upper + padding)
        parity_axis.set_xlabel(f"Actual {sensor}")
        parity_axis.set_ylabel(f"Predicted {sensor}")
        parity_axis.set_title(f"Actual vs predicted (R² {score:.3f})" if np.isfinite(score) else "Actual vs predicted (R² n/a)")

        engine_ids = sorted(rows["unit_id"].unique())
        centers = np.arange(len(engine_ids)) * 1.0
        actual_values = [rows.loc[rows["unit_id"] == engine_id, "Actual"].to_numpy() for engine_id in engine_ids]
        predicted_values = [rows.loc[rows["unit_id"] == engine_id, "Predicted"].to_numpy() for engine_id in engine_ids]
        actual_boxes = distribution_axis.boxplot(
            actual_values, positions=centers - 0.18, widths=0.32,
            patch_artist=True, showfliers=False, manage_ticks=False,
        )
        predicted_boxes = distribution_axis.boxplot(
            predicted_values, positions=centers + 0.18, widths=0.32,
            patch_artist=True, showfliers=False, manage_ticks=False,
        )
        for box in actual_boxes["boxes"]:
            box.set(facecolor="#78a9eb", alpha=0.8)
        for box in predicted_boxes["boxes"]:
            box.set(facecolor="#e4a05a", alpha=0.8)
        distribution_axis.set_xticks(centers, [str(engine_id) for engine_id in engine_ids], rotation=90)
        distribution_axis.set_xlabel("Held-out engine ID")
        distribution_axis.set_ylabel(sensor)
        distribution_axis.set_title("Cycle-level value spread by engine")
        distribution_axis.legend(
            handles=[Patch(facecolor="#78a9eb", label="Actual"), Patch(facecolor="#e4a05a", label="Predicted")],
            frameon=False,
        )
        figure.suptitle(f"{sensor}: held-out sensor reconstruction")
        figure.tight_layout()
        paths.append(save_figure(figure, Path(directory) / f"{sensor.lower()}_actual_vs_predicted.png"))
    return paths


def plot_sensor_cycle_windows_by_regime(
    predictions: pd.DataFrame,
    directory: Path,
    window_size: int = 50,
) -> tuple[list[Path], pd.DataFrame]:
    """Plot per-regime sensor timelines in short cycle windows with engine spread bands."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    directory = Path(directory)
    sensors = list(dict.fromkeys(predictions["Sensor"]))
    windowed = predictions.copy()
    windowed["Cycle_start"] = ((windowed["cycle"] - 1) // window_size) * window_size + 1
    windowed["Cycle_end"] = windowed["Cycle_start"] + window_size - 1
    paths = []
    metric_rows = []
    columns = 4
    rows = int(np.ceil(len(sensors) / columns))

    for (regime_id, cycle_start, cycle_end), window_rows in windowed.groupby(
        ["regime_id", "Cycle_start", "Cycle_end"], sort=True
    ):
        figure, axes = plt.subplots(rows, columns, figsize=(16, 3.0 * rows), squeeze=False)
        for axis, sensor in zip(axes.flat, sensors):
            sensor_rows = window_rows.loc[window_rows["Sensor"] == sensor]
            if sensor_rows.empty:
                axis.set_visible(False)
                continue
            actual = sensor_rows["Actual"].to_numpy(dtype=float)
            predicted = sensor_rows["Predicted"].to_numpy(dtype=float)
            score = float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 1e-12 else np.nan
            errors = actual - predicted
            metrics = {
                "R2": score,
                "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "MAE": float(np.mean(np.abs(errors))),
                "target_variance": float(np.var(actual)),
            }
            metric_rows.append({
                "Regime_ID": int(regime_id),
                "Sensor": sensor,
                "Cycle_start": int(cycle_start),
                "Cycle_end": int(cycle_end),
                "Rows": len(sensor_rows),
                "Engines": int(sensor_rows["unit_id"].nunique()),
                **metrics,
            })

            cycle_summary = sensor_rows.groupby("cycle").agg(
                Actual_mean=("Actual", "mean"),
                Actual_std=("Actual", "std"),
                Predicted_mean=("Predicted", "mean"),
                Predicted_std=("Predicted", "std"),
            ).fillna(0.0)
            cycles = cycle_summary.index.to_numpy(dtype=int)
            actual_mean = cycle_summary["Actual_mean"].to_numpy(dtype=float)
            actual_std = cycle_summary["Actual_std"].to_numpy(dtype=float)
            predicted_mean = cycle_summary["Predicted_mean"].to_numpy(dtype=float)
            predicted_std = cycle_summary["Predicted_std"].to_numpy(dtype=float)
            axis.plot(cycles, actual_mean, color="#26364a", linewidth=1.4, label="Actual")
            axis.fill_between(cycles, actual_mean - actual_std, actual_mean + actual_std, color="#26364a", alpha=0.12)
            axis.plot(cycles, predicted_mean, color="#326bff", linewidth=1.4, linestyle="--", label="Predicted")
            axis.fill_between(cycles, predicted_mean - predicted_std, predicted_mean + predicted_std, color="#326bff", alpha=0.14)
            score_label = f"R² {score:.3f}" if np.isfinite(score) else "R² n/a"
            axis.set_title(f"{sensor}: {score_label}")
            axis.set_xlim(cycle_start, cycle_end)
            axis.set_xlabel("Cycle")
            axis.set_ylabel("Sensor value")
            axis.grid(alpha=0.2)

        for axis in axes.flat[len(sensors):]:
            axis.set_visible(False)
        figure.suptitle(f"Regime {int(regime_id)} sensor actual vs predicted | cycles {int(cycle_start)}-{int(cycle_end)}")
        figure.legend(
            handles=[
                Line2D([0], [0], color="#26364a", label="Actual mean"),
                Line2D([0], [0], color="#326bff", linestyle="--", label="Predicted mean"),
            ],
            loc="lower center", ncol=2, frameon=False,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.97))
        paths.append(save_figure(
            figure,
            directory / f"regime_{int(regime_id)}_cycles_{int(cycle_start):03d}-{int(cycle_end):03d}.png",
        ))

    return paths, pd.DataFrame(metric_rows)


def plot_subset_rul_parity(datasets: dict[str, pd.DataFrame], predictions: dict[str, np.ndarray], path: Path) -> Path:
    """Plot pooled grouped-CV RUL actual-vs-predicted scores by subset."""
    subsets = list(datasets)
    columns = 2
    rows = int(np.ceil(len(subsets) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 5.5 * rows), squeeze=False)
    for axis, subset in zip(axes.flat, subsets):
        truth = datasets[subset]["RUL"].to_numpy(dtype=float)
        estimate = np.asarray(predictions[subset], dtype=float)
        score = float(r2_score(truth, estimate))
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        padding = (upper - lower) * 0.04 or 1.0
        axis.scatter(truth, estimate, s=5, alpha=0.15, color="#326bff", edgecolors="none", rasterized=True)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.2)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Actual RUL (cycles)")
        axis.set_ylabel("Predicted RUL (cycles)")
        axis.set_title(f"{subset}: grouped OOF R² {score:.3f}")
    figure.suptitle("C-MAPSS RUL actual vs predicted by subset", y=1.0)
    figure.tight_layout()
    return save_figure(figure, path)
