"""Run the C-MAPSS virtual-sensor, SPC, and RUL teaching workflow."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import CMapssLoader
from evaluation import RULBenchmark
from plotting import (
    plot_actual_vs_predicted,
    plot_failure_detection,
    plot_model_r2,
    plot_regime_scores,
    plot_sensor_parity_by_channel,
    plot_sensor_cycle_windows_by_regime,
    plot_sensor_parity_grid,
    plot_sensor_r2,
)
from regimes import RegimeDecoupler
from schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RESULTS_DIR, SENSOR_COLUMNS, SETTING_COLUMNS
from splitting import partition_healthy_and_lifecycle, split_engine_units
from targets import compute_piecewise_rul, evaluate_rul_predictions, regression_metrics
from virtual_sensor import VirtualSensor


def _sensor_channel_analysis(
    train_healthy: pd.DataFrame,
    test_lifecycle: pd.DataFrame,
    seed: int,
    table_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit healthy-window virtual sensors and score every physical channel on held-out engines."""
    metrics = []
    predictions = []
    for target in SENSOR_COLUMNS:
        print(f"[maintenance] Sensor analysis: {target}", flush=True)
        features = [column for column in SENSOR_COLUMNS if column != target]
        features.extend([*SETTING_COLUMNS, "cycle"])
        model = VirtualSensor(
            target_sensor=target,
            feature_cols=features,
            healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT,
            random_state=seed,
        ).fit(train_healthy)
        scored = model.predict(test_lifecycle)
        truth = scored[target].to_numpy(dtype=float)
        estimate = scored["prediction"].to_numpy(dtype=float)
        target_metrics = regression_metrics(truth, estimate)
        if target_metrics["target_variance"] <= 1e-12:
            target_metrics["R2"] = np.nan
        metrics.append({"Sensor": target, "Test_rows": len(scored), **target_metrics})
        predictions.append(pd.DataFrame({
            "unit_id": scored["unit_id"].to_numpy(),
            "cycle": scored["cycle"].to_numpy(),
            "regime_id": scored["regime_id"].to_numpy(),
            "Sensor": target,
            "Actual": truth,
            "Predicted": estimate,
        }))

    metric_frame = pd.DataFrame(metrics).sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame.to_csv(table_dir / "sensor_channel_metrics.csv", index=False)
    prediction_frame.to_csv(table_dir / "sensor_channel_predictions.csv", index=False)
    plot_sensor_r2(metric_frame, figure_dir / "sensor_r2_by_channel.png")
    plot_sensor_parity_grid(prediction_frame, figure_dir / "sensor_actual_vs_predicted_all_channels.png")
    plot_sensor_parity_by_channel(prediction_frame, figure_dir / "sensor_channels")
    _, cycle_metrics = plot_sensor_cycle_windows_by_regime(
        prediction_frame,
        figure_dir / "sensor_cycle_windows",
        window_size=50,
    )
    cycle_metrics.to_csv(table_dir / "sensor_cycle_window_metrics.csv", index=False)
    return metric_frame, prediction_frame


def run(subset: str, output_dir: Path, seed: int, include_lstm: bool = False) -> dict[str, pd.DataFrame]:
    """Execute the guide's core analyses and write their tables/figures."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"[maintenance] Loading {subset} data and building engine holdout", flush=True)
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    sensors = loader.informative_sensors(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, DEFAULT_HEALTHY_CYCLE_LIMIT)

    n_regimes = 6 if subset in {"FD002", "FD004"} else 1
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(parts["full_lifecycle_train"])
    train_full = regimes.transform(parts["full_lifecycle_train"])
    test_full = regimes.transform(parts["full_lifecycle_test"])
    healthy_train = regimes.transform(parts["healthy_train"])
    healthy_test = regimes.transform(parts["healthy_test"])

    variance = RegimeDecoupler.variance_decomposition(train_full, "T50")
    pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(table_dir / "t50_variance_decomposition.csv", index=False)
    pd.DataFrame(variance["clusters"]).to_csv(table_dir / "regime_summary.csv", index=False)

    print("[maintenance] Fitting T50 virtual sensor and scoring failure alarms", flush=True)
    virtual_features = [sensor for sensor in sensors if sensor != "T50"]
    virtual_model = VirtualSensor(
        target_sensor="T50", feature_cols=virtual_features,
        healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=seed,
    )
    residual_trace, anomalies, anomaly_summary = virtual_model.evaluate(healthy_train, test_full)
    residual_trace.to_csv(table_dir / "virtual_sensor_residuals.csv", index=False)
    anomalies.to_csv(table_dir / "fleet_anomalies.csv", index=False)
    pd.DataFrame([anomaly_summary]).to_csv(table_dir / "fleet_anomaly_summary.csv", index=False)
    plot_failure_detection(residual_trace, anomalies, figure_dir / "t50_failure_detection.png")

    healthy_prediction = virtual_model.predict(healthy_test)
    regime_rows = []
    for regime_id, rows in healthy_prediction.groupby("regime_id"):
        truth, prediction = rows["T50"].to_numpy(), rows["prediction"].to_numpy()
        regime_rows.append({
            "Regime_ID": int(regime_id), "Test_rows": len(rows),
            "R2": float(r2_score(truth, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
            "MAE": float(mean_absolute_error(truth, prediction)),
        })
    regime_scores = pd.DataFrame(regime_rows)
    regime_scores.to_csv(table_dir / "healthy_t50_by_regime.csv", index=False)
    plot_regime_scores(regime_scores, figure_dir / "healthy_t50_by_regime.png")

    print("[maintenance] Benchmarking RUL models on held-out engines", flush=True)

    rul_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    benchmark = RULBenchmark(random_state=seed)
    holdout, predictions = benchmark.benchmark_train_test(
        parts["full_lifecycle_train"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_train"])),
        parts["full_lifecycle_test"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_test"])),
        rul_features,
    )
    holdout.to_csv(table_dir / "rul_holdout_benchmark.csv", index=False)
    best_model = str(holdout.loc[holdout["R2"].idxmax(), "Model"])
    plot_model_r2(holdout, figure_dir / "rul_model_r2_comparison.png")
    plot_actual_vs_predicted(
        predictions["y_true"], predictions[best_model],
        figure_dir / "rul_best_model_actual_vs_predicted.png",
        f"RUL holdout: {best_model}", "RUL (cycles)",
    )
    test_prediction_frame = parts["full_lifecycle_test"][["unit_id", "cycle"]].reset_index(drop=True)
    test_prediction_frame["RUL_actual"] = predictions["y_true"]
    for model_name, values in predictions.items():
        if model_name != "y_true":
            test_prediction_frame[f"RUL_predicted_{model_name}"] = values
    test_prediction_frame.to_csv(table_dir / "rul_holdout_predictions.csv", index=False)

    sensor_metrics, _ = _sensor_channel_analysis(
        healthy_train, test_full, seed, table_dir, figure_dir
    )
    cv_frame = frame.loc[frame["unit_id"].isin(train_units)].copy()
    cv = benchmark.cross_validate(cv_frame, rul_features, n_splits=5)
    cv.to_csv(table_dir / "rul_grouped_cross_validation.csv", index=False)

    tables = {
        "regime_variance": pd.DataFrame([variance]),
        "virtual_sensor_regimes": regime_scores,
        "anomaly_summary": pd.DataFrame([anomaly_summary]),
        "rul_holdout": holdout,
        "rul_grouped_cv": cv,
        "sensor_channel_metrics": sensor_metrics,
    }
    if include_lstm:
        from models.lstm import PyTorchLSTMModel

        lstm_frame = frame.copy()
        lstm_train = lstm_frame.loc[lstm_frame["unit_id"].isin(train_units)]
        lstm_test = lstm_frame.loc[lstm_frame["unit_id"].isin(test_units)]
        lstm_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
        model = PyTorchLSTMModel(seq_length=15, hidden_dim=16, num_layers=1, batch_size=256, max_epochs=4, patience=2, random_state=seed)
        model.fit(lstm_train, lstm_features, "RUL")
        truth, estimate = model.predict(lstm_test)
        lstm_results = pd.DataFrame([{"Model": "PyTorch LSTM", **evaluate_rul_predictions(truth, estimate)}])
        lstm_results.to_csv(table_dir / "rul_lstm_holdout.csv", index=False)
        tables["rul_lstm"] = lstm_results
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD004")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "maintenance_guide")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--include-lstm", action="store_true", help="Run an additional sequence-model holdout comparison")
    args = parser.parse_args()
    tables = run(args.subset, args.output_dir, args.seed, args.include_lstm)
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")
    print(f"Tables written to {(args.output_dir / 'tables').resolve()}")
    print(f"Figures written to {(args.output_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
