"""Run the C-MAPSS virtual-sensor, SPC, and RUL teaching workflow."""

import argparse
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pdm_cmaps.data import CMapssLoader
from pdm_cmaps.evaluation import RULBenchmark
from pdm_cmaps.plotting import plot_regime_scores
from pdm_cmaps.regimes import RegimeDecoupler
from pdm_cmaps.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RESULTS_DIR, SENSOR_COLUMNS
from pdm_cmaps.splitting import partition_healthy_and_lifecycle, split_engine_units
from pdm_cmaps.targets import compute_piecewise_rul, evaluate_rul_predictions
from pdm_cmaps.virtual_sensor import VirtualSensor


def run(subset: str, output_dir: Path, seed: int, include_lstm: bool = False) -> dict[str, pd.DataFrame]:
    """Execute the guide's core analyses and write their tables/figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(output_dir / "t50_variance_decomposition.csv", index=False)
    pd.DataFrame(variance["clusters"]).to_csv(output_dir / "regime_summary.csv", index=False)

    virtual_features = [sensor for sensor in sensors if sensor != "T50"]
    virtual_model = VirtualSensor(
        target_sensor="T50", feature_cols=virtual_features,
        healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT, random_state=seed,
    )
    residual_trace, anomalies, anomaly_summary = virtual_model.evaluate(healthy_train, test_full)
    residual_trace.to_csv(output_dir / "virtual_sensor_residuals.csv", index=False)
    anomalies.to_csv(output_dir / "fleet_anomalies.csv", index=False)
    pd.DataFrame([anomaly_summary]).to_csv(output_dir / "fleet_anomaly_summary.csv", index=False)

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
    regime_scores.to_csv(output_dir / "healthy_t50_by_regime.csv", index=False)
    plot_regime_scores(regime_scores, output_dir / "healthy_t50_by_regime.png")

    rul_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
    benchmark = RULBenchmark(random_state=seed)
    holdout, predictions = benchmark.benchmark_train_test(
        parts["full_lifecycle_train"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_train"])),
        parts["full_lifecycle_test"].assign(RUL=compute_piecewise_rul(parts["full_lifecycle_test"])),
        rul_features,
    )
    holdout.to_csv(output_dir / "rul_holdout_benchmark.csv", index=False)
    cv_frame = frame.loc[frame["unit_id"].isin(train_units)].copy()
    cv = benchmark.cross_validate(cv_frame, rul_features, n_splits=5)
    cv.to_csv(output_dir / "rul_grouped_cross_validation.csv", index=False)

    tables = {
        "regime_variance": pd.DataFrame([variance]),
        "virtual_sensor_regimes": regime_scores,
        "anomaly_summary": pd.DataFrame([anomaly_summary]),
        "rul_holdout": holdout,
        "rul_grouped_cv": cv,
    }
    if include_lstm:
        from pdm_cmaps.models.lstm import PyTorchLSTMModel

        lstm_frame = frame.copy()
        lstm_train = lstm_frame.loc[lstm_frame["unit_id"].isin(train_units)]
        lstm_test = lstm_frame.loc[lstm_frame["unit_id"].isin(test_units)]
        lstm_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
        model = PyTorchLSTMModel(seq_length=15, hidden_dim=16, num_layers=1, batch_size=256, max_epochs=4, patience=2, random_state=seed)
        model.fit(lstm_train, lstm_features, "RUL")
        truth, estimate = model.predict(lstm_test)
        lstm_results = pd.DataFrame([{"Model": "PyTorch LSTM", **evaluate_rul_predictions(truth, estimate)}])
        lstm_results.to_csv(output_dir / "rul_lstm_holdout.csv", index=False)
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
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
