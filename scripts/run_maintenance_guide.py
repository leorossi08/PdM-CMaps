"""Run the C-MAPSS virtual-sensor, SPC, and RUL teaching workflow."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from anomaly import EnvelopeAnomalyDetector
from data import CMapssLoader
from evaluation import RULBenchmark
from plotting import (
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_with_envelope,
    plot_failure_detection,
    plot_geometric_envelope_detection,
    plot_minirocket_feature_filtering,
    plot_model_r2,
    plot_regime_scores,
    plot_sensor_parity_by_channel,
    plot_sensor_cycle_windows_by_regime,
    plot_sensor_parity_grid,
    plot_sensor_r2,
    plot_signal_filtering_comparison,
    plot_time_series_envelope_detection,
)
from regimes import RegimeDecoupler
from schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE, DEFAULT_RESULTS_DIR, SENSOR_COLUMNS, SETTING_COLUMNS
from splitting import partition_healthy_and_lifecycle, split_engine_units
from targets import compute_piecewise_rul, evaluate_rul_predictions, regression_metrics
from virtual_sensor import VirtualSensor


def _sensor_channel_analysis(
    train_healthy: pd.DataFrame,
    test_healthy: pd.DataFrame,
    test_lifecycle: pd.DataFrame,
    seed: int,
    table_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit healthy-window virtual sensors and score every physical channel on held-out engines."""
    metrics = []
    regime_metrics = []
    predictions = []
    healthy_predictions = []
    total_sensors = len(SENSOR_COLUMNS)
    for idx, target in enumerate(SENSOR_COLUMNS, start=1):
        features = [column for column in SENSOR_COLUMNS if column != target]
        features.extend([*SETTING_COLUMNS, "cycle"])
        model = VirtualSensor(
            target_sensor=target,
            feature_cols=features,
            healthy_cycle_limit=DEFAULT_HEALTHY_CYCLE_LIMIT,
            random_state=seed,
        ).fit(train_healthy)
        scored = model.predict(test_lifecycle)
        healthy_scored = model.predict(test_healthy)
        truth = scored[target].to_numpy(dtype=float)
        estimate = scored["prediction"].to_numpy(dtype=float)
        target_metrics = regression_metrics(truth, estimate)
        if target_metrics["target_variance"] <= 1e-12:
            target_metrics["R2"] = np.nan
        r2_display = f"{target_metrics['R2']:.4f}" if np.isfinite(target_metrics['R2']) else "n/a (constant)"
        is_multi_regime = "regime_id" in scored.columns and scored["regime_id"].nunique() > 1
        if is_multi_regime:
            within_r2s = []
            for _, r_group in scored.groupby("regime_id"):
                r_metrics = regression_metrics(
                    r_group[target].to_numpy(dtype=float),
                    r_group["prediction"].to_numpy(dtype=float),
                )
                if np.isfinite(r_metrics["R2"]) and r_metrics["target_variance"] > 1e-12:
                    within_r2s.append(r_metrics["R2"])
            within_r2_str = f" | Within-Regime R² = {np.mean(within_r2s):.4f}" if within_r2s else ""
            print(f"  [channels] [{idx:02d}/{total_sensors:02d}] {target:<10} -> Pooled R² = {r2_display}{within_r2_str}, RMSE = {target_metrics['RMSE']:.4f}", flush=True)
        else:
            print(f"  [channels] [{idx:02d}/{total_sensors:02d}] {target:<10} -> Full Lifecycle R² = {r2_display}, RMSE = {target_metrics['RMSE']:.4f}", flush=True)
        metrics.append({
            "Sensor": target,
            "Evaluation_scope": "full_lifecycle_pooled",
            "Test_rows": len(scored),
            **target_metrics,
        })
        predictions.append(pd.DataFrame({
            "unit_id": scored["unit_id"].to_numpy(),
            "cycle": scored["cycle"].to_numpy(),
            "regime_id": scored["regime_id"].to_numpy(),
            "Sensor": target,
            "Actual": truth,
            "Predicted": estimate,
        }))
        healthy_predictions.append(pd.DataFrame({
            "unit_id": healthy_scored["unit_id"].to_numpy(),
            "cycle": healthy_scored["cycle"].to_numpy(),
            "regime_id": healthy_scored["regime_id"].to_numpy(),
            "Sensor": target,
            "Actual": healthy_scored[target].to_numpy(dtype=float),
            "Predicted": healthy_scored["prediction"].to_numpy(dtype=float),
        }))

        for scope, scope_frame in (("healthy_window", healthy_scored), ("full_lifecycle", scored)):
            for regime_id, regime_rows in [("all", scope_frame), *scope_frame.groupby("regime_id", sort=True)]:
                scope_truth = regime_rows[target].to_numpy(dtype=float)
                scope_prediction = regime_rows["prediction"].to_numpy(dtype=float)
                scope_metrics = regression_metrics(scope_truth, scope_prediction)
                if scope_metrics["target_variance"] <= 1e-12:
                    scope_metrics["R2"] = np.nan
                regime_metrics.append({
                    "Evaluation_scope": scope,
                    "Regime_ID": regime_id,
                    "Sensor": target,
                    "Test_rows": len(regime_rows),
                    "Test_engines": int(regime_rows["unit_id"].nunique()),
                    **scope_metrics,
                })

    metric_frame = pd.DataFrame(metrics).sort_values("R2", ascending=False, na_position="last").reset_index(drop=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    healthy_prediction_frame = pd.concat(healthy_predictions, ignore_index=True)
    metric_frame.to_csv(table_dir / "sensor_channel_metrics.csv", index=False)
    pd.DataFrame(regime_metrics).to_csv(table_dir / "sensor_channel_metrics_by_regime_and_scope.csv", index=False)
    prediction_frame.to_csv(table_dir / "sensor_channel_predictions.csv", index=False)
    healthy_prediction_frame.to_csv(table_dir / "sensor_channel_healthy_predictions.csv", index=False)
    plot_sensor_r2(metric_frame, figure_dir / "sensor_r2_by_channel.png")
    plot_sensor_parity_grid(prediction_frame, figure_dir / "sensor_actual_vs_predicted_all_channels.png")
    plot_sensor_parity_by_channel(
        prediction_frame, figure_dir / "sensor_channels", scope_label="Full lifecycle"
    )
    plot_sensor_parity_by_channel(
        healthy_prediction_frame,
        figure_dir / "sensor_channels" / "healthy_window",
        scope_label="Healthy-window validation (cycles 1-50)",
    )
    _, cycle_metrics = plot_sensor_cycle_windows_by_regime(
        prediction_frame,
        figure_dir / "sensor_cycle_windows",
        window_size=50,
    )
    cycle_metrics.to_csv(table_dir / "sensor_cycle_window_metrics.csv", index=False)
    print(f"  [channels] Complete. Saved sensor metrics to {(table_dir / 'sensor_channel_metrics.csv').name}", flush=True)
    return metric_frame, prediction_frame


PRIMARY_TARGET_SENSORS = ("T50", "P30", "BPR", "Nc")


def run(
    subset: str,
    output_dir: Path,
    seed: int,
    include_lstm: bool = False,
    include_minirocket: bool = False,
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    target_sensors: tuple[str, ...] = PRIMARY_TARGET_SENSORS,
    sample_engine_id: int | None = 179,
    min_consecutive_breaches: int = 2,
) -> dict[str, pd.DataFrame]:
    """Execute the guide's core analyses and write their tables/figures."""
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================================", flush=True)
    print(f"[maintenance] STARTING PdM PIPELINE: Subset={subset} | Seed={seed} | Healthy limit={healthy_cycle_limit} cycles", flush=True)
    print(f"=======================================================================\n", flush=True)

    print(f"[maintenance] [STEP 1/7] Loading {subset} data and partitioning disjoint engine holdout...", flush=True)
    loader = CMapssLoader()
    frame = loader.load(subset)
    frame["RUL"] = compute_piecewise_rul(frame)
    sensors = loader.informative_sensors(frame)
    train_units, test_units = split_engine_units(frame, test_ratio=0.25, random_state=seed)
    parts = partition_healthy_and_lifecycle(frame, train_units, test_units, healthy_cycle_limit)
    print(f"  -> Total rows: {len(frame)} | Engines: {frame['unit_id'].nunique()} (Train: {len(train_units)}, Test: {len(test_units)})", flush=True)

    n_regimes = 6 if subset in {"FD002", "FD004"} else 1
    print(f"[maintenance] [STEP 2/7] Decoupling operating flight regimes ({n_regimes} clusters)...", flush=True)
    regimes = RegimeDecoupler(n_clusters=n_regimes, random_state=seed).fit(parts["full_lifecycle_train"])
    train_full = regimes.transform(parts["full_lifecycle_train"])
    test_full = regimes.transform(parts["full_lifecycle_test"])
    healthy_train = regimes.transform(parts["healthy_train"])
    healthy_test = regimes.transform(parts["healthy_test"])

    valid_targets = [s for s in target_sensors if s in sensors]
    if not valid_targets:
        valid_targets = ["T50"] if "T50" in sensors else [sensors[0]]

    target_unit_id = sample_engine_id if (sample_engine_id is not None and sample_engine_id in test_units) else int(test_units[0])
    print(f"  -> Primary sample engine for trace visualization: Engine #{target_unit_id}", flush=True)

    primary_variance = None
    primary_regime_scores = None
    primary_anomaly_summary = None

    print(f"[maintenance] [STEPS 3-5/7] Profiling Virtual Sensors & Envelopes across targets: {', '.join(valid_targets)}...", flush=True)
    for target_sensor in valid_targets:
        s_lower = target_sensor.lower()
        print(f"\n  [sensor: {target_sensor}] Decoupling variance and fitting Virtual Sensor...", flush=True)

        # 1. Variance Decomposition
        variance = RegimeDecoupler.variance_decomposition(train_full, target_sensor)
        pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(table_dir / f"{s_lower}_variance_decomposition.csv", index=False)
        if target_sensor == "T50":
            pd.DataFrame([variance]).drop(columns=["clusters"]).to_csv(table_dir / "t50_variance_decomposition.csv", index=False)
            primary_variance = variance

        between_pct = variance.get("Between_regime_pct", variance.get("pct_between_regime", 0.0))
        within_pct = variance.get("Within_regime_pct", variance.get("pct_within_regime", 100.0))
        base_r2 = variance.get("regime_baseline_r2", between_pct / 100.0 if between_pct else 0.0)
        print(f"    -> {target_sensor} variance: Between regimes = {between_pct:.1f}%, Within regime = {within_pct:.1f}% (Baseline R² = {base_r2:.4f})", flush=True)

        # 2. Virtual Sensor fitting & Fleet SPC
        virtual_features = [sensor for sensor in sensors if sensor != target_sensor]
        virtual_model = VirtualSensor(
            target_sensor=target_sensor, feature_cols=virtual_features,
            healthy_cycle_limit=healthy_cycle_limit, random_state=seed,
        )
        residual_trace, anomalies, anomaly_summary = virtual_model.evaluate(healthy_train, test_full)
        h_suffix = f"_h{healthy_cycle_limit}"
        residual_trace.to_csv(table_dir / f"virtual_sensor_residuals_{s_lower}.csv", index=False)
        residual_trace.to_csv(table_dir / f"virtual_sensor_residuals_{s_lower}{h_suffix}.csv", index=False)
        anomalies.to_csv(table_dir / f"fleet_anomalies_{s_lower}.csv", index=False)
        anomalies.to_csv(table_dir / f"fleet_anomalies_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([anomaly_summary]).to_csv(table_dir / f"fleet_anomaly_summary_{s_lower}.csv", index=False)
        pd.DataFrame([anomaly_summary]).to_csv(table_dir / f"fleet_anomaly_summary_{s_lower}{h_suffix}.csv", index=False)
        plot_failure_detection(residual_trace, anomalies, figure_dir / f"{s_lower}_failure_detection.png", healthy_cycle_limit=healthy_cycle_limit)
        if target_sensor == "T50":
            residual_trace.to_csv(table_dir / "virtual_sensor_residuals.csv", index=False)
            anomalies.to_csv(table_dir / "fleet_anomalies.csv", index=False)
            pd.DataFrame([anomaly_summary]).to_csv(table_dir / "fleet_anomaly_summary.csv", index=False)
            plot_failure_detection(residual_trace, anomalies, figure_dir / "t50_failure_detection.png", healthy_cycle_limit=healthy_cycle_limit)
            primary_anomaly_summary = anomaly_summary

        print(f"    -> Virtual Sensor {target_sensor}: Fleet detection = {anomaly_summary['detection_rate_pct']:.1f}% | Mean lead time = {anomaly_summary['mean_lead_time_cycles']:.1f} cycles", flush=True)

        # 3. Signal Filtering & MiniRocket for target_unit_id
        plot_signal_filtering_comparison(
            frame=frame,
            unit_id=target_unit_id,
            sensor=target_sensor,
            path=figure_dir / f"signal_filtering_ewma_{s_lower}.png",
            alphas=(0.1, 0.35, 0.7),
        )
        plot_minirocket_feature_filtering(
            frame=frame,
            unit_id=target_unit_id,
            path=figure_dir / f"minirocket_feature_filtering_{s_lower}.png",
            sensor=target_sensor,
            random_state=seed,
        )
        if target_sensor == "T50":
            plot_signal_filtering_comparison(
                frame=frame,
                unit_id=target_unit_id,
                sensor=target_sensor,
                path=figure_dir / "signal_filtering_ewma_comparison.png",
                alphas=(0.1, 0.35, 0.7),
            )
            plot_minirocket_feature_filtering(
                frame=frame,
                unit_id=target_unit_id,
                path=figure_dir / "minirocket_feature_filtering_padua.png",
                sensor=target_sensor,
                random_state=seed,
            )

        # 4. Envelope Anomaly Detection for target_unit_id
        envelope_detector = EnvelopeAnomalyDetector(
            k_sigma=3.0,
            healthy_cycle_limit=healthy_cycle_limit,
            window_size=5,
            min_consecutive_breaches=min_consecutive_breaches,
        )
        healthy_res_rows = residual_trace.loc[residual_trace["cycle"] <= healthy_cycle_limit]
        envelope_detector.calibrate(healthy_res_rows[target_sensor], healthy_res_rows["prediction"])

        sample_trace = residual_trace.loc[residual_trace["unit_id"] == target_unit_id].copy()
        scored_env, env_summary = envelope_detector.score_time_series_envelope(
            sample_trace, actual_col=target_sensor, pred_col="prediction",
        )
        scored_geom, geom_summary = envelope_detector.score_geometric_envelope(
            sample_trace, signal_col=target_sensor,
        )

        scored_env.to_csv(table_dir / f"envelope_scored_trace_{s_lower}.csv", index=False)
        scored_env.to_csv(table_dir / f"envelope_scored_trace_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary_{s_lower}.csv", index=False)
        pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary_{s_lower}{h_suffix}.csv", index=False)
        pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary_{s_lower}.csv", index=False)
        pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary_{s_lower}{h_suffix}.csv", index=False)
        if target_sensor == "T50":
            scored_env.to_csv(table_dir / "envelope_scored_trace.csv", index=False)
            scored_env.to_csv(table_dir / f"envelope_scored_trace{h_suffix}.csv", index=False)
            pd.DataFrame([env_summary]).to_csv(table_dir / "envelope_summary.csv", index=False)
            pd.DataFrame([env_summary]).to_csv(table_dir / f"envelope_summary{h_suffix}.csv", index=False)
            pd.DataFrame([geom_summary]).to_csv(table_dir / "geometric_envelope_summary.csv", index=False)
            pd.DataFrame([geom_summary]).to_csv(table_dir / f"geometric_envelope_summary{h_suffix}.csv", index=False)

        plot_time_series_envelope_detection(
            scored_env,
            env_summary,
            figure_dir / f"{s_lower}_time_series_envelope_detection.png",
            title=f"Detecção por Envelope de Séries Temporais: {target_sensor} (Estilo Power BI)",
            sensor_name=target_sensor,
            healthy_cycle_limit=healthy_cycle_limit,
        )
        plot_geometric_envelope_detection(
            scored_geom,
            geom_summary,
            figure_dir / f"{s_lower}_geometric_envelope_detection.png",
            title=f"Envelope Geométrico Digital (Hull): {target_sensor}",
            sensor_name=target_sensor,
            healthy_cycle_limit=healthy_cycle_limit,
        )
        plot_actual_vs_predicted_with_envelope(
            residual_trace[target_sensor],
            residual_trace["prediction"],
            figure_dir / f"{s_lower}_parity_envelope.png",
            f"Sensor Virtual {target_sensor}: Real vs Predito com Embrulho de Tolerância (±3σ)",
            value_label=f"{target_sensor}",
            k_sigma=3.0,
            cal_sigma=envelope_detector.sigma_e,
        )
        if target_sensor == "T50":
            plot_time_series_envelope_detection(
                scored_env,
                env_summary,
                figure_dir / "t50_time_series_envelope_detection.png",
                title="Detecção por Envelope de Séries Temporais (Estilo Power BI)",
                sensor_name="T50",
                healthy_cycle_limit=healthy_cycle_limit,
            )
            plot_geometric_envelope_detection(
                scored_geom,
                geom_summary,
                figure_dir / "t50_geometric_envelope_detection.png",
                title="Envelope Geométrico de Sinal Digital (Rolling Peak/Valley Hull)",
                sensor_name="T50",
                healthy_cycle_limit=healthy_cycle_limit,
            )
            plot_actual_vs_predicted_with_envelope(
                residual_trace["T50"],
                residual_trace["prediction"],
                figure_dir / "t50_parity_envelope.png",
                "Sensor Virtual T50: Real vs Predito com Embrulho de Tolerância (±3σ)",
                value_label="T50 (°C)",
                k_sigma=3.0,
                cal_sigma=envelope_detector.sigma_e,
            )

        print(f"    -> Time-series envelope ({target_sensor}): Alarm at cycle {env_summary['first_alarm_cycle']} | EOL: {env_summary['eol_cycle']} | Lead time: {env_summary['lead_time_cycles']} cycles", flush=True)

        # 5. Held-out healthy regime validation
        healthy_prediction = virtual_model.predict(healthy_test)
        regime_rows = []
        for regime_id, rows in healthy_prediction.groupby("regime_id"):
            truth, prediction = rows[target_sensor].to_numpy(), rows["prediction"].to_numpy()
            regime_rows.append({
                "Regime_ID": int(regime_id), "Test_rows": len(rows),
                "R2": float(r2_score(truth, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
                "MAE": float(mean_absolute_error(truth, prediction)),
            })
        regime_scores = pd.DataFrame(regime_rows)
        regime_scores.to_csv(table_dir / f"healthy_{s_lower}_by_regime.csv", index=False)
        regime_scores.to_csv(table_dir / f"healthy_{s_lower}_by_regime{h_suffix}.csv", index=False)
        plot_regime_scores(regime_scores, figure_dir / f"healthy_{s_lower}_by_regime.png")
        if target_sensor == "T50":
            regime_scores.to_csv(table_dir / "healthy_t50_by_regime.csv", index=False)
            regime_scores.to_csv(table_dir / f"healthy_t50_by_regime{h_suffix}.csv", index=False)
            plot_regime_scores(regime_scores, figure_dir / "healthy_t50_by_regime.png")
            primary_regime_scores = regime_scores

    print(f"[maintenance] [STEP 6/7] Profiling full sensor fleet across channels and operating regimes...", flush=True)
    sensor_metrics, _ = _sensor_channel_analysis(
        healthy_train, healthy_test, test_full, seed, table_dir, figure_dir
    )

    print(f"[maintenance] [STEP 7/7] Benchmarking RUL models on held-out engines (CatBoost, LightGBM, XGBoost, etc.)...", flush=True)

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
    plot_actual_vs_predicted_with_envelope(
        predictions["y_true"], predictions[best_model],
        figure_dir / "rul_best_model_parity_envelope.png",
        f"RUL Holdout: {best_model} com Embrulho de Tolerância (±3σ)",
        value_label="RUL (ciclos)",
        k_sigma=3.0,
    )
    test_prediction_frame = parts["full_lifecycle_test"][["unit_id", "cycle"]].reset_index(drop=True)
    test_prediction_frame["RUL_actual"] = predictions["y_true"]
    for model_name, values in predictions.items():
        if model_name != "y_true":
            test_prediction_frame[f"RUL_predicted_{model_name}"] = values
    test_prediction_frame.to_csv(table_dir / "rul_holdout_predictions.csv", index=False)

    cv_frame = frame.loc[frame["unit_id"].isin(train_units)].copy()
    cv = benchmark.cross_validate(cv_frame, rul_features, n_splits=5)
    table_dir.mkdir(parents=True, exist_ok=True)
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
        lstm_results = pd.DataFrame([{
            "Model": "PyTorch LSTM",
            **evaluate_rul_predictions(truth, estimate),
            "Fit_Time_s": round(model.fit_time, 3),
        }])
        lstm_results.to_csv(table_dir / "rul_lstm_holdout.csv", index=False)
        tables["rul_lstm"] = lstm_results

    if include_minirocket:
        print("[maintenance] Fitting MiniRocket + RidgeCV and benchmarking RUL on held-out engines", flush=True)
        from models.minirocket import MiniRocketRULModel

        mr_frame = frame.copy()
        mr_train = mr_frame.loc[mr_frame["unit_id"].isin(train_units)]
        mr_test = mr_frame.loc[mr_frame["unit_id"].isin(test_units)]
        mr_features = [*sensors, "setting_1", "setting_2", "setting_3", "cycle"]
        mr_model = MiniRocketRULModel(seq_length=15, num_kernels=1000, random_state=seed)
        mr_model.fit(mr_train, mr_features, "RUL")
        mr_truth, mr_estimate = mr_model.predict(mr_test)
        mr_results = pd.DataFrame([{
            "Model": "MiniRocket + RidgeCV",
            **evaluate_rul_predictions(mr_truth, mr_estimate),
            "Fit_Time_s": round(mr_model.fit_time, 3),
        }])
        combined_holdout = pd.concat([holdout, mr_results], ignore_index=True)
        combined_holdout.to_csv(table_dir / "rul_holdout_benchmark.csv", index=False)
        plot_model_r2(combined_holdout, figure_dir / "rul_model_r2_comparison.png", min_r2_cutoff=0.75)
        plot_actual_vs_predicted_with_envelope(
            mr_truth, mr_estimate,
            figure_dir / "rul_minirocket_parity_envelope.png",
            "RUL Holdout: MiniRocket + RidgeCV com Embrulho de Tolerância (±3σ)",
            value_label="RUL (ciclos)",
            k_sigma=3.0,
        )
        tables["rul_minirocket"] = mr_results
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD004")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "maintenance_guide")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--include-lstm", action="store_true", help="Run an additional sequence-model holdout comparison")
    parser.add_argument("--include-minirocket", action="store_true", help="Run an additional MiniRocket sequence-model holdout comparison")
    parser.add_argument("--healthy-cycle-limit", type=int, default=DEFAULT_HEALTHY_CYCLE_LIMIT, help="Healthy cycle limit used for virtual sensor calibration (default: 50)")
    parser.add_argument("--target-sensors", nargs="+", default=list(PRIMARY_TARGET_SENSORS), help="Target sensors to evaluate for virtual sensor and envelope detection (default: T50 P30 BPR Nc)")
    parser.add_argument("--min-consecutive-breaches", type=int, default=2, help="Minimum consecutive breaches required to trigger an envelope alarm (default: 2, Western Electric rule)")
    parser.add_argument("--unit-id", type=int, default=179, help="Engine unit ID to use for sample traces (default: 179)")
    args = parser.parse_args()
    tables = run(
        args.subset,
        args.output_dir,
        args.seed,
        args.include_lstm,
        args.include_minirocket,
        args.healthy_cycle_limit,
        target_sensors=tuple(args.target_sensors),
        sample_engine_id=args.unit_id,
        min_consecutive_breaches=args.min_consecutive_breaches,
    )
    print("\n=======================================================================", flush=True)
    print(f"[maintenance] PIPELINE EXECUTION COMPLETE! Summary of generated tables:", flush=True)
    print(f"=======================================================================", flush=True)
    for name, table in tables.items():
        print(f"  - Table: {name:<28} ({len(table)} rows)", flush=True)
    print(f"\n[maintenance] Tables written to: {(args.output_dir / 'tables').resolve()}", flush=True)
    print(f"[maintenance] Figures written to: {(args.output_dir / 'figures').resolve()}\n", flush=True)


if __name__ == "__main__":
    main()
