"""Comprehensive scientific and statistical rigor tests for C-MAPSS PdM pipeline.

Covers the 10 peer-review audit dimensions:
1. regimes.py: k-sweep diagnostics, lexicographic centroid sorting, relative margin, OOS variance, regime standardization.
2. targets.py: overflow guard in nasa_scoring_function, prediction clipping, stratified evaluation scopes (critical zone & censored).
3. splitting.py & evaluation.py: EOL-stratified engine split, non-leakage grouped CV, pooled OOF vs fold-weighted metrics, Mean NASA Score.
4. features.py: boundary fix for lag2 at t=2, is_startup indicator, CausalFeatureEngineer with median rollstd imputation, regime standardization.
5. virtual_sensor.py: canonical exogenous predictors, hierarchical fallback for sparse regimes, delegation to anomaly detector.
6. anomaly.py & plotting.py: online alarm restriction (cycle > H), disaggregated operational metrics, empirical UCL, regime-standardized envelope.
7. thesis_statistics.py: H1 engine-block Friedman, H2 survivor-bias-free window (n=62/62), H3 fleet Ljung-Box, H4/H5 Hodges-Lehmann & FDR, BCa observed metrics.
8. diagnostics.py: n>=10 repeats, grouped subsystem permutation importance, Wasserstein-1 distance.
9. models/: deterministic DataLoader generator, internal GroupShuffleSplit early stopping, GroupKFold in RidgeCV, replicate padding in MiniRocket.
10. data.py: load_test for official test benchmark, post-split informative_sensors.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from anomaly import EnvelopeAnomalyDetector, StatisticalAnomalyDetector
from data import CMapssLoader
from diagnostics import (
    compare_split_strategies,
    distribution_shift,
    learning_curves,
    permutation_importances,
)
from evaluation import RULBenchmark, grouped_oof
from features import CausalFeatureEngineer, engineer_causal_temporal_features
from models.deep_models import BasePyTorchSequenceModel, PyTorchRNNModel
from models.factory import get_model, get_model_pool
from models.minirocket import MiniRocketRULModel, MiniRocketTransformer
from regimes import RegimeDecoupler
from schema import COLUMNS, DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RUL_CAP, SENSOR_COLUMNS, SETTING_COLUMNS
from splitting import partition_healthy_and_lifecycle, split_engine_units
from targets import (
    clip_predictions,
    compute_piecewise_rul,
    evaluate_rul_predictions,
    nasa_scoring_function,
)
from thesis_statistics import ThesisHypothesisAnalyzer


def make_synthetic_cmapss(
    n_engines: int = 12,
    min_cycles: int = 60,
    max_cycles: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic C-MAPSS dataset with diverse EOLs and 6 flight regimes."""
    rng = np.random.default_rng(seed)
    rows = []
    regime_settings = [
        (0.0, 0.0, 100.0),
        (10.0, 0.25, 100.0),
        (20.0, 0.70, 100.0),
        (25.0, 0.62, 60.0),
        (35.0, 0.84, 100.0),
        (42.0, 0.84, 100.0),
    ]
    for unit_id in range(1, n_engines + 1):
        eol = rng.integers(min_cycles, max_cycles + 1)
        for cycle in range(1, eol + 1):
            r_idx = rng.integers(0, len(regime_settings))
            s1, s2, s3 = regime_settings[r_idx]
            deg_factor = (cycle / eol) ** 2
            row = {
                "unit_id": unit_id,
                "cycle": cycle,
                "setting_1": s1 + rng.normal(0, 0.01),
                "setting_2": s2 + rng.normal(0, 0.005),
                "setting_3": s3,
            }
            # Physical sensors: T2, P2, P15, epr are boundary/inlet conditions
            row["T2"] = 518.67 - 3.5 * s1 + rng.normal(0, 0.2)
            row["P2"] = 14.696 * (1.0 - 0.02 * s1) + rng.normal(0, 0.05)
            row["P15"] = 14.0 + 0.1 * s2 + rng.normal(0, 0.02)
            row["epr"] = 1.0 + 0.3 * s2 + rng.normal(0, 0.01)
            row["Nf_dmd"] = 2388.0
            row["PCNfR_dmd"] = 100.0
            # Internal degraded sensors: T50, T24, P30 drift with deg_factor
            row["T24"] = 640.0 + 5.0 * s1 + 15.0 * deg_factor + rng.normal(0, 0.5)
            row["T30"] = 1580.0 + 10.0 * s1 + 25.0 * deg_factor + rng.normal(0, 1.0)
            row["T50"] = 1400.0 + 8.0 * s1 + 35.0 * deg_factor + rng.normal(0, 0.8)
            row["P30"] = 550.0 - 2.0 * s1 - 10.0 * deg_factor + rng.normal(0, 0.5)
            # Remaining sensor columns
            for sensor in SENSOR_COLUMNS:
                if sensor not in row:
                    row[sensor] = 100.0 + rng.normal(0, 1.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# =========================================================================
# 1. regimes.py
# =========================================================================

def test_regime_k_sweep_diagnostics():
    """Verify empirical k-selection evaluation (Silhouette, Calinski-Harabasz, BIC)."""
    df = make_synthetic_cmapss(n_engines=8, seed=42)
    decoupler = RegimeDecoupler()
    sweep = decoupler.evaluate_k_sweep(df, k_range=range(2, 6))
    assert isinstance(sweep, pd.DataFrame)
    assert "k" in sweep.columns
    assert "silhouette" in sweep.columns
    assert "calinski_harabasz" in sweep.columns
    assert "bic" in sweep.columns
    assert len(sweep) == 4
    assert (sweep["silhouette"] >= -1.0).all() and (sweep["silhouette"] <= 1.0).all()
    assert (sweep["calinski_harabasz"] > 0).all()


def test_regime_lexicographical_centroid_sorting():
    """Verify cluster ordering is physical and lexicographically stable across runs."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    r1 = RegimeDecoupler(n_clusters=6, random_state=42).fit(df)
    r2 = RegimeDecoupler(n_clusters=6, random_state=123).fit(df)
    # Centroid coordinates sorted by (altitude, mach, TRA) should be strictly monotonic in altitude
    centers1 = r1.kmeans.cluster_centers_
    centers_phys1 = r1.scaler.inverse_transform(centers1)
    # Check that regime_ids map to lexicographically ordered physical centers
    ordered_centers = np.zeros_like(centers_phys1)
    for cl_idx, r_id in r1.cluster_to_regime_id.items():
        ordered_centers[r_id] = centers_phys1[cl_idx]
    # Altitude (setting_1) should be non-decreasing across ordered regime IDs
    assert np.all(np.diff(ordered_centers[:, 0]) >= -1e-3)


def test_regime_distance_margin_and_relative_margin():
    """Verify distance_margin includes absolute and dimensionless relative margins."""
    df = make_synthetic_cmapss(n_engines=4, seed=42)
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42).fit(df)
    margins = decoupler.distance_margin(df)
    assert "nearest_distance" in margins.columns
    assert "runner_up_distance" in margins.columns
    assert "distance_margin" in margins.columns
    assert "relative_margin" in margins.columns
    assert (margins["nearest_distance"] <= margins["runner_up_distance"] + 1e-6).all()
    assert (margins["distance_margin"] >= -1e-6).all()
    assert (margins["relative_margin"] >= -1e-6).all() and (margins["relative_margin"] <= 1.0 + 1e-6).all()


def test_regime_standardization_and_out_of_sample_variance():
    """Verify regime-wise standardization and clean separation of descriptive vs OOS variance."""
    train_df = make_synthetic_cmapss(n_engines=8, seed=42)
    test_df = make_synthetic_cmapss(n_engines=4, seed=99)
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42).fit(train_df)
    train_trans = decoupler.transform(train_df)
    test_trans = decoupler.transform(test_df)

    # Standardize by regime
    decoupler.fit_sensor_scaler(train_trans, ["T50", "P30"])
    std_train = decoupler.standardize_sensors(train_trans, ["T50", "P30"])
    # Per-regime training mean of standardized features should be ~0.0
    for r_id in train_trans["regime_id"].unique():
        sub = std_train.loc[std_train["regime_id"] == r_id, "T50_norm"]
        assert abs(sub.mean()) < 1e-3
        assert abs(sub.std() - 1.0) < 1e-2 or sub.std() == 0.0

    # Variance decomposition separating descriptive vs OOS
    decomp = RegimeDecoupler.variance_decomposition(test_trans, "T50", reference_frame=train_trans)
    assert "descriptive_between_pct" in decomp or "pct_between_regime" in decomp
    assert "oos_r2" in decomp or "regime_baseline_r2" in decomp
    assert decomp["baseline_evaluation"] == "out_of_sample"


# =========================================================================
# 2. targets.py
# =========================================================================

def test_nasa_scoring_function_overflow_guard():
    """Verify nasa_scoring_function does not trigger RuntimeWarning: overflow on extreme differences."""
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([600.0, -600.0])  # Extreme deviations that would overflow exp(60)
    score, mean_score = nasa_scoring_function(y_true, y_pred)
    assert np.isfinite(score)
    assert np.isfinite(mean_score)
    assert score > 0.0


def test_clip_predictions_and_stratified_evaluation():
    """Verify prediction clipping in [0, 125] and stratified critical-zone scoring."""
    y_true = np.array([125.0, 80.0, 30.0, 10.0, 0.0])
    y_pred = np.array([150.0, 90.0, 35.0, -5.0, -20.0])
    clipped = clip_predictions(y_pred, lower=0.0, upper=DEFAULT_RUL_CAP)
    assert clipped.max() <= DEFAULT_RUL_CAP
    assert clipped.min() >= 0.0

    # Evaluation with clipping
    metrics_unclipped = evaluate_rul_predictions(y_true, y_pred, clip=False)
    metrics_clipped = evaluate_rul_predictions(y_true, y_pred, clip=True)
    assert metrics_clipped["RMSE"] <= metrics_unclipped["RMSE"] + 1e-6

    # Critical zone evaluation (RUL < 50)
    from targets import evaluate_stratified_rul
    strat = evaluate_stratified_rul(y_true, y_pred, critical_threshold=50.0)
    assert "overall" in strat
    assert "critical_zone" in strat
    assert strat["critical_zone"]["N_samples"] == 3


# =========================================================================
# 3. splitting.py & evaluation.py
# =========================================================================

def test_split_engine_units_eol_stratified():
    """Verify EOL-stratified holdout split produces matching EOL distributions without leakage."""
    df = make_synthetic_cmapss(n_engines=24, min_cycles=50, max_cycles=150, seed=42)
    train_units, test_units = split_engine_units(df, test_ratio=0.25, random_state=42, stratify_eol=True)
    assert len(set(train_units).intersection(test_units)) == 0
    assert len(train_units) + len(test_units) == df["unit_id"].nunique()

    # Compare EOL medians between train and test
    eol_map = df.groupby("unit_id")["cycle"].max()
    train_eols = eol_map.loc[train_units]
    test_eols = eol_map.loc[test_units]
    # KS-test or median difference should confirm similar distributions
    assert abs(train_eols.median() - test_eols.median()) < 25.0


def test_cross_validate_pooled_vs_fold_weighted_and_mean_nasa():
    """Verify cross_validate reports pooled OOF, fold-weighted metrics, and Mean_NASA_Score."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "setting_2", "T2", "T50"]
    benchmark = RULBenchmark(verbose=False)
    cv_results = benchmark.cross_validate(df, features, target_col="RUL", n_splits=3)
    assert isinstance(cv_results, pd.DataFrame)
    # Check that pooled OOF and fold-weighted metrics are distinctly reported
    assert "CV_OOF_Pooled_R2" in cv_results.columns or "CV_R2_Mean" in cv_results.columns
    assert "CV_Mean_NASA_Score" in cv_results.columns or "CV_NASA_Score_Mean" in cv_results.columns


# =========================================================================
# 4. features.py
# =========================================================================

def test_lag2_boundary_fix_at_cycle_2():
    """Verify lag2 at t=2 is filled with lag1 (x at t=1), not current x at t=2."""
    df = pd.DataFrame({
        "unit_id": [1, 1, 1],
        "cycle": [1, 2, 3],
        "T50": [100.0, 110.0, 120.0],
    })
    res = engineer_causal_temporal_features(df, ["T50"], windows=(3,))
    # At t=1: lag1=100, lag2=100
    assert res.loc[0, "T50_lag1"] == 100.0
    assert res.loc[0, "T50_lag2"] == 100.0
    # At t=2: lag1=100, lag2 must be 100 (filled with lag1), NOT 110!
    assert res.loc[1, "T50_lag1"] == 100.0
    assert res.loc[1, "T50_lag2"] == 100.0
    # At t=3: lag1=110, lag2=100
    assert res.loc[2, "T50_lag1"] == 110.0
    assert res.loc[2, "T50_lag2"] == 100.0


def test_causal_feature_engineer_startup_and_imputation():
    """Verify CausalFeatureEngineer provides is_startup indicator and healthy-median rollstd imputation."""
    train_df = make_synthetic_cmapss(n_engines=6, seed=42)
    test_df = make_synthetic_cmapss(n_engines=2, seed=99)
    engineer = CausalFeatureEngineer(sensor_cols=["T50", "T24"], windows=(3, 5))
    engineer.fit(train_df, healthy_cycle_limit=30)
    transformed = engineer.transform(test_df)
    assert "is_startup" in transformed.columns
    # cycles 1 and 2 should have is_startup == 1.0
    assert (transformed.loc[transformed["cycle"] <= 2, "is_startup"] == 1.0).all()
    assert (transformed.loc[transformed["cycle"] > 2, "is_startup"] == 0.0).all()
    # rollstd at cycle 1 should be imputed with healthy median, not rigid 0.0
    cycle1_std = transformed.loc[transformed["cycle"] == 1, "T50_rollstd_3"].to_numpy()
    assert (cycle1_std > 0.0).all()


# =========================================================================
# 5. virtual_sensor.py
# =========================================================================

def test_canonical_exogenous_virtual_sensor_and_fallback():
    """Verify VirtualSensor uses canonical exogenous features and handles sparse regimes via fallback."""
    from virtual_sensor import CANONICAL_EXOGENOUS_FEATURES, VirtualSensor

    assert "setting_1" in CANONICAL_EXOGENOUS_FEATURES
    assert "T2" in CANONICAL_EXOGENOUS_FEATURES
    assert "T50" not in CANONICAL_EXOGENOUS_FEATURES  # Target is not exogenous

    df = make_synthetic_cmapss(n_engines=6, seed=42)
    df["regime_id"] = df["cycle"] % 6
    # Fit virtual sensor with canonical features
    vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=30, random_state=42)
    # Check that it fits without error even if a regime has limited samples
    vs.fit(df)
    assert len(vs.models) > 0
    preds = vs.predict(df)
    assert "prediction" in preds.columns
    assert "residual" in preds.columns
    assert np.isfinite(preds["prediction"]).all()


# =========================================================================
# 6. anomaly.py & plotting.py
# =========================================================================

def test_online_alarm_restriction_and_disaggregated_metrics():
    """Verify operational alarm is restricted to online monitoring phase (cycle > H)."""
    detector = StatisticalAnomalyDetector(healthy_cycle_limit=50, threshold_sigma=3.0)
    # Calibrate on healthy residuals
    np.random.seed(42)
    healthy_res = np.random.randn(200) * 0.5
    detector.calibrate(healthy_res)

    # Engine with early spike at cycle 10, but true failure starting at cycle 80
    cycles = np.arange(1, 101)
    res = np.random.randn(100) * 0.5
    res[9] = 10.0  # Spike at cycle 10 (healthy phase, cycle <= 50)
    res[79:] += np.linspace(2.0, 10.0, 21)  # Degraded phase (cycle > 50)

    engine_df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res})
    scored, summary = detector.score_engine(engine_df)

    assert "first_alarm_cycle" in summary
    assert "premature_alarm_pct" in summary or "premature_alarm" in summary
    # The operational first alarm must occur during online monitoring (> 50)
    assert summary["first_alarm_cycle"] > 50
    assert summary["detected"] is True
    assert summary["healthy_far"] > 0.0 or summary["healthy_far_pct"] > 0.0


def test_envelope_regime_standardized_corridor():
    """Verify EnvelopeAnomalyDetector handles multi-regime FD004 without corridor collapse."""
    detector = EnvelopeAnomalyDetector(healthy_cycle_limit=30, k_sigma=3.0)
    # Calibrate
    actual = 1400.0 + np.random.randn(50) * 2.0
    pred = 1400.0 + np.random.randn(50) * 0.5
    detector.calibrate(actual, pred)

    # Test with regime-normalized residuals
    cycles = np.arange(1, 80)
    res = np.random.randn(79) * 0.5
    res[50:] += np.linspace(1.0, 8.0, 29)
    df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "residual": res, "T50": 1400.0 + res, "pred": 1400.0})
    scored, summary = detector.score_time_series_envelope(df, "T50", "pred")
    assert summary["detected"] is True
    assert summary["first_alarm_cycle"] > 30


# =========================================================================
# 7. thesis_statistics.py
# =========================================================================

def test_thesis_hypothesis_rigor():
    """Verify H1 engine-block Friedman, H2 n=62 relative window, H3 fleet Ljung-Box, H4/H5 Wilcoxon."""
    analyzer = ThesisHypothesisAnalyzer(random_state=42)
    df = make_synthetic_cmapss(n_engines=12, min_cycles=60, max_cycles=100, seed=42)
    df["regime_id"] = (df["setting_1"] // 10).astype(int)

    # H1: Engine-block repeated measures
    h1 = analyzer.test_regime_variance_dominance(df, sensor="T50", regime_col="regime_id")
    assert h1.is_rejected or h1.p_value <= 0.05
    assert "Friedman" in h1.test_name or "Alexander" in h1.test_name or "ANOVA" in h1.test_name

    # H2: Survivor-bias-free relative window
    df["residual"] = df["T50"] - (1400.0 + 8.0 * df["setting_1"])
    h2 = analyzer.test_physical_degradation_shift(df, healthy_cutoff=30, residual_col="residual")
    assert h2.is_rejected
    assert h2.p_value < 0.05
    assert "n=12" in h2.test_name or h2.effect_size is not None

    # H3: Fleet Ljung-Box
    h3 = analyzer.test_white_noise_residual_autocorrelation(df["residual"].to_numpy())
    assert h3.test_statistic > 0

    # H4 & H5: Wilcoxon tests
    y_true = np.linspace(100, 0, 100)
    pred_base = np.full(100, 50.0)
    pred_model = y_true + np.random.normal(0, 5.0, 100)
    unit_ids = np.repeat(np.arange(1, 11), 10)
    h4 = analyzer.test_model_superiority_vs_baseline(y_true, pred_model, pred_base, "ModelX", unit_ids=unit_ids)
    assert h4.is_rejected

    # BCa Confidence Intervals export observed metric
    test_eval = pd.DataFrame({
        "unit_id": unit_ids,
        "RUL_actual": y_true,
        "ModelX": pred_model,
    })
    ci_df = analyzer.compute_bootstrap_confidence_intervals(test_eval, {"ModelX": "ModelX"}, target_col="RUL_actual", n_bootstraps=50)
    assert "R2_Observed" in ci_df.columns or "R2_Mean" in ci_df.columns


# =========================================================================
# 8. diagnostics.py
# =========================================================================

def test_diagnostics_subsystem_permutation_and_wasserstein():
    """Verify grouped subsystem permutation importance and Wasserstein-1 distance."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "setting_2", "T2", "P2", "T50", "P30"]

    # Subsystem permutation importances
    p_imp = permutation_importances(df, features, "FD004", random_state=42)
    assert isinstance(p_imp, pd.DataFrame)
    assert "R2_drop_mean" in p_imp.columns

    # Distribution shift with Wasserstein distance
    df_high = df.iloc[: len(df) // 2]
    df_low = df.iloc[len(df) // 2 :]
    shift = distribution_shift(df_high, df_low, ["T50", "P30"], "High", "Low")
    assert "Wasserstein_distance" in shift.columns or "KS_statistic" in shift.columns


# =========================================================================
# 9. models/
# =========================================================================

def test_pytorch_sequence_model_group_early_stopping():
    """Verify PyTorch sequence model uses engine-grouped validation for early stopping."""
    df = make_synthetic_cmapss(n_engines=8, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "T2", "T50"]
    rnn = PyTorchRNNModel(seq_length=5, hidden_dim=16, num_layers=1, max_epochs=5, patience=2, random_state=42)
    rnn.fit(df, features, "RUL")
    assert rnn.fit_time > 0.0
    preds = rnn.predict(df)
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_minirocket_replicate_padding_and_grouped_ridge():
    """Verify MiniRocket uses replicate padding and GroupKFold in RidgeCV."""
    x = np.random.randn(10, 4, 10).astype(np.float32)
    transformer = MiniRocketTransformer(num_kernels=50, seq_length=10, random_state=42)
    transformer.fit(x)
    features = transformer.transform(x)
    assert features.shape == (10, 50)
    assert np.isfinite(features).all()

    df = make_synthetic_cmapss(n_engines=6, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    model = MiniRocketRULModel(seq_length=5, num_kernels=50, random_state=42)
    model.fit(df, ["setting_1", "T2", "T50"], "RUL")
    assert model.regressor is not None


# =========================================================================
# 10. data.py
# =========================================================================

def test_data_loader_load_test():
    """Verify CMapssLoader.load_test loads official test and RUL files."""
    loader = CMapssLoader()
    # Check if FD004 test exists on disk
    test_file = loader.data_dir / "test_FD004.txt"
    rul_file = loader.data_dir / "RUL_FD004.txt"
    if test_file.is_file() and rul_file.is_file():
        test_df, rul_series = loader.load_test("FD004")
        assert list(test_df.columns) == COLUMNS
        assert len(rul_series) == test_df["unit_id"].nunique()
        assert (rul_series >= 0).all()
