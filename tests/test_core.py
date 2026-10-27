"""Focused contracts for shared C-MAPSS data and analysis components."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from data import CMapssLoader
from diagnostics import feature_signal
from evaluation import grouped_oof
from features import engineer_causal_temporal_features
from regimes import RegimeDecoupler
from splitting import split_engine_units
from targets import compute_piecewise_rul, nasa_scoring_function
from schema import COLUMNS, SENSOR_COLUMNS


def sample_frame(engine_count: int = 10) -> pd.DataFrame:
    rows = []
    for unit_id in range(1, engine_count + 1):
        for cycle in range(1, 6):
            row = {column: 0.0 for column in COLUMNS}
            row.update({"unit_id": unit_id, "cycle": cycle, "setting_1": unit_id % 3, "setting_2": 0.2, "setting_3": 100.0})
            for index, sensor in enumerate(SENSOR_COLUMNS):
                row[sensor] = float(unit_id + cycle * (index + 1))
            rows.append(row)
    return pd.DataFrame(rows)


def test_loader_reads_named_fd001_schema():
    frame = CMapssLoader().load("FD001")
    assert list(frame.columns) == COLUMNS
    assert frame["unit_id"].nunique() == 100


def test_engine_holdout_has_no_shared_units():
    frame = sample_frame()
    train, test = split_engine_units(frame, test_ratio=0.3, random_state=4)
    assert not set(train).intersection(test)
    assert len(train) + len(test) == frame["unit_id"].nunique()


def test_capped_rul_reaches_zero_and_caps_at_limit():
    frame = sample_frame(engine_count=2)
    rul = compute_piecewise_rul(frame, rul_max=3)
    assert rul.max() == 3.0
    assert rul.min() == 0.0


def test_temporal_features_do_not_cross_engine_boundaries():
    frame = sample_frame(engine_count=2)
    features = engineer_causal_temporal_features(frame, ["T50"], windows=(3,))
    first_rows = features.groupby("unit_id", sort=False).head(1)
    assert (first_rows["T50_lag1"] == first_rows["T50"]).all()
    assert (first_rows["T50_delta"] == 0.0).all()


def test_variance_decomposition_adds_to_total():
    frame = sample_frame()
    frame["regime_id"] = frame["unit_id"] % 3
    result = RegimeDecoupler.variance_decomposition(frame, "T50")
    assert np.isclose(result["total_var"], result["between_regime_var"] + result["within_regime_var"])
    assert np.isclose(result["pct_between_regime"] + result["pct_within_regime"], 100.0)
    assert result["Between_regime_pct"] == result["pct_between_regime"]
    assert result["Within_regime_pct"] == result["pct_within_regime"]
    assert 0.0 <= result["regime_baseline_r2"] <= 1.0


def test_late_rul_error_has_larger_nasa_penalty():
    truth = np.array([50.0])
    early, _ = nasa_scoring_function(truth, np.array([45.0]))
    late, _ = nasa_scoring_function(truth, np.array([55.0]))
    assert late > early


def test_grouped_oof_covers_rows_without_engine_leakage():
    frame = sample_frame(engine_count=10)
    frame["RUL"] = compute_piecewise_rul(frame)
    features = ["cycle", "setting_1", "T2"]
    estimator = ExtraTreesRegressor(n_estimators=5, random_state=2)
    folds, prediction = grouped_oof(frame, features, estimator=estimator, n_splits=5)
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction).all()
    assert len(folds) == 5


def test_feature_signal_handles_constant_features_without_runtime_warning():
    frame = sample_frame(engine_count=3)
    frame["RUL"] = compute_piecewise_rul(frame)
    frame["constant"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = feature_signal(frame, ["constant", "cycle"], "test")
    constant_correlation = result.loc[result["Feature"] == "constant", "Abs_correlation"].iloc[0]
    assert np.isnan(constant_correlation)
    assert np.isfinite(result.loc[result["Feature"] == "cycle", "Abs_correlation"]).all()


def test_loader_rejects_duplicate_engine_cycle_rows():
    frame = sample_frame(engine_count=2)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate engine-cycle"):
        CMapssLoader.validate(duplicated)


def test_model_pool_includes_padua_algorithms():
    from models.factory import get_model, get_model_pool

    pool = get_model_pool()
    assert "CatBoost" in pool
    assert "XGBoost" in pool
    assert "LightGBM" in pool

    X = np.random.randn(20, 4)
    y = np.random.randn(20)

    for name in ["catboost", "xgboost", "lightgbm"]:
        model = get_model(name, n_estimators=5, max_depth=3, iterations=5, depth=3)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)
        assert np.isfinite(pred).all()


def test_envelope_anomaly_detector():
    from anomaly import EnvelopeAnomalyDetector

    detector = EnvelopeAnomalyDetector(k_sigma=2.5, healthy_cycle_limit=30, window_size=4, min_consecutive_breaches=2)
    np.random.seed(42)
    healthy_actual = 1400.0 + np.random.randn(30) * 1.5
    healthy_pred = 1400.0 + np.random.randn(30) * 0.2
    detector.calibrate(healthy_actual, healthy_pred)

    assert detector.sigma_e > 0
    assert detector.mu_amp > 0

    cycles = np.arange(1, 101)
    actual = 1400.0 + np.random.randn(100) * 1.5
    # Add severe thermal runaway / degradation starting at cycle 75
    actual[74:] += np.linspace(4.0, 30.0, 26)
    pred = np.full(100, 1400.0)

    df = pd.DataFrame({"unit_id": 1, "cycle": cycles, "T50": actual, "pred": pred})
    scored_env, summary_env = detector.score_time_series_envelope(df, "T50", "pred")

    assert "envelope_upper" in scored_env.columns
    assert "envelope_lower" in scored_env.columns
    assert summary_env["detected"] is True
    assert summary_env["first_alarm_cycle"] is not None
    assert summary_env["first_alarm_cycle"] >= 70
    assert summary_env["eol_cycle"] == 100
    assert summary_env["lead_time_cycles"] == 100 - summary_env["first_alarm_cycle"]

    scored_geom, summary_geom = detector.score_geometric_envelope(df, "T50")
    assert "geom_upper_hull" in scored_geom.columns
    assert "geom_lower_hull" in scored_geom.columns
    assert summary_geom["detected"] is True
    assert summary_geom["first_alarm_cycle"] is not None

