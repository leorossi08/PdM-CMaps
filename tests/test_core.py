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
