"""Per-engine temporal features that do not cross engine boundaries."""

import numpy as np
import pandas as pd


class CausalFeatureEngineer:
    """Stateful temporal feature engineer that learns healthy baseline variance for startup imputation."""

    def __init__(self, sensor_cols: list[str], windows: tuple[int, ...] = (3, 5)):
        self.sensor_cols = list(sensor_cols)
        self.windows = tuple(windows)
        self.healthy_rollstd_medians_: dict[tuple[str, int], float] = {}

    def fit(self, train_frame: pd.DataFrame, healthy_cycle_limit: int = 50) -> "CausalFeatureEngineer":
        """Compute median rolling std in healthy training phase for initial imputation."""
        healthy = train_frame.loc[train_frame["cycle"] <= healthy_cycle_limit].sort_values(["unit_id", "cycle"])
        for sensor in self.sensor_cols:
            grouped = healthy.groupby("unit_id", sort=False)[sensor]
            for window in self.windows:
                roll_std = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=2).std()
                ).dropna()
                median_std = float(roll_std.median()) if not roll_std.empty else 0.01
                if not np.isfinite(median_std) or median_std < 1e-6:
                    median_std = 0.01
                self.healthy_rollstd_medians_[(sensor, window)] = median_std
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add lags, deltas, startup indicator, and rolling metrics with learned startup imputation."""
        result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
        result["is_startup"] = (result["cycle"] <= 2).astype(float)
        for sensor in self.sensor_cols:
            grouped = result.groupby("unit_id", sort=False)[sensor]
            lag1 = grouped.shift(1)
            lag2 = grouped.shift(2)
            result[f"{sensor}_lag1"] = lag1.fillna(result[sensor])
            # Boundary fix at t=2: fill lag2 with lag1 first, then current sensor
            result[f"{sensor}_lag2"] = lag2.fillna(lag1).fillna(result[sensor])
            result[f"{sensor}_delta"] = (result[sensor] - lag1).fillna(0.0)
            for window in self.windows:
                fallback_std = self.healthy_rollstd_medians_.get((sensor, window), 0.0)
                result[f"{sensor}_rollmean_{window}"] = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(result[sensor])
                result[f"{sensor}_rollstd_{window}"] = grouped.transform(
                    lambda values: values.shift(1).rolling(window, min_periods=2).std()
                ).fillna(fallback_std)
        return result

    def fit_transform(self, train_frame: pd.DataFrame, healthy_cycle_limit: int = 50) -> pd.DataFrame:
        """Fit on train_frame and transform it."""
        return self.fit(train_frame, healthy_cycle_limit=healthy_cycle_limit).transform(train_frame)


def engineer_causal_temporal_features(
    frame: pd.DataFrame,
    sensor_cols: list[str],
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    """Add lags, deltas, and trailing statistics; startup values use current/past data only."""
    result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
    for sensor in sensor_cols:
        grouped = result.groupby("unit_id", sort=False)[sensor]
        lag1 = grouped.shift(1)
        lag2 = grouped.shift(2)
        result[f"{sensor}_lag1"] = lag1.fillna(result[sensor])
        # Boundary fix at t=2: fill lag2 with lag1 (x at t=1), NOT x at t=2
        result[f"{sensor}_lag2"] = lag2.fillna(lag1).fillna(result[sensor])
        result[f"{sensor}_delta"] = (result[sensor] - lag1).fillna(0.0)
        for window in windows:
            result[f"{sensor}_rollmean_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(result[sensor])
            result[f"{sensor}_rollstd_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=2).std()
            ).fillna(0.0)
    return result
