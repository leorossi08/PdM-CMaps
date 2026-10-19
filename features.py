"""Per-engine temporal features that do not cross engine boundaries."""

import pandas as pd


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
        result[f"{sensor}_lag2"] = lag2.fillna(result[sensor])
        result[f"{sensor}_delta"] = (result[sensor] - lag1).fillna(0.0)
        for window in windows:
            result[f"{sensor}_rollmean_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(result[sensor])
            result[f"{sensor}_rollstd_{window}"] = grouped.transform(
                lambda values: values.shift(1).rolling(window, min_periods=2).std()
            ).fillna(0.0)
    return result


def apply_causal_target_smoothing(
    frame: pd.DataFrame,
    target_sensor: str,
    windows: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    """Add within-engine trailing means that include the current available reading."""
    result = frame.sort_values(["unit_id", "cycle"]).copy().reset_index(drop=True)
    grouped = result.groupby("unit_id", sort=False)[target_sensor]
    for window in windows:
        result[f"{target_sensor}_smooth_{window}"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
    return result
