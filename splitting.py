"""Engine-level data partitions and grouped cross-validation helpers."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE


def split_engine_units(
    frame: pd.DataFrame,
    test_ratio: float = 0.25,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted train/test engine IDs with no engine overlap."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between zero and one")
    units = np.asarray(frame["unit_id"].unique()).copy()
    if len(units) < 2:
        raise ValueError("At least two engines are required for a holdout split")
    np.random.default_rng(random_state).shuffle(units)
    n_test = min(len(units) - 1, max(1, int(round(len(units) * test_ratio))))
    return np.sort(units[n_test:]), np.sort(units[:n_test])


def partition_healthy_and_lifecycle(
    frame: pd.DataFrame,
    train_units: np.ndarray,
    test_units: np.ndarray,
    healthy_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
) -> dict[str, pd.DataFrame]:
    """Create healthy-window and full-lifecycle frames for each engine partition."""
    overlap = set(train_units).intersection(test_units)
    if overlap:
        raise ValueError(f"Train/test engine overlap: {sorted(overlap)}")
    ordered = frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    train_mask = ordered["unit_id"].isin(train_units)
    test_mask = ordered["unit_id"].isin(test_units)
    return {
        "healthy_train": ordered.loc[train_mask & (ordered["cycle"] <= healthy_limit)].copy().reset_index(drop=True),
        "healthy_test": ordered.loc[test_mask & (ordered["cycle"] <= healthy_limit)].copy().reset_index(drop=True),
        "full_lifecycle_train": ordered.loc[train_mask].copy().reset_index(drop=True),
        "full_lifecycle_test": ordered.loc[test_mask].copy().reset_index(drop=True),
    }


def grouped_folds(frame: pd.DataFrame, n_splits: int = 5):
    """Yield GroupKFold indices while preserving each engine as an indivisible group."""
    groups = frame["unit_id"].to_numpy()
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError(f"n_splits must be between 2 and {len(unique_groups)} engines")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(frame))
    for train_idx, test_idx in splitter.split(placeholder, groups=groups):
        if set(groups[train_idx]).intersection(groups[test_idx]):
            raise AssertionError("Grouped fold split an engine across train and test")
        yield train_idx, test_idx
