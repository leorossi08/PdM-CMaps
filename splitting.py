"""Engine-level data partitions and grouped cross-validation helpers."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE


def split_engine_units(
    frame: pd.DataFrame,
    test_ratio: float = 0.25,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify_eol: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted train/test engine IDs with no engine overlap and optional EOL stratification."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between zero and one")
    units = np.asarray(frame["unit_id"].unique()).copy()
    if len(units) < 2:
        raise ValueError("At least two engines are required for a holdout split")

    if stratify_eol:
        eol_series = frame.groupby("unit_id")["cycle"].max().loc[units]
        n_unique_eol = eol_series.nunique()
        if n_unique_eol > 1:
            q = min(5, len(units) // 4)
            bins = None
            if q > 1:
                try:
                    bins = pd.qcut(eol_series, q=q, labels=False, duplicates="drop")
                except Exception:
                    bins = None
            if bins is not None and bins.nunique() > 1:
                rng = np.random.default_rng(random_state)
                train_list, test_list = [], []
                for _, b_units in bins.groupby(bins):
                    u_arr = b_units.index.to_numpy().copy()
                    rng.shuffle(u_arr)
                    n_b_test = max(0, int(round(len(u_arr) * test_ratio)))
                    test_list.extend(u_arr[:n_b_test])
                    train_list.extend(u_arr[n_b_test:])
                if not test_list and len(train_list) > 1:
                    test_list.append(train_list.pop())
                elif not train_list and len(test_list) > 1:
                    train_list.append(test_list.pop())
                return np.sort(np.asarray(train_list)), np.sort(np.asarray(test_list))

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
