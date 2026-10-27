"""Reusable holdout and engine-grouped model evaluation."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from models.factory import get_model_pool
from schema import DEFAULT_RANDOM_STATE
from targets import evaluate_rul_predictions, regression_metrics


def grouped_oof(
    frame: pd.DataFrame,
    features: list[str],
    estimator=None,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    target_col: str = "RUL",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit an estimator in GroupKFold and return fold metrics and aligned OOF predictions."""
    x = frame[features].to_numpy(dtype=float)
    y = frame[target_col].to_numpy(dtype=float)
    groups = frame["unit_id"].to_numpy()
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError(f"n_splits must be between 2 and {len(unique_groups)} engines")
    splitter = GroupKFold(n_splits=n_splits)
    prediction = np.full(len(frame), np.nan)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if set(groups[train_idx]).intersection(groups[valid_idx]):
            raise AssertionError("Engine leakage detected in grouped fold")
        model = clone(estimator) if estimator is not None else get_model_pool(random_state)["Extra Trees"]
        model.fit(x[train_idx], y[train_idx])
        prediction[valid_idx] = model.predict(x[valid_idx])
        rows.append({
            "Fold": fold,
            "Engines": int(np.unique(groups[valid_idx]).size),
            **regression_metrics(y[valid_idx], prediction[valid_idx]),
        })
    if not np.isfinite(prediction).all():
        raise AssertionError("Grouped OOF predictions do not cover every row")
    return pd.DataFrame(rows), prediction


class RULBenchmark:
    """Compare the shared RUL model pool using holdout or grouped-fold evaluation."""

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, model_pool=None, verbose: bool = True):
        self.random_state = random_state
        self.model_pool = model_pool
        self.verbose = verbose

    def get_models(self) -> dict:
        """Return fresh model instances for each benchmark run."""
        if self.model_pool is not None:
            return {name: clone(model) for name, model in self.model_pool.items()}
        return get_model_pool(self.random_state)

    def benchmark_train_test(
        self,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "RUL",
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Fit models on training engines and score a disjoint held-out fleet."""
        x_train = train_frame[feature_cols].to_numpy(dtype=float)
        y_train = train_frame[target_col].to_numpy(dtype=float)
        x_test = test_frame[feature_cols].to_numpy(dtype=float)
        y_test = test_frame[target_col].to_numpy(dtype=float)
        predictions = {"y_true": y_test}
        baseline = np.full(len(y_test), y_train.mean())
        predictions["Baseline (Mean RUL)"] = baseline
        records = [{"Model": "Baseline (Mean RUL)", **evaluate_rul_predictions(y_test, baseline), "Fit_Time_s": 0.0}]
        models = self.get_models()
        total_models = len(models)
        for idx, (name, estimator) in enumerate(models.items(), start=1):
            if self.verbose:
                print(f"  [benchmark] [{idx}/{total_models}] Fitting '{name}' on {len(x_train)} samples...", flush=True)
            started = time.perf_counter()
            estimator.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            prediction = np.clip(estimator.predict(x_test), 0.0, None)
            predictions[name] = prediction
            metrics = evaluate_rul_predictions(y_test, prediction)
            if self.verbose:
                print(f"  [benchmark] -> '{name}' done in {fit_seconds:.2f}s | R²: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}", flush=True)
            records.append({"Model": name, **metrics, "Fit_Time_s": round(fit_seconds, 3)})
        return pd.DataFrame(records), predictions

    def cross_validate(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "RUL",
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Report mean and standard deviation of metrics across engine-grouped folds."""
        x = frame[feature_cols].to_numpy(dtype=float)
        y = frame[target_col].to_numpy(dtype=float)
        groups = frame["unit_id"].to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        records = []
        models = self.get_models()
        total_models = len(models)
        for idx, (name, prototype) in enumerate(models.items(), start=1):
            if self.verbose:
                print(f"  [grouped-cv] [{idx}/{total_models}] Running {n_splits}-fold grouped CV for '{name}'...", flush=True)
            fold_metrics = []
            for train_idx, test_idx in splitter.split(x, y, groups):
                model = clone(prototype).fit(x[train_idx], y[train_idx])
                prediction = np.clip(model.predict(x[test_idx]), 0.0, None)
                fold_metrics.append(evaluate_rul_predictions(y[test_idx], prediction))
            summary_row = {
                "Model": name,
                **{f"CV_{metric}_Mean": float(np.mean([item[metric] for item in fold_metrics])) for metric in ("R2", "RMSE", "MAE", "NASA_Score")},
                **{f"CV_{metric}_Std": float(np.std([item[metric] for item in fold_metrics], ddof=1)) for metric in ("R2", "RMSE", "MAE", "NASA_Score")},
            }
            if self.verbose:
                print(f"  [grouped-cv] -> '{name}' CV Mean R²: {summary_row['CV_R2_Mean']:.4f} (±{summary_row['CV_R2_Std']:.4f}) | RMSE: {summary_row['CV_RMSE_Mean']:.2f}", flush=True)
            records.append(summary_row)
        return pd.DataFrame(records)
