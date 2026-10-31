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

        # Conditional baseline: Mean RUL by cycle from training set
        cycle_mean_map = train_frame.groupby("cycle")[target_col].mean().to_dict()
        default_mean = float(y_train.mean())
        baseline_cycle = test_frame["cycle"].map(cycle_mean_map).fillna(default_mean).to_numpy(dtype=float)
        predictions["Baseline (Mean RUL by cycle)"] = baseline_cycle
        records.append({
            "Model": "Baseline (Mean RUL by cycle)",
            **evaluate_rul_predictions(y_test, baseline_cycle),
            "Fit_Time_s": 0.0,
        })

        models = self.get_models()
        total_models = len(models)
        for idx, (name, estimator) in enumerate(models.items(), start=1):
            if self.verbose:
                print(f"  [benchmark] [{idx}/{total_models}] Fitting '{name}' on {len(x_train)} samples...", flush=True)
            started = time.perf_counter()
            if hasattr(estimator, "fit") and not hasattr(estimator, "predict_proba") and type(estimator).__name__ in [
                "MiniRocketRULModel", "PyTorchRNNModel", "PyTorchLSTMModel", "PyTorchTransformerModel", "PyTorchCNNModel"
            ]:
                estimator.fit(train_frame, feature_cols, target_col)
                fit_seconds = time.perf_counter() - started
                if hasattr(estimator, "predict_values"):
                    prediction = estimator.predict_values(test_frame)
                else:
                    prediction = estimator.predict(test_frame)
            else:
                estimator.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - started
                prediction = estimator.predict(x_test)
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
        unique_groups = np.unique(groups)
        if n_splits < 2 or n_splits > len(unique_groups):
            raise ValueError(f"n_splits must be between 2 and {len(unique_groups)} engines")
        splitter = GroupKFold(n_splits=n_splits)
        records = []
        models = self.get_models()
        total_models = len(models)
        for idx, (name, prototype) in enumerate(models.items(), start=1):
            if self.verbose:
                print(f"  [grouped-cv] [{idx}/{total_models}] Running {n_splits}-fold grouped CV for '{name}'...", flush=True)
            fold_metrics = []
            oof_prediction = np.full(len(frame), np.nan)
            for train_idx, test_idx in splitter.split(x, y, groups):
                if set(groups[train_idx]).intersection(groups[test_idx]):
                    raise AssertionError("Engine leakage detected across cross-validation folds")
                if hasattr(prototype, "fit") and not hasattr(prototype, "predict_proba") and type(prototype).__name__ in [
                    "MiniRocketRULModel", "PyTorchRNNModel", "PyTorchLSTMModel", "PyTorchTransformerModel", "PyTorchCNNModel"
                ]:
                    import copy
                    model = copy.deepcopy(prototype)
                    train_sub = frame.iloc[train_idx].copy().reset_index(drop=True)
                    test_sub = frame.iloc[test_idx].copy().reset_index(drop=True)
                    model.fit(train_sub, feature_cols, target_col)
                    if hasattr(model, "predict_values"):
                        pred_fold = model.predict_values(test_sub)
                    else:
                        pred_fold = model.predict(test_sub)
                else:
                    model = clone(prototype).fit(x[train_idx], y[train_idx])
                    pred_fold = model.predict(x[test_idx])
                oof_prediction[test_idx] = pred_fold
                fold_metrics.append(evaluate_rul_predictions(y[test_idx], pred_fold))
            pooled = evaluate_rul_predictions(y, oof_prediction)
            summary_row = {
                "Model": name,
                "CV_R2_Mean": float(pooled["R2"]),
                "CV_RMSE_Mean": float(pooled["RMSE"]),
                "CV_MAE_Mean": float(pooled["MAE"]),
                "CV_NASA_Score_Mean": float(pooled["NASA_Score"]),
                "CV_Mean_NASA_Score": float(pooled["Mean_NASA_Score"]),
                "CV_R2_Std": float(np.std([item["R2"] for item in fold_metrics], ddof=1)),
                "CV_RMSE_Std": float(np.std([item["RMSE"] for item in fold_metrics], ddof=1)),
                "CV_MAE_Std": float(np.std([item["MAE"] for item in fold_metrics], ddof=1)),
                "CV_NASA_Score_Std": float(np.std([item["NASA_Score"] for item in fold_metrics], ddof=1)),
                "CV_OOF_Pooled_R2": float(pooled["R2"]),
                "CV_OOF_Pooled_RMSE": float(pooled["RMSE"]),
                "CV_OOF_Pooled_MAE": float(pooled["MAE"]),
                "CV_OOF_Pooled_NASA_Score": float(pooled["NASA_Score"]),
                "CV_OOF_Pooled_Mean_NASA_Score": float(pooled["Mean_NASA_Score"]),
                **{f"Fold_Mean_{m}": float(np.mean([item[m] for item in fold_metrics])) for m in ("R2", "RMSE", "MAE", "Mean_NASA_Score")},
                **{f"Fold_Std_{m}": float(np.std([item[m] for item in fold_metrics], ddof=1)) for m in ("R2", "RMSE", "MAE", "Mean_NASA_Score")},
                "CV_Aggregation": "pooled OOF; predictions are not clipped",
            }
            if self.verbose:
                print(f"  [grouped-cv] -> '{name}' CV Mean R²: {summary_row['CV_R2_Mean']:.4f} (±{summary_row['CV_R2_Std']:.4f}) | RMSE: {summary_row['CV_RMSE_Mean']:.2f}", flush=True)
            records.append(summary_row)
        return pd.DataFrame(records)
