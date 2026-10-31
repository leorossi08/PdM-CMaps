"""RUL targets and regression metrics shared by both analyses."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from schema import DEFAULT_RUL_CAP


def compute_piecewise_rul(
    frame: pd.DataFrame,
    rul_max: int = DEFAULT_RUL_CAP,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.Series:
    """Compute min(RUL cap, engine final cycle minus current cycle)."""
    if rul_max < 1:
        raise ValueError("rul_max must be positive")
    end_of_life = frame.groupby(unit_col)[cycle_col].transform("max")
    return (end_of_life - frame[cycle_col]).clip(lower=0, upper=rul_max).astype(float)


def clip_predictions(y_pred, lower: float = 0.0, upper: float = float(DEFAULT_RUL_CAP)) -> np.ndarray:
    """Clip RUL predictions to physical bounds [lower, upper]."""
    return np.clip(np.asarray(y_pred, dtype=float), lower, upper)


def nasa_scoring_function(y_true, y_pred) -> tuple[float, float]:
    """Return total and mean asymmetric PHM score with overflow protection."""
    difference = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    diff_clipped = np.clip(difference, -500.0, 500.0)
    per_row = np.where(
        diff_clipped < 0,
        np.exp(-diff_clipped / 13.0) - 1.0,
        np.exp(diff_clipped / 10.0) - 1.0,
    )
    return float(per_row.sum()), float(per_row.mean())


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R², RMSE, MAE, and target variance on the scored observations."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "target_variance": float(np.var(truth)),
    }


def evaluate_rul_predictions(
    y_true,
    y_pred,
    clip: bool = False,
    rul_cap: float = float(DEFAULT_RUL_CAP),
) -> dict[str, float]:
    """Calculate symmetric regression metrics and asymmetric RUL costs with optional clipping."""
    truth = np.asarray(y_true, dtype=float)
    prediction = clip_predictions(y_pred, 0.0, rul_cap) if clip else np.asarray(y_pred, dtype=float)
    total, mean = nasa_scoring_function(truth, prediction)
    return {
        **{key: round(value, 6) for key, value in regression_metrics(truth, prediction).items()},
        "NASA_Score": round(total, 2),
        "Mean_NASA_Score": round(mean, 3),
        "Late_Pred_Pct": round(float(np.mean(prediction > truth) * 100.0), 1),
    }


def evaluate_stratified_rul(
    y_true,
    y_pred,
    critical_threshold: float = 50.0,
    rul_cap: float = float(DEFAULT_RUL_CAP),
) -> dict[str, dict[str, float]]:
    """Evaluate RUL predictions across full holdout and critical degradation zone (RUL < critical_threshold)."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    overall = evaluate_rul_predictions(truth, prediction, clip=False)
    overall["N_samples"] = len(truth)

    critical_mask = truth < critical_threshold
    if critical_mask.any():
        critical_zone = evaluate_rul_predictions(truth[critical_mask], prediction[critical_mask], clip=False)
        critical_zone["N_samples"] = int(np.sum(critical_mask))
    else:
        critical_zone = {"N_samples": 0}

    return {
        "overall": overall,
        "critical_zone": critical_zone,
    }
