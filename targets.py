"""RUL targets and regression metrics shared by both analyses."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .schema import DEFAULT_RUL_CAP


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


def nasa_scoring_function(y_true, y_pred) -> tuple[float, float]:
    """Return total and mean asymmetric PHM score, penalizing late predictions more."""
    difference = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    per_row = np.where(
        difference < 0,
        np.exp(-difference / 13.0) - 1.0,
        np.exp(difference / 10.0) - 1.0,
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


def evaluate_rul_predictions(y_true, y_pred) -> dict[str, float]:
    """Calculate symmetric regression metrics and asymmetric RUL costs."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    total, mean = nasa_scoring_function(truth, prediction)
    return {
        **{key: round(value, 4 if key == "R2" else 3) for key, value in regression_metrics(truth, prediction).items()},
        "NASA_Score": round(total, 2),
        "Mean_NASA_Score": round(mean, 3),
        "Late_Pred_Pct": round(float(np.mean(prediction > truth) * 100.0), 1),
    }
