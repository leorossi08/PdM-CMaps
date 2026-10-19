"""Healthy-window virtual sensor training and held-out fleet residuals."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from anomaly import StatisticalAnomalyDetector
from models.factory import get_model
from schema import DEFAULT_HEALTHY_CYCLE_LIMIT


class VirtualSensor:
    """Fit one regression model per operating regime using healthy training rows."""

    def __init__(
        self,
        target_sensor: str = "T50",
        feature_cols: list[str] | None = None,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        estimator=None,
        random_state: int = 42,
    ):
        self.target_sensor = target_sensor
        self.feature_cols = feature_cols
        self.healthy_cycle_limit = healthy_cycle_limit
        self.estimator = estimator if estimator is not None else get_model(
            "extra_trees", n_estimators=60, max_depth=12, random_state=random_state
        )
        self.models: dict[int, object] = {}

    def fit(self, train_frame: pd.DataFrame) -> "VirtualSensor":
        """Fit regime-specific estimators on cycles within the healthy window."""
        if "regime_id" not in train_frame:
            raise ValueError("train_frame must have regime_id assigned by a train-fitted regime model")
        features = self.feature_cols or [
            column for column in train_frame.columns
            if column not in {self.target_sensor, "unit_id", "cycle", "regime_id", "RUL", "subpart"}
        ]
        self.feature_cols = list(features)
        healthy = train_frame.loc[train_frame["cycle"] <= self.healthy_cycle_limit]
        self.models = {}
        for regime_id, rows in healthy.groupby("regime_id"):
            if len(rows) < 30:
                continue
            model = clone(self.estimator)
            model.fit(rows[self.feature_cols], rows[self.target_sensor])
            self.models[int(regime_id)] = model
        if not self.models:
            raise ValueError("No regime has at least 30 healthy training rows")
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return rows supported by a fitted regime model with prediction/residual columns."""
        if not self.models:
            raise RuntimeError("Fit the virtual sensor before predicting")
        if "regime_id" not in frame:
            raise ValueError("frame must have regime_id assigned")
        parts = []
        for regime_id, model in self.models.items():
            rows = frame.loc[frame["regime_id"] == regime_id].copy()
            if rows.empty:
                continue
            rows["prediction"] = model.predict(rows[self.feature_cols])
            rows["residual"] = rows[self.target_sensor] - rows["prediction"]
            parts.append(rows)
        if not parts:
            return frame.iloc[0:0].assign(prediction=pd.Series(dtype=float), residual=pd.Series(dtype=float))
        return pd.concat(parts).sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    def evaluate(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Fit, calibrate on held-out healthy cycles, then score lifecycle anomalies."""
        self.fit(train_frame)
        healthy_test = test_frame.loc[test_frame["cycle"] <= self.healthy_cycle_limit]
        calibration_predictions = self.predict(healthy_test)
        lifecycle_predictions = self.predict(test_frame)
        calibrated_parts = []
        for regime_id, rows in lifecycle_predictions.groupby("regime_id", sort=False):
            calibration_rows = calibration_predictions.loc[calibration_predictions["regime_id"] == regime_id]
            if calibration_rows.empty:
                continue
            detector = StatisticalAnomalyDetector(healthy_cycle_limit=self.healthy_cycle_limit)
            detector.calibrate(calibration_rows["residual"].to_numpy())
            part = rows.copy()
            part["z_score"] = detector.compute_z_scores(part["residual"].to_numpy())
            calibrated_parts.append(part)
        if not calibrated_parts:
            raise ValueError("No held-out healthy rows were available to calibrate residuals")
        scored = pd.concat(calibrated_parts, ignore_index=True).sort_values(["unit_id", "cycle"])
        scored["ewma_abs_z"] = np.nan
        records = []
        for unit_id, engine in scored.groupby("unit_id", sort=False):
            ordered = engine.sort_values("cycle")
            ewma = pd.Series(np.abs(ordered["z_score"].to_numpy())).ewm(alpha=0.35, adjust=False).mean().to_numpy()
            scored.loc[ordered.index, "ewma_abs_z"] = ewma
            healthy = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
            far = float(np.mean(np.abs(ordered["z_score"].to_numpy()[healthy]) >= 3.0)) if healthy.any() else 0.0
            alarm_positions = np.flatnonzero(ewma >= 3.0)
            eol = int(ordered["cycle"].max())
            alarm_cycle = int(ordered.iloc[alarm_positions[0]]["cycle"]) if alarm_positions.size else None
            records.append({
                "unit_id": int(unit_id), "eol_cycle": eol, "first_alarm_cycle": alarm_cycle,
                "lead_time_cycles": eol - alarm_cycle if alarm_cycle is not None else 0,
                "healthy_far": far, "detected": alarm_cycle is not None,
            })
        per_engine = pd.DataFrame(records)
        summary = {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
        }
        return scored.reset_index(drop=True), per_engine, summary
