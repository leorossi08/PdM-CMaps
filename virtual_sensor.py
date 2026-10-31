"""Healthy-window virtual sensor training and held-out fleet residuals."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from anomaly import StatisticalAnomalyDetector
from models.factory import get_model
from schema import DEFAULT_HEALTHY_CYCLE_LIMIT

CANONICAL_EXOGENOUS_FEATURES = [
    "setting_1", "setting_2", "setting_3", "T2", "P2", "P15", "epr", "Nf_dmd", "PCNfR_dmd"
]


class VirtualSensor:
    """Fit one regression model per operating regime using healthy training rows."""

    def __init__(
        self,
        target_sensor: str = "T50",
        feature_cols: list[str] | None = None,
        mode: str = "exogenous",
        threshold_sigma: float = 3.0,
        ewma_alpha: float = 0.35,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
        estimator=None,
        random_state: int = 42,
    ):
        self.target_sensor = target_sensor
        self.feature_cols = feature_cols
        self.mode = mode
        self.threshold_sigma = threshold_sigma
        self.ewma_alpha = ewma_alpha
        self.healthy_cycle_limit = healthy_cycle_limit
        self.estimator = estimator if estimator is not None else get_model(
            "extra_trees", n_estimators=60, max_depth=12, random_state=random_state
        )
        self.models: dict[int, object] = {}
        self.detectors: dict[int, StatisticalAnomalyDetector] = {}
        self.ewma_detector: StatisticalAnomalyDetector | None = None
        self.calibration_predictions_: pd.DataFrame | None = None
        self.fallback_model: object | None = None
        self.fallback_detector: StatisticalAnomalyDetector | None = None

    def fit(self, train_frame: pd.DataFrame) -> "VirtualSensor":
        """Fit regime-specific estimators on cycles within the healthy window with hierarchical fallback."""
        if "regime_id" not in train_frame:
            raise ValueError("train_frame must have regime_id assigned by a train-fitted regime model")
        if self.feature_cols is not None:
            features = list(self.feature_cols)
        elif self.mode == "exogenous":
            features = [c for c in CANONICAL_EXOGENOUS_FEATURES if c in train_frame.columns and c != self.target_sensor]
            if not features:
                features = [
                    column for column in train_frame.columns
                    if column not in {self.target_sensor, "unit_id", "cycle", "regime_id", "RUL", "subpart"}
                ]
        else:
            features = [
                column for column in train_frame.columns
                if column not in {self.target_sensor, "unit_id", "cycle", "regime_id", "RUL", "subpart"}
            ]
        self.feature_cols = list(features)
        healthy = train_frame.loc[train_frame["cycle"] <= self.healthy_cycle_limit]
        if healthy.empty:
            healthy = train_frame

        # Train global hierarchical fallback model across all healthy data
        self.fallback_model = clone(self.estimator)
        self.fallback_model.fit(healthy[self.feature_cols], healthy[self.target_sensor].to_numpy(dtype=float))
        fb_resid = healthy[self.target_sensor].to_numpy(dtype=float) - self.fallback_model.predict(healthy[self.feature_cols])
        self.fallback_detector = StatisticalAnomalyDetector(
            threshold_sigma=self.threshold_sigma,
            ewma_alpha=self.ewma_alpha,
            healthy_cycle_limit=self.healthy_cycle_limit,
        )
        self.fallback_detector.calibrate(fb_resid)

        self.models = {}
        self.detectors = {}
        calibration_parts = []

        all_regimes = sorted(train_frame["regime_id"].unique())
        for regime_id in all_regimes:
            rows = healthy.loc[healthy["regime_id"] == regime_id]
            groups = rows["unit_id"].to_numpy() if not rows.empty else np.array([])
            n_engines = np.unique(groups).size if groups.size else 0

            if len(rows) < 30 or n_engines < 2:
                warnings.warn(
                    f"Regime {regime_id} has {len(rows)} samples and {n_engines} engines (< 30 samples or < 2 engines); "
                    "using global hierarchical fallback estimator.",
                    UserWarning,
                )
                self.models[int(regime_id)] = self.fallback_model
                self.detectors[int(regime_id)] = self.fallback_detector
                if not rows.empty:
                    oof_rows = rows.copy()
                    pred = self.fallback_model.predict(rows[self.feature_cols])
                    oof_rows["prediction"] = pred
                    oof_rows["residual"] = rows[self.target_sensor] - pred
                    calibration_parts.append(oof_rows)
                continue

            n_splits = min(5, n_engines)
            oof_prediction = np.full(len(rows), np.nan)
            reg_features = rows[self.feature_cols]
            target = rows[self.target_sensor].to_numpy(dtype=float)

            for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(reg_features, target, groups):
                oof_model = clone(self.estimator)
                oof_model.fit(reg_features.iloc[train_idx], target[train_idx])
                oof_prediction[valid_idx] = oof_model.predict(reg_features.iloc[valid_idx])

            if not np.isfinite(oof_prediction).all():
                nan_mask = ~np.isfinite(oof_prediction)
                oof_prediction[nan_mask] = self.fallback_model.predict(reg_features.iloc[nan_mask])

            model = clone(self.estimator)
            model.fit(reg_features, target)
            self.models[int(regime_id)] = model
            oof_rows = rows.copy()
            oof_rows["prediction"] = oof_prediction
            oof_rows["residual"] = target - oof_prediction
            calibration_parts.append(oof_rows)
            detector = StatisticalAnomalyDetector(
                threshold_sigma=self.threshold_sigma,
                ewma_alpha=self.ewma_alpha,
                healthy_cycle_limit=self.healthy_cycle_limit,
            )
            detector.calibrate(oof_rows["residual"].to_numpy())
            self.detectors[int(regime_id)] = detector

        if not self.models:
            raise ValueError("No regime could be fitted or calibrated")

        if calibration_parts:
            self.calibration_predictions_ = pd.concat(calibration_parts, ignore_index=True).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
            calibration_z = []
            for regime_id, rows in self.calibration_predictions_.groupby("regime_id", sort=False):
                det = self.detectors.get(int(regime_id), self.fallback_detector)
                z_scores = det.compute_z_scores(rows["residual"].to_numpy())
                calibration_z.append(z_scores)
            self.ewma_detector = StatisticalAnomalyDetector(
                threshold_sigma=self.threshold_sigma,
                ewma_alpha=self.ewma_alpha,
                healthy_cycle_limit=self.healthy_cycle_limit,
            )
            self.ewma_detector.calibrate_ewma_distribution(np.concatenate(calibration_z))
        else:
            self.ewma_detector = self.fallback_detector

        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return rows supported by a fitted regime model with prediction/residual columns."""
        if not self.models and self.fallback_model is None:
            raise RuntimeError("Fit the virtual sensor before predicting")
        if "regime_id" not in frame:
            raise ValueError("frame must have regime_id assigned")
        parts = []
        for regime_id, rows in frame.groupby("regime_id"):
            model = self.models.get(int(regime_id), self.fallback_model)
            part = rows.copy()
            part["prediction"] = model.predict(part[self.feature_cols])
            part["residual"] = part[self.target_sensor] - part["prediction"]
            parts.append(part)
        if not parts:
            return frame.iloc[0:0].assign(prediction=pd.Series(dtype=float), residual=pd.Series(dtype=float))
        result = pd.concat(parts).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        return result

    def evaluate(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Fit on training engines and score lifecycle anomalies on held-out engines."""
        self.fit(train_frame)
        lifecycle_predictions = self.predict(test_frame)
        calibrated_parts = []
        for regime_id, rows in lifecycle_predictions.groupby("regime_id", sort=False):
            detector = self.detectors.get(int(regime_id), self.fallback_detector)
            part = rows.copy()
            part["z_score"] = detector.compute_z_scores(part["residual"].to_numpy())
            calibrated_parts.append(part)
        if not calibrated_parts:
            raise ValueError("No held-out healthy rows were available to calibrate residuals")
        scored = pd.concat(calibrated_parts, ignore_index=True).sort_values(["unit_id", "cycle"])
        if self.ewma_detector is None:
            raise RuntimeError("Fit the virtual sensor before scoring EWMA")
        scored["ewma_abs_z"] = np.nan
        records = []
        for unit_id, engine in scored.groupby("unit_id", sort=False):
            ordered = engine.sort_values("cycle")
            ewma = self.ewma_detector.ewma_abs_z(ordered["z_score"].to_numpy())
            scored.loc[ordered.index, "ewma_abs_z"] = ewma
            limits = self.ewma_detector.ewma_upper_limits(len(ewma))
            scored.loc[ordered.index, "ewma_ucl"] = limits

            healthy_mask = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
            monitoring_mask = ordered["cycle"].to_numpy() > self.healthy_cycle_limit

            far = float(np.mean(ewma[healthy_mask] >= limits[healthy_mask])) if healthy_mask.any() else 0.0
            premature = bool(np.any(ewma[healthy_mask] >= limits[healthy_mask])) if healthy_mask.any() else False

            # Online alarms restricted to cycle > healthy_cycle_limit
            online_alarm_positions = np.flatnonzero((ewma >= limits) & monitoring_mask)
            eol = int(ordered["cycle"].max())
            alarm_cycle = int(ordered.iloc[online_alarm_positions[0]]["cycle"]) if online_alarm_positions.size else None
            lead_time = eol - alarm_cycle if alarm_cycle is not None else 0

            records.append({
                "unit_id": int(unit_id),
                "eol_cycle": eol,
                "first_alarm_cycle": alarm_cycle,
                "lead_time_cycles": lead_time,
                "healthy_far": far,
                "premature_alarm": premature,
                "detected": alarm_cycle is not None,
            })
        per_engine = pd.DataFrame(records)
        detected_engines = per_engine.loc[per_engine["detected"]]
        mean_lead_time_detected = float(detected_engines["lead_time_cycles"].mean()) if not detected_engines.empty else 0.0

        summary = {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_lead_time_detected_cycles": mean_lead_time_detected,
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
            "premature_alarm_pct": float(per_engine["premature_alarm"].mean() * 100.0),
        }
        return scored.reset_index(drop=True), per_engine, summary
