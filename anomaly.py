"""Healthy-residual calibration and persistent EWMA anomaly detection."""

import numpy as np
import pandas as pd

from schema import DEFAULT_HEALTHY_CYCLE_LIMIT


class StatisticalAnomalyDetector:
    """Calibrate residual location/scale and summarize per-engine EWMA alarms."""

    def __init__(
        self,
        threshold_sigma: float = 3.0,
        ewma_alpha: float = 0.35,
        healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT,
    ):
        if threshold_sigma <= 0 or not 0 < ewma_alpha <= 1:
            raise ValueError("threshold_sigma must be positive and ewma_alpha in (0, 1]")
        self.threshold_sigma = threshold_sigma
        self.ewma_alpha = ewma_alpha
        self.healthy_cycle_limit = healthy_cycle_limit
        self.mu_e = 0.0
        self.sigma_e = 1.0

    def calibrate(self, healthy_residuals) -> "StatisticalAnomalyDetector":
        """Estimate mean and standard deviation from healthy calibration residuals."""
        residuals = np.asarray(healthy_residuals, dtype=float)
        if residuals.size == 0 or not np.isfinite(residuals).all():
            raise ValueError("healthy_residuals must be a non-empty finite array")
        self.mu_e = float(residuals.mean())
        self.sigma_e = float(residuals.std())
        if self.sigma_e < 1e-8:
            self.sigma_e = 1.0
        return self

    def compute_z_scores(self, residuals) -> np.ndarray:
        """Standardize residuals against the fitted healthy baseline."""
        return (np.asarray(residuals, dtype=float) - self.mu_e) / self.sigma_e

    def score_engine(self, engine_frame: pd.DataFrame, residual_col: str = "residual") -> tuple[pd.DataFrame, dict]:
        """Add Z/EWMA columns and return the first alarm and healthy-window FAR."""
        ordered = engine_frame.sort_values("cycle").copy().reset_index(drop=True)
        z = self.compute_z_scores(ordered[residual_col].to_numpy())
        ewma = pd.Series(np.abs(z)).ewm(alpha=self.ewma_alpha, adjust=False).mean().to_numpy()
        healthy = ordered["cycle"].to_numpy() <= self.healthy_cycle_limit
        far = float(np.mean(np.abs(z[healthy]) >= self.threshold_sigma)) if healthy.any() else 0.0
        alarms = np.flatnonzero(ewma >= self.threshold_sigma)
        first_cycle = int(ordered.iloc[alarms[0]]["cycle"]) if alarms.size else None
        eol = int(ordered["cycle"].max())
        ordered["z_score"] = z
        ordered["ewma_abs_z"] = ewma
        summary = {
            "unit_id": int(ordered["unit_id"].iloc[0]),
            "eol_cycle": eol,
            "first_alarm_cycle": first_cycle,
            "lead_time_cycles": eol - first_cycle if first_cycle is not None else 0,
            "healthy_far": far,
            "detected": first_cycle is not None,
        }
        return ordered, summary

    def evaluate_fleet(self, fleet_frame: pd.DataFrame, residual_col: str = "residual") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Return scored residual rows, per-engine results, and fleet summary."""
        scored = []
        summaries = []
        for _, engine in fleet_frame.groupby("unit_id", sort=False):
            engine_scores, summary = self.score_engine(engine, residual_col)
            scored.append(engine_scores)
            summaries.append(summary)
        per_engine = pd.DataFrame(summaries)
        return pd.concat(scored, ignore_index=True), per_engine, {
            "total_engines": len(per_engine),
            "detection_rate_pct": float(per_engine["detected"].mean() * 100.0),
            "mean_lead_time_cycles": float(per_engine["lead_time_cycles"].mean()),
            "mean_healthy_false_alarm_pct": float(per_engine["healthy_far"].mean() * 100.0),
        }
