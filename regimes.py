"""Operating-regime clustering and sensor variance decomposition."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from schema import DEFAULT_RANDOM_STATE, SETTING_COLUMNS


class RegimeDecoupler:
    """Fit operating-condition clusters on training rows and assign stable ordered IDs."""

    def __init__(self, n_clusters: int = 6, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_to_regime_id: dict[int, int] = {}

    def fit(self, train_frame: pd.DataFrame) -> "RegimeDecoupler":
        """Fit KMeans using operating settings from the training partition only."""
        self.kmeans.fit(train_frame[SETTING_COLUMNS].to_numpy(dtype=float))
        order = np.argsort(self.kmeans.cluster_centers_[:, 0])
        self.cluster_to_regime_id = {int(cluster): rank for rank, cluster in enumerate(order)}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Copy rows and add their nearest fitted ``regime_id``."""
        if not self.cluster_to_regime_id:
            raise RuntimeError("Fit the regime model before transforming rows")
        result = frame.copy()
        clusters = self.kmeans.predict(result[SETTING_COLUMNS].to_numpy(dtype=float))
        result["regime_id"] = [self.cluster_to_regime_id[int(cluster)] for cluster in clusters]
        return result

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on the supplied training rows and return their regime labels."""
        return self.fit(train_frame).transform(train_frame)

    def distance_margin(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return nearest/runner-up centroid distances and their ambiguity margin."""
        if not self.cluster_to_regime_id:
            raise RuntimeError("Fit the regime model before measuring distances")
        distances = self.kmeans.transform(frame[SETTING_COLUMNS].to_numpy(dtype=float))
        ordered_clusters = np.argsort(self.kmeans.cluster_centers_[:, 0])
        ordered = distances[:, ordered_clusters]
        ranked = np.sort(ordered, axis=1)
        return pd.DataFrame({
            "nearest_distance": ranked[:, 0],
            "runner_up_distance": ranked[:, 1],
            "distance_margin": ranked[:, 1] - ranked[:, 0],
        }, index=frame.index)

    @staticmethod
    def variance_decomposition(frame: pd.DataFrame, target: str) -> dict:
        """Decompose population variance into regime-mean and within-regime parts."""
        if "regime_id" not in frame:
            raise ValueError("frame must contain regime_id; assign regimes using a training-fitted model")
        values = frame[target].to_numpy(dtype=float)
        total = float(np.var(values))
        means = frame.groupby("regime_id")[target].mean()
        fitted = frame["regime_id"].map(means).to_numpy(dtype=float)
        between = float(np.var(fitted))
        within = float(np.var(values - fitted))
        summaries = frame.groupby("regime_id").agg(
            n_samples=(target, "size"), mean_target=(target, "mean"), std_target=(target, "std"),
        ).reset_index()
        return {
            "target": target,
            "total_var": total,
            "total_sd": float(np.std(values)),
            "between_regime_var": between,
            "within_regime_var": within,
            "pct_between_regime": between / total * 100.0 if total else 0.0,
            "pct_within_regime": within / total * 100.0 if total else 0.0,
            "Between_regime_pct": between / total * 100.0 if total else 0.0,
            "Within_regime_pct": within / total * 100.0 if total else 0.0,
            "regime_baseline_r2": between / total if total else 0.0,
            "regime_baseline_rmse": float(np.sqrt(within)),
            "clusters": summaries.to_dict(orient="records"),
        }

    @staticmethod
    def decompose_sensors(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        """Return same-population variance shares for several sensors."""
        rows = []
        for sensor in sensors:
            result = RegimeDecoupler.variance_decomposition(frame, sensor)
            rows.append({
                "Sensor": sensor,
                "Total_variance": result["total_var"],
                "Between_regime_variance": result["between_regime_var"],
                "Within_regime_variance": result["within_regime_var"],
                "Between_regime_pct": result["pct_between_regime"],
                "Within_regime_pct": result["pct_within_regime"],
            })
        return pd.DataFrame(rows)
