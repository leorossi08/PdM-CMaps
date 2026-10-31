"""Operating-regime clustering and sensor variance decomposition."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from schema import DEFAULT_RANDOM_STATE, SETTING_COLUMNS


class RegimeDecoupler:
    """Fit operating-condition clusters on training rows and assign stable ordered IDs."""

    def __init__(self, n_clusters: int = 6, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.cluster_to_regime_id: dict[int, int] = {}
        self.regime_sensor_stats_: dict[int, dict[str, tuple[float, float]]] = {}

    def evaluate_k_sweep(self, train_frame: pd.DataFrame, k_range=range(2, 10)) -> pd.DataFrame:
        """Evaluate clustering quality metrics across a range of k values on training settings."""
        settings = self.scaler.fit_transform(train_frame[SETTING_COLUMNS].to_numpy(dtype=float))
        records = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10).fit(settings)
            labels = km.labels_
            sil = float(silhouette_score(settings, labels)) if k > 1 and len(np.unique(labels)) > 1 else np.nan
            ch = float(calinski_harabasz_score(settings, labels)) if k > 1 and len(np.unique(labels)) > 1 else np.nan
            gmm = GaussianMixture(n_components=k, random_state=self.random_state, max_iter=100).fit(settings)
            bic = float(gmm.bic(settings))
            records.append({
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "bic": bic,
            })
        return pd.DataFrame(records)

    def fit(self, train_frame: pd.DataFrame) -> "RegimeDecoupler":
        """Fit KMeans using operating settings from the training partition and sort lexicographically."""
        settings = self.scaler.fit_transform(train_frame[SETTING_COLUMNS].to_numpy(dtype=float))
        self.kmeans.fit(settings)
        centers_phys = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        # Lexicographical sorting on physical coordinates: (round(alt, 0), round(mach, 2), round(TRA, 0))
        order = sorted(
            range(self.n_clusters),
            key=lambda idx: (
                round(float(centers_phys[idx, 0]), 0),
                round(float(centers_phys[idx, 1]), 2),
                round(float(centers_phys[idx, 2]), 0),
            ),
        )
        self.cluster_to_regime_id = {int(cluster): rank for rank, cluster in enumerate(order)}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Copy rows and add their nearest fitted ``regime_id``."""
        if not self.cluster_to_regime_id:
            raise RuntimeError("Fit the regime model before transforming rows")
        result = frame.copy()
        clusters = self.kmeans.predict(self.scaler.transform(result[SETTING_COLUMNS].to_numpy(dtype=float)))
        result["regime_id"] = [self.cluster_to_regime_id[int(cluster)] for cluster in clusters]
        return result

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on the supplied training rows and return their regime labels."""
        return self.fit(train_frame).transform(train_frame)

    def fit_sensor_scaler(self, train_frame: pd.DataFrame, sensors: list[str]) -> "RegimeDecoupler":
        """Compute and store per-regime mean and standard deviation for the given sensors on training data."""
        if "regime_id" not in train_frame:
            raise ValueError("train_frame must contain regime_id")
        self.regime_sensor_stats_ = {}
        for r_id, group in train_frame.groupby("regime_id"):
            stats_dict = {}
            for sensor in sensors:
                vals = group[sensor].to_numpy(dtype=float)
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                if std_val < 1e-8:
                    std_val = 1.0
                stats_dict[sensor] = (mean_val, std_val)
            self.regime_sensor_stats_[int(r_id)] = stats_dict
        return self

    def standardize_sensors(self, frame: pd.DataFrame, sensors: list[str], suffix: str = "_norm") -> pd.DataFrame:
        """Standardize sensor readings using fitted training regime statistics."""
        if not self.regime_sensor_stats_:
            raise RuntimeError("Call fit_sensor_scaler before standardize_sensors")
        if "regime_id" not in frame:
            raise ValueError("frame must contain regime_id")
        result = frame.copy()
        for sensor in sensors:
            norm_col = f"{sensor}{suffix}"
            vals = result[sensor].to_numpy(dtype=float)
            r_ids = result["regime_id"].to_numpy(dtype=int)
            out = np.empty_like(vals)
            for i in range(len(vals)):
                r_id = r_ids[i]
                if r_id in self.regime_sensor_stats_ and sensor in self.regime_sensor_stats_[r_id]:
                    m, s = self.regime_sensor_stats_[r_id][sensor]
                    out[i] = (vals[i] - m) / s
                else:
                    out[i] = vals[i]
            result[norm_col] = out
        return result

    def distance_margin(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return nearest/runner-up centroid distances and their ambiguity margin."""
        if not self.cluster_to_regime_id:
            raise RuntimeError("Fit the regime model before measuring distances")
        distances = self.kmeans.transform(self.scaler.transform(frame[SETTING_COLUMNS].to_numpy(dtype=float)))
        ranked = np.sort(distances, axis=1)
        nearest = ranked[:, 0]
        runner_up = ranked[:, 1]
        abs_margin = runner_up - nearest
        rel_margin = 1.0 - (nearest / (runner_up + 1e-9))
        return pd.DataFrame({
            "nearest_distance": nearest,
            "runner_up_distance": runner_up,
            "distance_margin": abs_margin,
            "relative_margin": np.clip(rel_margin, 0.0, 1.0),
        }, index=frame.index)

    @staticmethod
    def variance_decomposition(frame: pd.DataFrame, target: str, reference_frame: pd.DataFrame | None = None) -> dict:
        """Decompose variance; optionally score fixed regime means from reference_frame OOS."""
        if "regime_id" not in frame:
            raise ValueError("frame must contain regime_id; assign regimes using a training-fitted model")
        values = frame[target].to_numpy(dtype=float)
        total = float(np.var(values))
        descriptive_means = frame.groupby("regime_id")[target].mean()
        fitted = frame["regime_id"].map(descriptive_means).to_numpy(dtype=float)
        between = float(np.var(fitted))
        within = float(np.var(values - fitted))
        baseline_means = (reference_frame if reference_frame is not None else frame).groupby("regime_id")[target].mean()
        baseline_prediction = frame["regime_id"].map(baseline_means).to_numpy(dtype=float)
        if not np.isfinite(baseline_prediction).all():
            raise ValueError("reference_frame must provide a mean for every regime in frame")
        baseline_rmse = float(np.sqrt(np.mean((values - baseline_prediction) ** 2)))
        summaries = frame.groupby("regime_id").agg(
            n_samples=(target, "size"), mean_target=(target, "mean"),
            std_target=(target, lambda values: values.std(ddof=0)),
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
            "descriptive_between_pct": between / total * 100.0 if total else 0.0,
            "descriptive_within_pct": within / total * 100.0 if total else 0.0,
            "regime_baseline_r2": 1.0 - float(np.mean((values - baseline_prediction) ** 2)) / total if total else 0.0,
            "regime_baseline_rmse": baseline_rmse,
            "oos_r2": 1.0 - float(np.mean((values - baseline_prediction) ** 2)) / total if total else 0.0,
            "oos_rmse": baseline_rmse,
            "baseline_evaluation": "out_of_sample" if reference_frame is not None else "in_sample",
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
