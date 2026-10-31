"""Native PyTorch MiniRocket causal feature extractor and regularized linear regression pipeline."""

import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .deep_models import build_causal_engine_sequences


class MiniRocketTransformer(BaseEstimator, TransformerMixin):
    """MiniRocket multivariate time-series feature extractor in native PyTorch.

    Extracts deterministic convolutional features using length-9 zero-sum kernels,
    causal left-padding, and Proportion of Positive Values (PPV) pooling.
    """

    def __init__(
        self,
        num_kernels: int = 1000,
        seq_length: int = 15,
        dilations: tuple[int, ...] = (1, 2, 4),
        max_channels: int = 3,
        random_state: int | None = 42,
        device: str = "cpu",
        batch_size: int = 2048,
    ):
        self.num_kernels = num_kernels
        self.seq_length = seq_length
        self.dilations = dilations
        self.max_channels = max_channels
        self.random_state = random_state
        self.device = torch.device(device)
        self.batch_size = batch_size

    @staticmethod
    def generate_base_kernels() -> np.ndarray:
        """Generate the 84 canonical length-9 patterns with weights in {-1, 2} summing to zero."""
        combs = list(itertools.combinations(range(9), 3))
        base = np.full((len(combs), 9), -1.0, dtype=np.float32)
        for i, idxs in enumerate(combs):
            base[i, list(idxs)] = 2.0
        return base

    def _format_input(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input array to a float32 torch.Tensor with shape (samples, channels, time)."""
        if isinstance(X, np.ndarray):
            tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            tensor = X.float()
        else:
            tensor = torch.tensor(X, dtype=torch.float32)

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected 3D input (samples, channels, time) or (samples, time, channels), "
                f"got shape {tuple(tensor.shape)}"
            )

        if hasattr(self, "in_channels_"):
            if tensor.shape[1] != self.in_channels_ and tensor.shape[2] == self.in_channels_:
                tensor = tensor.transpose(1, 2)
        else:
            if tensor.shape[1] == self.seq_length and tensor.shape[2] != self.seq_length:
                tensor = tensor.transpose(1, 2)

        return tensor

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | None = None) -> "MiniRocketTransformer":
        """Generate kernel parameters and extract quantile threshold biases from training sequences."""
        tensor_x = self._format_input(X)
        self.in_channels_ = tensor_x.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.base_kernels_ = self.generate_base_kernels()

        dilations = (1, 2, 4) if self.dilations is None else tuple(self.dilations)
        active_dilations = tuple(d for d in dilations if 8 * d <= max(8, int(self.seq_length * 2.5)))
        if not active_dilations:
            active_dilations = (1,)
        self.active_dilations_ = active_dilations

        n_dil = len(active_dilations)
        base_count = self.num_kernels // n_dil
        remainder = self.num_kernels % n_dil
        counts = [base_count + (1 if i < remainder else 0) for i in range(n_dil)]

        dilation_groups = []
        all_weights = {}

        for d, k_count in zip(active_dilations, counts):
            if k_count == 0:
                continue
            weights_d = np.zeros((k_count, self.in_channels_, 9), dtype=np.float32)
            for i in range(k_count):
                pattern_idx = rng.integers(0, 84)
                pattern = self.base_kernels_[pattern_idx]
                n_c = rng.integers(1, min(self.max_channels, self.in_channels_) + 1)
                selected_channels = rng.choice(self.in_channels_, size=n_c, replace=False)
                for c in selected_channels:
                    weights_d[i, c, :] = pattern

            weights_tensor = torch.from_numpy(weights_d).to(self.device)
            dilation_groups.append((d, weights_tensor))
            all_weights[d] = weights_d

        self.weights_ = all_weights

        n_samples = tensor_x.shape[0]
        sample_size = min(n_samples, 1000)
        if n_samples > sample_size:
            sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
            x_sub = tensor_x[sample_indices].to(self.device)
        else:
            x_sub = tensor_x.to(self.device)

        fitted_dilation_groups = []
        biases_dict = {}

        with torch.no_grad():
            for d, weights_tensor in dilation_groups:
                k_count = weights_tensor.shape[0]
                pad_x = F.pad(x_sub, (8 * d, 0))
                out_d = F.conv1d(pad_x, weights_tensor, dilation=d)
                out_flat = out_d.transpose(0, 1).reshape(k_count, -1).cpu().numpy()

                quantiles = rng.uniform(0.1, 0.9, size=k_count)
                sorted_vals = np.sort(out_flat, axis=1)
                q_indices = np.clip(
                    (quantiles * (sorted_vals.shape[1] - 1)).astype(int),
                    0,
                    sorted_vals.shape[1] - 1,
                )
                b_d = sorted_vals[np.arange(k_count), q_indices].astype(np.float32)

                bias_tensor = torch.from_numpy(b_d).to(self.device)
                fitted_dilation_groups.append((d, weights_tensor, bias_tensor))
                biases_dict[d] = b_d

        self.dilation_groups_ = fitted_dilation_groups
        self.biases_ = biases_dict
        return self

    def convolve(self, X: np.ndarray | torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute causal convolution output tensors for each active dilation group."""
        if not hasattr(self, "dilation_groups_"):
            raise RuntimeError("MiniRocketTransformer is not fitted yet. Call fit() first.")
        tensor_x = self._format_input(X)
        outputs = {}
        with torch.no_grad():
            for d, weights_tensor, _ in self.dilation_groups_:
                batch_x = tensor_x.to(self.device)
                pad_x = F.pad(batch_x, (8 * d, 0))
                outputs[d] = F.conv1d(pad_x, weights_tensor, dilation=d)
        return outputs

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute Proportion of Positive Values (PPV) features for each input sequence."""
        if not hasattr(self, "dilation_groups_"):
            raise RuntimeError("MiniRocketTransformer is not fitted yet. Call fit() first.")

        tensor_x = self._format_input(X)
        n_samples = tensor_x.shape[0]
        if n_samples == 0:
            return np.empty((0, self.num_kernels), dtype=np.float32)

        all_ppv = []
        with torch.no_grad():
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_x = tensor_x[start_idx:end_idx].to(self.device)
                batch_features = []
                for d, weights_tensor, bias_tensor in self.dilation_groups_:
                    pad_x = F.pad(batch_x, (8 * d, 0), mode="replicate")
                    out_d = F.conv1d(pad_x, weights_tensor, dilation=d)
                    ppv_d = (out_d > bias_tensor.view(1, -1, 1)).float().mean(dim=-1)
                    batch_features.append(ppv_d)
                batch_ppv = torch.cat(batch_features, dim=1).cpu().numpy()
                all_ppv.append(batch_ppv)

        return np.vstack(all_ppv)


class MiniRocketRULModel:
    """Scikit-learn-style RUL pipeline combining engine sequence creation, MiniRocket PPV features, and RidgeCV."""

    def __init__(
        self,
        seq_length: int = 15,
        num_kernels: int = 1000,
        dilations: tuple[int, ...] = (1, 2, 4),
        max_channels: int = 3,
        alphas: np.ndarray | None = None,
        random_state: int = 42,
        device: str = "cpu",
        batch_size: int = 2048,
    ):
        self.seq_length = seq_length
        self.num_kernels = num_kernels
        self.dilations = dilations
        self.max_channels = max_channels
        self.alphas = np.logspace(-2, 4, 7) if alphas is None else np.asarray(alphas)
        self.random_state = random_state
        self.device = device
        self.batch_size = batch_size
        self.scaler: StandardScaler | None = None
        self.transformer: MiniRocketTransformer | None = None
        self.feature_scaler: StandardScaler | None = None
        self.regressor: RidgeCV | None = None
        self.feature_cols: list[str] = []
        self.target_col: str = ""
        self.fit_time: float = 0.0

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> "MiniRocketRULModel":
        """Form engine-local sequences, extract PPV features, and train a regularized Ridge regressor."""
        started = time.perf_counter()
        self.feature_cols = list(feature_cols)
        self.target_col = target_col

        x, y, self.scaler = build_causal_engine_sequences(
            frame, self.feature_cols, target_col, self.seq_length, fit_scaler=True,
        )
        x_trans = np.transpose(x, (0, 2, 1))

        self.transformer = MiniRocketTransformer(
            num_kernels=self.num_kernels,
            seq_length=self.seq_length,
            dilations=self.dilations,
            max_channels=self.max_channels,
            random_state=self.random_state,
            device=self.device,
            batch_size=self.batch_size,
        )
        ppv = self.transformer.fit_transform(x_trans)
        self.feature_scaler = StandardScaler()
        ppv_scaled = self.feature_scaler.fit_transform(ppv)

        groups = frame["unit_id"].to_numpy()
        n_groups = len(np.unique(groups))
        cv_splits = min(5, n_groups)
        if cv_splits >= 2:
            cv = list(GroupKFold(n_splits=cv_splits).split(ppv_scaled, y, groups=groups))
            self.regressor = RidgeCV(alphas=self.alphas, cv=cv)
        else:
            self.regressor = RidgeCV(alphas=self.alphas)
        self.regressor.fit(ppv_scaled, y)

        self.fit_time = time.perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned true targets and sequence-endpoint predictions."""
        if self.regressor is None or self.scaler is None or self.transformer is None or self.feature_scaler is None:
            raise RuntimeError("Fit the MiniRocketRULModel before calling predict")

        x, y, _ = build_causal_engine_sequences(
            frame, self.feature_cols, self.target_col, self.seq_length,
            scaler=self.scaler, fit_scaler=False,
        )
        x_trans = np.transpose(x, (0, 2, 1))
        ppv = self.transformer.transform(x_trans)
        ppv_scaled = self.feature_scaler.transform(ppv)
        prediction = self.regressor.predict(ppv_scaled)
        return y, prediction

    def predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Return only the prediction array."""
        _, prediction = self.predict(frame)
        return prediction
