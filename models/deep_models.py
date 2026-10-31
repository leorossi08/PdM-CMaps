"""Modular PyTorch Deep Sequence Models for Predictive Maintenance (PdM) RUL Estimation.

Provides unified, modular architectures for:
1. Recurrent Neural Networks (RNN / LSTM / GRU)
2. 1D Convolutional Neural Networks (CNN)
3. Time Series Transformers (Multi-Head Self-Attention)

Follows OOP best practices: common sequence windowing, data loaders, training loops,
device management (CUDA/CPU), early stopping, and scikit-learn-compatible API.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_causal_engine_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    seq_length: int = 15,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """Build causal sliding windows of length `seq_length` without crossing engine boundaries.
    
    Uses causal left-padding with the first available cycle for early cycles (t < seq_length),
    ensuring that EVERY row in the input DataFrame receives an aligned prediction.
    """
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    ordered = frame.sort_values(["unit_id", "cycle"]).copy()
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    
    feature_vals = scaler.fit_transform(ordered[feature_cols]) if fit_scaler else scaler.transform(ordered[feature_cols])
    scaled = pd.DataFrame(feature_vals, columns=feature_cols, index=ordered.index)
    scaled["unit_id"] = ordered["unit_id"].to_numpy()
    scaled["cycle"] = ordered["cycle"].to_numpy()

    has_target = target_col is not None and target_col in ordered.columns
    targets_all = ordered[target_col].astype(float) if has_target else None

    sequences = []
    labels = [] if has_target else None

    for _, group in scaled.groupby("unit_id", sort=False):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(features)
        grp_targets = targets_all.loc[group.index].to_numpy(dtype=np.float32) if has_target else None

        for t in range(n_rows):
            start = t - seq_length + 1
            if start < 0:
                pad_len = -start
                pad = np.repeat(features[:1], pad_len, axis=0)
                sub = features[: t + 1]
                seq = np.vstack([pad, sub])
            else:
                seq = features[start : t + 1]
            sequences.append(seq)
            if has_target:
                labels.append(grp_targets[t])

    seq_arr = np.asarray(sequences, dtype=np.float32)
    lbl_arr = np.asarray(labels, dtype=np.float32) if has_target else None
    return seq_arr, lbl_arr, scaler


# =========================================================================
# Neural Network Architectures (PyTorch Modules)
# =========================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RNNRegressorNet(nn.Module):
    """Recurrent Neural Network (LSTM or GRU) with regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: Literal["lstm", "gru"] = "lstm",
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # Take last time step
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class CNNRegressorNet(nn.Module):
    """1D Temporal Convolutional Neural Network with Adaptive Pooling & MLP head."""

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(num_filters * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv1D expects (Batch, Channels, SeqLen)
        x_trans = x.transpose(1, 2)
        feats = self.conv_block(x_trans)
        pooled = self.pool(feats).squeeze(-1)
        return self.head(pooled).squeeze(-1)


class TransformerRegressorNet(nn.Module):
    """Time Series Transformer Encoder with Multi-Head Self-Attention and regression head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen, Features)
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer_encoder(encoded)
        # Global average pooling over sequence length
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# =========================================================================
# Base PyTorch Sequence Model (Scikit-Learn Compatible Wrapper)
# =========================================================================

def get_safe_device() -> torch.device:
    """Return CUDA device only if hardware compute capability is supported by PyTorch, else CPU."""
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


class BasePyTorchSequenceModel:
    """Base class providing sequence batching, training loop, early stopping, and device management."""

    def __init__(
        self,
        seq_length: int = 15,
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        if device is None:
            self.device = get_safe_device()
        else:
            self.device = torch.device(device)

        self.net: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fit_time: float = 0.0

    def _build_network(self, input_dim: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_network")

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str = "RUL") -> "BasePyTorchSequenceModel":
        """Fit model with engine-preserving causal sequences, deterministic generator, and grouped early stopping."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.feature_cols = list(feature_cols)

        started = time.perf_counter()

        # Engine-grouped internal validation partition (15%) for uncorrupted early stopping
        n_engines = frame["unit_id"].nunique()
        has_val = n_engines >= 4
        if has_val:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
            tr_idx, val_idx = next(gss.split(frame, groups=frame["unit_id"]))
            tr_frame = frame.iloc[tr_idx].copy().reset_index(drop=True)
            val_frame = frame.iloc[val_idx].copy().reset_index(drop=True)
        else:
            tr_frame = frame
            val_frame = None

        x_tr, y_tr, self.scaler = build_causal_engine_sequences(
            tr_frame, self.feature_cols, target_col, self.seq_length, fit_scaler=True,
        )
        self.target_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
        scaled_y_tr = self.target_scaler.transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()

        gen = torch.Generator().manual_seed(self.random_state)
        dataset_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(scaled_y_tr))
        loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True, generator=gen)

        loader_val = None
        if val_frame is not None:
            x_val, y_val, _ = build_causal_engine_sequences(
                val_frame, self.feature_cols, target_col, self.seq_length, scaler=self.scaler, fit_scaler=False,
            )
            scaled_y_val = self.target_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).ravel()
            loader_val = DataLoader(
                TensorDataset(torch.from_numpy(x_val), torch.from_numpy(scaled_y_val)),
                batch_size=self.batch_size * 2,
                shuffle=False,
            )

        self.net = self._build_network(len(self.feature_cols)).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        waiting = 0

        for _ in range(self.max_epochs):
            self.net.train()
            total_tr_loss = 0.0
            for batch_x, batch_y in loader_tr:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
                optimizer.step()
                total_tr_loss += float(loss.item()) * len(batch_y)

            # Evaluate early stopping on internal engine validation split
            if loader_val is not None:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in loader_val:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        val_loss += float(loss_fn(self.net(batch_x), batch_y).item()) * len(batch_y)
                epoch_loss = val_loss / len(loader_val.dataset)
            else:
                epoch_loss = total_tr_loss / len(loader_tr.dataset)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                waiting = 0
            else:
                waiting += 1
                if waiting >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.fit_time = time.perf_counter() - started
        return self

    def predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Alias for predict returning raw array."""
        return self.predict(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict aligned target values for every row in the DataFrame."""
        if self.net is None or self.scaler is None or self.target_scaler is None:
            raise RuntimeError("Model must be fitted before predict")

        x, _, _ = build_causal_engine_sequences(
            frame, self.feature_cols, None, self.seq_length, scaler=self.scaler, fit_scaler=False,
        )
        self.net.eval()
        preds = []
        loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=self.batch_size * 2, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                scaled_pred = self.net(batch_x).cpu().numpy()
                preds.append(scaled_pred)

        concatenated = np.concatenate(preds)
        unscaled = self.target_scaler.inverse_transform(concatenated.reshape(-1, 1)).ravel()
        return unscaled


class PyTorchRNNModel(BasePyTorchSequenceModel):
    """Modular Recurrent Neural Network (LSTM or GRU) Regressor."""

    def __init__(
        self,
        seq_length: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: Literal["lstm", "gru"] = "lstm",
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type

    def _build_network(self, input_dim: int) -> nn.Module:
        return RNNRegressorNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
        )


class PyTorchCNNModel(BasePyTorchSequenceModel):
    """1D Temporal Convolutional Neural Network Regressor."""

    def __init__(
        self,
        seq_length: int = 15,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return CNNRegressorNet(
            input_dim=input_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )


class PyTorchTransformerModel(BasePyTorchSequenceModel):
    """Time Series Transformer Regressor with Multi-Head Attention."""

    def __init__(
        self,
        seq_length: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return TransformerRegressorNet(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
