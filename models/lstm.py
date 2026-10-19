"""Engine-local sequence construction and a compact PyTorch LSTM regressor."""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_engine_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 15,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale rows and form chronological windows without crossing engine boundaries."""
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    ordered = frame.sort_values(["unit_id", "cycle"]).copy()
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    values = scaler.fit_transform(ordered[feature_cols]) if fit_scaler else scaler.transform(ordered[feature_cols])
    scaled = pd.DataFrame(values, columns=feature_cols, index=ordered.index)
    scaled["unit_id"] = ordered["unit_id"].to_numpy()
    scaled["cycle"] = ordered["cycle"].to_numpy()
    sequences, labels = [], []
    for unit_id, group in scaled.groupby("unit_id", sort=False):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets = ordered.loc[group.index, target_col].to_numpy(dtype=np.float32)
        if len(features) < seq_length:
            pad = np.repeat(features[:1], seq_length - len(features), axis=0)
            sequences.append(np.vstack([pad, features]))
            labels.append(targets[-1])
        else:
            for end in range(seq_length, len(features) + 1):
                sequences.append(features[end - seq_length:end])
                labels.append(targets[end - 1])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32), scaler


class LSTMRegressorNet(nn.Module):
    """LSTM encoder followed by a scalar regression head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1]).squeeze(-1)


class PyTorchLSTMModel:
    """Scikit-learn-style wrapper that retains fitted scalers and the trained network."""

    def __init__(
        self,
        seq_length: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.002,
        batch_size: int = 128,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device(device)
        self.net = None
        self.scaler = None
        self.target_scaler = None
        self.feature_cols: list[str] = []
        self.target_col = ""
        self.fit_time = 0.0

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> "PyTorchLSTMModel":
        """Train on engine-local windows; early stopping monitors training loss."""
        torch.manual_seed(self.random_state)
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        started = time.perf_counter()
        x, y, self.scaler = build_engine_sequences(
            frame, self.feature_cols, target_col, self.seq_length, fit_scaler=True,
        )
        self.target_scaler = StandardScaler().fit(y.reshape(-1, 1))
        scaled_y = self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(scaled_y)), batch_size=self.batch_size, shuffle=True)
        self.net = LSTMRegressorNet(len(self.feature_cols), self.hidden_dim, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best_loss, best_state, waiting = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch_y)
            epoch_loss = total / len(loader.dataset)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
                waiting = 0
            else:
                waiting += 1
                if waiting >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_time = time.perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned target and sequence-endpoint predictions."""
        if self.net is None or self.scaler is None or self.target_scaler is None:
            raise RuntimeError("Fit the LSTM before calling predict")
        x, y, _ = build_engine_sequences(
            frame, self.feature_cols, self.target_col, self.seq_length,
            scaler=self.scaler, fit_scaler=False,
        )
        self.net.eval()
        with torch.no_grad():
            scaled = self.net(torch.from_numpy(x).to(self.device)).cpu().numpy()
        prediction = self.target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
        return y, prediction
