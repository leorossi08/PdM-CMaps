"""Automated tests for the native PyTorch MiniRocket implementation."""

import numpy as np
import pytest
import torch

from models.minirocket import MiniRocketRULModel, MiniRocketTransformer
from targets import compute_piecewise_rul
from tests.test_core import sample_frame


def test_minirocket_kernel_properties():
    """Validate that base kernels and multi-channel kernels satisfy length 9, zero sum, and {-1, 2} weights."""
    base_kernels = MiniRocketTransformer.generate_base_kernels()
    assert base_kernels.shape == (84, 9)
    assert np.all(np.isin(base_kernels, [-1.0, 2.0]))
    assert np.allclose(base_kernels.sum(axis=1), 0.0)
    for row in base_kernels:
        assert (row == 2.0).sum() == 3
        assert (row == -1.0).sum() == 6

    # Test multi-channel kernel assignment
    transformer = MiniRocketTransformer(num_kernels=84, seq_length=15, random_state=42)
    dummy_x = np.random.randn(10, 5, 15).astype(np.float32)
    transformer.fit(dummy_x)

    assert hasattr(transformer, "weights_")
    for d, weights in transformer.weights_.items():
        assert weights.shape[1] == 5  # channels
        assert weights.shape[2] == 9  # length
        for k in range(weights.shape[0]):
            kernel = weights[k]
            # Kernel sum must be zero
            assert np.isclose(kernel.sum(), 0.0)
            active_channels = (np.abs(kernel).sum(axis=1) > 0)
            assert 1 <= active_channels.sum() <= 3
            for c_idx in np.where(active_channels)[0]:
                c_weights = kernel[c_idx]
                assert set(np.unique(c_weights)) == {-1.0, 2.0}
                assert np.isclose(c_weights.sum(), 0.0)


def test_minirocket_causality():
    """Ensure that causal left-padding guarantees that output at time t does not depend on future timestamps."""
    transformer = MiniRocketTransformer(num_kernels=50, seq_length=15, dilations=(1, 2), random_state=42)
    # Fit on initial dummy data
    init_x = np.random.randn(20, 4, 15).astype(np.float32)
    transformer.fit(init_x)

    # Construct two series: identical up to t=9, diverging from t=10 onwards
    x1 = np.random.randn(1, 4, 15).astype(np.float32)
    x2 = x1.copy()
    x2[:, :, 10:] = np.random.randn(1, 4, 5).astype(np.float32) + 10.0

    conv1 = transformer.convolve(x1)
    conv2 = transformer.convolve(x2)

    for d in transformer.active_dilations_:
        out1 = conv1[d]
        out2 = conv2[d]
        # Timestamps 0 to 9 must be bitwise/numerically identical
        assert torch.allclose(out1[..., :10], out2[..., :10], atol=1e-6), (
            f"Future leakage detected for dilation {d} at timestamps before t=10"
        )
        # Timestamps 10 to 14 must differ because future values changed
        assert not torch.allclose(out1[..., 10:], out2[..., 10:]), (
            f"Expected difference at t >= 10 for dilation {d}"
        )


def test_minirocket_shape_and_determinism():
    """Validate output dimensions, bounded [0, 1] range, and reproducibility with random_state."""
    n_samples, n_channels, seq_len = 25, 6, 15
    num_kernels = 120
    x = np.random.randn(n_samples, n_channels, seq_len).astype(np.float32)

    transformer1 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=42)
    ppv1 = transformer1.fit_transform(x)

    assert ppv1.shape == (n_samples, num_kernels)
    assert np.all(ppv1 >= 0.0) and np.all(ppv1 <= 1.0)
    assert np.isfinite(ppv1).all()

    # Reproducibility check with identical random_state
    transformer2 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=42)
    ppv2 = transformer2.fit_transform(x)
    assert np.allclose(ppv1, ppv2)

    # Different random_state should yield different feature activations
    transformer3 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=999)
    ppv3 = transformer3.fit_transform(x)
    assert not np.allclose(ppv1, ppv3)


def test_minirocket_fit_predict_pipeline():
    """Test the complete MiniRocketRULModel fit/predict pipeline on sample engine cycles."""
    frame = sample_frame(engine_count=6)
    frame["RUL"] = compute_piecewise_rul(frame)
    features = ["setting_1", "setting_2", "setting_3", "T50", "T24", "P30"]

    model = MiniRocketRULModel(seq_length=4, num_kernels=60, random_state=42)

    # Ensure calling predict before fit raises RuntimeError
    with pytest.raises(RuntimeError, match="Fit the MiniRocketRULModel before calling predict"):
        model.predict(frame)

    model.fit(frame, features, "RUL")
    assert model.fit_time > 0.0
    assert model.regressor is not None

    y_true, y_pred = model.predict(frame)
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_pred).all()

    values = model.predict_values(frame)
    assert np.array_equal(y_pred, values)
