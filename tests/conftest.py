"""Pytest configuration and fixtures for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


@pytest.fixture
def sample_weights() -> dict[str, npt.NDArray[np.float32]]:
    """Create sample model weights for testing.

    Returns:
        Dictionary of sample weights
    """
    np.random.seed(42)
    return {
        "layer1": np.random.randn(10, 5).astype(np.float32),
        "layer2": np.random.randn(5, 3).astype(np.float32),
        "bias1": np.random.randn(5).astype(np.float32),
        "bias2": np.random.randn(3).astype(np.float32),
    }


@pytest.fixture
def float32_array() -> npt.NDArray[np.float32]:
    """Create a sample float32 array for testing.

    Returns:
        Random float32 numpy array
    """
    np.random.seed(42)
    return np.random.randn(100, 50).astype(np.float32)


@pytest.fixture
def int8_array() -> npt.NDArray[np.int8]:
    """Create a sample int8 array for testing.

    Returns:
        Random int8 numpy array
    """
    np.random.seed(42)
    return np.random.randint(-127, 128, size=(100, 50), dtype=np.int8)


@pytest.fixture
def temp_model_file(tmp_path: Path) -> Path:
    """Create a temporary model file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to temporary model file
    """
    model_file = tmp_path / "test_model.onnx"
    # Create a dummy file with some content
    model_file.write_bytes(b"dummy model content" * 1000)
    return model_file


@pytest.fixture
def prediction_target_arrays() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Create sample prediction and target arrays for testing.

    Returns:
        Tuple of (prediction, target) arrays
    """
    np.random.seed(42)
    target = np.random.randn(100).astype(np.float32)
    # Create prediction with some noise
    prediction = target + np.random.randn(100).astype(np.float32) * 0.1
    return prediction, target


@pytest.fixture
def classification_arrays() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Create sample classification arrays for testing.

    Returns:
        Tuple of (prediction, target) class labels
    """
    np.random.seed(42)
    target = np.random.randint(0, 10, size=100, dtype=np.int_)
    # Create prediction with 90% accuracy
    prediction = target.copy()
    # Flip 10% of predictions
    flip_indices = np.random.choice(100, size=10, replace=False)
    prediction[flip_indices] = (prediction[flip_indices] + 1) % 10
    return prediction, target
