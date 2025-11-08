"""Utility functions for AI edge computing and TinyML.

This module provides helper functions for model analysis, metrics calculation,
and common operations in edge AI development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def get_model_size(model_path: Path | str) -> float:
    """Get the size of a model file in megabytes.

    Args:
        model_path: Path to the model file

    Returns:
        Model size in megabytes

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If path is invalid
    """
    path = Path(model_path)

    if not path.exists():
        msg = f"Model file not found: {path}"
        raise FileNotFoundError(msg)

    if not path.is_file():
        msg = f"Path is not a file: {path}"
        raise ValueError(msg)

    size_bytes = path.stat().st_size
    return size_bytes / (1024 * 1024)


def calculate_compression_ratio(
    original_size: float,
    compressed_size: float,
) -> float:
    """Calculate compression ratio.

    Args:
        original_size: Original model size in MB
        compressed_size: Compressed model size in MB

    Returns:
        Compression ratio (original_size / compressed_size)

    Raises:
        ValueError: If sizes are invalid
    """
    if original_size <= 0:
        msg = f"Original size must be positive, got {original_size}"
        raise ValueError(msg)

    if compressed_size <= 0:
        msg = f"Compressed size must be positive, got {compressed_size}"
        raise ValueError(msg)

    return original_size / compressed_size


def calculate_mse(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> float:
    """Calculate mean squared error between prediction and target.

    Args:
        prediction: Predicted values
        target: Target values

    Returns:
        Mean squared error

    Raises:
        ValueError: If arrays have different shapes or are empty
    """
    if prediction.shape != target.shape:
        msg = f"Arrays must have same shape, got {prediction.shape} and {target.shape}"
        raise ValueError(msg)

    if prediction.size == 0:
        msg = "Cannot calculate MSE for empty arrays"
        raise ValueError(msg)

    return float(np.mean((prediction - target) ** 2))


def calculate_mae(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> float:
    """Calculate mean absolute error between prediction and target.

    Args:
        prediction: Predicted values
        target: Target values

    Returns:
        Mean absolute error

    Raises:
        ValueError: If arrays have different shapes or are empty
    """
    if prediction.shape != target.shape:
        msg = f"Arrays must have same shape, got {prediction.shape} and {target.shape}"
        raise ValueError(msg)

    if prediction.size == 0:
        msg = "Cannot calculate MAE for empty arrays"
        raise ValueError(msg)

    return float(np.mean(np.abs(prediction - target)))


def calculate_accuracy(
    prediction: npt.NDArray[np.int_],
    target: npt.NDArray[np.int_],
) -> float:
    """Calculate classification accuracy.

    Args:
        prediction: Predicted class labels
        target: Target class labels

    Returns:
        Accuracy as a percentage (0-100)

    Raises:
        ValueError: If arrays have different shapes or are empty
    """
    if prediction.shape != target.shape:
        msg = f"Arrays must have same shape, got {prediction.shape} and {target.shape}"
        raise ValueError(msg)

    if prediction.size == 0:
        msg = "Cannot calculate accuracy for empty arrays"
        raise ValueError(msg)

    return float(np.mean(prediction == target) * 100)


def estimate_inference_time(
    model_size_mb: float,
    target_device: str = "cortex-m4",
) -> float:
    """Estimate inference time for a model on a target device.

    This is a simplified estimation based on model size and device capabilities.
    Actual inference time depends on many factors including model architecture,
    quantization, and hardware-specific optimizations.

    Args:
        model_size_mb: Model size in megabytes
        target_device: Target device type (cortex-m4, cortex-m7, jetson-nano, etc.)

    Returns:
        Estimated inference time in milliseconds

    Raises:
        ValueError: If model_size_mb is invalid or device is unknown
    """
    if model_size_mb <= 0:
        msg = f"Model size must be positive, got {model_size_mb}"
        raise ValueError(msg)

    # Simplified performance factors (ops/ms) for different devices
    device_performance: dict[str, float] = {
        "cortex-m4": 0.5,      # Very limited performance
        "cortex-m7": 1.0,      # Better performance
        "esp32": 0.8,          # IoT device
        "jetson-nano": 10.0,   # Edge AI device
        "raspberry-pi-4": 5.0, # Single-board computer
    }

    performance_factor = device_performance.get(target_device.lower())
    if performance_factor is None:
        msg = f"Unknown device: {target_device}"
        raise ValueError(msg)

    # Estimate based on model size (simplified model)
    base_time_ms = model_size_mb * 10  # Base estimation
    return base_time_ms / performance_factor


def calculate_memory_footprint(
    model_params: int,
    activation_size: int,
    batch_size: int = 1,
    bits_per_param: int = 32,
) -> float:
    """Calculate total memory footprint in megabytes.

    Args:
        model_params: Number of model parameters
        activation_size: Size of activation tensors (elements)
        batch_size: Batch size for inference
        bits_per_param: Bits per parameter (8, 16, or 32)

    Returns:
        Total memory footprint in megabytes

    Raises:
        ValueError: If parameters are invalid
    """
    if model_params <= 0:
        msg = f"Model params must be positive, got {model_params}"
        raise ValueError(msg)

    if activation_size < 0:
        msg = f"Activation size cannot be negative, got {activation_size}"
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"Batch size must be positive, got {batch_size}"
        raise ValueError(msg)

    if bits_per_param not in {8, 16, 32}:
        msg = f"Bits per param must be 8, 16, or 32, got {bits_per_param}"
        raise ValueError(msg)

    # Calculate model memory
    bytes_per_param = bits_per_param / 8
    model_memory_bytes = model_params * bytes_per_param

    # Calculate activation memory
    activation_memory_bytes = activation_size * batch_size * 4  # Assuming FP32 activations

    # Total memory
    total_bytes = model_memory_bytes + activation_memory_bytes
    return total_bytes / (1024 * 1024)


def format_model_stats(
    model_size_mb: float,
    params_count: int,
    compression_ratio: float | None = None,
) -> str:
    """Format model statistics as a readable string.

    Args:
        model_size_mb: Model size in megabytes
        params_count: Number of parameters
        compression_ratio: Optional compression ratio

    Returns:
        Formatted statistics string
    """
    stats = [
        f"Model Size: {model_size_mb:.2f} MB",
        f"Parameters: {params_count:,}",
    ]

    if compression_ratio is not None:
        stats.append(f"Compression Ratio: {compression_ratio:.2f}x")

    return " | ".join(stats)
