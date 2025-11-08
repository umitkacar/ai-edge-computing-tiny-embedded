"""Tests for utils module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ai_edge_tinyml.utils import (
    calculate_accuracy,
    calculate_compression_ratio,
    calculate_mae,
    calculate_memory_footprint,
    calculate_mse,
    estimate_inference_time,
    format_model_stats,
    get_model_size,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_get_model_size_valid_file(self, temp_model_file: Path) -> None:
        """Test getting size of a valid model file."""
        size_mb = get_model_size(temp_model_file)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_get_model_size_with_string_path(self, temp_model_file: Path) -> None:
        """Test with string path instead of Path object."""
        size_mb = get_model_size(str(temp_model_file))
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_get_model_size_nonexistent_file(self, tmp_path: Path) -> None:
        """Test with non-existent file."""
        nonexistent = tmp_path / "does_not_exist.onnx"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            get_model_size(nonexistent)

    def test_get_model_size_directory(self, tmp_path: Path) -> None:
        """Test with directory instead of file."""
        with pytest.raises(ValueError, match="Path is not a file"):
            get_model_size(tmp_path)


class TestCalculateCompressionRatio:
    """Tests for calculate_compression_ratio function."""

    def test_valid_compression(self) -> None:
        """Test with valid compression values."""
        ratio = calculate_compression_ratio(100.0, 25.0)
        assert ratio == 4.0

    def test_no_compression(self) -> None:
        """Test when sizes are equal."""
        ratio = calculate_compression_ratio(50.0, 50.0)
        assert ratio == 1.0

    def test_expansion(self) -> None:
        """Test when compressed size is larger."""
        ratio = calculate_compression_ratio(50.0, 100.0)
        assert ratio == 0.5

    @pytest.mark.parametrize(
        ("original", "compressed", "error_match"),
        [
            (0.0, 50.0, "Original size must be positive"),
            (-10.0, 50.0, "Original size must be positive"),
            (100.0, 0.0, "Compressed size must be positive"),
            (100.0, -20.0, "Compressed size must be positive"),
        ],
    )
    def test_invalid_sizes(self, original: float, compressed: float, error_match: str) -> None:
        """Test with invalid size values."""
        with pytest.raises(ValueError, match=error_match):
            calculate_compression_ratio(original, compressed)


class TestCalculateMSE:
    """Tests for calculate_mse function."""

    def test_identical_arrays(self) -> None:
        """Test MSE with identical arrays."""
        arr = np.random.randn(100).astype(np.float32)
        mse = calculate_mse(arr, arr)
        assert mse == 0.0

    def test_different_arrays(
        self,
        prediction_target_arrays: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ) -> None:
        """Test MSE with different arrays."""
        prediction, target = prediction_target_arrays
        mse = calculate_mse(prediction, target)
        assert isinstance(mse, float)
        assert mse > 0

    def test_different_shapes(self) -> None:
        """Test with arrays of different shapes."""
        arr1 = np.random.randn(100).astype(np.float32)
        arr2 = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="Arrays must have same shape"):
            calculate_mse(arr1, arr2)

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        arr1 = np.array([], dtype=np.float32)
        arr2 = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Cannot calculate MSE for empty arrays"):
            calculate_mse(arr1, arr2)


class TestCalculateMAE:
    """Tests for calculate_mae function."""

    def test_identical_arrays(self) -> None:
        """Test MAE with identical arrays."""
        arr = np.random.randn(100).astype(np.float32)
        mae = calculate_mae(arr, arr)
        assert mae == 0.0

    def test_different_arrays(
        self,
        prediction_target_arrays: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ) -> None:
        """Test MAE with different arrays."""
        prediction, target = prediction_target_arrays
        mae = calculate_mae(prediction, target)
        assert isinstance(mae, float)
        assert mae > 0

    def test_different_shapes(self) -> None:
        """Test with arrays of different shapes."""
        arr1 = np.random.randn(100).astype(np.float32)
        arr2 = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="Arrays must have same shape"):
            calculate_mae(arr1, arr2)

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        arr1 = np.array([], dtype=np.float32)
        arr2 = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Cannot calculate MAE for empty arrays"):
            calculate_mae(arr1, arr2)


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Test with perfect predictions."""
        arr = np.random.randint(0, 10, size=100, dtype=np.int_)
        accuracy = calculate_accuracy(arr, arr)
        assert accuracy == 100.0

    def test_zero_accuracy(self) -> None:
        """Test with all wrong predictions."""
        target = np.zeros(100, dtype=np.int_)
        prediction = np.ones(100, dtype=np.int_)
        accuracy = calculate_accuracy(prediction, target)
        assert accuracy == 0.0

    def test_partial_accuracy(
        self,
        classification_arrays: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
    ) -> None:
        """Test with partial accuracy."""
        prediction, target = classification_arrays
        accuracy = calculate_accuracy(prediction, target)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 100.0
        assert accuracy == 90.0  # Based on fixture creating 90% accuracy

    def test_different_shapes(self) -> None:
        """Test with arrays of different shapes."""
        arr1 = np.random.randint(0, 10, size=100, dtype=np.int_)
        arr2 = np.random.randint(0, 10, size=50, dtype=np.int_)

        with pytest.raises(ValueError, match="Arrays must have same shape"):
            calculate_accuracy(arr1, arr2)

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        arr1 = np.array([], dtype=np.int_)
        arr2 = np.array([], dtype=np.int_)

        with pytest.raises(ValueError, match="Cannot calculate accuracy for empty arrays"):
            calculate_accuracy(arr1, arr2)


class TestEstimateInferenceTime:
    """Tests for estimate_inference_time function."""

    @pytest.mark.parametrize(
        ("device", "expected_range"),
        [
            ("cortex-m4", (100, 500)),
            ("cortex-m7", (50, 250)),
            ("jetson-nano", (5, 50)),
            ("raspberry-pi-4", (10, 100)),
        ],
    )
    def test_estimate_for_devices(self, device: str, expected_range: tuple[float, float]) -> None:
        """Test inference time estimation for different devices."""
        model_size = 10.0  # 10 MB model
        time_ms = estimate_inference_time(model_size, device)

        assert isinstance(time_ms, float)
        assert time_ms > 0
        min_time, max_time = expected_range
        assert min_time <= time_ms <= max_time

    def test_invalid_model_size(self) -> None:
        """Test with invalid model size."""
        with pytest.raises(ValueError, match="Model size must be positive"):
            estimate_inference_time(0.0)

        with pytest.raises(ValueError, match="Model size must be positive"):
            estimate_inference_time(-10.0)

    def test_unknown_device(self) -> None:
        """Test with unknown device."""
        with pytest.raises(ValueError, match="Unknown device"):
            estimate_inference_time(10.0, "unknown-device")


class TestCalculateMemoryFootprint:
    """Tests for calculate_memory_footprint function."""

    def test_basic_calculation(self) -> None:
        """Test basic memory footprint calculation."""
        footprint = calculate_memory_footprint(
            model_params=1_000_000,
            activation_size=100_000,
            batch_size=1,
            bits_per_param=32,
        )
        assert isinstance(footprint, float)
        assert footprint > 0

    @pytest.mark.parametrize(
        ("bits", "expected_multiplier"),
        [
            (8, 0.25),
            (16, 0.5),
            (32, 1.0),
        ],
    )
    def test_different_quantization(self, bits: int, expected_multiplier: float) -> None:
        """Test memory footprint with different bit widths."""
        base_footprint = calculate_memory_footprint(
            model_params=1_000_000,
            activation_size=0,
            batch_size=1,
            bits_per_param=32,
        )

        quantized_footprint = calculate_memory_footprint(
            model_params=1_000_000,
            activation_size=0,
            batch_size=1,
            bits_per_param=bits,
        )

        # Footprint should scale with bit width
        assert abs(quantized_footprint - base_footprint * expected_multiplier) < 0.01

    def test_batch_size_scaling(self) -> None:
        """Test that batch size affects activation memory."""
        footprint_batch_1 = calculate_memory_footprint(
            model_params=1_000_000,
            activation_size=100_000,
            batch_size=1,
            bits_per_param=32,
        )

        footprint_batch_4 = calculate_memory_footprint(
            model_params=1_000_000,
            activation_size=100_000,
            batch_size=4,
            bits_per_param=32,
        )

        # Batch size should increase activation memory
        assert footprint_batch_4 > footprint_batch_1

    @pytest.mark.parametrize(
        ("params", "activation", "batch", "bits", "error_match"),
        [
            (0, 1000, 1, 32, "Model params must be positive"),
            (-1000, 1000, 1, 32, "Model params must be positive"),
            (1000, -100, 1, 32, "Activation size cannot be negative"),
            (1000, 1000, 0, 32, "Batch size must be positive"),
            (1000, 1000, 1, 7, "Bits per param must be 8, 16, or 32"),
        ],
    )
    def test_invalid_parameters(
        self,
        params: int,
        activation: int,
        batch: int,
        bits: int,
        error_match: str,
    ) -> None:
        """Test with invalid parameters."""
        with pytest.raises(ValueError, match=error_match):
            calculate_memory_footprint(params, activation, batch, bits)


class TestFormatModelStats:
    """Tests for format_model_stats function."""

    def test_basic_formatting(self) -> None:
        """Test basic stats formatting."""
        stats = format_model_stats(10.5, 1_000_000)
        assert "10.50 MB" in stats
        assert "1,000,000" in stats

    def test_with_compression_ratio(self) -> None:
        """Test formatting with compression ratio."""
        stats = format_model_stats(10.5, 1_000_000, compression_ratio=4.0)
        assert "10.50 MB" in stats
        assert "1,000,000" in stats
        assert "4.00x" in stats

    def test_without_compression_ratio(self) -> None:
        """Test formatting without compression ratio."""
        stats = format_model_stats(10.5, 1_000_000)
        assert "10.50 MB" in stats
        assert "1,000,000" in stats
        assert "Compression Ratio" not in stats
