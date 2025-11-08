"""Tests for quantization module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ai_edge_tinyml.quantization import (
    QuantizationConfig,
    QuantizationMode,
    QuantizationStrategy,
    Quantizer,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = QuantizationConfig()
        assert config.mode == QuantizationMode.INT8
        assert config.strategy == QuantizationStrategy.PER_TENSOR
        assert config.calibration_samples == 100
        assert config.per_channel is False
        assert config.symmetric is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = QuantizationConfig(
            mode=QuantizationMode.INT4,
            strategy=QuantizationStrategy.PER_CHANNEL,
            calibration_samples=200,
            per_channel=True,
            symmetric=False,
        )
        assert config.mode == QuantizationMode.INT4
        assert config.strategy == QuantizationStrategy.PER_CHANNEL
        assert config.calibration_samples == 200
        assert config.per_channel is True
        assert config.symmetric is False

    def test_invalid_calibration_samples(self) -> None:
        """Test validation of calibration samples."""
        with pytest.raises(ValueError, match="calibration_samples must be positive"):
            QuantizationConfig(calibration_samples=0)

        with pytest.raises(ValueError, match="calibration_samples must be positive"):
            QuantizationConfig(calibration_samples=-10)


class TestQuantizer:
    """Tests for Quantizer class."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        quantizer = Quantizer()
        assert quantizer.config.mode == QuantizationMode.INT8
        assert quantizer.config.symmetric is True

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = QuantizationConfig(mode=QuantizationMode.INT4)
        quantizer = Quantizer(config)
        assert quantizer.config.mode == QuantizationMode.INT4

    @pytest.mark.parametrize(
        ("mode", "expected_dtype"),
        [
            (QuantizationMode.INT8, np.int8),
            (QuantizationMode.INT4, np.int8),
            (QuantizationMode.FLOAT16, np.float16),
        ],
    )
    def test_quantize_modes(
        self,
        float32_array: npt.NDArray[np.float32],
        mode: QuantizationMode,
        expected_dtype: type,
    ) -> None:
        """Test different quantization modes."""
        config = QuantizationConfig(mode=mode)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        assert quantized.dtype == expected_dtype
        assert quantized.shape == float32_array.shape

    def test_quantize_int8_symmetric(self, float32_array: npt.NDArray[np.float32]) -> None:
        """Test INT8 symmetric quantization."""
        config = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        assert quantized.dtype == np.int8
        assert np.all(quantized >= -127)
        assert np.all(quantized <= 127)

        # Check that scale and zero point are set correctly
        scale, zero_point = quantizer.get_quantization_params()
        assert scale > 0
        assert zero_point == 0  # Symmetric should have zero_point = 0

    def test_quantize_int8_asymmetric(self, float32_array: npt.NDArray[np.float32]) -> None:
        """Test INT8 asymmetric quantization."""
        config = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=False)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        assert quantized.dtype == np.int8
        assert np.all(quantized >= -128)
        assert np.all(quantized <= 127)

        # Check quantization params
        scale, _ = quantizer.get_quantization_params()
        assert scale > 0

    def test_quantize_dequantize_roundtrip(self, float32_array: npt.NDArray[np.float32]) -> None:
        """Test quantize-dequantize roundtrip."""
        config = QuantizationConfig(mode=QuantizationMode.INT8)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        dequantized = quantizer.dequantize(quantized)

        assert dequantized.shape == float32_array.shape
        assert dequantized.dtype == np.float32

        # Check that values are reasonably close
        mse = np.mean((float32_array - dequantized) ** 2)
        assert mse < 1.0  # MSE should be relatively small

    def test_quantize_float16(self, float32_array: npt.NDArray[np.float32]) -> None:
        """Test FLOAT16 quantization."""
        config = QuantizationConfig(mode=QuantizationMode.FLOAT16)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        assert quantized.dtype == np.float16
        assert quantized.shape == float32_array.shape

        # Dequantize should work correctly
        dequantized = quantizer.dequantize(quantized)
        assert dequantized.dtype == np.float32

    def test_quantize_empty_array(self) -> None:
        """Test quantization of empty array."""
        quantizer = Quantizer()
        empty_array = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="Cannot quantize empty weights array"):
            quantizer.quantize(empty_array)

    def test_dequantize_empty_array(self) -> None:
        """Test dequantization of empty array."""
        quantizer = Quantizer()
        empty_array = np.array([], dtype=np.int8)

        with pytest.raises(ValueError, match="Cannot dequantize empty weights array"):
            quantizer.dequantize(empty_array)

    def test_calculate_quantization_error(self, float32_array: npt.NDArray[np.float32]) -> None:
        """Test quantization error calculation."""
        quantizer = Quantizer()

        quantized = quantizer.quantize(float32_array)
        error = quantizer.calculate_quantization_error(float32_array, quantized)

        assert isinstance(error, float)
        assert error >= 0
        assert error < 1.0  # Error should be relatively small

    @pytest.mark.parametrize(
        "mode",
        [
            QuantizationMode.INT8,
            QuantizationMode.INT4,
            QuantizationMode.FLOAT16,
            QuantizationMode.DYNAMIC,
        ],
    )
    def test_all_modes_work(
        self,
        float32_array: npt.NDArray[np.float32],
        mode: QuantizationMode,
    ) -> None:
        """Test that all quantization modes work without errors."""
        config = QuantizationConfig(mode=mode)
        quantizer = Quantizer(config)

        quantized = quantizer.quantize(float32_array)
        dequantized = quantizer.dequantize(quantized)

        assert quantized.shape == float32_array.shape
        assert dequantized.shape == float32_array.shape
        assert dequantized.dtype == np.float32
