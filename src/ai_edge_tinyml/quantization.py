"""Quantization utilities for model compression.

This module provides tools for quantizing neural network models to reduce
their size and improve inference speed on edge devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class QuantizationMode(Enum):
    """Quantization modes."""

    INT8 = "int8"
    INT4 = "int4"
    FLOAT16 = "float16"
    DYNAMIC = "dynamic"


class QuantizationStrategy(Enum):
    """Quantization strategies."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    ASYMMETRIC = "asymmetric"
    SYMMETRIC = "symmetric"


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        mode: Quantization mode (int8, int4, float16, dynamic)
        strategy: Quantization strategy
        calibration_samples: Number of calibration samples for static quantization
        per_channel: Whether to use per-channel quantization
        symmetric: Whether to use symmetric quantization
    """

    mode: QuantizationMode = QuantizationMode.INT8
    strategy: QuantizationStrategy = QuantizationStrategy.PER_TENSOR
    calibration_samples: int = 100
    per_channel: bool = False
    symmetric: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.calibration_samples <= 0:
            msg = f"calibration_samples must be positive, got {self.calibration_samples}"
            raise ValueError(msg)


class Quantizer:
    """Quantize neural network models for edge deployment.

    This class implements various quantization techniques including INT8, INT4,
    and FP16 quantization with support for both symmetric and asymmetric strategies.

    Examples:
        >>> config = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        >>> quantizer = Quantizer(config)
        >>> quantized_weights = quantizer.quantize(weights)
        >>> dequantized = quantizer.dequantize(quantized_weights)
    """

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        """Initialize the quantizer.

        Args:
            config: Quantization configuration. If None, uses default config.
        """
        self.config = config or QuantizationConfig()
        self._scale: float = 1.0
        self._zero_point: int = 0

    def quantize(
        self,
        weights: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int8] | npt.NDArray[np.float16]:
        """Quantize weights using configured quantization mode.

        Args:
            weights: Floating-point weights to quantize

        Returns:
            Quantized weights

        Raises:
            ValueError: If weights are invalid
        """
        if weights.size == 0:
            msg = "Cannot quantize empty weights array"
            raise ValueError(msg)

        match self.config.mode:
            case QuantizationMode.INT8:
                return self._quantize_int8(weights)
            case QuantizationMode.INT4:
                return self._quantize_int4(weights)
            case QuantizationMode.FLOAT16:
                return self._quantize_float16(weights)
            case QuantizationMode.DYNAMIC:
                return self._quantize_dynamic(weights)

    def dequantize(
        self,
        quantized_weights: npt.NDArray[np.int8] | npt.NDArray[np.float16],
    ) -> npt.NDArray[np.float32]:
        """Dequantize weights back to float32.

        Args:
            quantized_weights: Quantized weights

        Returns:
            Dequantized floating-point weights

        Raises:
            ValueError: If quantized_weights are invalid
        """
        if quantized_weights.size == 0:
            msg = "Cannot dequantize empty weights array"
            raise ValueError(msg)

        match self.config.mode:
            case QuantizationMode.FLOAT16:
                return quantized_weights.astype(np.float32)
            case QuantizationMode.INT8 | QuantizationMode.INT4 | QuantizationMode.DYNAMIC:
                # Type narrowing for mypy - quantized weights are int8 for these modes
                int_weights = quantized_weights.astype(np.int8)
                return self._dequantize_int(int_weights)

    def _quantize_int8(self, weights: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
        """Quantize to INT8.

        Args:
            weights: Input weights

        Returns:
            INT8 quantized weights
        """
        if self.config.symmetric:
            # Symmetric quantization: range [-127, 127]
            max_val = np.abs(weights).max()
            self._scale = max_val / 127.0 if max_val > 0 else 1.0
            self._zero_point = 0
            quantized = np.round(weights / self._scale).clip(-127, 127)
        else:
            # Asymmetric quantization: range [-128, 127]
            min_val, max_val = weights.min(), weights.max()
            self._scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            self._zero_point = int(np.round(-min_val / self._scale - 128))
            quantized = np.round(weights / self._scale + self._zero_point).clip(-128, 127)

        return quantized.astype(np.int8)

    def _quantize_int4(self, weights: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
        """Quantize to INT4 (stored as INT8).

        Args:
            weights: Input weights

        Returns:
            INT4 quantized weights (stored as INT8)
        """
        # INT4 range: [-8, 7] for symmetric, [-8, 7] for asymmetric
        if self.config.symmetric:
            max_val = np.abs(weights).max()
            self._scale = max_val / 7.0 if max_val > 0 else 1.0
            self._zero_point = 0
            quantized = np.round(weights / self._scale).clip(-7, 7)
        else:
            min_val, max_val = weights.min(), weights.max()
            self._scale = (max_val - min_val) / 15.0 if max_val > min_val else 1.0
            self._zero_point = int(np.round(-min_val / self._scale - 8))
            quantized = np.round(weights / self._scale + self._zero_point).clip(-8, 7)

        return quantized.astype(np.int8)

    def _quantize_float16(self, weights: npt.NDArray[np.float32]) -> npt.NDArray[np.float16]:
        """Quantize to FLOAT16.

        Args:
            weights: Input weights

        Returns:
            FLOAT16 quantized weights
        """
        return weights.astype(np.float16)

    def _quantize_dynamic(self, weights: npt.NDArray[np.float32]) -> npt.NDArray[np.int8]:
        """Apply dynamic quantization.

        Args:
            weights: Input weights

        Returns:
            Dynamically quantized weights
        """
        # For dynamic quantization, use per-tensor INT8 with symmetric strategy
        return self._quantize_int8(weights)

    def _dequantize_int(
        self,
        quantized_weights: npt.NDArray[np.int8],
    ) -> npt.NDArray[np.float32]:
        """Dequantize integer quantized weights.

        Args:
            quantized_weights: Quantized weights

        Returns:
            Dequantized weights
        """
        return (quantized_weights.astype(np.float32) - self._zero_point) * self._scale

    def get_quantization_params(self) -> tuple[float, int]:
        """Get quantization parameters (scale and zero point).

        Returns:
            Tuple of (scale, zero_point)
        """
        return (self._scale, self._zero_point)

    def calculate_quantization_error(
        self,
        original: npt.NDArray[np.float32],
        quantized: npt.NDArray[np.int8] | npt.NDArray[np.float16],
    ) -> float:
        """Calculate quantization error (MSE).

        Args:
            original: Original float32 weights
            quantized: Quantized weights

        Returns:
            Mean squared error between original and dequantized weights
        """
        dequantized = self.dequantize(quantized)
        return float(np.mean((original - dequantized) ** 2))
