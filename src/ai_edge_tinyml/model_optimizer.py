"""Model optimization utilities for edge deployment.

This module provides tools for optimizing neural network models for deployment
on edge devices and embedded systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class OptimizationLevel(Enum):
    """Model optimization levels."""

    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    ULTRA = "ultra"


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for model optimization.

    Attributes:
        level: Optimization level to apply
        target_size_mb: Target model size in megabytes
        preserve_accuracy: Whether to preserve model accuracy
        prune_threshold: Threshold for weight pruning (0.0 to 1.0)
        enable_fusion: Enable operator fusion optimizations
    """

    level: OptimizationLevel = OptimizationLevel.BASIC
    target_size_mb: float | None = None
    preserve_accuracy: bool = True
    prune_threshold: float = 0.01
    enable_fusion: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.prune_threshold <= 1.0:
            msg = f"prune_threshold must be between 0.0 and 1.0, got {self.prune_threshold}"
            raise ValueError(msg)

        if self.target_size_mb is not None and self.target_size_mb <= 0:
            msg = f"target_size_mb must be positive, got {self.target_size_mb}"
            raise ValueError(msg)


class ModelProtocol(Protocol):
    """Protocol for model objects."""

    def get_weights(self) -> dict[str, npt.NDArray[np.float32]]:
        """Get model weights."""
        ...

    def set_weights(self, weights: dict[str, npt.NDArray[np.float32]]) -> None:
        """Set model weights."""
        ...

    def save(self, path: Path) -> None:
        """Save model to file."""
        ...


class ModelOptimizer:
    """Optimize neural network models for edge deployment.

    This class provides various optimization techniques including pruning,
    quantization-aware training, and model compression.

    Examples:
        >>> optimizer = ModelOptimizer(OptimizationConfig(level=OptimizationLevel.AGGRESSIVE))
        >>> optimized_model = optimizer.optimize(model)
        >>> compression_ratio = optimizer.get_compression_ratio()
    """

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        """Initialize the model optimizer.

        Args:
            config: Optimization configuration. If None, uses default config.
        """
        self.config = config or OptimizationConfig()
        self._original_size: float = 0.0
        self._optimized_size: float = 0.0

    def optimize(self, model: ModelProtocol) -> ModelProtocol:
        """Optimize a model using configured optimization techniques.

        Args:
            model: Model to optimize

        Returns:
            Optimized model

        Raises:
            ValueError: If model is invalid or optimization fails
        """
        weights = model.get_weights()
        self._original_size = self._calculate_size(weights)

        # Apply optimization based on level
        if self.config.level == OptimizationLevel.NONE:
            return model

        optimized_weights = self._apply_pruning(weights)

        if self.config.enable_fusion:
            optimized_weights = self._apply_fusion(optimized_weights)

        model.set_weights(optimized_weights)
        self._optimized_size = self._calculate_size(optimized_weights)

        return model

    def _apply_pruning(
        self,
        weights: dict[str, npt.NDArray[np.float32]],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Apply weight pruning to model weights.

        Args:
            weights: Model weights dictionary

        Returns:
            Pruned weights dictionary
        """
        pruned_weights: dict[str, npt.NDArray[np.float32]] = {}

        for name, weight in weights.items():
            # Prune small weights
            mask = np.abs(weight) > self.config.prune_threshold
            pruned_weights[name] = weight * mask

        return pruned_weights

    def _apply_fusion(
        self,
        weights: dict[str, npt.NDArray[np.float32]],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Apply operator fusion optimizations.

        Args:
            weights: Model weights dictionary

        Returns:
            Fused weights dictionary
        """
        # Placeholder for fusion logic
        return weights

    def _calculate_size(self, weights: dict[str, npt.NDArray[np.float32]]) -> float:
        """Calculate total size of weights in MB.

        Args:
            weights: Model weights dictionary

        Returns:
            Total size in megabytes
        """
        total_bytes = sum(w.nbytes for w in weights.values())
        return total_bytes / (1024 * 1024)

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio achieved.

        Returns:
            Compression ratio (original_size / optimized_size)

        Raises:
            RuntimeError: If optimization hasn't been run yet
        """
        if self._original_size == 0 or self._optimized_size == 0:
            msg = "Must run optimize() before calculating compression ratio"
            raise RuntimeError(msg)

        return self._original_size / self._optimized_size

    def get_size_reduction_mb(self) -> float:
        """Calculate size reduction in MB.

        Returns:
            Size reduction in megabytes

        Raises:
            RuntimeError: If optimization hasn't been run yet
        """
        if self._original_size == 0 or self._optimized_size == 0:
            msg = "Must run optimize() before calculating size reduction"
            raise RuntimeError(msg)

        return self._original_size - self._optimized_size
