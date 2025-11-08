"""AI Edge TinyML - State-of-the-art toolkit for edge AI and embedded systems.

This package provides tools and utilities for deploying AI models on edge devices
and embedded systems, including model compression, quantization, and optimization.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ümit Kacar"
__email__ = "umitkacar@users.noreply.github.com"

from ai_edge_tinyml.model_optimizer import ModelOptimizer
from ai_edge_tinyml.quantization import Quantizer, QuantizationConfig
from ai_edge_tinyml.utils import get_model_size, calculate_compression_ratio

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "ModelOptimizer",
    "Quantizer",
    "QuantizationConfig",
    "get_model_size",
    "calculate_compression_ratio",
]
