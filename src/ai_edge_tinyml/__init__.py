"""AI Edge TinyML - State-of-the-art toolkit for edge AI and embedded systems.

This package provides tools and utilities for deploying AI models on edge devices
and embedded systems, including model compression, quantization, and optimization.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Ümit Kacar"
__email__ = "umitkacar@users.noreply.github.com"

from ai_edge_tinyml.model_optimizer import ModelOptimizer
from ai_edge_tinyml.quantization import QuantizationConfig, Quantizer
from ai_edge_tinyml.utils import calculate_compression_ratio, get_model_size

__all__ = [
    "ModelOptimizer",
    "QuantizationConfig",
    "Quantizer",
    "__author__",
    "__email__",
    "__version__",
    "calculate_compression_ratio",
    "get_model_size",
]
