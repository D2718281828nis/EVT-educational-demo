"""Dataset generation utilities for the educational EVT notebooks."""

from .data_generator import (
    KuramotoTimeSeries,
    generate_fractal_extreme_series,
    generate_kuramoto_time_series,
)

__all__ = [
    "KuramotoTimeSeries",
    "generate_fractal_extreme_series",
    "generate_kuramoto_time_series",
]
