"""Synthetic datasets used by the EVT educational demo."""

from .data_generator import (
    KuramotoTimeSeries,
    generate_fractal_extreme_series,
    generate_kuramoto_time_series,
)
from .visualization import plot_evt_source_result

__all__ = [
    "KuramotoTimeSeries",
    "generate_fractal_extreme_series",
    "generate_kuramoto_time_series",
    "plot_evt_source_result",
]
