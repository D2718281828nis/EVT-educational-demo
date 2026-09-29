"""Synthetic datasets used by the EVT educational demo."""

from .data_generator import (
    AdditiveFractalSeries,
    FractalExtremeComponents,
    KuramotoPropagationRecording,
    KuramotoPropagationSeries,
    KuramotoTimeSeries,
    generate_additive_fractal_series,
    generate_fractal_extreme_series,
    generate_fractal_extreme_series_components,
    generate_kuramoto_propagation_recording,
    generate_kuramoto_propagation_series,
    generate_kuramoto_time_series,
)
from .visualization import plot_additive_fractal_series_components, plot_evt_source_result

__all__ = [
    "AdditiveFractalSeries",
    "FractalExtremeComponents",
    "KuramotoPropagationRecording",
    "KuramotoPropagationSeries",
    "KuramotoTimeSeries",
    "generate_additive_fractal_series",
    "generate_fractal_extreme_series",
    "generate_fractal_extreme_series_components",
    "generate_kuramoto_propagation_recording",
    "generate_kuramoto_propagation_series",
    "generate_kuramoto_time_series",
    "plot_additive_fractal_series_components",
    "plot_evt_source_result",
]
