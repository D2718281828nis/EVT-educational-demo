"""Plotting helpers for synthetic example datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .data_generator import KuramotoTimeSeries


def plot_fractal_extreme_series(
    values: ArrayLike,
    event_mask: ArrayLike,
    hurst_exponent: float,
    output_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot the coloured-noise series and highlight its injected event."""
    values_array = np.asarray(values)
    mask_array = np.asarray(event_mask, dtype=bool)
    if values_array.ndim != 1 or mask_array.shape != values_array.shape:
        raise ValueError("values and event_mask must be equally sized 1-D arrays")
    figure, axes = plt.subplots(figsize=(11, 4))
    axes.plot(values_array, linewidth=0.8, label="signal")
    axes.fill_between(
        np.arange(values_array.size),
        values_array.min(),
        values_array.max(),
        where=mask_array,
        alpha=0.2,
        color="tab:red",
        label="injected extreme",
    )
    axes.set(title=f"Fractal-noise example (H={hurst_exponent:g})", xlabel="sample", ylabel="value")
    axes.legend()
    _save(figure, output_path)
    return figure, axes


def plot_kuramoto_time_series(
    series: KuramotoTimeSeries,
    output_path: str | Path | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot oscillator signals together with their phase coherence."""
    time = np.arange(series.values.shape[0]) * series.time_step
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time, series.values, linewidth=0.7, alpha=0.75)
    axes[0].set(ylabel="sin(phase)", title="Coupled Kuramoto oscillators")
    axes[1].plot(time, series.coherence, color="black")
    axes[1].set(xlabel="time", ylabel="coherence", ylim=(0.0, 1.05))
    figure.tight_layout()
    _save(figure, output_path)
    return figure, axes


def _save(figure: Figure, output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
