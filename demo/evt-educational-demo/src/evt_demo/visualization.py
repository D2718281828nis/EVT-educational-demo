"""Plots for the generated educational datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_generator import KuramotoTimeSeries


def _save(fig, path: str | Path | None) -> None:
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")


def plot_fractal_extreme_series(values, event_mask, hurst_exponent, save_path=None):
    """Plot a fractal-noise realization and its injected event."""
    series = np.asarray(values, dtype=float)
    mask = np.asarray(event_mask, dtype=bool)
    if series.ndim != 1 or mask.shape != series.shape:
        raise ValueError("values and event_mask must be aligned one-dimensional arrays")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series, color="#355c7d", lw=0.8, label="fractal-noise series")
    ax.scatter(np.flatnonzero(mask), series[mask], s=18, color="#c06c84", label="injected extreme")
    ax.set(title=f"Fractal extreme series (H={hurst_exponent:.2f})", xlabel="timestamp", ylabel="value")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_kuramoto_time_series(series: KuramotoTimeSeries, save_path=None):
    """Plot oscillator observations together with synchronization coherence."""
    time = np.arange(len(series.values)) * series.time_step
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, series.values, lw=0.6, alpha=0.75)
    axes[0].axvline(series.event_onset * series.time_step, color="#c06c84", ls="--", label="source perturbation")
    axes[0].set(ylabel="sin(phase)", title="Synchronized Kuramoto channels")
    axes[0].legend()
    axes[1].plot(time, series.coherence, color="#355c7d")
    axes[1].set(xlabel="time", ylabel="coherence", ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes
