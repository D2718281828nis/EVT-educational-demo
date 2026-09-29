"""Plotting helpers for synthetic example datasets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from graph_evt_agent import PipelineResult

from .data_generator import AdditiveFractalSeries, KuramotoPropagationSeries, KuramotoTimeSeries


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


def plot_additive_fractal_series_components(
    series: AdditiveFractalSeries,
    output_path: str | Path | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Show where and how an :class:`AdditiveFractalSeries` was built.

    Four stacked panels: the three additive components on their own, and
    their sum (the actual series) with the event window and onset marked, so
    the construction is visible rather than only stated.
    """
    n_points = series.values.shape[0]
    time = np.arange(n_points)
    window_end = min(series.event_onset + series.event_duration - 1, n_points - 1)

    figure, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True, constrained_layout=True)

    axes[0].plot(time, series.oscillators, color="#7c3aed", linewidth=0.9)
    periods = ", ".join(f"{p:g}" for p in series.oscillator_periods)
    axes[0].set(title=f"1. Oscillators: sum of 3 sine waves (periods {periods}), deterministic", ylabel="value")

    axes[1].plot(time, series.extreme_injection, color="#dc2626", linewidth=0.9)
    axes[1].axvspan(series.event_onset, window_end, color="#dc2626", alpha=0.08)
    axes[1].set(title=f"2. Extreme injection: Hurst-colored burst (H={series.hurst_exponent:g}) "
                      f"from event_onset={series.event_onset} -- what EVT is meant to catch",
               ylabel="value")

    axes[2].plot(time, series.simple_noise, color="#334155", linewidth=0.6)
    axes[2].set(title="3. Simple stochastic: i.i.d. Gaussian noise, unstructured, everywhere", ylabel="value")

    axes[3].plot(time, series.values, color="#0f766e", linewidth=0.8, label="sum = full series")
    axes[3].fill_between(time, series.values.min(), series.values.max(), where=series.event_mask,
                         color="#dc2626", alpha=0.12, label="event window")
    axes[3].axvline(series.event_onset, color="#dc2626", linestyle="--", linewidth=1,
                    label=f"event_onset ({series.event_onset})")
    axes[3].set(title="4. Sum: oscillators + extreme injection + simple noise", xlabel="sample", ylabel="value")
    axes[3].legend(loc="upper left", fontsize=8)

    for axis in axes:
        axis.grid(alpha=.2)
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


def plot_kuramoto_propagation_series(
    series: KuramotoPropagationSeries,
    channel_names: list[str] | tuple[str, ...] | None = None,
    output_path: str | Path | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot a propagation series with the source channel and event front marked.

    The top panel highlights the source channel's trace against the rest.
    The bottom panel shows the ground-truth ``channel_event_mask``, with
    channels ordered by their arrival delay from the source, so the
    propagation front is visible directly rather than only in metadata.
    """
    n_channels = series.values.shape[1]
    names = list(channel_names or [f"channel_{index:02d}" for index in range(n_channels)])
    if len(names) != n_channels:
        raise ValueError("channel_names must match the number of channels")
    time = np.arange(series.values.shape[0]) * series.time_step

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    for channel in range(n_channels):
        is_source = channel == series.source_channel
        axes[0].plot(
            time, series.values[:, channel],
            color="black" if is_source else "tab:blue",
            linewidth=1.6 if is_source else 0.6,
            alpha=1.0 if is_source else 0.5,
            zorder=3 if is_source else 1,
            label=f"{names[channel]} (source)" if is_source else None,
        )
    window_end = min(series.event_onset + series.event_duration - 1, len(time) - 1)
    axes[0].axvspan(time[series.event_onset], time[window_end], color="tab:red", alpha=0.12,
                    label="event window at source")
    axes[0].set(title="Kuramoto propagation series (source channel highlighted)", ylabel="value")
    axes[0].legend(loc="upper left")

    order = np.argsort(series.channel_arrival_delay)
    image = axes[1].imshow(
        series.channel_event_mask[:, order].T.astype(float),
        aspect="auto", origin="lower", cmap="Reds", vmin=0, vmax=1,
        extent=(time[0], time[-1], -0.5, n_channels - 0.5),
    )
    axes[1].set(
        yticks=np.arange(n_channels), yticklabels=[names[index] for index in order],
        xlabel="time", ylabel="channel (sorted by arrival delay)",
        title="Ground-truth propagation front",
    )
    figure.colorbar(image, ax=axes[1], label="event active (ground truth)")
    _save(figure, output_path)
    return figure, axes


def plot_evt_source_result(
    values: ArrayLike,
    result: PipelineResult,
    *,
    timestamps: ArrayLike | None = None,
    channel_names: list[str] | tuple[str, ...] | None = None,
    baseline_size: int | None = None,
    context: int = 80,
    output_path: str | Path | None = None,
) -> tuple[Figure, NDArray[np.object_]]:
    """Visualize the complete EVT detection-to-source result.

    The first panel shows *why and when* EVT raised an alarm.  The second shows
    standardized channel responses around that time, while the third exposes
    the source-localization probabilities instead of reporting only an argmax.
    The function deliberately consumes a ``PipelineResult`` so the figure is a
    view of the library result, not a second implementation of EVT.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] != len(result.detection.indicator):
        raise ValueError("values must be time × channel and match the EVT indicator")
    if context < 1:
        raise ValueError("context must be positive")

    time = np.arange(len(data)) if timestamps is None else np.asarray(timestamps)
    if time.ndim != 1 or len(time) != len(data):
        raise ValueError("timestamps must be a 1-D array matching values")
    names = list(channel_names or result.input_profile.channel_names)
    if len(names) != data.shape[1]:
        names = [f"channel_{index:02d}" for index in range(data.shape[1])]

    detection = result.detection
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(time, detection.indicator, color="tab:blue", label="EVT indicator")
    axes[0].axhline(
        detection.alarm_threshold,
        color="tab:red",
        linestyle="--",
        label="POT/GPD alarm threshold",
    )
    if baseline_size is not None:
        if not 1 <= baseline_size <= len(data):
            raise ValueError("baseline_size must identify a non-empty prefix")
        axes[0].axvspan(time[0], time[baseline_size - 1], color="0.8", alpha=0.35,
                        label="baseline")
    axes[0].set(ylabel="indicator", title="1. EVT detection")

    if not detection.detected or detection.time_index is None:
        axes[0].text(0.5, 0.85, "No persistent EVT alarm", transform=axes[0].transAxes,
                     ha="center", color="tab:red")
        for axis, message in zip(axes[1:], ("No event window", "Source localization skipped")):
            axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")
            axis.set_axis_off()
        axes[0].legend(loc="best")
        _save(figure, output_path)
        return figure, axes

    event = detection.time_index
    axes[0].axvline(time[event], color="black", linewidth=1.5,
                    label=f"detected at index {event}")
    axes[0].legend(loc="best", ncols=2)

    start, stop = max(0, event - context), min(len(data), event + context + 1)
    standardized = np.abs(data - detection.location) / np.maximum(detection.scale, 1e-12)
    image = axes[1].imshow(
        standardized[start:stop].T,
        aspect="auto",
        origin="lower",
        extent=(start, stop - 1, -0.5, data.shape[1] - 0.5),
        cmap="magma",
    )
    axes[1].axvline(event, color="cyan", linestyle="--", linewidth=1.5)
    axes[1].set(yticks=np.arange(data.shape[1]), yticklabels=names,
                ylabel="channel", title="2. Standardized response around the alarm")
    figure.colorbar(image, ax=axes[1], label="absolute robust z-score")

    ranking = result.ranking
    if ranking is None:
        axes[2].text(0.5, 0.5, "Source localization skipped", transform=axes[2].transAxes,
                     ha="center", va="center")
        axes[2].set_axis_off()
    else:
        node_ids = np.asarray(ranking.node_ids, dtype=int)
        probabilities = np.asarray(ranking.probabilities, dtype=float)
        colors = ["tab:red" if node == ranking.source else "tab:blue" for node in node_ids]
        axes[2].bar(np.arange(len(node_ids)), probabilities, color=colors)
        axes[2].set(
            xticks=np.arange(len(node_ids)),
            xticklabels=[names[node] for node in node_ids],
            ylabel="probability",
            title=f"3. Initial-source ranking — estimate: {names[ranking.source]}",
        )
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].set_ylim(0, max(1.0, probabilities.max() * 1.1))

    axes[-1].set_xlabel("ranked channels" if ranking is not None else "")
    _save(figure, output_path)
    return figure, axes


def _save(figure: Figure, output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
