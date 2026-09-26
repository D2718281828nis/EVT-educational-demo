"""Deterministic generators for the example datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class KuramotoTimeSeries:
    """Observations and diagnostics from a Kuramoto oscillator simulation."""

    values: NDArray[np.float64]
    coherence: NDArray[np.float64]
    time_step: float


def generate_fractal_extreme_series(
    n_points: int = 1_500,
    hurst_exponent: float = 0.75,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Create unit-variance coloured noise with one smooth extreme event.

    The background is synthesized in the Fourier domain with a power spectrum
    proportional to ``1 / f ** (2H - 1)``.  This is a pedagogical approximation
    to fractional Gaussian noise, not an exact fGn sampler.
    """
    if n_points < 20:
        raise ValueError("n_points must be at least 20")
    if not 0.0 < hurst_exponent < 1.0:
        raise ValueError("hurst_exponent must be between 0 and 1")

    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(n_points)
    amplitudes = np.zeros_like(frequencies)
    amplitudes[1:] = frequencies[1:] ** (-(hurst_exponent - 0.5))
    spectrum = (rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)) * amplitudes
    background = np.fft.irfft(spectrum, n=n_points)
    background = (background - background.mean()) / background.std()

    duration = max(5, n_points // 20)
    low = n_points // 4
    high = n_points - low - duration
    onset = int(rng.integers(low, high + 1))
    event_mask = np.zeros(n_points, dtype=bool)
    event_mask[onset : onset + duration] = True
    pulse = np.sin(np.linspace(0.0, np.pi, duration)) ** 2
    values = background.copy()
    values[event_mask] += 8.0 * pulse
    return values, event_mask


def generate_kuramoto_time_series(
    n_points: int = 1_500,
    n_channels: int = 12,
    source_channel: int = 0,
    seed: int = 42,
    *,
    coupling: float = 2.0,
    time_step: float = 0.02,
) -> KuramotoTimeSeries:
    """Simulate globally coupled oscillators and return their sine signals."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    if not 0 <= source_channel < n_channels:
        raise ValueError("source_channel must identify an existing channel")
    if coupling < 0.0 or time_step <= 0.0:
        raise ValueError("coupling must be non-negative and time_step positive")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, n_channels)
    natural_frequencies = rng.normal(1.0, 0.12, n_channels)
    # Give the nominated channel a small, reproducible frequency lead.
    natural_frequencies[source_channel] += 0.2
    values = np.empty((n_points, n_channels), dtype=float)
    coherence = np.empty(n_points, dtype=float)

    for index in range(n_points):
        values[index] = np.sin(phases)
        order_parameter = np.mean(np.exp(1j * phases))
        coherence[index] = abs(order_parameter)
        phase_difference = phases[np.newaxis, :] - phases[:, np.newaxis]
        interaction = np.sin(phase_difference).mean(axis=1)
        phases += time_step * (natural_frequencies + coupling * interaction)

    return KuramotoTimeSeries(values=values, coherence=coherence, time_step=time_step)
