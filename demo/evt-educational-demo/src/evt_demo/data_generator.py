"""Reproducible synthetic datasets used by the educational notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KuramotoTimeSeries:
    """Simulation output for coupled oscillators with one perturbed source."""

    values: np.ndarray
    phases: np.ndarray
    coherence: np.ndarray
    time_step: float
    source_channel: int
    event_onset: int


def generate_fractal_extreme_series(
    n_points: int = 1_500,
    hurst_exponent: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized fractal noise with one smooth extreme episode.

    The background is synthesized in the frequency domain with power spectrum
    ``S(f) ~ f**(1 - 2H)``, the fractional-Gaussian-noise relationship.  This is
    a finite educational approximation rather than an exact fGn sampler.
    """
    if n_points < 200:
        raise ValueError("n_points must be at least 200")
    if not 0 < hurst_exponent < 1:
        raise ValueError("hurst_exponent must be between 0 and 1")

    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(n_points)
    spectrum = np.zeros(frequencies.size, dtype=complex)
    amplitude = frequencies[1:] ** (0.5 - hurst_exponent)
    spectrum[1:] = amplitude * (
        rng.normal(size=amplitude.size) + 1j * rng.normal(size=amplitude.size)
    )
    values = np.fft.irfft(spectrum, n=n_points)
    values = (values - values.mean()) / values.std(ddof=0)

    duration = max(12, n_points // 60)
    onset_low = n_points // 2
    onset_high = n_points - duration - max(20, n_points // 20)
    onset = int(rng.integers(onset_low, onset_high + 1))
    event_mask = np.zeros(n_points, dtype=bool)
    event_mask[onset : onset + duration] = True
    pulse = np.sin(np.linspace(0, np.pi, duration + 2)[1:-1])
    values[event_mask] += 8.0 * pulse
    return values, event_mask


def generate_kuramoto_time_series(
    n_points: int = 1_500,
    n_channels: int = 12,
    source_channel: int = 0,
    seed: int = 42,
    *,
    coupling: float = 2.0,
    time_step: float = 0.05,
) -> KuramotoTimeSeries:
    """Simulate noisy globally coupled Kuramoto oscillators and a source kick."""
    if n_points < 200:
        raise ValueError("n_points must be at least 200")
    if n_channels < 3:
        raise ValueError("n_channels must be at least 3")
    if not 0 <= source_channel < n_channels:
        raise ValueError("source_channel is outside the channel range")
    if coupling <= 0 or time_step <= 0:
        raise ValueError("coupling and time_step must be positive")

    rng = np.random.default_rng(seed)
    natural_frequency = rng.normal(1.0, 0.08, n_channels)
    phases = np.empty((n_points, n_channels), dtype=float)
    phases[0] = rng.uniform(-np.pi, np.pi, n_channels)
    onset = n_points * 2 // 3

    for index in range(1, n_points):
        previous = phases[index - 1]
        interaction = np.sin(previous[None, :] - previous[:, None]).mean(axis=1)
        noise = rng.normal(0.0, 0.025, n_channels)
        phases[index] = previous + time_step * (
            natural_frequency + coupling * interaction
        ) + noise
        if index == onset:
            phases[index, source_channel] += np.pi

    values = np.sin(phases)
    coherence = np.abs(np.exp(1j * phases).mean(axis=1))
    return KuramotoTimeSeries(
        values=values,
        phases=phases,
        coherence=coherence,
        time_step=time_step,
        source_channel=source_channel,
        event_onset=onset,
    )
