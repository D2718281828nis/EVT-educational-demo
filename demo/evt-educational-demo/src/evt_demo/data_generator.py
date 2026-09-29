"""Deterministic generators for the example datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from graph_evt_agent.input import TimeSeriesInput


@dataclass(frozen=True)
class KuramotoTimeSeries:
    """Observations and diagnostics from a Kuramoto oscillator simulation."""

    values: NDArray[np.float64]
    coherence: NDArray[np.float64]
    time_step: float


@dataclass(frozen=True)
class KuramotoPropagationSeries:
    """A Kuramoto recording with one Hurst/DFA-shaped extreme event injected
    at a source channel and propagated to every other channel."""

    values: NDArray[np.float64]
    coherence: NDArray[np.float64]
    time_step: float
    source_channel: int
    event_onset: int
    event_duration: int
    channel_arrival_delay: NDArray[np.int64]
    event_mask: NDArray[np.bool_]
    channel_event_mask: NDArray[np.bool_]


def _colored_noise(n_points: int, hurst_exponent: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Unit-variance colored noise via the same 1/f-style spectral shaping
    :func:`generate_fractal_extreme_series` uses for its background."""
    frequencies = np.fft.rfftfreq(n_points)
    amplitudes = np.zeros_like(frequencies)
    amplitudes[1:] = frequencies[1:] ** (-(hurst_exponent - 0.5))
    spectrum = (rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)) * amplitudes
    noise = np.fft.irfft(spectrum, n=n_points)
    return (noise - noise.mean()) / noise.std()


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


def generate_kuramoto_propagation_series(
    n_points: int = 1_500,
    n_channels: int = 12,
    source_channel: int = 0,
    seed: int = 42,
    *,
    coupling: float = 2.0,
    time_step: float = 0.02,
    hurst_exponent: float = 0.75,
    event_magnitude_in_sigma: float = 8.0,
    delay_per_hop: int = 5,
    decay_per_hop: float = 0.65,
) -> KuramotoPropagationSeries:
    """Inject one Hurst/DFA-shaped extreme event that starts at ``source_channel``
    and propagates with ring-topology delay and decay to every other channel.

    The background is the same coupled-oscillator simulation as
    :func:`generate_kuramoto_time_series`. The injected event is a tapered
    burst of ``hurst_exponent``-colored noise -- the same spectral-shaping
    technique :func:`generate_fractal_extreme_series` uses for its 1-D
    background -- so the same Hurst/DFA parameter that shapes the 1-D dataset
    also controls the tail heaviness an EVT detector sees here. Each channel
    receives a delayed (``delay_per_hop`` samples per ring hop from the
    source) and attenuated (``decay_per_hop`` per hop) copy of that burst, so
    a graph-aware EVT/source-ranking pipeline has a genuine propagation front
    to localize back to ``source_channel``.
    """
    if n_points < 20:
        raise ValueError("n_points must be at least 20")
    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")
    if not 0 <= source_channel < n_channels:
        raise ValueError("source_channel must identify an existing channel")
    if not 0.0 < hurst_exponent < 1.0:
        raise ValueError("hurst_exponent must be between 0 and 1")

    base = generate_kuramoto_time_series(
        n_points, n_channels, source_channel, seed, coupling=coupling, time_step=time_step,
    )
    values = base.values.copy()

    # A stream independent of the oscillator simulation's own RNG consumption,
    # so this event injection never perturbs generate_kuramoto_time_series.
    event_rng = np.random.default_rng(np.random.SeedSequence([seed, n_channels, source_channel]))

    duration = max(5, n_points // 20)
    hops = np.minimum(
        np.abs(np.arange(n_channels) - source_channel),
        n_channels - np.abs(np.arange(n_channels) - source_channel),
    )
    channel_arrival_delay = (hops * delay_per_hop).astype(np.int64)
    max_delay = int(channel_arrival_delay.max())

    low = n_points // 4
    high = n_points - duration - max_delay
    if high <= low:
        raise ValueError("n_points is too small for the requested channel count and delay_per_hop")
    onset = int(event_rng.integers(low, high + 1))

    pulse = _colored_noise(duration, hurst_exponent, event_rng)
    taper = np.sin(np.linspace(0.0, np.pi, duration)) ** 2
    burst = event_magnitude_in_sigma * pulse * taper

    channel_event_mask = np.zeros((n_points, n_channels), dtype=bool)
    for channel in range(n_channels):
        start = onset + int(channel_arrival_delay[channel])
        stop = start + duration
        gain = decay_per_hop ** hops[channel]
        values[start:stop, channel] += gain * burst
        channel_event_mask[start:stop, channel] = True

    return KuramotoPropagationSeries(
        values=values,
        coherence=base.coherence,
        time_step=time_step,
        source_channel=source_channel,
        event_onset=onset,
        event_duration=duration,
        channel_arrival_delay=channel_arrival_delay,
        event_mask=channel_event_mask.any(axis=1),
        channel_event_mask=channel_event_mask,
    )


@dataclass(frozen=True)
class KuramotoPropagationRecording:
    """A long Kuramoto recording holding several independently-sourced
    Hurst/DFA-shaped propagation events -- the multi-episode counterpart of
    :class:`KuramotoPropagationSeries`, built the way
    :func:`graph_evt_agent.synthetic.make_recording` extends
    :func:`graph_evt_agent.synthetic.make_propagation_episode`, but on a real
    Kuramoto oscillator background instead of the library's simpler
    noise-plus-sine synthetic model."""

    values: NDArray[np.float64]
    coherence: NDArray[np.float64]
    time_step: float
    event_times: tuple[int, ...]
    sources: tuple[int, ...]
    channel_names: tuple[str, ...]
    channel_arrival_delays: tuple[NDArray[np.int64], ...]
    event_duration: int

    def as_input(self) -> TimeSeriesInput:
        """Wrap values with channel names for the pipeline, labeller or evaluator."""
        return TimeSeriesInput(self.values, channel_names=self.channel_names)


def generate_kuramoto_propagation_recording(
    n_events: int = 4,
    n_channels: int = 12,
    seed: int | None = None,
    *,
    baseline_size: int = 300,
    gap: int = 150,
    coupling: float = 2.0,
    time_step: float = 0.02,
    hurst_exponent: float = 0.75,
    event_magnitude_in_sigma: float = 8.0,
    delay_per_hop: int = 5,
    decay_per_hop: float = 0.65,
) -> KuramotoPropagationRecording:
    """A long recording holding ``n_events`` well-separated Kuramoto
    propagation events, each with an independently drawn source channel.

    Same generative family as :func:`generate_kuramoto_propagation_series`
    (real Kuramoto oscillator background, Hurst/DFA-colored propagating
    bursts) instead of the library's built-in synthetic ``make_recording``,
    so training data for :class:`~graph_evt_agent.learning.ProcessModelTrainer`
    matches the distribution a real Kuramoto recording actually has.
    """
    if n_events < 1:
        raise ValueError("n_events must be positive")
    if gap < 60:
        raise ValueError("gap must be at least 60 samples")
    if n_channels < 2:
        raise ValueError("n_channels must be at least 2")

    length = baseline_size + gap * n_events + gap // 2
    base = generate_kuramoto_time_series(
        length, n_channels, source_channel=0, seed=seed, coupling=coupling, time_step=time_step,
    )
    values = base.values.copy()
    channel_names = tuple(f"channel_{index:02d}" for index in range(n_channels))

    event_rng = np.random.default_rng(np.random.SeedSequence([0 if seed is None else seed, n_channels, n_events]))
    duration = max(5, min(gap // 3, length // 20))
    event_times = tuple(baseline_size + gap // 2 + gap * index for index in range(n_events))
    sources = tuple(int(channel) for channel in event_rng.integers(0, n_channels, n_events))

    channel_indices = np.arange(n_channels)
    channel_arrival_delays = []
    for onset, source in zip(event_times, sources):
        hops = np.minimum(np.abs(channel_indices - source), n_channels - np.abs(channel_indices - source))
        delays = (hops * delay_per_hop).astype(np.int64)
        pulse = _colored_noise(duration, hurst_exponent, event_rng)
        taper = np.sin(np.linspace(0.0, np.pi, duration)) ** 2
        burst = event_magnitude_in_sigma * pulse * taper
        for channel in range(n_channels):
            start = onset + int(delays[channel])
            if start >= length:
                continue
            stop = min(length, start + duration)
            gain = decay_per_hop ** hops[channel]
            values[start:stop, channel] += gain * burst[: stop - start]
        channel_arrival_delays.append(delays)

    return KuramotoPropagationRecording(
        values=values,
        coherence=base.coherence,
        time_step=time_step,
        event_times=event_times,
        sources=sources,
        channel_names=channel_names,
        channel_arrival_delays=tuple(channel_arrival_delays),
        event_duration=duration,
    )
