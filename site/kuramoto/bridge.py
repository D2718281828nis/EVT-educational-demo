"""Glue between the interactive page and ``evt_demo.data_generator``.

Runs inside Pyodide (see ``worker.js``). Every number the page draws comes
from the repository's own generator; this module only packs its outputs into
little-endian float64 bytes so the worker can hand them to the page.
"""

from __future__ import annotations

import json

import numpy as np

from evt_demo import data_generator as gen


def _pack(array) -> bytes:
    return np.ascontiguousarray(array, dtype="<f8").ravel().tobytes()


def defaults(n_channels: int, source_channel: int, seed: int) -> str:
    """The seeded natural frequencies and initial phases the generator draws."""
    base = gen.generate_kuramoto_time_series(2, n_channels, source_channel, seed)
    return json.dumps({
        "natural_frequencies": base.natural_frequencies.tolist(),
        "initial_phases": base.initial_phases.tolist(),
    })


def run(params_json: str) -> dict:
    p = json.loads(params_json)
    args = (int(p["n_points"]), int(p["n_channels"]), int(p["source_channel"]), int(p["seed"]))
    common = dict(
        coupling=float(p["coupling"]),
        time_step=float(p["time_step"]),
        natural_frequencies=np.array(p["natural_frequencies"], dtype=float),
        initial_phases=np.array(p["initial_phases"], dtype=float),
    )
    base = gen.generate_kuramoto_time_series(*args, **common)
    n_points, n_channels = base.values.shape

    noise = p["noise"]
    if noise["enabled"]:
        series = gen.generate_kuramoto_propagation_series(
            *args, **common,
            hurst_exponent=float(noise["hurst_exponent"]),
            event_magnitude_in_sigma=float(noise["event_magnitude_in_sigma"]),
            delay_per_hop=int(noise["delay_per_hop"]),
            decay_per_hop=float(noise["decay_per_hop"]),
            event_onset=int(noise["event_onset"]),
            event_duration=int(noise["event_duration"]),
        )
        values = series.values
        channel_event_mask = series.channel_event_mask
        arrival = series.channel_arrival_delay
        meta = {"event_onset": series.event_onset, "event_duration": series.event_duration}
    else:
        values = base.values
        channel_event_mask = np.zeros_like(values, dtype=bool)
        arrival = np.zeros(n_channels)
        meta = {"event_onset": None, "event_duration": 0}

    meta.update(n_points=n_points, n_channels=n_channels, time_step=base.time_step)
    return {
        "meta": json.dumps(meta),
        "values": _pack(values),
        "background": _pack(base.values),
        "coherence": _pack(base.coherence),
        "channel_event_mask": _pack(channel_event_mask),
        "arrival": _pack(arrival),
    }
