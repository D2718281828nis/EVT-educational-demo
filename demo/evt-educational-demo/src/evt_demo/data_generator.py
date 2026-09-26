"""Synthetic multichannel graph time series with a propagating extreme event."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraphTimeSeries:
    values: np.ndarray  # [time, nodes]
    adjacency: np.ndarray  # [nodes, nodes]
    groups: np.ndarray  # structural node groups
    source_node: int
    event_onset: int
    arrival_times: np.ndarray


def _graph_distances(adjacency: np.ndarray, source: int) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf)
    distance[source] = 0
    frontier = [source]
    while frontier:
        node = frontier.pop(0)
        for neighbour in np.flatnonzero(adjacency[node]):
            if not np.isfinite(distance[neighbour]):
                distance[neighbour] = distance[node] + 1
                frontier.append(int(neighbour))
    return distance


def generate_graph_time_series(
    n_points: int = 1_200,
    n_nodes: int = 18,
    n_groups: int = 3,
    phi: float = 0.72,
    sigma: float = 0.7,
    event_onset: int = 850,
    source_node: int | None = None,
    magnitude: float = 8.0,
    seed: int = 42,
) -> GraphTimeSeries:
    """Generate correlated AR(1) channels and an event spreading over graph edges."""
    if n_points < 300 or n_nodes < 6 or not 2 <= n_groups <= n_nodes:
        raise ValueError("Need n_points>=300, n_nodes>=6 and 2<=n_groups<=n_nodes")
    if not -1 < phi < 1 or sigma <= 0 or magnitude <= 0:
        raise ValueError("Require abs(phi)<1 and positive sigma/magnitude")
    if not n_points // 2 <= event_onset < n_points - 80:
        raise ValueError("event_onset must leave baseline and event windows")
    rng = np.random.default_rng(seed)
    groups = np.arange(n_nodes) * n_groups // n_nodes
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for group in range(n_groups):
        ids = np.flatnonzero(groups == group)
        for left, right in zip(ids[:-1], ids[1:]):
            adjacency[left, right] = adjacency[right, left] = 1
    for group in range(n_groups - 1):
        left = np.flatnonzero(groups == group)[-1]
        right = np.flatnonzero(groups == group + 1)[0]
        adjacency[left, right] = adjacency[right, left] = 1
    source = int(rng.integers(n_nodes) if source_node is None else source_node)
    if not 0 <= source < n_nodes:
        raise ValueError("source_node is outside graph")
    innovations = rng.normal(0, sigma, (n_points, n_nodes))
    shared = rng.normal(0, sigma * 0.35, (n_points, n_groups))
    values = np.zeros_like(innovations)
    values[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2), n_nodes)
    for time in range(1, n_points):
        values[time] = phi * values[time - 1] + innovations[time] + shared[time, groups]
    distances = _graph_distances(adjacency, source).astype(int)
    arrival = event_onset + 3 * distances
    width = 18
    for node in range(n_nodes):
        start = arrival[node]
        stop = min(start + width, n_points)
        pulse = np.sin(np.linspace(0, np.pi, stop - start))
        values[start:stop, node] += magnitude * np.exp(-0.22 * distances[node]) * pulse
    return GraphTimeSeries(values, adjacency, groups, source, event_onset, arrival)


def generate_time_series_with_extreme_events(
    n_points: int = 2_000,
    phi: float = 0.8,
    sigma: float = 1.0,
    n_extreme_events: int = 3,
    extreme_magnitude: float = 8.0,
    extreme_duration: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the one-channel series used by early versions of the notebooks.

    This compatibility API remains intentionally independent of the graph generator:
    it lets previously downloaded notebooks run while the current notebooks use
    :func:`generate_graph_time_series` for the two-component graph method.
    """
    if n_points < 100 or not -1 < phi < 1 or sigma <= 0:
        raise ValueError("Require n_points>=100, abs(phi)<1 and sigma>0")
    if n_extreme_events < 0 or extreme_duration < 1 or extreme_magnitude <= 0:
        raise ValueError("Invalid extreme-event parameters")
    warmup = max(20, extreme_duration)
    if n_extreme_events * extreme_duration > n_points - warmup:
        raise ValueError("Too many non-overlapping event points")
    rng = np.random.default_rng(seed)
    stationary_std = sigma / np.sqrt(1 - phi**2)
    values = np.empty(n_points, dtype=float)
    values[0] = rng.normal(0, stationary_std)
    innovations = rng.normal(0, sigma, n_points - 1)
    for time in range(1, n_points):
        values[time] = phi * values[time - 1] + innovations[time - 1]
    mask = np.zeros(n_points, dtype=bool)
    candidates = np.arange(warmup, n_points - extreme_duration + 1)
    starts: list[int] = []
    while len(starts) < n_extreme_events:
        if not len(candidates):
            raise ValueError("Cannot place non-overlapping events")
        start = int(rng.choice(candidates))
        starts.append(start)
        candidates = candidates[np.abs(candidates - start) >= extreme_duration]
    pulse = np.sin(np.linspace(0, np.pi, extreme_duration + 2)[1:-1])
    for start in starts:
        values[start : start + extreme_duration] += (
            extreme_magnitude * stationary_std * pulse
        )
        mask[start : start + extreme_duration] = True
    return values, mask


# The original task used the singular spelling; keep both public spellings.
generate_time_series_with_extreme_event = generate_time_series_with_extreme_events
